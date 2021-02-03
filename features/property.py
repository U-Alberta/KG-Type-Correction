""" 
property
"""

import pickle
import torch
import random
from torch import nn
from collections import Counter

F = nn.functional
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_vocab(dataset, col=2):
    prop_to_id = {
        'PAD': 0,
        'UNK': 1
    }
    id_to_prop = {
        0: 'PAD',
        1: 'UNK'
    }
    # build vocabulary
    counter = Counter()
    for row in dataset:
        props = row[col]
        counter.update(props)
    props_to_keep = [x[0] for x in counter.most_common() if x[1] > 20]
    print(props_to_keep[:10])

    cur_id = 1
    for c in props_to_keep:
        cur_id += 1
        prop_to_id[c] = cur_id
        id_to_prop[cur_id] = c
    return prop_to_id, id_to_prop


class Dataloader:
    _dataset = None
    all_types = None
    type_to_id = None
    id_to_type = None
    tokenizer = None
    prop_to_id = None
    id_to_prop = None


    def __init__(self, split='train', batchsize=128):
        self.batchsize = batchsize
        if Dataloader._dataset is None:
            with open('data/raw-prop-features.pkl', 'rb') as f:
                Dataloader._dataset = pickle.load(f)
            Dataloader.all_types = list(set(x[1] for x in Dataloader._dataset))
            Dataloader.all_types.sort()
            Dataloader.type_to_id = {x: i for i, x in enumerate(Dataloader.all_types)}
            Dataloader.id_to_type = {i: x for x, i in self.type_to_id.items()}

            Dataloader.prop_to_id, Dataloader.id_to_prop = build_vocab(Dataloader._dataset)
        
        split_point = int(len(Dataloader._dataset) * 0.9)
        if split == 'train':
            self.pairs = Dataloader._dataset[:split_point]
        else:
            self.pairs = Dataloader._dataset[split_point:]
        
        self.i = 0
        self.max_iterations = (len(self.pairs) + batchsize - 1) // batchsize

    def __len__(self):
        return self.max_iterations
    
    def __iter__(self):
        random.shuffle(self.pairs)
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= self.max_iterations:
            raise StopIteration
        
        xs = []
        offsets = []
        ys = []
        pairs = self.pairs[self.i * self.batchsize: (self.i + 1) * self.batchsize]
        for _, typename, props in pairs:
            ids = [Dataloader.prop_to_id.get(c, 1) for c in props]
            offsets.append(len(xs))
            xs.extend(ids)
            ys.append(self.type_to_id[typename])
        
        self.i += 1

        return torch.tensor(xs), torch.tensor(offsets), torch.tensor(ys)
    

class Model(nn.Module):
    def __init__(self, num_props, num_classes, feature_only=False):
        HIDDEN_SIZE = 256
        super().__init__()
        self.feature_only = feature_only
        self.num_props = num_props
        self.num_classes = num_classes
        
        self.embedding = nn.EmbeddingBag(self.num_props, HIDDEN_SIZE)
        self.hidden1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.hidden2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.classifier = nn.Linear(HIDDEN_SIZE, num_classes)
    
    def forward(self, x, offsets):
        x = self.embedding(x, offsets)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        if self.feature_only:
            return x
        else:
            return self.classifier(x)

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

def validate(model):
    loader = Dataloader(split='val')
    val_loss = 0.
    n = 0
    criterion = nn.CrossEntropyLoss()

    for xs, offsets, ys in loader:
        xs, offsets, ys = xs.to(device), offsets.to(device), ys.to(device)
        logits = model(xs, offsets)
        loss = criterion(logits, ys)
        val_loss += loss.item()
        n += 1
    
    print(f"val loss: {val_loss / n}")


def train():
    LOG_FREQ = 200
    SAVE_FREQ = 2000

    loader = Dataloader()
    model = Model(len(loader.prop_to_id), len(loader.all_types)).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    validate(model)

    total_it = 0
    for ep in range(30):
        running_loss = 0
        running_acc = 0
        for i, (xs, offsets, ys) in enumerate(loader):
            xs, ys = xs.to(device), ys.to(device)
            offsets = offsets.to(device)
            total_it += 1
            optim.zero_grad()
            logits = model(xs, offsets)
            loss = criterion(logits, ys)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_acc += (ys == logits.argmax(dim=-1)).type(torch.FloatTensor).mean().item()

            if i % LOG_FREQ == LOG_FREQ-1:
                print(f"Train {i:05d}/{ep:05d}  Loss {running_loss / LOG_FREQ:.4f}  Acc {running_acc / LOG_FREQ: .4f}")
                running_acc = 0
                running_loss = 0.

        model.save('data/property-model.pyt')
        validate(model)

if __name__ == '__main__':
    train()
