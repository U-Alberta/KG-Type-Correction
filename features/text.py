import pickle
import torch
import random
from transformers import BertTokenizer, BertModel
from torch import nn
from tqdm import tqdm

F = nn.functional
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Dataloader:
    _dataset = None
    all_types = None
    type_to_id = None
    id_to_type = None
    tokenizer = None

    def __init__(self, split='train', batchsize=32, shuffle=True):
        self.batchsize = batchsize
        self.shuffle = shuffle
        if Dataloader._dataset is None:
            with open('data/raw-text-features.pkl', 'rb') as f:
                Dataloader._dataset = pickle.load(f)
            Dataloader.all_types = list(set(x[1] for x in Dataloader._dataset))
            Dataloader.all_types.sort()
            Dataloader.type_to_id = {x: i for i, x in enumerate(Dataloader.all_types)}
            Dataloader.id_to_type = {i: x for x, i in self.type_to_id.items()}
            Dataloader.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        split_point = int(len(Dataloader._dataset) * 0.99)
        if split == 'train':
            self.pairs = Dataloader._dataset[:split_point]
        elif split == 'val':
            self.pairs = Dataloader._dataset[split_point:]
        else:
            self.pairs = Dataloader._dataset
        
        self.i = 0
        self.max_iterations = (len(self.pairs) + batchsize - 1) // batchsize

    def __len__(self):
        return self.max_iterations
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.pairs)
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= self.max_iterations:
            raise StopIteration
        
        ys = []
        pairs = self.pairs[self.i * self.batchsize: (self.i + 1) * self.batchsize]
        for _, typename, _ in pairs:
            ys.append(self.type_to_id[typename])
        
        texts = [x[2] for x in pairs]
        encoded = Dataloader.tokenizer.batch_encode_plus(
                texts,
                max_length=64,
                pad_to_max_length=True
        )
        masks = encoded['attention_mask']
        xs = encoded['input_ids']
        self.i += 1
        return torch.tensor(xs), torch.tensor(masks), torch.tensor(ys, dtype=torch.long)


class Model(nn.Module):
    def __init__(self, num_classes, feature_only=False, preload=False):
        super().__init__()
        self.feature_only = feature_only
        self.num_classes = num_classes


        if preload:
            self.preload = torch.load('features/data/text-features.pkl')
        else:
            self.preload = None
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, x, masks, idxs=None):
        if self.preload is not None:
            return self.preload[idxs].to(device)
        if self.feature_only:
            with torch.no_grad():
                x = self.bert(x, attention_mask=masks)[1]
            return x
        else:
            x = self.bert(x, attention_mask=masks)[1]
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

    for xs, masks, ys in loader:
        xs, masks, ys = xs.to(device), masks.to(device), ys.to(device)
        logits = model(xs, masks)
        loss = criterion(logits, ys)
        val_loss += loss.item()
        n += 1
    
    print(f"val loss: {val_loss / n}")
 
def train():
    LOG_FREQ = 200
    SAVE_FREQ = 2000

    loader = Dataloader()
    model = Model(len(loader.all_types)).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)

    validate(model)

    total_it = 0
    for ep in range(2):
        running_loss = 0
        running_acc = 0
        for i, (xs, masks, ys) in enumerate(loader):
            xs, ys = xs.to(device), ys.to(device)
            masks = masks.to(device)
            total_it += 1
            optim.zero_grad()
            logits = model(xs, masks)
            loss = criterion(logits, ys)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_acc += (ys == logits.argmax(dim=-1)).type(torch.FloatTensor).mean().item()

            if i % LOG_FREQ== LOG_FREQ-1:
                print(f"Train {i:05d}/{ep:05d}  Loss {running_loss / LOG_FREQ:.4f}  Acc {running_acc / LOG_FREQ: .4f}")
                running_acc = 0
                running_loss = 0.
            if i % SAVE_FREQ == SAVE_FREQ - 1:
                model.save('data/text-model.pyt')
        
        validate(model)

def export_features():
    loader = Dataloader(split='all', shuffle=False, batchsize=2048)
    model = Model(len(loader.all_types)).to(device)
    model.load('data/text-model.pyt')
    model.feature_only = True
    model.eval()

    features = []
    for i, (xs, masks, _) in tqdm(enumerate(loader), total=len(loader)):
        xs, masks = xs.to(device), masks.to(device)
        with torch.no_grad():
            feat = model(xs, masks).detach().cpu().tolist()
            features.extend(feat)
    features = torch.tensor(features)
    with open('data/text-features.pkl', 'wb') as f:
        torch.save(features, f)


if __name__ == '__main__':
    train()
    export_features()
