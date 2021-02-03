import random
from torch import nn
import torch
import numpy as np
import pickle
from pprint import pprint
from transformers import BertTokenizer
import sys
import sqlite3
import numpy as np
import json

from features.property import build_vocab as build_prop_vocab
from features.surface import build_vocab as build_surface_vocab
from features.surface import Model as SurfaceModel
from features.property import Model as PropertyModel
from features.text import Model as TextModel
from vat import VATLoss

F = nn.functional
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')
conn = sqlite3.connect("data/corpus.db")

gold_label_cache = {} # entity-type pairs annotated so far
al_mode = 'scratch'


def manual_seed():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)


class Dataset(torch.utils.data.Dataset):
    _dataset = None
    type_to_id = None
    id_to_type = None
    char_to_id, id_to_type = None, None
    prop_to_id, id_to_prop = None, None
    tokenizer = None

    def __init__(self, split='train', use_gold=False, size_limit=None):
        self.use_gold = use_gold
        if Dataset._dataset is None:
            with open('features/data/raw-text-features.pkl', 'rb') as f:
                text_features = pickle.load(f)
            with open('features/data/raw-prop-features.pkl', 'rb') as f:
                prop_features = pickle.load(f)
            Dataset._dataset = []
            for idx, (tf, pf) in enumerate(zip(text_features, prop_features)):
                # name, type name, abstract, props
                Dataset._dataset.append((pf[0], pf[1], tf[2], pf[2], idx))

            Dataset.all_types = list(set(x[1] for x in Dataset._dataset))
            Dataset.all_types.sort()
            Dataset.type_to_id = {x: i for i,
                                  x in enumerate(Dataset.all_types)}
            Dataset.id_to_type = {i: x for x, i in self.type_to_id.items()}

            Dataset.char_to_id, Dataset.id_to_char = build_surface_vocab(
                Dataset._dataset)
            Dataset.prop_to_id, Dataset.id_to_prop = build_prop_vocab(
                Dataset._dataset, 3)

            Dataset.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')

        filtered_dataset = []
        # Because of the overlap and ambiguity, we merge Biomolecule w/ ChemicalSubstance, and Area w/ Place
        # Note this is applied to both train and test set.
        for x in Dataset._dataset:
            if x[1] == 'Biomolecule':
                x = (x[0], 'ChemicalSubstance', x[2], x[3], x[4])
            elif x[0].startswith('List_of_'):
                x = (x[0], 'List', x[2], x[3], x[4])
            elif x[1] == 'Area':
                x = (x[0], 'Place', x[2], x[3], x[4])
            filtered_dataset.append(x)
        Dataset._dataset = filtered_dataset
        split_point = int(len(Dataset._dataset) * 0.97)
        test_split_point = int(len(Dataset._dataset) * 0.99)
        if split == 'train':
            self.pairs = Dataset._dataset[:split_point]
        elif split == 'val':
            self.pairs = Dataset._dataset[split_point:test_split_point]
        elif split == 'test':
            self.pairs = Dataset._dataset[test_split_point:]
        elif split == 'query': # due to high computation cost, active sampling is done on this subset
            self.pairs = random.sample(Dataset._dataset[:split_point], 2000)
        elif split == 'gold': # contains pairs so far annotated in this session
            self.pairs = Dataset._dataset[:split_point]
            self.pairs = [x for x in self.pairs if x[0] in gold_label_cache]
            print(len(self.pairs))
        elif split == 'full':
            self.pairs = Dataset._dataset[:]
        elif split == 'all-gold': # contains all annotated pairs in the history
            with open('data/ssl-ds.json') as f:
                ssl_ds = json.load(f)
            for k, v in ssl_ds:
                gold_label_cache[k] = v
            self.pairs = Dataset._dataset[:split_point]
            self.pairs = [x for x in self.pairs if x[0] in gold_label_cache]

        if size_limit is not None and len(self.pairs) > size_limit:
            self.pairs = random.sample(self.pairs, size_limit)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        name, typename, text, props, idx = pair

        if self.use_gold and name in gold_label_cache:
            typename = gold_label_cache[name]

        prop_x = [Dataset.prop_to_id.get(c, 1) for c in props]
        surface_x = [Dataset.char_to_id.get(c, 1) for c in name]
        encoded = Dataset.tokenizer.encode_plus(
            text,
            max_length=64,
            pad_to_max_length=True
        )
        text_mask = encoded['attention_mask']
        text_x = encoded['input_ids']
        y = self.type_to_id[typename]

        # convert y to one-hot
        one_hot = torch.zeros(len(self.type_to_id))
        one_hot[y] = 1.

        sample = {
            'idx': idx,
            'prop_x': prop_x,
            'surface_x': surface_x,
            'text_x': text_x,
            'text_mask': text_mask,
            'y': one_hot,
            'y_idx': y,
            'name': name,
            'typename': typename
        }
        return sample


class Model(nn.Module):
    def __init__(self, num_classes, num_chars, num_props):
        self.num_classes = num_classes
        self.use_transition = False
        self.use_snm = False

        FEATURE_SIZE = 768 + 256 + 64
        HIDDEN_SIZE = 512
        super().__init__()
        # BERT weights are frozen, due to limited computation power (preload=True)
        self.text_model = TextModel(
            num_classes, feature_only=True, preload=True)
        self.prop_model = PropertyModel(
            num_props, num_classes, feature_only=True)
        self.surface_model = SurfaceModel(
            num_chars, num_classes, feature_only=True)

        self.proj = nn.Linear(FEATURE_SIZE, HIDDEN_SIZE)
        self.linears = nn.ModuleList(
            [nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(2)])

        self.classifier = nn.Linear(HIDDEN_SIZE, num_classes)
        self.simple_transition = nn.Parameter(
            10 * torch.ones(self.num_classes, ), requires_grad=True)
        self.confidence = {}

    def forward(self, prop_xs=None, prop_offsets=None, surface_xs=None, text_xs=None, text_masks=None, idxs=None, xs=None,
                feature_only=False, return_embedding=False, **kwargs):
        if xs is None:
            text_features = self.text_model(text_xs, text_masks, idxs)
            prop_features = self.prop_model(prop_xs, prop_offsets)
            surface_features = self.surface_model(surface_xs)
            x = torch.cat([text_features, prop_features,
                           surface_features], dim=-1)
        else:
            x = xs

        if feature_only:
            return x

        x = self.proj(x)
        for i, l in enumerate(self.linears):
            x = self.linears[i//2](x) + l(x)
            x = F.relu(x)

        if return_embedding:
            return x

        logits = self.classifier(x)
        return logits

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


def collate_samples(samples):
    prop_xs = []
    prop_offsets = []
    for x in samples:
        prop_offsets.append(len(prop_xs))
        prop_xs.extend(x['prop_x'])

    batch = {
        'idxs': torch.tensor([x['idx'] for x in samples]),
        'ys': torch.stack([x['y'] for x in samples]),
        'y_idxs': torch.tensor([x['y_idx'] for x in samples]),
        'prop_xs': torch.LongTensor(prop_xs),
        'prop_offsets': torch.tensor(prop_offsets),
        'surface_xs': nn.utils.rnn.pad_sequence([torch.tensor(x['surface_x']) for x in samples]),
        'text_xs': torch.tensor([x['text_x'] for x in samples]),
        'text_masks': torch.tensor([x['text_mask'] for x in samples]),
        'names': [x['name'] for x in samples],
        'typenames': [x['typename'] for x in samples]
    }
    return batch


def move_batch(batch):
    for k, v in batch.items():
        if k not in ('names', 'typenames'):
            batch[k] = v.to(device)


def validate(model):
    model.eval()
    dataset = Dataset(split='val')
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, num_workers=0, collate_fn=collate_samples)
    val_loss = 0.
    val_acc = 0
    n = 0
    criterion = nn.BCEWithLogitsLoss()

    for batch in loader:
        move_batch(batch)
        ys = batch['ys']
        logits = model(**batch)
        loss = criterion(logits, batch['ys'])
        val_loss += loss.item()

        ys_pred = logits.argmax(dim=-1)
        val_acc += (ys_pred == batch['y_idxs']).sum().item()

        n += len(ys)

    val_loss /= len(loader)
    val_acc /= n
    print(f"Val loss {val_loss}")
    print(f"Val acc  {val_acc}")

    model.train()


def export():
    dataset = Dataset(split='test')
    loader = torch.utils.data.DataLoader(
        dataset, 128, shuffle=True, num_workers=0, collate_fn=collate_samples)
    model = Model(
        num_classes=len(dataset.type_to_id),
        num_props=len(dataset.prop_to_id),
        num_chars=len(dataset.char_to_id)
    ).to(device)

    model.load(f"checkpoints/{model_name}.pyt")
    predictions = []  # name, typename, cor?
    for batch in loader:
        move_batch(batch)
        y_idxs = batch['y_idxs']
        logits = model(**batch)
        for name, tname, y_idx, logit in zip(batch['names'], batch['typenames'], y_idxs, logits):
            pred = int(logit[y_idx] < 0)
            predictions.append((name, tname, pred))
    with open(f'results/{model_name}_predictions.tsv', 'w') as f:
        for row in predictions:
            f.write("\t".join(map(str, row)))
            f.write("\n")
    # evaluate
    with open('data/test-set.tsv') as f:
        annotations = [x.split('\t') for x in f][1:]
    annotations = [x for x in annotations if x[3]]
    annotation_map = dict([(x[0], int(x[2] != x[3])) for x in annotations])

    tp = 0
    fp = 0
    fn = 0
    for name, tname, pred in predictions:
        if name not in annotation_map:
            continue
        if pred == 1 and annotation_map[name] == 1:
            tp += 1
        elif pred == 1 and annotation_map[name] == 0:
            fp += 1
        elif pred == 0 and annotation_map[name] == 1:
            fn += 1
    print("Precision", tp / (tp + fp))
    print("Recall", tp / (tp + fn))


def rank_uncertainty(model):
    dataset = Dataset('query')
    loader = torch.utils.data.DataLoader(
        dataset, 512, shuffle=True, num_workers=0, collate_fn=collate_samples)

    crit = torch.nn.BCEWithLogitsLoss(reduction='none')

    examples = []

    model.eval()
    for _, batch in enumerate(loader):
        move_batch(batch)
        logits = model(**batch)

        probs = logits.sigmoid()
        loss = crit(logits, probs).mean(dim=-1)

        for x, tname, loss in zip(batch['names'], batch['typenames'], loss.tolist()):
            if x not in gold_label_cache:
                examples.append((x, tname, loss))
    model.train()
    examples.sort(key=lambda x: (x[-1], random.random()), reverse=True)
    return examples


def rank_err_reduction(model):
    dataset = Dataset('query')
    loader = torch.utils.data.DataLoader(
        dataset, 32, shuffle=True, num_workers=0, collate_fn=collate_samples)

    crit = torch.nn.BCELoss(reduction='none')

    examples = []

    model.eval()
    for _, batch in enumerate(loader):
        move_batch(batch)

        is_gold_label = torch.tensor(
            [x in gold_label_cache for x in batch['names']], dtype=torch.float).to(device)
        is_gold_label = is_gold_label.view(-1, 1).repeat(1, model.num_classes)
        logits = model(**batch)
        # MAP noise model
        probs = logits.sigmoid()
        retain_probs = model.simple_transition.sigmoid()
        retain_probs = torch.max(retain_probs, is_gold_label)
        adjusted_probs = retain_probs * probs + (1-retain_probs) * (1 - probs)
        base_loss = crit(adjusted_probs * batch['ys'], batch['ys']).sum(dim=-1)

        base_grads = []
        for l in base_loss:
            l.backward(retain_graph=True)
            base_grads.append([x.grad.clone().view(-1)
                               for x in model.parameters() if x.grad is not None])

        # label loss
        probs_p = (probs * batch['ys']).sum(dim=-1)
        labelled_loss = probs_p * (crit(torch.ones_like(probs) * batch['ys'], batch['ys']).sum(dim=-1)) \
            + (1-probs_p) * (crit(torch.zeros_like(probs)
                                  * batch['ys'], batch['ys']).sum(dim=-1))

        labelled_grads = []
        for l in labelled_loss:
            l.backward(retain_graph=True)
            labelled_grads.append([x.grad.clone().view(-1)
                                   for x in model.parameters() if x.grad is not None])

        # score examples
        for x, tname, bgrad, lgrad in zip(batch['names'], batch['typenames'], base_grads, labelled_grads):
            score = 0.
            for bg, lg in zip(bgrad, lgrad):
                score += (lg - bg).pow(2).sum()
            score = score ** 0.5
            if x not in gold_label_cache:
                examples.append((x, tname, score))
    model.train()
    examples.sort(key=lambda x: (x[-1], random.random()), reverse=True)
    return examples


def run_query(model, size, strategy='uncertainty'):
    print("Ranking examples")
    assert strategy in ('uncertainty', 'err_reduction')

    if strategy == 'uncertainty':
        examples = rank_uncertainty(model, size)
    else:
        examples = rank_err_reduction(model, size)

    samples = examples[:size]
    i = 0
    for ent, tname, _ in samples:
        if len(gold_label_cache) >= 200:
            return
        print(f"{i} / {len(samples)}")
        i += 1
        label = query_single(ent, tname)
        gold_label_cache[ent] = label


def finetune(model):
    print("Start finetuning")
    dataset = Dataset(split='gold', use_gold=True)
    loader = torch.utils.data.DataLoader(
        dataset, 128, shuffle=True, num_workers=0, collate_fn=collate_samples)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optim = torch.optim.Adam(model.parameters())
    model.train()
    vat_loss = VATLoss()
    for _, batch in enumerate(loader):
        move_batch(batch)
        optim.zero_grad()

        xs = model(**batch, feature_only=True)
        lds = vat_loss(model, xs)
        logits = model(xs=xs)
        pos_weight = 1. + (batch['ys'] * 8.)
        loss = (pos_weight * criterion(logits, batch['ys'])).mean() + .1 * lds
        loss.backward()
        optim.step()

        del batch
    validate(model)


def train():
    LOG_FREQ = 10
    VAL_FREQ = 2000
    QUERY_FREQ = 400
    QUERY_SIZE = 20

    with open('data/priors.pkl', 'rb') as f:
        priors = pickle.load(f)
    mean_prior = np.mean(list(priors.values()))

    dataset = Dataset(use_gold=True)
    loader = torch.utils.data.DataLoader(
        dataset, 128, shuffle=True, num_workers=0, collate_fn=collate_samples)
    model = Model(
        num_classes=len(dataset.type_to_id),
        num_props=len(dataset.prop_to_id),
        num_chars=len(dataset.char_to_id)
    ).to(device)

    criterion = nn.BCELoss(reduction='none')
    optim = torch.optim.Adam(model.parameters())
    vat_loss = VATLoss()

    total_it = 0
    for ep in range(3):
        model.train()
        running_loss = 0
        running_acc = 0

        for i, batch in enumerate(loader):
            if total_it % QUERY_FREQ == 0:
                print(
                    f"Querying {len(gold_label_cache)} - {len(gold_label_cache) + QUERY_SIZE}")

                run_query(model, QUERY_SIZE)
                for _ in range(2):
                    finetune(model) # fine-tune on annotated pairs

            move_batch(batch)

            batch_priors = [
                0.5 + priors.get(n, mean_prior)/2 for n in batch['names']]
            batch_priors = torch.tensor(
                batch_priors, dtype=torch.float).view(-1, 1).repeat(1, model.num_classes)

            is_gold_label = torch.tensor(
                [x in gold_label_cache for x in batch['names']], dtype=torch.float).to(device)
            is_gold_label = is_gold_label.view(-1,
                                               1).repeat(1, model.num_classes)

            batch_priors = torch.max(0.8 * batch_priors, 2. * is_gold_label)

            total_it += 1
            optim.zero_grad()

            xs = model(**batch, feature_only=True)

            lds = vat_loss(model, xs)
            logits = model(xs=xs)
            # MAP noise model
            probs = logits.sigmoid()
            retain_probs = model.simple_transition.sigmoid()
            retain_probs = torch.max(retain_probs, is_gold_label)
            adjusted_probs = retain_probs * probs + \
                (1-retain_probs) * (1 - probs)
            # noise model + prior
            loss = (batch_priors * criterion(adjusted_probs,
                                             batch['ys'])).mean() + .1 *lds
            # noise model
            # loss = (criterion(adjusted_probs, batch['ys'])).mean()
            # vanilla
            # loss = (criterion(probs, batch['ys'])).mean() + .1 * lds
            loss.backward()
            optim.step()

            running_loss += loss.item()

            del batch

            if i % LOG_FREQ == LOG_FREQ-1:
                print(
                    f"Train {i:05d}/{ep:05d}  Loss {running_loss / LOG_FREQ:.4f}  Acc {running_acc / LOG_FREQ: .4f}")
                running_acc = 0
                running_loss = 0.

            if i % VAL_FREQ == VAL_FREQ-1:
                validate(model)
        validate(model)
        save_path = f'checkpoints/{model_name}_{ep}.pyt'
        print(f'Save to {save_path}')
        model.save(save_path)


def query_single(entity, orig_label):
    with conn:
        record = conn.execute(
            "SELECT gold_label FROM annotations WHERE name=?", [entity]).fetchone()
        if record:
            return record[0]
    with conn:
        abstract = conn.execute("SELECT abstract FROM entities WHERE name=?", [
                                entity]).fetchone()[0]
        print(f"NAME: {entity}")
        print(f"TYPE: {orig_label}")
        print(f"ABST: {abstract}")

        corr = input("CORR? ")
        label = orig_label
        if corr != 'y':
            label = input("ANNO? ")
            while label not in Dataset.all_types:
                label = input("ANNO?")

        conn.execute(
            "INSERT INTO annotations (name, orig_label, gold_label) VALUES (?, ?, ?)",
            (entity, orig_label, label)
        )
        return label


def obtain_prior_confidence():
    import pickle
    from sklearn.metrics.pairwise import cosine_similarity
    dataset = Dataset(split='full')
    with open('data/glove-trimmed.pkl', 'rb') as f:
        glove = pickle.load(f)
    i = 0
    type_to_glove = {
        "chemicalsubstance": "chemical",
        "meanoftransportation": "transportation"
    }
    priors = {}
    for row in dataset:
        i += 1
        entity, typename = row['name'], row['typename']
        with conn:
            record = conn.execute(
                "SELECT hypernym FROM hypernyms WHERE entity=?", (entity,)).fetchone()
            if record:
                hypernym = record[0]
                typename = type_to_glove.get(
                    typename.lower(), typename.lower())
                typeemb = glove.get(typename)
                hypernymemb = glove.get(hypernym)
                if not (hypernymemb is None or typeemb is None):
                    prior = cosine_similarity(typeemb.reshape(
                        1, -1), hypernymemb.reshape(1, -1))[0][0]
                    priors[entity] = prior
    with open('data/priors.pkl', 'wb') as f:
        pickle.dump(priors, f)


if __name__ == '__main__':
    import sys

    print('Usage: python3', sys.argv[0], 'model-name [eval]')

    model_name = sys.argv[1]
    manual_seed()
    if 'eval' in sys.argv:
        print("Start evaluation...")
        export()
    else:
        print("Start training...")
        train()
