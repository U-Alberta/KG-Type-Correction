"""
Create unlabelled dataset for semi-supervised learning
"""

import pickle
from taxonomy import get_taxonomy
import random


with open('data/t2e.pkl', 'rb') as f:
    t2e = pickle.load(f)

taxonomy = get_taxonomy()
coarse_types = set()
for n in taxonomy.values():
    if n.level == 1:
        coarse_types.add(n.name)
print(coarse_types)

coarse_t2e = {}
for t, v in t2e.items():
    if t not in taxonomy:
        continue
    parent_types = taxonomy[t].get_all_parents()
    coarse_type = None
    for p in parent_types:
        if p in coarse_types:
            coarse_type = p
            break
    if coarse_type is None:
        continue
    if not coarse_type in coarse_t2e:
        coarse_t2e[coarse_type] = []
    coarse_t2e[coarse_type].extend(v)

for k, v in coarse_t2e.items():
    coarse_t2e[k] = list(set(v))

freqs = dict([(k, len(v)) for k, v in coarse_t2e.items()])
freqs['Others'] = 0

SAMPLE_PER_CLASS = 10000

coarse_t2e['Others'] = []
for k, v in freqs.items():
    if v <= SAMPLE_PER_CLASS:
        freqs['Others'] += v
        coarse_t2e['Others'].extend(coarse_t2e[k])
        del coarse_t2e[k]

all_pairs = [] # entity, type
for k, v in coarse_t2e.items():
    if len(v) > SAMPLE_PER_CLASS:
        v = random.sample(v, SAMPLE_PER_CLASS)
    for e in v:
        all_pairs.append((e, k))

random.shuffle(all_pairs)
if len(all_pairs) > 500_000:
    all_pairs = all_pairs[:500_000]

with open('coarse_dataset.pkl', 'wb') as f:
    pickle.dump(all_pairs, f)
