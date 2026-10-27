"""Whole-trial FACED adapter; preserves official numerical EA and split behavior."""
from pathlib import Path
import pickle
import warnings
import numpy as np
import torch
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from nets.functionals import euler_align
from nets.utils.data import DomainDataset, StratifiedDomainDataLoader
from common import inside, sha, write_json, read_json

LABELS = np.repeat(np.arange(9), [3, 3, 3, 3, 4, 3, 3, 3, 3]).astype(np.int64)

class TrialDataset(DomainDataset):
    def __init__(self, cache, indices):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.array = np.load(cache / 'X.npy', mmap_mode='r')
        all_rows = read_json(cache / 'ids.json')
        self.rows = [all_rows[int(i)] for i in self.indices]
        self.labels = torch.tensor([r['label'] for r in self.rows], dtype=torch.int64)
        self.domains = torch.tensor([r['domain'] for r in self.rows], dtype=torch.int64)
        self.pending_ids = []
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        self.pending_ids.append(int(self.indices[index]))
        return [dict(inputs=torch.from_numpy(np.array(self.array[self.indices[index]], copy=True)),
                     domains=self.domains[index]), self.labels[index]]

class RecordingLoader:
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
        self.batch_ids = []
    def __iter__(self):
        self.dataset.pending_ids.clear()
        for batch in self.loader:
            self.batch_ids = self.dataset.pending_ids.copy()
            self.dataset.pending_ids.clear()
            yield batch
    def __len__(self):
        return len(self.loader)

def loaders(cache, fold):
    train, val = TrialDataset(cache, fold['train']), TrialDataset(cache, fold['val'])
    lt = StratifiedDomainDataLoader(train, batch_size=50, domains_per_batch=5, shuffle=True, drop_last=False)
    lv = torch.utils.data.DataLoader(val, batch_size=len(val), num_workers=0)
    return RecordingLoader(lt), RecordingLoader(lv)

