"""Author training steps reused verbatim; outer lifecycle adds recovery and audit."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from model import HEEGNet
from bmlr_head import attach_head
from nets.trainer import Trainer
from nets.callbacks import EarlyStopping, MomentumBatchNormScheduler
from nets.batchnorm import BatchNormDispersion
from common import (sha, write_json, read_json, capture_rng, restore_rng, save_pt, load_pt,
                    bn_extra, restore_bn_extra, finite_model, PACKAGE)
from data import TrialDataset, loaders, RecordingLoader
from checkpoint import restore_tensors, topology
from optimizer_checkpoint import restore_optimizer

def build_model(cfg, device):
    model = HEEGNet(chunk_size=7500, num_electrodes=32, num_classes=9,
                   domains=torch.arange(123), bnorm_dispersion=BatchNormDispersion.SCALAR,
                   domain_adaptation=True, device=device, dtype=torch.float64,
                   **cfg['model']).to(device=device, dtype=torch.float64)
    return attach_head(model, 'lorentz')

def geometry(model, features):
    x = features.detach()
    radius = model.manifold.dist0(x)
    k = float(model.manifold.k.detach().cpu())
    residual = -x[:, 0].square() + x[:, 1:].square().sum(-1) + k
    if not torch.isfinite(x).all() or not torch.isfinite(radius).all():
        raise FloatingPointError('Nonfinite actual Lorentz geometry')
    return dict(features=x.cpu(), radius=radius.cpu(), lorentz_residual=residual.cpu(),
                k=k, sectional_curvature=-1.0/k, ROOT=[k**.5, 'remaining coordinates zero'],
                tap='forward return before official BMLR, after Lorentz pooling and flatten',
                ball_norm=(x[:,1:]/(x[:,:1]+k**.5)).norm(dim=-1).cpu(),
                projection_events=None, cones='not applicable')

def model_state(model):
    state = deepcopy(model.state_dict())
    return dict(model=state, bn_extra=bn_extra(model), alias_topology=topology(model, state),
                training_flags={name:m.training for name,m in model.named_modules()})

def restore_model(model, state):
    if 'head_initialization' in state:model.head_initialization=deepcopy(state['head_initialization'])
    if 'alias_topology' in state and topology(model, state['model']) != state['alias_topology']:
        raise RuntimeError('Serialized checkpoint alias metadata differs from stored tensors')
    restore_tensors(model, state['model'])
    restore_bn_extra(model, state['bn_extra'])
    if 'training_flags' in state:
        modules = dict(model.named_modules())
        if set(modules) != set(state['training_flags']):
            raise RuntimeError('Checkpoint module mode keys differ')
        for name, flag in state['training_flags'].items():
            modules[name].training = flag

class RestorableEarlyStopping(EarlyStopping):
    def on_fit_end(self, trainer, net):
        # Same author selection/trigger/save; only the checkpoint copy operation
        # is replaced so BN buffers recover the selected storage topology.
        path = Path(self.tempdir.name) / 'es_state_dict.pt'
        if self.counter >= 0 and path.exists():
            restore_tensors(net, load_pt(path, next(net.parameters()).device))

class RetainedTrainer(Trainer):
    """train_epoch/test are inherited unchanged from fixed author snapshot."""
    def __init__(self, cfg, device, directory, binding):
        self.es = RestorableEarlyStopping(metric='val_loss', higher_is_better=False, patience=20, verbose=False)
        self.scheduler = MomentumBatchNormScheduler(epochs=90, bs0=50, bs=10, tau0=.85)
        super().__init__(max_epochs=100, min_epochs=70, callbacks=[self.es, self.scheduler],
                         loss=torch.nn.CrossEntropyLoss(), device=device, dtype=torch.float64,
                         swd_weight=.01, lr=.001, weight_decay=.001)
        self.cfg = cfg
        self.directory = Path(directory)
        self.binding = binding
        self.directory.mkdir(parents=True, exist_ok=True)
        self.active_loader = None
        self.native_first = None
        self.native_val_first = None
        self.active_stage = 'train'

    def train_epoch(self, model, train_dataloader):
        from objective import swd_loss
        model.train()
        for batch_idx,batch in enumerate(train_dataloader):
            for cb in self.callbacks:cb.on_train_batch_start(self,model,batch,batch_idx)
            features,y=batch
            features['inputs']=features['inputs'].to(dtype=self.dtype_,device=self.device_)
            y=y.to(self.device_)
            pred,x=model(**features)
            ce=self.loss_fn(pred,y);swd=swd_loss(x,features['domains'],model.manifold)
            loss=ce+self.swd_weight*swd
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError('Nonfinite native ball gradient')
            self.optimizer.step();self.optimizer.zero_grad();self.current_step+=1
            if self.current_step<=3 or self.current_step%25==0:
                write_json(self.directory/'progress.json',dict(epoch=self.current_epoch,step=self.current_step,
                    loss=float(loss.detach()),ce=float(ce.detach()),hhsw=float(swd.detach()),
                    ids=train_dataloader.batch_ids,binding=self.binding))

    def snapshot(self, model, next_epoch, resume_allowed=True):
        return dict(**model_state(model), optimizer=deepcopy(self.optimizer.state_dict()),
                    rng=capture_rng(), epoch=self.current_epoch, next_epoch=next_epoch,
                    step=self.current_step, records=deepcopy(self.records),
                    early_stopping=dict(counter=self.es.counter, best_score=self.es.best_score,
                                        best_epoch=self.es.best_epoch), stop_fit=self.stop_fit_,
                    scheduler=dict(eta=self.scheduler.eta_, eta_test=self.scheduler.eta_test_),
                    binding=self.binding, config=self.cfg, head_initialization=model.head_initialization, resume_allowed=resume_allowed,
                    granularity='completed epoch boundary' if resume_allowed else 'diagnostic only')

    def fit_retained(self, model, train_loader, val_loader, resume=None):
        self.optimizer = model.configure_optimizers(lr=self.lr, weight_decay=self.weight_decay)
        for cb in self.callbacks:
            cb.on_fit_start(self, model)
        start = 0
        if resume is not None:
            state = load_pt(resume, self.device_)
            if state['binding'] != self.binding or not state['resume_allowed']:
                raise RuntimeError('Checkpoint binding/boundary mismatch')
            restore_model(model, state)
            self.optimizer_restore_audit = restore_optimizer(self.optimizer, state['optimizer'])
            self.current_epoch, self.current_step = state['epoch'], state['step']
            self.records, self.stop_fit_ = state['records'], state['stop_fit']
            for key, value in state['early_stopping'].items():
                setattr(self.es, key, value)
            self.scheduler.eta_ = state['scheduler']['eta']
            self.scheduler.eta_test_ = state['scheduler']['eta_test']
            selected = load_pt(self.directory / 'selected.pt', self.device_)
            torch.save(selected['model'], Path(self.es.tempdir.name) / 'es_state_dict.pt')
            restore_rng(state['rng'])
            start = state['next_epoch']
        else:
            save_pt(self.directory / 'initial.pt', self.snapshot(model, 0, False))

        def observed_forward(module, args, output):
            g=geometry(module,output[1])
            ids=self.active_loader.batch_ids.copy()
            with (self.directory/'native_geometry.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(epoch=self.current_epoch,step=self.current_step,stage=self.active_stage,
                    indices=ids,radius=g['radius'].tolist(),ball_norm=g['ball_norm'].tolist(),bmlr=module.lmlp.last_audit,
                    projection_events=g['projection_events']))+'\n')
            if module.training and self.native_first is None:
                self.native_first = dict(**g,
                                         indices=self.active_loader.batch_ids.copy(),
                                         epoch=self.current_epoch, before_update_step=self.current_step,
                                         mode='native_train')
            if self.active_stage=='val' and self.native_val_first is None:
                self.native_val_first=dict(**g,indices=ids,epoch=self.current_epoch,step=self.current_step,mode='native_val_BUFFER')
        handle = model.register_forward_hook(observed_forward)
        original_step = self.optimizer.step
        def step(*args, **kwargs):
            out = original_step(*args, **kwargs)
            if self.current_step == 0:
                finite_model(model)
                state = self.snapshot(model, 0, False)
                state['step'] = 1
                state['native_first'] = self.native_first
                save_pt(self.directory / 'first_update.pt', state)
            return out
        self.optimizer.step = step
        try:
            for epoch in range(start, self.epochs):
                if self.stop_fit_:
                    break
                self.current_epoch = epoch
                self.native_first = None
                self.native_val_first = None
                for cb in self.callbacks:
                    cb.on_train_epoch_start(self, model)
                self.active_loader = train_loader
                self.active_stage = 'train'
                self.train_epoch(model, train_loader)
                self.active_stage = 'train_eval'
                trn = {f'trn_{k}': v for k, v in self.test(model, train_loader).items()}
                self.active_loader = val_loader
                self.active_stage = 'val'
                val = {f'val_{k}': v for k, v in self.test(model, val_loader).items()}
                self.log_dict(trn); self.log_dict(val)
                finite_model(model)
                for cb in self.callbacks:
                    cb.on_train_epoch_end(self, model)
                state = self.snapshot(model, epoch + 1)
                # Each completed epoch supports native state recovery; no historical-state fabrication.
                save_pt(self.directory / ('epoch_%03d.pt' % epoch), state)
                save_pt(self.directory / 'last.pt', state)
                if self.es.best_epoch == epoch:
                    save_pt(self.directory / 'selected.pt', state)
                if self.native_first is not None:
                    save_pt(self.directory / ('native_first_epoch_%03d.pt' % epoch), self.native_first)
                if self.native_val_first is not None:
                    save_pt(self.directory / ('native_val_epoch_%03d.pt' % epoch), self.native_val_first)
                write_json(self.directory / 'history.json', self.records)
                print(json.dumps(dict(epoch=epoch, step=self.current_step, **trn, **val)), flush=True)
                if self.stop_fit_:
                    break
        finally:
            handle.remove()
            self.optimizer.step = original_step
        for cb in self.callbacks:
            cb.on_fit_end(self, model)
        # Author ES restores tensors; use matching selected non-tensor BN settings for inference/recovery.
        selected = load_pt(self.directory / 'selected.pt', self.device_)
        restore_bn_extra(model, selected['bn_extra'])

def export_geometry(cfg, cache, checkpoint, output, device, indices=None):
    state = load_pt(checkpoint, device)
    model = build_model(cfg, device)
    restore_model(model, state); model.eval()
    all_rows = read_json(cache / 'ids.json')
    if indices is None:
        if 'indices' in state:
            indices = state['indices']
        else:
            split = read_json(cache / 'splits.json')[state['binding']['fold']]
            indices = split['train'] + split['val']
    ds = TrialDataset(cache, indices)
    loader = RecordingLoader(torch.utils.data.DataLoader(ds, batch_size=4, num_workers=0))
    output.mkdir(parents=True, exist_ok=False)
    saved = []
    before = model_state(model)
    with torch.no_grad():
        for batch_ix, (f, labels) in enumerate(loader):
            logits, features = model(f['inputs'].to(device, dtype=torch.float64), f['domains'])
            name = 'batch_%05d.pt' % batch_ix
            save_pt(output / name, dict(**geometry(model, features), ids=[all_rows[i] for i in loader.batch_ids],
                                        logits=logits.cpu(), labels=labels, mode='eval_BUFFER',
                                        checkpoint_sha256=sha(checkpoint)))
            saved.append(name)
    if any(not torch.equal(v, model.state_dict()[k]) for k, v in before['model'].items()):
        raise RuntimeError('Geometry export mutated model statistics')
    write_json(output / 'index.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                                          binding=state['binding'], batches=saved, n=len(indices),
                                          mode='eval_BUFFER; no REFIT; separate source/target state',
                                          cone_metrics='not applicable'))
