"""Stdlib behavioral tests: stub objective, NOT a PyTorch/model experiment."""
from pathlib import Path
import ast
import copy
import json
import math
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loss_logging import (LossLogger, collect, read_jsonl, read_json, write_json,
                          optimizer_groups, export_logs)


class Detached:
    def __init__(self, value): self.value = value
    def cpu(self): return self
    def item(self): return self.value
    def __float__(self): return float(self.value)


class Scalar:
    """Dual-number scalar: analytic derivative; detach discards it completely."""
    def __init__(self, value, derivative=0., parameter=None):
        self.value, self.derivative, self.parameter = value, derivative, parameter
    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return Scalar(self.value + other.value, self.derivative + other.derivative, self.parameter or other.parameter)
    __radd__ = __add__
    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return Scalar(self.value * other.value,
                      self.derivative * other.value + self.value * other.derivative,
                      self.parameter or other.parameter)
    __rmul__ = __mul__
    def detach(self): return Detached(self.value)
    def backward(self): self.parameter.grad = self.derivative


class Finite:
    def __init__(self, value): self.value = math.isfinite(value.value if isinstance(value, Scalar) else value)
    def __bool__(self): return self.value
    def all(self): return self.value


class Batch:
    def __init__(self, n): self.shape = (n,)
    def to(self, *args, **kwargs): return self


class Model:
    def __init__(self):
        self.p = types.SimpleNamespace(value=1.2, grad=None, requires_grad=True)
        self.manifold, self.calls = None, 0
    def train(self): pass
    def parameters(self): return [self.p]
    def __call__(self, **features):
        self.calls += 1
        return Scalar(self.p.value, 1., self.p), Scalar(self.p.value * 2., 2., self.p)


class Optimizer:
    def __init__(self, model):
        self.p, self.gradients = model.p, []
        self.param_groups = [dict(params=[], lr=.001), dict(params=[self.p], lr=.001)]
    def step(self):
        self.gradients.append(self.p.grad)
        self.p.value -= self.param_groups[1]['lr'] * self.p.grad
    def zero_grad(self): self.p.grad = None


class Loader:
    def __iter__(self):
        for count in (3, 2, 4):
            self.batch_ids = list(range(count))
            yield dict(inputs=Batch(count), domains=list(range(count))), Batch(count)


def method(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RetainedTrainer')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'train_epoch')
    scope = dict(torch=types.SimpleNamespace(isfinite=Finite), write_json=write_json,
                 optimizer_groups=optimizer_groups)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    return scope['train_epoch']


def observe(logger, epoch, step, n=2, ce=1., hhsw=5.):
    logger.record_step(epoch=epoch, step=step, ce=ce, hhsw=hhsw, weight=.01,
        weighted=.01*hhsw, total=ce+.01*hhsw, sample_ids=list(range(n)),
        domains=[0, 1], sample_count=n,
        groups=[dict(group=0, parameter_tensors=0, trainable_tensors=0, empty=True, lr=.001),
                dict(group=1, parameter_tensors=1, trainable_tensors=1, empty=False, lr=.001)])


def finish(logger, ce=1.):
    cursor = logger.prepare_epoch(dict(trn_loss=ce, trn_score=.5), dict(val_loss=ce+.2, val_score=.4))
    logger.commit()
    return copy.deepcopy(cursor)


class LossTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_actual_d1_d2_train_method_observation_equivalence_with_stub_objective(self):
        for variant, module_name in [('D1', 'objective'), ('D2', 'ball_hhsw')]:
            outputs = []
            for name, path, logging in [
                    ('old', ROOT/'provenance/loss-v1'/f'{variant}_engine.py.txt', False),
                    ('new', ROOT/'variants'/variant/'engine.py', True)]:
                random.seed(9123)
                model = Model(); opt = Optimizer(model)
                directory = self.root / variant / name
                directory.mkdir(parents=True)
                logger = LossLogger(directory, {'synthetic': True}) if logging else None
                values = []
                def hhsw(x, domains, manifold):
                    v = x*x + random.random()
                    values.append(v.value)
                    return v
                trainer = types.SimpleNamespace(callbacks=[], dtype_='stub', device_='stub',
                    loss_fn=lambda pred, y: pred*pred, swd_weight=.01,
                    optimizer=opt, current_step=0, current_epoch=0, directory=directory,
                    binding={'synthetic': True}, loss_log=logger)
                with patch.dict(sys.modules, {module_name: types.SimpleNamespace(swd_loss=hhsw)}):
                    method(path)(trainer, model, Loader())
                outputs.append((model.p.value, opt.gradients, values, random.getstate(), model.calls))
                if logging:
                    finish(logger)
                    bs, es, report = collect(directory/'loss_logs')
                    self.assertEqual(len(bs), 3)
                    self.assertEqual([b['hhsw_raw'] for b in bs], values)
                    self.assertEqual([b['sample_count'] for b in bs], [3, 2, 4])
                    for b in bs:
                        self.assertEqual(b['total_loss'], b['ce']+b['hhsw_weighted'])
            self.assertEqual(outputs[0], outputs[1])

    def test_variable_batch_aggregation_missing_eval_and_rng(self):
        random.seed(22); before = random.getstate()
        log = LossLogger(self.root, {})
        observe(log, 0, 1, n=2, ce=1., hhsw=10.)
        observe(log, 0, 2, n=6, ce=3., hhsw=30.)
        finish(log)
        bs, es, _ = collect(log.root)
        e = es[0]
        self.assertEqual(e['ce'], 2.)
        self.assertEqual(e['ce_sample_weighted'], 2.5)
        self.assertEqual(e['hhsw_raw'], 20.)
        self.assertAlmostEqual(e['total_loss'], 2.2)
        self.assertAlmostEqual(e['total_loss'], e['ce']+e['hhsw_weighted'])
        self.assertIsNone(e['eval_hhsw']); self.assertIsNone(e['eval_total_loss'])
        self.assertEqual(e['eval_objective_status'], 'not_computed')
        self.assertEqual(bs[0]['optimizer_groups'][0]['parameter_tensors'], 0)
        self.assertEqual(before, random.getstate())

    def test_resume_replay_partial_line_and_checkpoint_marker_lag(self):
        old = LossLogger(self.root, {})
        observe(old, 0, 1); cursor0 = finish(old)
        observe(old, 1, 2, ce=999.)  # completed write, uncommitted abandoned update
        with (old.directory/'batches.jsonl').open('ab') as f: f.write(b'{"partial":')
        raw_before = (old.directory/'batches.jsonl').read_bytes()
        resumed = LossLogger(self.root, {}, next_epoch=1, step=1, resume_cursor=cursor0)
        observe(resumed, 1, 2, ce=2.)
        cursor1 = resumed.prepare_epoch(dict(trn_loss=2., trn_score=.5), dict(val_loss=2., val_score=.5))
        # Simulate last.pt saved with cursor1, process dies before marker commit.
        self.assertEqual(len(collect(resumed.root)[0]), 1)
        again = LossLogger(self.root, {}, next_epoch=2, step=2, resume_cursor=cursor1)
        bs, es, report = collect(again.root)
        self.assertEqual([b['ce'] for b in bs], [1., 2.])
        self.assertEqual([b['global_step'] for b in bs], [1, 2])
        self.assertEqual(len(es), 2)
        self.assertTrue(any(w['issue']=='incomplete_final_line_preserved' for w in report['warnings']))
        self.assertTrue(any(w['issue']=='uncommitted_tail_excluded' for w in report['warnings']))
        self.assertEqual(raw_before, (old.directory/'batches.jsonl').read_bytes())
        # Roll back to an earlier complete checkpoint; later branch stays on disk.
        branch = LossLogger(self.root, {}, next_epoch=1, step=1, resume_cursor=cursor0)
        observe(branch, 1, 2, ce=3.); finish(branch)
        bs, _, report = collect(branch.root)
        self.assertEqual([b['ce'] for b in bs], [1., 3.])
        self.assertIn(resumed.segment, report['excluded_segments'])
        export_logs(self.root, self.root/'export')
        self.assertEqual(len(read_jsonl(self.root/'export/batches.jsonl')[0]), 2)

    def test_corruption_and_missing_cursor_fail_loudly(self):
        with self.assertRaises(ValueError): LossLogger(self.root, {}, next_epoch=1, step=3)
        log = LossLogger(self.root, {})
        observe(log, 0, 1); finish(log)
        with (log.directory/'batches.jsonl').open('ab') as f: f.write(b'{bad}\n')
        with self.assertRaisesRegex(ValueError, 'Malformed complete'): collect(log.root)

    def test_write_failure_is_not_swallowed(self):
        log = LossLogger(self.root, {})
        with patch('loss_logging.append_record', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'): observe(log, 0, 1)

    def test_automatic_plot_cpu_child_and_failure_does_not_raise(self):
        from loss_plot import automatic_plot
        with patch('loss_plot.subprocess.run', return_value=types.SimpleNamespace(returncode=1)) as process:
            automatic_plot(self.root)
        self.assertEqual(process.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(read_json(self.root/'loss_plots/automatic_plot_status.json')['returncode'], 1)
        self.assertNotIn('matplotlib', sys.modules)
        self.assertNotIn('torch', sys.modules)
        with patch('loss_plot.subprocess.run', side_effect=OSError('subprocess unavailable')):
            automatic_plot(self.root)
        self.assertEqual(read_json(self.root/'loss_plots/automatic_plot_status.json')['status'], 'failed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
