"""Append-only loss observations. Stdlib only; never samples or recomputes losses.

JSONL is authoritative; per-segment CSV is an incremental convenience copy.
Only a checkpoint-bound committed lineage is exported as the primary curve.
"""
from pathlib import Path
import csv
import io
import json
import math
import os
import time

SCHEMA = 'loss-log-v2'
BATCH_FIELDS = ['schema', 'segment', 'epoch', 'global_step', 'sample_count',
                'ce', 'hhsw_raw', 'hhsw_weight', 'hhsw_weighted', 'total_loss',
                'hhsw_aggregation', 'domain_count', 'domain_ids', 'sample_ids',
                'sample_id_reference', 'optimizer_groups', 'mode']
EPOCH_FIELDS = ['schema', 'segment', 'epoch', 'first_step', 'global_step',
                'optimizer_steps', 'sample_count', 'ce', 'hhsw_raw',
                'hhsw_weighted', 'total_loss', 'ce_sample_weighted',
                'aggregation', 'domain_counts', 'train_eval_ce', 'train_eval_ba',
                'sourceval_ce', 'sourceval_ba', 'eval_hhsw', 'eval_total_loss',
                'eval_objective_status', 'eval_ce_aggregation']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def scalar(value):
    # Tensor observation only: detach before conversion; no arithmetic on tensors.
    if hasattr(value, 'detach'):
        value = value.detach().cpu().item()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite loss observation')
    return value


def plain_list(values):
    if hasattr(values, 'detach'):
        values = values.detach().cpu().tolist()
    return [int(v) for v in values]


def optimizer_groups(optimizer):
    """Call before optimizer.step(), after any existing batch callbacks."""
    return [dict(group=i, parameter_tensors=len(g['params']),
                 trainable_tensors=sum(bool(p.requires_grad) for p in g['params']),
                 empty=len(g['params']) == 0, lr=scalar(g['lr']))
            for i, g in enumerate(optimizer.param_groups)]


def csv_text(rows, fields, header=False):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise')
    if header:
        writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                         if isinstance(v, (list, dict)) else v for k, v in row.items()})
    return stream.getvalue()


def append_record(directory, kind, row, fields, durable=False):
    # Do not catch write errors: losing required observations is a run failure.
    with (directory / (kind + '.jsonl')).open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        if durable:
            os.fsync(f.fileno())
    path = directory / (kind + '.csv')
    header = not path.exists()
    with path.open('a', encoding='utf-8', newline='') as f:
        f.write(csv_text([row], fields, header))
        f.flush()
        if durable:
            os.fsync(f.fileno())


def read_jsonl(path):
    """Ignore only a non-newline-terminated final fragment; retain its bytes."""
    rows, warnings = [], []
    path = Path(path)
    if not path.exists():
        return rows, warnings
    offset = 0
    with path.open('rb') as f:
        for line in f:
            if not line.endswith(b'\n'):
                warnings.append(dict(path=str(path), byte_offset=offset,
                                     bytes=len(line), issue='incomplete_final_line_preserved'))
                break
            try:
                rows.append(json.loads(line))
            except (ValueError, UnicodeError) as e:
                raise ValueError(f'Malformed complete JSONL line at {path}:{offset}') from e
            offset += len(line)
    return rows, warnings


def aggregate_epoch(rows, train_eval, sourceval):
    if not rows or len({r['epoch'] for r in rows}) != 1:
        raise ValueError('Expected one nonempty training epoch')
    n, samples = len(rows), sum(r['sample_count'] for r in rows)
    result = {k: math.fsum(r[k] for r in rows) / n
              for k in ('ce', 'hhsw_raw', 'hhsw_weighted', 'total_loss')}
    if not math.isclose(result['total_loss'], result['ce'] + result['hhsw_weighted'],
                        rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Inconsistent training loss components')
    return dict(schema=SCHEMA, segment=rows[0]['segment'], epoch=rows[0]['epoch'],
                first_step=rows[0]['global_step'], global_step=rows[-1]['global_step'],
                optimizer_steps=n, sample_count=samples, **result,
                ce_sample_weighted=math.fsum(r['ce'] * r['sample_count'] for r in rows) / samples,
                aggregation='optimizer_step_equal_mean; NOT recomputed epoch pointcloud HHSW',
                domain_counts=sorted({r['domain_count'] for r in rows}),
                train_eval_ce=scalar(train_eval['trn_loss']), train_eval_ba=scalar(train_eval['trn_score']),
                sourceval_ce=scalar(sourceval['val_loss']), sourceval_ba=scalar(sourceval['val_score']),
                eval_hhsw=None, eval_total_loss=None, eval_objective_status='not_computed',
                eval_ce_aggregation='author test(): equal mean of batch CE; BA across concatenated labels')


class LossLogger:
    def __init__(self, run_dir, binding, next_epoch=0, step=0, resume_cursor=None):
        self.root = Path(run_dir) / 'loss_logs'
        self.root.mkdir(parents=True, exist_ok=True)
        self.parents = []
        if resume_cursor is not None:
            if resume_cursor['schema'] != SCHEMA:
                raise ValueError('Unknown checkpoint loss-log schema')
            if (resume_cursor['next_epoch'], resume_cursor['step']) != (next_epoch, step):
                raise ValueError('Loss cursor differs from checkpoint boundary')
            self.parents = resume_cursor['lineage']
            # Validate the exact retained prefix before attaching a new branch.
            collect(self.root, resume_cursor)
            write_json(self.root / 'committed.json', resume_cursor)
        elif next_epoch or step:
            raise ValueError('Checkpoint lacks loss cursor; v1 history cannot be invented')
        elif (self.root / 'committed.json').exists():
            raise FileExistsError('Fresh loss log cannot overwrite an existing committed run')
        self.segment = f'{time.time_ns()}-p{os.getpid()}'  # no Python/NumPy/Torch RNG
        self.directory = self.root / 'segments' / self.segment
        self.directory.mkdir(parents=True, exist_ok=False)
        self.start_epoch, self.start_step = next_epoch, step
        self.last_step = step
        self.pending = []
        self.cursor = resume_cursor
        write_json(self.directory / 'segment.json', dict(schema=SCHEMA, segment=self.segment,
            binding=binding, start_epoch=next_epoch, start_step=step,
            parent_cursor=resume_cursor, hhsw_aggregation='sum over unique domains; mean over 1000 projections per domain',
            sample_id_reference='integer indices into frozen cache ids.json; hashes in binding',
            csv_authority='JSONL is authoritative; CSV may lag if interrupted between writes'))

    def record_step(self, *, epoch, step, ce, hhsw, weight, weighted, total,
                    sample_ids, domains, groups, sample_count):
        if step != self.last_step + 1 or sample_count <= 0:
            raise ValueError('Loss steps must be contiguous and batches nonempty')
        if self.pending and epoch != self.pending[0]['epoch']:
            raise ValueError('Previous epoch has not been prepared')
        ids, domain_ids = plain_list(sample_ids), sorted(set(plain_list(domains)))
        if len(ids) != sample_count or not domain_ids:
            raise ValueError('Loss sample/domain identity mismatch')
        row = dict(schema=SCHEMA, segment=self.segment, epoch=int(epoch), global_step=int(step),
            sample_count=int(sample_count), ce=scalar(ce), hhsw_raw=scalar(hhsw),
            hhsw_weight=scalar(weight), hhsw_weighted=scalar(weighted), total_loss=scalar(total),
            hhsw_aggregation='sum_over_domains; mean_over_1000_projections_per_domain',
            domain_count=len(domain_ids), domain_ids=domain_ids, sample_ids=ids,
            sample_id_reference='frozen cache ids.json indices; see segment binding',
            optimizer_groups=groups, mode='native_train_before_update')
        if not math.isclose(row['total_loss'], row['ce'] + row['hhsw_weighted'], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Step total is not CE + weighted HHSW')
        append_record(self.directory, 'batches', row, BATCH_FIELDS)
        self.pending.append(row)
        self.last_step = step

    def prepare_epoch(self, train_eval, sourceval):
        row = aggregate_epoch(self.pending, train_eval, sourceval)
        # Ensure all batches are durable before a checkpoint can reference them.
        for suffix in ('jsonl', 'csv'):
            with (self.directory / ('batches.' + suffix)).open('ab') as f:
                f.flush()
                os.fsync(f.fileno())
        append_record(self.directory, 'epochs', row, EPOCH_FIELDS, durable=True)
        self.cursor = dict(schema=SCHEMA, next_epoch=row['epoch'] + 1, step=self.last_step,
            lineage=self.parents + [dict(segment=self.segment, start_epoch=self.start_epoch,
                start_step=self.start_step, last_epoch=row['epoch'], last_step=self.last_step)])
        self.pending = []
        return self.cursor

    def commit(self):
        # Called only after last.pt was atomically saved with this same cursor.
        write_json(self.root / 'committed.json', self.cursor)


def collect(root, cursor=None):
    """Read-only, bounded checkpoint lineage. Never selects the newest tail by time."""
    root = Path(root)
    if cursor is None:
        marker = root / 'committed.json'
        if not marker.exists():
            raise FileNotFoundError('No committed loss epochs; v1 gaps / partial first epoch cannot be reconstructed')
        cursor = read_json(marker)
    if cursor.get('schema') != SCHEMA or not cursor.get('lineage'):
        raise ValueError('Invalid loss cursor')
    batches, epochs, warnings, used = [], [], [], set()
    step, epoch = 0, 0
    for part in cursor['lineage']:
        name = part['segment']
        if Path(name).name != name or name in used:
            raise ValueError('Unsafe or repeated log segment')
        used.add(name)
        directory = root / 'segments' / name
        meta = read_json(directory / 'segment.json')
        if (part['start_step'], part['start_epoch']) != (step, epoch):
            raise ValueError('Noncontiguous loss lineage')
        if (meta['start_step'], meta['start_epoch']) != (step, epoch):
            raise ValueError('Segment boundary mismatch')
        bs, bw = read_jsonl(directory / 'batches.jsonl')
        es, ew = read_jsonl(directory / 'epochs.jsonl')
        warnings += bw + ew
        chosen_b = [r for r in bs if r['global_step'] <= part['last_step']]
        chosen_e = [r for r in es if r['epoch'] <= part['last_epoch']]
        if [r['global_step'] for r in chosen_b] != list(range(step + 1, part['last_step'] + 1)):
            raise ValueError('Missing or duplicated committed steps')
        if [r['epoch'] for r in chosen_e] != list(range(epoch, part['last_epoch'] + 1)):
            raise ValueError('Missing or duplicated committed epochs')
        for r in chosen_b + chosen_e:
            if r['segment'] != name or r['schema'] != SCHEMA:
                raise ValueError('Record provenance mismatch')
        for r in chosen_e:
            members = [b for b in chosen_b if b['epoch'] == r['epoch']]
            if (len(members) != r['optimizer_steps'] or not members
                    or members[0]['global_step'] != r['first_step']
                    or members[-1]['global_step'] != r['global_step']):
                raise ValueError('Epoch/step coverage mismatch')
        if not chosen_e or chosen_e[-1]['global_step'] != part['last_step']:
            raise ValueError('Cursor is not a completed epoch boundary')
        if len(chosen_b) != len(bs) or len(chosen_e) != len(es):
            warnings.append(dict(segment=name, issue='uncommitted_tail_excluded',
                                 batch_rows=len(bs)-len(chosen_b), epoch_rows=len(es)-len(chosen_e)))
        batches += chosen_b
        epochs += chosen_e
        step, epoch = part['last_step'], part['last_epoch'] + 1
    if (cursor['step'], cursor['next_epoch']) != (step, epoch):
        raise ValueError('Cursor totals mismatch')
    excluded = sorted(p.name for p in (root / 'segments').iterdir() if p.is_dir() and p.name not in used)
    return batches, epochs, dict(schema=SCHEMA, cursor=cursor, warnings=warnings,
        excluded_segments=excluded, policy='checkpoint committed prefix only; raw tails/branches retained',
        crash_window='If last.pt saved before marker update, offline view lags; --resume restores its cursor. No torch/model read by plotter.')


def export_logs(run_dir, output):
    batches, epochs, report = collect(Path(run_dir) / 'loss_logs')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for kind, rows, fields in (('batches', batches, BATCH_FIELDS), ('epochs', epochs, EPOCH_FIELDS)):
        (output / (kind + '.csv')).write_text(csv_text(rows, fields, True), encoding='utf-8', newline='')
        (output / (kind + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows), encoding='utf-8')
    write_json(output / 'selection.json', report)
    return batches, epochs, report
