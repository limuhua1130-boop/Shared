"""Offline plots from committed logs, plus an isolated CPU child launcher."""
from pathlib import Path
import json
import os
import subprocess
import sys
from loss_logging import export_logs, write_json


def plot_loss(run_dir, output):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    logs = run_dir / 'loss_logs'
    if output == logs or logs in output.parents or output == run_dir:
        raise ValueError('Plot output must not overwrite the source log directory')
    batches, epochs, selection = export_logs(run_dir, output)
    # Export remains useful if the plotting dependency is missing.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError('CSV/JSONL exported, but plot requires matplotlib==3.9.2; install requirements-plot.txt') from e
    missing = []

    def series(ax, rows, x, key, label, **style):
        values = [r.get(key) for r in rows]
        absent = sum(v is None for v in values)
        if absent:
            missing.append(dict(field=key, missing=absent, rows=len(rows)))
        if absent == len(values):
            ax.text(.04, .90 - .09 * len(ax.texts), label + ': not computed / missing',
                    transform=ax.transAxes, fontsize=9)
            return
        ax.plot(x, [float('nan') if v is None else v for v in values], label=label, **style)

    files = []
    for resolution, train_rows, train_x in (
            ('epoch', epochs, [r['epoch'] + 1 for r in epochs]),
            ('step', batches, [r['global_step'] for r in batches])):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
        main, raw, ev, lr = axes.flat
        fig.suptitle('D1/D2 loss observations | committed lineage | ' + run_dir.name, fontsize=14)
        for key, label, color in [('total_loss', 'Total = CE + weighted HHSW', '#173f5f'),
                                  ('ce', 'CE', '#007f78'), ('hhsw_weighted', '0.01 x raw HHSW', '#ca6f1e')]:
            series(main, train_rows, train_x, key, label, color=color, linewidth=1.3)
        main.set_title('Train mode: ' + ('optimizer-step equal mean' if resolution == 'epoch' else 'every optimizer step'))
        series(raw, train_rows, train_x, 'hhsw_raw', 'Raw HHSW (domain sum)', color='#7952a1', linewidth=1.2)
        raw.set_title('Raw HHSW | separate scale')
        ex = [r['epoch'] + 1 for r in epochs]
        series(ev, epochs, ex, 'train_eval_ce', 'train_eval CE', color='#007f78')
        series(ev, epochs, ex, 'sourceval_ce', 'sourceval CE', color='#c03d56')
        ev.set_title('Existing eval forwards: CE only')
        ev.text(.02, .02, 'Eval HHSW / total: not computed', transform=ev.transAxes, fontsize=9)
        group_ids = sorted({g['group'] for b in batches for g in b.get('optimizer_groups', [])})
        empty = []
        for gid in group_ids:
            pairs = [(b['global_step'], g) for b in batches for g in b.get('optimizer_groups', []) if g['group'] == gid]
            if all(g['empty'] for _, g in pairs):
                empty.append(gid)
                continue
            lr.plot([x for x, _ in pairs], [g['lr'] if not g['empty'] else float('nan') for _, g in pairs],
                    label=f'group {gid}: actual LR', linewidth=1.4)
        if empty:
            lr.text(.02, .02, 'Empty groups (0 parameters): ' + ', '.join(map(str, empty)), transform=lr.transAxes, fontsize=9)
        if not group_ids:
            lr.text(.05, .5, 'Optimizer LR: missing', transform=lr.transAxes)
            missing.append(dict(field='optimizer_groups', missing=len(batches), rows=len(batches)))
        lr.set_title('Actual LR before update | nonempty groups')
        for ax in (main, raw):
            ax.set_xlabel('Epoch (1-based)' if resolution == 'epoch' else 'Global optimizer step')
            ax.set_ylabel('Loss')
        ev.set_xlabel('Epoch (1-based)'); ev.set_ylabel('CE')
        lr.set_xlabel('Global optimizer step'); lr.set_ylabel('Learning rate')
        for ax in axes.flat:
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=8, loc='best')
        try:
            for extension in ('png', 'pdf'):
                path = output / f'loss_{resolution}.{extension}'
                fig.savefig(path, dpi=160)
                files.append(path.name)
        finally:
            plt.close(fig)
    result = dict(status='success', matplotlib=matplotlib.__version__, files=files,
                  batches=len(batches), epochs=len(epochs), missing=missing,
                  source_selection='selection.json', no_model_or_torch_import=True,
                  CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
                  eval_hhsw_total='not_computed', warning_count=len(selection['warnings']))
    write_json(output / 'plot_status.json', result)
    return result


def automatic_plot(run_dir):
    """Parent uses stdlib only; failures cannot change successful train/score exits."""
    run_dir = Path(run_dir).resolve()
    output = run_dir / 'loss_plots'
    command = [sys.executable, str(Path(__file__).with_name('share.py')), 'plot-loss',
               '--run-dir', str(run_dir), '--output', str(output)]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'plot_process.log').open('a', encoding='utf-8') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        status = dict(status='success' if result.returncode == 0 else 'failed',
                      returncode=result.returncode, command=command, CUDA_VISIBLE_DEVICES='',
                      training_exit_unchanged=True)
        write_json(output / 'automatic_plot_status.json', status)
        if result.returncode:
            print('Loss plot failed; training succeeded. See ' + str(output / 'plot_process.log'), file=sys.stderr, flush=True)
    except Exception as e:
        # Reporting a plot failure must itself never hide the training result.
        print(f'Loss plot failed; training result unchanged: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
        try:
            write_json(output / 'automatic_plot_status.json', dict(status='failed', error=repr(e),
                       command=command, CUDA_VISIBLE_DEVICES='', training_exit_unchanged=True))
        except Exception as report_error:
            print(f'Cannot write plot failure status: {report_error}', file=sys.stderr, flush=True)
