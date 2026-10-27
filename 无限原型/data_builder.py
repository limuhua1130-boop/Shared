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

def prepare_data(raw_root, cache, cfg):
    raw_root = Path(raw_root).expanduser().resolve()
    cache = Path(cache).expanduser().resolve()
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError('Use a new empty cache directory; existing caches use preflight')
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / 'prepared.json'
    # Author load_dataset sorts literal subject_<integer> directory names.
    subjects = sorted(range(123), key=lambda i: 'subject_' + str(i))
    shape = (123 * 28, 32, 7500)
    X = np.lib.format.open_memmap(cache / 'X.npy', mode='w+', dtype=np.float64, shape=shape)
    rows, raw_files = [], []
    for pos, subject in enumerate(subjects):
        source = inside(raw_root / ('sub%03d.pkl' % subject), raw_root)
        with source.open('rb') as stream:
            a = pickle.load(stream)  # Known local scientific input, never an arbitrary downloaded pickle.
        if not isinstance(a, np.ndarray) or a.shape != (28, 32, 7500):
            raise ValueError('Expected ndarray[28,32,7500]: ' + str(source))
        if a.shape != (28, 32, 7500) or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError('Unexpected shape/dtype/nonfinite raw data: ' + str(source))
        aligned = euler_align(a)
        if not np.isfinite(aligned).all():
            raise ValueError('EA produced nonfinite values; no covariance regularization or sample removal')
        # Author demo assigns EA directly to the original float64 X, including
        # NumPy's complex-to-real cast. Preserve that operation, not a new EA rule.
        real_norm = float(np.linalg.norm(aligned.real.ravel()))
        imag_norm = float(np.linalg.norm(aligned.imag.ravel()))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            X[pos*28:(pos+1)*28] = aligned
        assigned = X[pos*28:(pos+1)*28]
        assignment_finite = bool(np.isfinite(assigned).all())
        # NumPy1.26 author assignment was independently verified by03 against
        # aligned.real. Compare stored float64 bits, including signed zero.
        expected_real = np.asarray(aligned.real, dtype=np.float64)
        equals_real_bits = bool(np.array_equal(assigned.view(np.uint64), expected_real.view(np.uint64)))
        assignment = dict(subject=subject, aligned_dtype=str(aligned.dtype),
                          aligned_is_complex=bool(np.iscomplexobj(aligned)),
                          aligned_real_max_abs=float(np.max(np.abs(aligned.real))),
                          aligned_imag_max_abs=float(np.max(np.abs(aligned.imag))),
                          aligned_imag_to_real_L2_ratio=(imag_norm/real_norm if real_norm > 0 and np.isfinite(real_norm) and np.isfinite(imag_norm) else None),
                          ratio_undefined_or_norm_overflow=not (real_norm > 0 and np.isfinite(real_norm) and np.isfinite(imag_norm)),
                          assignment_dtype=str(assigned.dtype), assignment_finite=assignment_finite,
                          assignment_equals_real_bits=equals_real_bits,
                          warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                          policy='author direct NumPy assignment into float64; no imaginary threshold')
        # Persist every subject immediately, including a failing post-cast check.
        write_json(cache / ('ea_assignment_subject_%03d.json' % subject), assignment)
        if not assignment_finite or not equals_real_bits:
            raise ValueError('EA float64 assignment failed finite/bitwise storage check: ' + str(subject))
        raw_files.append(dict(subject=subject, path=str(source), sha256=sha(source),
                              dtype=str(a.dtype), shape=list(a.shape), ea_assignment=assignment))
        for video in range(28):
            rows.append(dict(index=len(rows), subject=subject, session=1, video=video,
                             window='whole_trial', start=0, end=7500, label=int(LABELS[video]),
                             sample_id='s%03d-session1-video%02d-whole' % (subject, video)))
        X.flush()
        print('prepared subject %03d (%d/123)' % (subject, pos+1), flush=True)
    del X
    domain_names = [str(r['subject']) + '_1' for r in rows]
    encoder = LabelEncoder().fit(domain_names)
    domains = encoder.transform(domain_names)
    labels = np.array([r['label'] for r in rows])
    subject_array = np.array([r['subject'] for r in rows])
    for r, d in zip(rows, domains):
        r['domain'] = int(d)
    write_json(cache / 'ids.json', rows)
    # Explicit reproducibility adaptation; original demo left NumPy unseeded.
    rng = np.random.RandomState(cfg['numpy_split_seed'])
    folds = []
    dummy = np.zeros((len(rows), 1))
    for fold, (fit, test) in enumerate(GroupKFold(10).split(dummy, labels, subject_array)):
        split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=rng)
        tr, va = next(split.split(dummy[fit], labels[fit]))
        train, val = fit[tr], fit[va]
        assert not set(subject_array[test]) & set(subject_array[fit])
        assert not set(train) & set(val)
        assert len(set(train) | set(val) | set(test)) == len(rows)
        assert set(labels[train]) == set(range(9))
        folds.append(dict(fold=fold, train=train.tolist(), val=val.tolist(), test=test.tolist(),
                          train_subjects=sorted(set(subject_array[train].tolist())),
                          target_subjects=sorted(set(subject_array[test].tolist()))))
    write_json(cache / 'splits.json', folds)
    write_json(marker, dict(raw_root=str(raw_root), raw_files=raw_files, shape=list(shape),
                           domain_classes=encoder.classes_.tolist(), data_protocol_sha256=sha(Path(__file__).parent / 'data_protocol.json'),
                           source_preprocessing_evidence='local raw ndarray; official preprocessing equivalence unverified',
                           sampling_hz_author=250, sampling_hz_file_metadata=None, units=None,
                           applied_filter=None, applied_resampling=None, segmentation='none; whole trial',
                           EA='official euler_align plus author direct float64 assignment, full subject/session before splits',
                           cache_hashes={p: sha(cache / p) for p in ['X.npy', 'ids.json', 'splits.json']}))
    return cache
