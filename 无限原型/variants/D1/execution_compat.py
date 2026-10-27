"""Validated03 v3 copying bridge, unchanged operational body; no old whitelist."""
from pathlib import Path
import copy,json,sys,hashlib
ADAPTER_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FINAL_RECEIPTS=[]
def install(torch,work,manifest,mode):
    original_deepcopy=copy.deepcopy
    # ScriptFunction is immutable code, like a Python function. Parameters,
    # buffers and module objects still undergo the unmodified deep-copy path.
    copy._deepcopy_dispatch[torch.jit.ScriptFunction] = lambda value, memo: value
    counters = dict(model_clones_checked=0, model_tensor_leaves_checked=0, optimizer_copies_checked=0)
    def tensor_tree_no_alias(x, y):
        if isinstance(x, torch.Tensor):
            assert isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape
            assert torch.equal(x, y)
            if x.numel():
                assert x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr()
        elif isinstance(x, dict):
            assert isinstance(y, dict) and x.keys() == y.keys()
            for key in x: tensor_tree_no_alias(x[key], y[key])
        elif isinstance(x, (tuple, list)):
            assert type(x) is type(y) and len(x) == len(y)
            for a, b in zip(x, y): tensor_tree_no_alias(a, b)
    def checked_deepcopy(value, memo=None, *rest):
        cloned = original_deepcopy(value, memo, *rest)
        if isinstance(value, torch.nn.Module):
            assert cloned is not value
            original = dict(value.named_parameters(remove_duplicate=False))
            original.update({'buffer:'+k:v for k,v in value.named_buffers(remove_duplicate=False)})
            copied = dict(cloned.named_parameters(remove_duplicate=False))
            copied.update({'buffer:'+k:v for k,v in cloned.named_buffers(remove_duplicate=False)})
            tensor_tree_no_alias(original, copied)
            alias_map = {}
            for key, tensor in original.items():
                if not tensor.numel(): continue
                identity = (str(tensor.device), tensor.untyped_storage().data_ptr())
                copied_identity = (str(copied[key].device), copied[key].untyped_storage().data_ptr())
                assert alias_map.setdefault(identity, copied_identity) == copied_identity
            counters['model_clones_checked'] += 1
            counters['model_tensor_leaves_checked'] += len(original)
        elif isinstance(value, dict) and 'state' in value and 'param_groups' in value:
            tensor_tree_no_alias(value['state'], cloned['state'])
            counters['optimizer_copies_checked'] += 1
        return cloned
    copy.deepcopy = checked_deepcopy
    for name in ['engine', 'workflow', 'validate']:
        module = sys.modules.get(name)
        if module is not None and getattr(module, 'deepcopy', None) is original_deepcopy:
            module.deepcopy = checked_deepcopy
    comparator_patch = False
    if mode == 'algebra':
        import validate
        import head_checks
        import numpy as np
        def exact_rng_equal(x, y):
            if isinstance(x, np.ndarray):
                return isinstance(y, np.ndarray) and x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
            if type(x) is not type(y):
                return False
            if isinstance(x, dict):
                return x.keys() == y.keys() and all(exact_rng_equal(x[k], y[k]) for k in x)
            if isinstance(x, (list, tuple)):
                return len(x) == len(y) and all(exact_rng_equal(a, b) for a, b in zip(x, y))
            return validate.exact_state_equal(x, y)
        # Exact dtype/shape/value comparison, no tolerance/assertion relaxation.
        head_checks.exact_state_equal = exact_rng_equal
        comparator_patch = True
    dest = work/'execution_adapter_v2'
    dest.mkdir(exist_ok=True)
    def receipt():
        (dest/(mode+'.json')).write_text(json.dumps(dict(
            adapter_SHA256=ADAPTER_SHA, package_manifest_SHA256=manifest,
            immutable_ScriptFunction_identity_copy=True,
            diagnostic_exact_numpy_comparator=comparator_patch,
            mutable_copy_checks=counters,
            applies_to_all_run_modes=True,
            official_model_loss_optimizer_files_modified=False), indent=2)+'\n')
    receipt()
    FINAL_RECEIPTS.append(receipt)
    return receipt
