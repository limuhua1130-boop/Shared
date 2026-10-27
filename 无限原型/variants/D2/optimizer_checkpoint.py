"""Restore geoopt native integer step counters after Adam's loading coercion."""
from copy import deepcopy
import torch

def exact_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.detach().cpu(), b.detach().cpu()))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact_state_equal(x, y) for x, y in zip(a, b))
    return a == b

def restore_optimizer(optimizer, saved):
    if (type(optimizer).__module__ != 'geoopt.optim.radam'
            or type(optimizer).__name__ != 'RiemannianAdam'):
        raise TypeError('Compatibility restoration is restricted to the fixed external RiemannianAdam')
    parameters_before = [[id(p) for p in g['params']] for g in optimizer.param_groups]
    step_implementation = type(optimizer).step
    # Native PyTorch loading is still responsible for device/parameter mapping.
    # Do not let load_state_dict mutate the evidence object through shallow keys.
    optimizer.load_state_dict(deepcopy(saved))
    if len(saved['param_groups']) != len(optimizer.param_groups):
        raise RuntimeError('Optimizer group count changed')
    mapping = {}
    for source_group, destination_group in zip(saved['param_groups'], optimizer.param_groups):
        if len(source_group['params']) != len(destination_group['params']):
            raise RuntimeError('Optimizer parameter group membership changed')
        for saved_id, parameter in zip(source_group['params'], destination_group['params']):
            if saved_id in mapping and mapping[saved_id] is not parameter:
                raise RuntimeError('Ambiguous optimizer parameter identity')
            mapping[saved_id] = parameter
    audit = dict(optimizer_class='geoopt.optim.radam.RiemannianAdam',
                 policy='restore checkpoint native Python int step after parent Adam load; no moment or step algorithm change',
                 step_type_restorations=[])
    for saved_id, saved_state in saved['state'].items():
        if saved_id not in mapping:
            raise RuntimeError('Optimizer state is not bound to a saved parameter')
        live = optimizer.state[mapping[saved_id]]
        if 'step' not in saved_state:
            continue
        original = saved_state['step']
        loaded = live['step']
        if type(original) is int:
            if original < 0:
                raise RuntimeError('Negative saved step')
            if isinstance(loaded, torch.Tensor):
                if (loaded.ndim != 0 or not bool(torch.isfinite(loaded))
                        or loaded.item() != original):
                    raise RuntimeError('Step numeric value changed during library loading; refusing lossy recovery')
                audit['step_type_restorations'].append(dict(parameter_state_id=saved_id,
                    saved_type='int', loaded_type='Tensor', loaded_dtype=str(loaded.dtype),
                    loaded_value=loaded.item(), restored_type='int', restored_value=original))
                # Use the original saved integer, never round a Tensor or zero the counter.
                live['step'] = original
            elif type(loaded) is not int or loaded != original:
                raise RuntimeError('Unexpected step type/value after optimizer loading')
        # Tensor-valued saved steps are not silently normalized; exact comparison
        # below rejects any changed representation or numerical value.
    if not exact_state_equal(optimizer.state_dict(), saved):
        raise RuntimeError('Optimizer strict state differs after native step-type restoration')
    if [[id(p) for p in g['params']] for g in optimizer.param_groups] != parameters_before:
        raise RuntimeError('Optimizer loading replaced or reordered parameter objects')
    if type(optimizer).step is not step_implementation:
        raise RuntimeError('Optimizer step implementation changed')
    audit['strict_full_state_equal'] = True
    audit['parameter_objects_and_order_preserved'] = True
    audit['optimizer_step_implementation_unchanged'] = True
    return audit

