"""Restore saved buffer storage relations without replacing any parameter object."""
import torch

def storage_key(tensor):
    return (str(tensor.device), tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
            tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))

def alias_groups(named_tensors):
    groups = {}
    for name, value in named_tensors.items():
        groups.setdefault(storage_key(value), []).append(name)
    return sorted(sorted(names) for names in groups.values() if len(names) > 1)

def buffer_names(model):
    state_keys = set(model.state_dict())
    return [n for n, _ in model.named_buffers(remove_duplicate=False) if n in state_keys]

def parameter_names(model):
    return [n for n, _ in model.named_parameters(remove_duplicate=False)]

def topology(model, state=None):
    state = model.state_dict() if state is None else state
    return dict(buffers=alias_groups({n:state[n] for n in buffer_names(model)}),
                parameters=alias_groups({n:state[n] for n in parameter_names(model)}))

def author_scalar_variance_transition(model, name, current, saved):
    # Author SCALAR BN initializes [1], but REFIT assigns frechet_variance's
    # scalar result directly. Restore either exact saved shape; never reshape it.
    parts = name.split('.')
    if (len(parts) != 4 or parts[:2] != ['bn', 'batchnorm']
            or not parts[2].isdigit() or not 0 <= int(parts[2]) < 123
            or parts[3] not in ('running_var', 'running_var_test')):
        return False
    module = model.get_submodule('.'.join(parts[:-1]))
    return (type(module).__module__ == 'ball_bn'
            and type(module).__name__ == 'BallDomainStats'
            and getattr(getattr(module, 'dispersion', None), 'name', None) == 'SCALAR'
            and parts[3] in module._buffers
            and current.dtype == saved.dtype == torch.float64
            and tuple(current.shape) in ((), (1,))
            and tuple(saved.shape) in ((), (1,))
            and current.numel() == saved.numel() == 1
            and bool(torch.isfinite(saved).all()))

def restore_tensors(model, saved):
    current = model.state_dict()
    if set(current) != set(saved):
        raise RuntimeError('Strict checkpoint key mismatch')
    registered_buffers = set(buffer_names(model))
    for n in current:
        if current[n].dtype != saved[n].dtype:
            raise RuntimeError('Strict checkpoint shape/dtype mismatch: ' + n)
        if current[n].shape != saved[n].shape:
            if (n not in registered_buffers
                    or not author_scalar_variance_transition(model, n, current[n], saved[n])):
                raise RuntimeError('Strict checkpoint shape/dtype mismatch: ' + n)
    before_parameters = {n:id(p) for n,p in model.named_parameters(remove_duplicate=False)}
    before_parameter_aliases = topology(model)['parameters']
    expected = topology(model, saved)
    if expected['parameters'] != before_parameter_aliases:
        raise RuntimeError('Saved parameter sharing differs; buffer repair must not change parameters')

    # One new registered buffer per saved EXACT tensor view. This keeps aliases
    # in unvisited domains and separates the trained mean/test-mean pairs.
    # No parameter replacement, no assign=True, no change to author BN operations.
    restored = {}
    registrations = {}
    for name in buffer_names(model):
        parent, _, local_name = name.rpartition('.')
        module = model.get_submodule(parent) if parent else model
        source = saved[name]
        key = (storage_key(source), str(current[name].device))
        if key not in restored:
            restored[key] = source.detach().to(device=current[name].device).clone()
        registration = (id(module), local_name)
        if registration in registrations and registrations[registration] != key:
            raise RuntimeError('Conflicting saved views for one registered buffer: ' + name)
        registrations[registration] = key
        module._buffers[local_name] = restored[key]

    model.load_state_dict(saved, strict=True)
    actual = model.state_dict()
    if any(actual[n].shape != saved[n].shape or actual[n].dtype != saved[n].dtype
           or not torch.equal(actual[n].detach().cpu(), saved[n].detach().cpu()) for n in saved):
        raise RuntimeError('Strict checkpoint tensor shape/dtype/values differ after buffer restoration')
    if topology(model) != expected:
        raise RuntimeError('Strict checkpoint buffer/parameter alias topology differs')
    if {n:id(p) for n,p in model.named_parameters(remove_duplicate=False)} != before_parameters:
        raise RuntimeError('Checkpoint restoration replaced parameter objects')

