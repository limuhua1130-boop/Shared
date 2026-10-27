"""Local return-type bridge for unchanged author Trainer.test .item() call."""
from functools import wraps
import struct
import numpy as np

def install():
    import nets.trainer as module
    from sklearn.metrics import balanced_accuracy_score
    current=module.balanced_accuracy_score
    if getattr(current,'_heegnet_float_item_bridge',False):
        if current._original_balanced_accuracy is not balanced_accuracy_score:
            raise RuntimeError('Unexpected pre-existing metric bridge')
        return dict(installed=True,already_installed=True,scope='nets.trainer.balanced_accuracy_score only')
    if current is not balanced_accuracy_score:
        raise RuntimeError('Unknown author metric binding; do not replace it')
    @wraps(current)
    def wrapped(*args,**kwargs):
        value=current(*args,**kwargs)
        if type(value) is float:
            result=np.float64(value)
            if struct.pack('!d',result.item())!=struct.pack('!d',value):
                raise RuntimeError('Metric return-type bridge changed IEEE64 value')
            return result
        if type(value) is np.float64:
            return value
        raise TypeError('Unexpected balanced-accuracy scalar type: '+str(type(value)))
    wrapped._heegnet_float_item_bridge=True
    wrapped._original_balanced_accuracy=current
    module.balanced_accuracy_score=wrapped
    return dict(installed=True,already_installed=False,scope='nets.trainer.balanced_accuracy_score only',
                original_function=current.__module__+'.'+current.__name__,
                policy='Python float to numpy.float64 with exact IEEE64 value; no metric calculation change')

def check_values():
    import nets.trainer as module
    from sklearn.metrics import balanced_accuracy_score
    install()
    cases=[([0,1,2],[0,1,2]),([0,0,1,1,2,2],[0,1,1,2,2,0]),
           ([0,0,0,0,1,2],[0,0,1,1,1,1])]
    results=[]
    for true,pred in cases:
        original=balanced_accuracy_score(true,pred)
        wrapped=module.balanced_accuracy_score(true,pred)
        assert type(wrapped) is np.float64
        assert struct.pack('!d',wrapped.item())==struct.pack('!d',float(original))
        results.append(dict(original_type=type(original).__name__,wrapped_type=type(wrapped).__name__,
                            value=wrapped.item(),bitwise_value_equal=True))
    assert not getattr(balanced_accuracy_score,'_heegnet_float_item_bridge',False)
    return dict(passed=True,cases=results,sklearn_global_function_unmodified=True,
                author_source_modified=False,model_forward=False,optimizer_updates=0)
