"""Only canonicalize int/string domain-map keys; all remaining types/identities strict."""
from copy import deepcopy
import json,re
def no_duplicate_pairs(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('Duplicate JSON key: '+k)
  d[k]=v
 return d
def read_json(path):return json.loads(path.read_text(),object_pairs_hook=no_duplicate_pairs)
def canonical(binding):
 if type(binding)is not dict or any(type(k)is not str for k in binding):raise ValueError('Binding dictionary schema')
 b=deepcopy(binding);source=b.get('target_ids_by_domain');out={}
 if type(source)is not dict or not source:raise ValueError('Target domain map missing')
 for k,v in source.items():
  if type(k)is int and 0<=k<123:key=str(k)
  elif type(k)is str and re.fullmatch(r'0|[1-9][0-9]*',k) and int(k)<123:key=k
  else:raise ValueError('Noncanonical domain key')
  if key in out:raise ValueError('Colliding domain representations')
  if type(v)is not list or any(type(i)is not int or not 0<=i<3444 for i in v) or len(v)!=len(set(v)):raise ValueError('Invalid domain ID list')
  out[key]=v
 b['target_ids_by_domain']=out;return b
def strict_equal(a,b):
 if type(a)is not type(b):return False
 if isinstance(a,dict):return a.keys()==b.keys() and all(type(k)is str for k in a) and all(strict_equal(a[k],b[k]) for k in a)
 if isinstance(a,list):return len(a)==len(b) and all(strict_equal(x,y) for x,y in zip(a,b))
 if type(a) in [str,int,bool,type(None)]:return a==b
 raise ValueError('Unexpected non-JSON identity type')
def require_binding(a,b):
 if not strict_equal(canonical(a),canonical(b)):raise ValueError('Actual binding mismatch')
def regression(real_binding):
 b=deepcopy(real_binding);j=json.loads(json.dumps(b));require_binding(b,j);checks=['actual_JSON_roundtrip_accept']
 changes={
 'EXP':lambda x:x.update(experiment_number='WRONG'),
 'fold':lambda x:x.update(fold=99),
 'sourceSHA':lambda x:x.update(source_selected_sha256='0'*64),
 'ID_value':lambda x:x['target_ids_by_domain'][next(iter(x['target_ids_by_domain']))].__setitem__(0,3443),
 'ID_order':lambda x:x['target_ids_by_domain'][next(iter(x['target_ids_by_domain']))].reverse(),
 'ID_count':lambda x:x['target_ids_by_domain'][next(iter(x['target_ids_by_domain']))].pop(),
 'bool_fold':lambda x:x.update(fold=bool(x['fold'])),
 'list_type':lambda x:x.update(target_order=tuple(x['target_order'])),
 'domain_key':lambda x:x['target_ids_by_domain'].update({'0123':[]}),
 }
 for name,change in changes.items():
  bad=deepcopy(j);change(bad)
  try:require_binding(b,bad)
  except (ValueError,TypeError):checks.append(name+'_reject')
  else:raise AssertionError('False acceptance: '+name)
 bad=deepcopy(j);key=next(iter(bad['target_ids_by_domain']));bad['target_ids_by_domain'][int(key)]=bad['target_ids_by_domain'][key]
 try:canonical(bad)
 except ValueError:checks.append('int_string_collision_reject')
 else:raise AssertionError('Key collision accepted')
 try:json.loads('{"a":1,"a":2}',object_pairs_hook=no_duplicate_pairs)
 except ValueError:checks.append('duplicate_JSON_key_reject')
 else:raise AssertionError('Duplicate JSON key accepted')
 return checks
