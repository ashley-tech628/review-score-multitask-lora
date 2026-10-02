"""Read safetensors metadata without loading tensors or unpickling .pt files."""
import argparse
import json
from pathlib import Path
import struct


def inspect(path):
    path=Path(path)
    with (path/'adapter_model.safetensors').open('rb') as stream:
        prefix=stream.read(8)
        if len(prefix)!=8:raise ValueError('Truncated safetensors header')
        length=struct.unpack('<Q',prefix)[0]
        if length>16*1024*1024:raise ValueError('Unexpectedly large header')
        header=json.loads(stream.read(length))
    keys=[k for k in header if k!='__metadata__']
    config=json.loads((path/'adapter_config.json').read_text(encoding='utf-8'))
    state=json.loads((path/'trainer_state.json').read_text(encoding='utf-8')) if (path/'trainer_state.json').exists() else {}
    head=[k for k in keys if 'regressor' in k or 'head' in k or 'classifier' in k]
    return {'tensor_count':len(keys),'head_tensor_keys':head,'modules_to_save':config.get('modules_to_save'),
            'global_step':state.get('global_step'),'epoch':state.get('epoch'),
            'best_metric':state.get('best_metric'),
            'finding':'No regression-head tensors found' if not head else 'Head-like tensor names found; inspect shapes and loader compatibility',
            'note':'Metadata inspection only; no inference or arbitrary pickle loading'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint')
    print(json.dumps(inspect(p.parse_args().checkpoint),indent=2))
