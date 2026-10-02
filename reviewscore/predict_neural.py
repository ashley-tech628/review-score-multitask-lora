"""Inference from a complete new serving bundle, never a legacy adapter alone."""
import argparse
import json
import torch
from . import TARGETS
from .neural import load_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--text',required=True)
    p.add_argument('--local-only',action='store_true')
    a=p.parse_args()
    if not a.text.strip():p.error('Text must not be empty')
    model,tokenizer=load_bundle(a.bundle,a.local_only)
    batch=tokenizer([a.text],truncation=True,max_length=128,return_tensors='pt')
    with torch.no_grad():values=model(**{k:v for k,v in batch.items() if k in ['input_ids','attention_mask']})[0].tolist()
    print(json.dumps({'normalized_scores':dict(zip(TARGETS,values)),'note':'Regression estimates, not confidence probabilities'},indent=2))


if __name__=='__main__':main()
