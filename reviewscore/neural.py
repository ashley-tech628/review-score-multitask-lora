"""Optional LoRA extension: trainable head, masked loss, complete serving bundle.

Requires torch, transformers, peft and safetensors. Not executed in the packaging
environment. The historical adapter is deliberately not accepted as a bundle.
"""
import json
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import save_file, load_file
from . import TARGETS


def masked_mse(predictions, labels):
    """Equal weight for each observed task; absent labels never become zeros."""
    observed = torch.isfinite(labels)
    clean = torch.where(observed, labels, torch.zeros_like(labels))
    counts = observed.sum(0)
    per_task = (((predictions - clean) ** 2) * observed).sum(0) / counts.clamp_min(1)
    active = counts > 0
    if not active.any():
        raise ValueError('Batch contains no observed labels')
    return per_task[active].mean()


class ScoreModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        # The head lives outside PEFT so freezing the encoder cannot freeze it.
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(hidden, hidden // 2),
                                  nn.ReLU(), nn.Linear(hidden // 2, 5), nn.Sigmoid())

    def forward(self, input_ids, attention_mask):
        return self.head(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0])


def create(base='distilbert-base-uncased', revision=None, local_only=False):
    encoder = AutoModel.from_pretrained(base, revision=revision, local_files_only=local_only)
    encoder = get_peft_model(encoder, LoraConfig(r=16, lora_alpha=32, lora_dropout=.1,
                                              target_modules=['q_lin', 'v_lin'], bias='none'))
    model = ScoreModel(encoder)
    assert all(p.requires_grad for p in model.head.parameters())
    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision, local_files_only=local_only)
    return model, tokenizer


def save_bundle(model, tokenizer, path, base, revision=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(path / 'adapter', safe_serialization=True)
    tokenizer.save_pretrained(path / 'tokenizer')
    save_file({k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()}, str(path / 'head.safetensors'))
    revision = getattr(model.encoder.config, '_commit_hash', None) or revision
    (path / 'bundle.json').write_text(json.dumps({'version': 1, 'base': base, 'revision': revision,
                                                'targets': list(TARGETS), 'max_length': 128}, indent=2), encoding='utf-8')


def load_bundle(path, local_only=False):
    path = Path(path)
    manifest = json.loads((path / 'bundle.json').read_text(encoding='utf-8'))
    if manifest.get('version') != 1 or manifest.get('targets') != list(TARGETS):
        raise ValueError('Unsupported bundle version or target order')
    if not (path / 'head.safetensors').exists():
        raise ValueError('Missing regression head; an adapter alone is not a serving bundle')
    encoder = AutoModel.from_pretrained(manifest['base'], revision=manifest['revision'], local_files_only=local_only)
    model = ScoreModel(PeftModel.from_pretrained(encoder, path / 'adapter'))
    model.head.load_state_dict(load_file(str(path / 'head.safetensors')), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(path / 'tokenizer', local_files_only=True)
    model.eval()
    return model, tokenizer
