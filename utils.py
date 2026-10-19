"""Shared device, checkpoint, reproducibility, and serialization helpers."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from model import MNISTCNN


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(name)


def load_model(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'No checkpoint at {path}. Run python train.py first.')
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('architecture') != 'MNISTCNN-v1':
        raise ValueError('Unsupported checkpoint architecture.')
    model = MNISTCNN().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')
