import argparse
import warnings
from pathlib import Path
import yaml
import sys
sys.path.append(str(Path(__file__).parents[1]))
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from torch.utils.data import Dataset, DataLoader
##TODO FIX PATHS
from glm.dataset.inference_utils import ModRoPE
from glm.model.model_registry import model_registry
from glm.train.tokenizer import tokenizer_registry
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd




sequences = read_fasta(args.fasta)
sequences = [seq for seq in sequences if len(seq) >= 1200]
device = 'cuda:0'
with open(args.config_yaml, 'r') as infile:
        run_configs = yaml.safe_load(infile)

        
# Rotary config depends on train_config collate_fn_args !!
rotary_config = run_configs['data']['valid_config']['gtdb_rep_mlm']['collate_fn_args']['rotary_config']
        
# Override config for inference setting
run_configs['model']['transformer_layer']['attn_method'] = 'torch' 
run_configs['training']['pl_strategy'] = {'class': 'Single','args':{'device': 0, }}
    
# Load and connect tokenizer
tokenizer_class = tokenizer_registry[run_configs['tokenizer']['class']]
if run_configs['tokenizer']['args'] is None:
    tokenizer = tokenizer_class.from_architecture(run_configs['tokenizer']['from_architecture'])
else:
    tokenizer = tokenizer_class(**run_configs['tokenizer']['args'])
run_configs['data']['tokenizer'] = tokenizer
    
# Load RoPE
modrope = ModRoPE(rotary_config)

# Load model
model = model_registry[run_configs['model_class']](run_configs, tokenizer)
ckpt = torch.load(args.ckpt)['state_dict']
ckpt = {k.replace('model._orig_mod.', ''): v for k, v in ckpt.items()}
model.load_state_dict(ckpt, strict=False)
model = model.to(device)
