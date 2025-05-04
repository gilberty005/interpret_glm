import os
from typing import Optional, List, Tuple, Dict, Union
import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import multiprocessing
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
from model.model_registry import model_registry
from model.tokenizer import tokenizer_registry
from dataset.datamodule_registry import datamodule_registry
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from dataset.inference_utils import ModRoPE
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelSummary  
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
import torch.nn as nn
from esm.modules import ESM1bLayerNorm, RobertaLMHead, TransformerLayer
import lightning.pytorch as pl
import esm
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import os
from types import SimpleNamespace

import pytorch_lightning as pl_lightning
import wandb
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
from transformers import AutoTokenizer, AutoModelForMaskedLM

def create_file(dir: str, file_name: str) -> None:
    if not os.path.isdir(dir):
        raise ValueError(f"The specified directory '{dir}' does not exist.")

    file_path = os.path.join(dir, file_name)
    try:
        open(file_path, "x").close()
    except FileExistsError:
        pass


def get_layer_activations(
    tokenizer: PreTrainedTokenizer,
    plm: PreTrainedModel,
    seqs: list[str],
    layer: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Get the activations of a specific layer in a pre-trained model.

    Args:
        tokenizer: The tokenizer to use.
        plm: The pre-trained model to get activations from.
        seqs: The list of sequences.
        layer: The layer index to get activations from (0-based).
        device: The device to use.

    Returns:
        The tensor of activations for the specified layer.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(seqs, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = plm(**inputs, output_hidden_states=True)
        #if not hasattr(outputs, 'hidden_states') or not outputs.hidden_states:
            #raise ValueError("Model did not return hidden states.")
    layer_acts = outputs.hidden_states[layer]
    del outputs
    return layer_acts

class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        k: int = 128,
        auxk: int = 256,
        batch_size: int = 256,
        dead_steps_threshold: int = 2000,
    ):
        """
        Initialize the Sparse Autoencoder.

        Args:
            d_model: Dimension of the pLM model.
            d_hidden: Dimension of the SAE hidden layer.
            k: Number of top-k activations to keep.
            auxk: Number of auxiliary activations.
            dead_steps_threshold: How many examples of inactivation before we consider
                a hidden dim dead.

        Adapted from https://github.com/tylercosgrove/sparse-autoencoder-mistral7b/blob/main/sae.py
        based on 'Scaling and evaluating sparse autoencoders' (Gao et al. 2024) https://arxiv.org/pdf/2406.04093
        """
        super().__init__()

        self.w_enc = nn.Parameter(torch.empty(d_model, d_hidden))
        self.w_dec = nn.Parameter(torch.empty(d_hidden, d_model))

        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.auxk = auxk
        self.batch_size = batch_size

        self.dead_steps_threshold = dead_steps_threshold #/ batch_size

        # TODO: Revisit to see if this is the best way to initialize
        nn.init.kaiming_uniform_(self.w_enc, a=math.sqrt(5))
        self.w_dec.data = self.w_enc.data.T.clone()
        self.w_dec.data /= self.w_dec.data.norm(dim=0)

        # Initialize dead neuron tracking. For each hidden dimension, save the
        # index of the example at which it was last activated.
        self.register_buffer("stats_last_nonzero", torch.zeros(d_hidden, dtype=torch.long))

    def topK_activation(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Apply top-k activation to the input tensor.

        Args:
            x: (BATCH_SIZE, D_EMBED, D_MODEL) input tensor to apply top-k activation on.
            k: Number of top activations to keep.

        Returns:
            torch.Tensor: Tensor with only the top k activations preserved,and others
            set to zero.

        This function performs the following steps:
        1. Find the top k values and their indices in the input tensor.
        2. Apply ReLU activation to these top k values.
        3. Create a new tensor of zeros with the same shape as the input.
        4. Scatter the activated top k values back into their original positions.
        """
        topk = torch.topk(x, k=k, dim=-1, sorted=False)
        values = F.relu(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result
    
    def batch_topk_activation(
        self,
        x: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        
        B = x.shape[0]
        total_keep = k * B
        flat = x.view(-1)
        total_keep = min(total_keep, flat.numel())
        topk = torch.topk(flat, k=total_keep, dim=0, largest=True, sorted=False)
        activated = F.relu(topk.values)
        out_flat = torch.zeros_like(flat)
        out_flat[topk.indices] = activated

        return out_flat.view_as(x)
    
    def LN(
        self, x: torch.Tensor, eps: float = 1e-5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Layer Normalization to the input tensor.

        Args:
            x: Input tensor to be normalized.
            eps: A small value added to the denominator for numerical stability.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The normalized tensor.
                - The mean of the input tensor.
                - The standard deviation of the input tensor.

        TODO: Is eps = 1e-5 the best value?
        """
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def auxk_mask_fn(self) -> torch.Tensor:
        """
        Create a mask for dead neurons.

        Returns:
            torch.Tensor: A boolean tensor of shape (D_HIDDEN,) where True indicates
                a dead neuron.
        """
        dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
        return dead_mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Sparse Autoencoder. If there are dead neurons, compute the
        reconstruction using the AUXK auxiliary hidden dims as well.

        Args:
            x: (BATCH_SIZE, D_EMBED, D_MODEL) input tensor to the SAE.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The reconstructed activations via top K hidden dims.
                - If there are dead neurons, the auxiliary activations via top AUXK
                    hidden dims; otherwise, None.
                - The number of dead neurons.
        """
        x, mu, std = self.LN(x)
        x = x - self.b_pre

        pre_acts = x @ self.w_enc + self.b_enc

        # latents: (BATCH_SIZE, D_EMBED, D_HIDDEN)
        #latents = self.batch_topk_activation(pre_acts, k=self.k)
        latents = self.topK_activation(pre_acts, k=self.k)
        # `(latents == 0)` creates a boolean tensor element-wise from `latents`.
        # `.all(dim=(0, 1))` preserves D_HIDDEN and does the boolean `all`
        # operation across BATCH_SIZE and D_EMBED. Finally, `.long()` turns
        # it into a vector of 0s and 1s of length D_HIDDEN.
        #
        # self.stats_last_nonzero is a vector of length D_HIDDEN. Doing
        # `*=` with `M = (latents == 0).all(dim=(0, 1)).long()` has the effect
        # of: if M[i] = 0, self.stats_last_nonzero[i] is cleared to 0, and then
        # immediately incremented; if M[i] = 1, self.stats_last_nonzero[i] is
        # unchanged. self.stats_last_nonzero[i] means "for how many consecutive
        # iterations has hidden dim i been zero".
        #self.stats_last_nonzero *= (latents == 0).all(dim=(0, 1)).long()
        #self.stats_last_nonzero += 1
        
        fired = latents.ne(0).any(dim=(0,1))    # True for any unit that had a non-zero activation
        self.stats_last_nonzero = torch.where(
            fired,
            torch.zeros_like(self.stats_last_nonzero),    # reset to 0 on fire
            self.stats_last_nonzero + 1                   # otherwise increment
        )
        
        dead_mask = self.auxk_mask_fn()
        num_dead = dead_mask.sum().item()

        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu

        if num_dead > 0:
            k_aux = min(x.shape[-1] // 2, num_dead)
            k_aux = self.auxk
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            #auxk_acts = self.batch_topk_activation(auxk_latents, k=k_aux)
            auxk_acts = self.topK_activation(auxk_latents, k=k_aux)
            num_nonzero_auxk = torch.count_nonzero(auxk_acts).item()
            auxk = auxk_acts @ self.w_dec + self.b_pre
            auxk = auxk * std + mu
        else:
            auxk = None
        
        return recons, auxk, num_dead

    @torch.no_grad()
    def forward_val(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Sparse Autoencoder for validation.

        Args:
            x: (BATCH_SIZE, D_EMBED, D_MODEL) input tensor to the SAE.

        Returns:
            torch.Tensor: The reconstructed activations via top K hidden dims.
        """
        x, mu, std = self.LN(x)
        x = x - self.b_pre
        pre_acts = x @ self.w_enc + self.b_enc
        #latents = self.batch_topk_activation(pre_acts, self.k)
        latents = self.topK_activation(pre_acts, self.k)
        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu
        return recons

    @torch.no_grad()
    def norm_weights(self) -> None:
        """
        Normalize the weights of the Sparse Autoencoder.
        """
        self.w_dec.data /= self.w_dec.data.norm(dim=0)

    @torch.no_grad()
    def norm_grad(self) -> None:
        """
        Normalize the gradient of the weights of the Sparse Autoencoder.
        """
        dot_products = torch.sum(self.w_dec.data * self.w_dec.grad, dim=0)
        self.w_dec.grad.sub_(self.w_dec.data * dot_products.unsqueeze(0))

    @torch.no_grad()
    def get_acts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the activations of the Sparse Autoencoder.

        Args:
            x: (BATCH_SIZE, D_EMBED, D_MODEL) input tensor to the SAE.

        Returns:
            torch.Tensor: The activations of the Sparse Autoencoder.
        """
        x, _, _ = self.LN(x)
        x = x - self.b_pre
        pre_acts = x @ self.w_enc + self.b_enc
        latents = self.topK_activation(pre_acts, self.k)
        return latents

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, mu, std = self.LN(x)
        x = x - self.b_pre
        acts = x @ self.w_enc + self.b_enc
        return acts, mu, std

    @torch.no_grad()
    def decode(self, acts: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        latents = self.topK_activation(acts, self.k)

        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu
        return recons


def loss_fn(
    x: torch.Tensor, recons: torch.Tensor, auxk: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the loss function for the Sparse Autoencoder.

    Args:
        x: (BATCH_SIZE, D_EMBED, D_MODEL) input tensor to the SAE.
        recons: (BATCH_SIZE, D_EMBED, D_MODEL) reconstructed activations via top K
            hidden dims.
        auxk: (BATCH_SIZE, D_EMBED, D_MODEL) auxiliary activations via top AUXK
            hidden dims. See A.2. in https://arxiv.org/pdf/2406.04093.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The MSE loss.
            - The auxiliary loss.
    """
    mse_scale = 1
    auxk_coeff = 1.0/32.0 # TODO: Is this the best coefficient?

    mse_loss = mse_scale * F.mse_loss(recons, x)
    if auxk is not None:
        auxk_loss = auxk_coeff * F.mse_loss(auxk, x - recons).nan_to_num(0)
    else:
        auxk_loss = torch.tensor(0.0)
    return mse_loss, auxk_loss
 
"""#Validation Metrics"""
def diff_cross_entropy(orig_logits, recons_logits, tokens):
    """
    Calculates the difference in cross-entropy between two sets of logits.

    This function is used in the validation step of the SAE model.
    """
    orig_logits = orig_logits.view(-1, orig_logits.size(-1))
    recons_logits = recons_logits.view(-1, recons_logits.size(-1))
    tokens = tokens.view(-1)
    orig_loss = F.cross_entropy(orig_logits, tokens).mean().item()
    recons_loss = F.cross_entropy(recons_logits, tokens).mean().item()
    return recons_loss - orig_loss


def calc_diff_cross_entropy(seq, layer, esm2_model, sae_model):
    """
    Calculates the difference in cross-entropy when splicing in the SAE model.
    Wrapper around diff_cross_entropy.

    Args:
        seq: A string representing the sequence.
        layer: The layer of the ESM model to use.
        esm2_model: The ESM model.
        sae_model: The SAE model.
    
    Returns:
        float: The difference in cross-entropy.
    """
    tokens, esm_layer_acts = esm2_model.get_layer_activations(seq, layer)
    recons, auxk, num_dead = sae_model(esm_layer_acts)
    logits_recon = esm2_model.get_sequence(recons, layer)
    logits_orig = esm2_model.get_sequence(esm_layer_acts, layer)

    return diff_cross_entropy(logits_orig, logits_recon, tokens)


def calc_loss_recovered(seq, layer, esm2_model, sae_model):
    """
    Calculates the "loss recovered": 1- \frac{CE(recons) - CE(orig)}{CE(zeros) - CE(orig)}.
    Wrapper around diff_cross_entropy.

    Args:
        seq: A string representing the sequence.
        layer: The layer of the ESM model to use.
        esm2_model: The ESM model.
        sae_model: The SAE model.
    
    Returns:
        float: The loss recovered.
    """
    tokens, esm_layer_acts = esm2_model.get_layer_activations(seq, layer)
    recons, auxk, num_dead = sae_model(esm_layer_acts)
    logits_recon = esm2_model.get_sequence(recons, layer)
    logits_orig = esm2_model.get_sequence(esm_layer_acts, layer)
    
    zeros_act = torch.zeros_like(esm_layer_acts)
    logits_zeros = esm2_model.get_sequence(zeros_act, layer)

    diff_CE = diff_cross_entropy(logits_orig, logits_recon, tokens)
    diff_CE_zeros = diff_cross_entropy(logits_orig, logits_zeros, tokens)

    return 1 - (diff_CE / diff_CE_zeros)

"""#SAE Module"""
def get_glm_model(args, alphabet):
    #change this
    with open(args.config_yaml, 'r') as infile:
        run_configs = yaml.safe_load(infile)

    rotary_config = run_configs['data']['valid_config']['gtdb_rep_mlm']['collate_fn_args']['rotary_config']
    run_configs['model']['transformer_layer']['attn_method'] = 'torch'
    run_configs['training']['pl_strategy'] = {'class': 'Single','args':{'device': 0, }}
    modrope = ModRoPE(rotary_config)
    glm_model = model_registry[run_configs['model_class']](run_configs, args.tokenizer)
    ckpt = torch.load(args.ckpt)['state_dict']
    ckpt = {k.replace('model._orig_mod.', ''): v for k, v in ckpt.items()}
    glm_model.load_state_dict(ckpt, strict=False)
    glm_model.eval()
    for param in glm_model.parameters():
        param.requires_grad = False
    glm_model.cuda()

    return glm_model

class SAELightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.layer_to_use = args.layer_to_use
        self.sae_model = SparseAutoencoder(
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            k=args.k,
            auxk=args.auxk,
            batch_size=args.batch_size,
            dead_steps_threshold=args.dead_steps_threshold,
        )
        self.alphabet = args.tokenizer
        self.validation_step_outputs = []
        self.glm_model = get_glm_model(self.args, self.alphabet)

    def forward(self, x):
        return self.sae_model(x)

    def training_step(self, batch, batch_idx):
        seqs, noised_tokens, tokens, noise_mask = batch
        batch_size = len(tokens)
        # Use the pre-initialized esm2_model
        #to do edit this
        with torch.no_grad():
            tokens, esm_layer_acts = self.glm_model.get_layer_activations(tokens, self.layer_to_use)
        recons, auxk, num_dead = self(esm_layer_acts)
        mse_loss, auxk_loss = loss_fn(esm_layer_acts, recons, auxk)
        
        loss = mse_loss + auxk_loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train_mse_loss",
            mse_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train_auxk_loss",
            auxk_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "num_dead_neurons",
            num_dead,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        #val_seqs = batch["Sequence"]
        seqs, noised_tokens, tokens, noise_mask = batch
        batch_size = len(noised_tokens)
        # Use the pre-initialized esm2_model
        
        with torch.no_grad():
            esm2_model = get_glm_model(
                self.args, self.alphabet
            )

        diff_CE_all = torch.zeros(batch_size, device=self.device)
        mse_loss_all = torch.zeros(batch_size, device=self.device)
        
        # Running inference one sequence at a time
        for i, seq in enumerate(tokens):
            tokens, esm_layer_acts = esm2_model.get_layer_activations(
                seq, self.layer_to_use
            )

            # Calculate MSE
            recons = self.sae_model.forward_val(esm_layer_acts)
            mse_loss, auxk_loss = loss_fn(esm_layer_acts, recons, None)
            mse_loss_all[i] = mse_loss

            # Calculate difference in cross-entropy
            orig_logits = esm2_model.get_sequence(esm_layer_acts, self.layer_to_use)
            spliced_logits = esm2_model.get_sequence(recons, self.layer_to_use)
            diff_CE = diff_cross_entropy(orig_logits, spliced_logits, tokens)
            diff_CE_all[i] = diff_CE

        val_metrics = {
            "val_loss": mse_loss_all.mean(),  # Log val_loss here
            "mse_loss": mse_loss_all.mean(),
            "diff_cross_entropy": diff_CE_all.mean(),
        }
        
        self.log("val_loss", val_metrics["val_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        # Return batch-level metrics for aggregation
        self.validation_step_outputs.append(val_metrics)
        return val_metrics
    
    def on_validation_epoch_end(self):
        # Aggregate metrics across batches
        avg_diff_cross_entropy = torch.stack(
            [x["diff_cross_entropy"] for x in self.validation_step_outputs]
        ).mean()
        avg_mse_loss = torch.stack([x["mse_loss"] for x in self.validation_step_outputs]).mean()

        # Log aggregated metrics
        self.log(
            "avg_mse_loss", avg_mse_loss, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "avg_diff_cross_entropy",
            avg_diff_cross_entropy,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)

    def on_after_backward(self):
        self.sae_model.norm_weights()
        self.sae_model.norm_grad()

"""#Training"""
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run interpretability experiments for GLM SAE')
    parser.add_argument(
        '--config_yaml_path',
        type=str,
        required=True,
        help='Path to the YAML configuration file'
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        required=True,
        help='Path to the checkpoint file'
    )
    parser.add_argument(
        '--layer',
        type=int,
        required=True,
        help='layer to use'
    )
    parser.add_argument(
        '--param',
        type=str,
        required=True,
        help='size of model'
    )
    parser.add_argument(
        '--lr',
        type=float,
        required=True,
        help='learning rate'
    )
    parser.add_argument(
        '--d_hidden',
        type=int,
        required=True,
        help='hidden dim of SAE'
    )
    parser.add_argument(
        '--k',
        type=int,
        required=True,
        help='top k for reconstruction'
    )
    parser.add_argument(
        '--auxk',
        type=int,
        required=True,
        help='k for auxiliary loss'
    )
    args_cli = parser.parse_args()

    config_yaml_path = args_cli.config_yaml_path
    ckpt_path = args_cli.ckpt_path
    wandb.login(key="")
    with open(config_yaml_path, 'r') as infile:
        run_configs = yaml.safe_load(infile)
    # Load and connect tokenizer
    tokenizer_class = tokenizer_registry[run_configs['tokenizer']['class']]
    if run_configs['tokenizer']['args'] is None:
        tokenizer = tokenizer_class.from_architecture(run_configs['tokenizer']['from_architecture'])
    else:
        tokenizer = tokenizer_class(**run_configs['tokenizer']['args'])
    run_configs['data']['tokenizer'] = tokenizer
    # Define the arguments manually
    args = SimpleNamespace(
        config_yaml=config_yaml_path,
        ckpt=ckpt_path,
        layer_to_use=args_cli.layer,
        d_model=1024,
        d_hidden=args_cli.d_hidden,
        batch_size=32,
        lr=args_cli.lr,
        k=args_cli.k,
        auxk=args_cli.auxk,
        dead_steps_threshold=2000,
        max_epochs=1,
        num_devices=1,
        tokenizer=tokenizer,
    )

    args.output_dir = f"results_{args.layer_to_use}_dim{args.d_hidden}_k{args.k}"

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    sae_name = f"glm_lr_{args.lr}_{args_cli.param}_sae_l{args.layer_to_use}_sae{args.d_hidden}_k{args.k}_auxk{args.auxk}"

    wandb_logger = WandbLogger(
        project="interpretability",
        name=sae_name,
        save_dir=os.path.join(args.output_dir, "wandb"),
    )

    #change this
    with open(args.config_yaml, 'r') as infile:
        run_configs = yaml.safe_load(infile)

        
    # Rotary config depends on train_config collate_fn_args !!
    rotary_config = run_configs['data']['valid_config']['gtdb_rep_mlm']['collate_fn_args']['rotary_config']
        
    model = SAELightningModule(args)
    run_configs["data"]["tokenizer"] = tokenizer

    lit_data = datamodule_registry[run_configs["datamodule_class"]](
            **run_configs["data"]
        )


    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename=sae_name + "-{step}-{val_loss:.2f}",
        save_top_k=-1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=list(range(args.num_devices)),
        strategy="ddp" if args.num_devices > 1 else "auto",
        logger=wandb_logger,
        log_every_n_steps=10,
        val_check_interval=500,
        limit_val_batches=10,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
    )
    print(trainer)
    trainer.fit(model, lit_data)
    trainer.test(model, lit_data)

    wandb.finish()

if __name__ == '__main__':
    main()
