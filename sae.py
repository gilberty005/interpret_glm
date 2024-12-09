# -*- coding: utf-8 -*-
"""SAE.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GsY1Qjxg6Gswj42n2WlBzrMav7_lSuLG

#Utils
"""

import os
from typing import Optional, List, Tuple, Dict, Union
import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

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
    seqs: List[str],
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
        if not hasattr(outputs, 'hidden_states') or not outputs.hidden_states:
            raise ValueError("Model did not return hidden states.")
        layer_acts = outputs.hidden_states[layer]

    return layer_acts


def train_val_test_split(
    df: pd.DataFrame, train_frac: float = 0.9
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the sequences into training, validation, and test sets. train_frac specifies
    the fraction of examples to use for training; the rest is split evenly between
    validation and test.

    Doing this by samples so it's stochastic.

    Args:
        seqs: The sequences to split.
        train_frac: The fraction of examples to use for training.

    Returns:
        A tuple containing the training, validation, and test sets.
    """
    is_train = pd.Series(
        np.random.choice([True, False], size=len(df), p=[train_frac, 1 - train_frac])
    )
    seqs_train = df[is_train]
    seqs_val_test = df[~is_train].reset_index(drop=True)

    is_val = pd.Series(np.random.choice([True, False], size=len(seqs_val_test), p=[0.1, 0.9]))
    seqs_val = seqs_val_test[is_val]
    seqs_test = seqs_val_test[~is_val]
    return seqs_train, seqs_val, seqs_test


def parse_swissprot_annotation(annotation_str: str, header: str) -> List[dict]:
    """
    Parse a SwissProt annotation string like this:

    ```
    MOTIF 119..132; /note="JAMM motif"; /evidence="ECO:0000255|PROSITE-ProRule:PRU01182"
    ```
    where MOTIF is the header argument.

    Returns:
        [{
            "start": 119,
            "end": 132,
            "note": "JAMM motif",
            "evidence": "ECO:0000255|PROSITE-ProRule:PRU01182",
        }]
    """
    res = []
    occurrences = [o for o in annotation_str.split(header + " ") if len(o) > 0]
    for o in occurrences:
        parts = [p for p in o.split("; /") if len(p) > 0]

        pos_part = parts[0]
        coords = pos_part.split("..")

        annotations_dict = {}
        for part in parts[1:]:
            key, value = part.split("=", 1)
            annotations_dict[key] = value.replace('"', "").replace(";", "").strip()

        try:
            list(map(int, coords))
        except ValueError:
            continue
        if len(annotations_dict) == 0:
            continue

        res.append(
            {
                "start": int(coords[0]),
                "end": int(coords[1]) if len(coords) > 1 else int(coords[0]),
                **annotations_dict,
            }
        )
    return res

"""#Data Module"""

# !pip install pytorch_lightning

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset


class PolarsDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {"Sequence": row["sequence"], "Entry": row["id"]}


# Data Module
class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, num_workers=7):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        df = pd.read_parquet(self.data_path)
        self.train_data, self.val_data, self.test_data = train_val_test_split(df)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            PolarsDataset(self.train_data), 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            PolarsDataset(self.val_data), 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            PolarsDataset(self.test_data), 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
        )

"""#SAE Model"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


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

        self.dead_steps_threshold = dead_steps_threshold / batch_size

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
        self.stats_last_nonzero *= (latents == 0).all(dim=(0, 1)).long()
        self.stats_last_nonzero += 1

        dead_mask = self.auxk_mask_fn()
        num_dead = dead_mask.sum().item()

        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu

        if num_dead > 0:
            k_aux = min(x.shape[-1] // 2, num_dead)

            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts = self.topK_activation(auxk_latents, k=k_aux)

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
    auxk_coeff = 1.0 / 32.0  # TODO: Is this the best coefficient?

    mse_loss = mse_scale * F.mse_loss(recons, x)
    if auxk is not None:
        auxk_loss = auxk_coeff * F.mse_loss(auxk, x - recons).nan_to_num(0)
    else:
        auxk_loss = torch.tensor(0.0)
    return mse_loss, auxk_loss


def estimate_loss(
    plm: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer: int,
    sae_model: SparseAutoencoder,
    examples_set: pd.DataFrame,
    sample_size: int = 100,
):
    """
    Estimate the loss of the Sparse Autoencoder using a set of examples.

    Args:
        sae_model: The Sparse Autoencoder model.
        examples_set: The examples set to estimate the loss on.
        sample_size: The number of examples to sample.

    Returns:
        float: The estimated loss.
    """
    samples = examples_set.sample(sample_size)
    test_losses = []
    seqs = [row["Sequence"] for _, row in samples.iterrows()]
    layer_acts = get_layer_activations(tokenizer=tokenizer, plm=plm, seqs=seqs, layer=layer)

    recons = sae_model.forward_val(layer_acts)
    mse_loss, _ = loss_fn(layer_acts, recons)
    test_losses.append(mse_loss.item())

    del layer_acts
    return np.mean(test_losses)

"""#ESM Wrapper"""

# !pip install esm
# !pip install fair-esm

import pytorch_lightning as pl
import torch
import torch.nn as nn
from esm.modules import ESM1bLayerNorm, RobertaLMHead, TransformerLayer

class ESM2Model(pl.LightningModule):
    def __init__(self, num_layers, embed_dim, attention_heads, alphabet, token_dropout):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout
        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def load_esm_ckpt(self, esm_pretrained):
        ckpt = {}
        model_data = torch.load(esm_pretrained)["model"]
        for k in model_data:
            if "lm_head" in k:
                ckpt[k.replace("encoder.", "")] = model_data[k]
            else:
                ckpt[k.replace("encoder.sentence_encoder.", "")] = model_data[k]
        self.load_state_dict(ckpt)

    def compose_input(self, list_tuple_seq):
      _, _, batch_tokens = self.batch_converter(list_tuple_seq)
      device = next(self.parameters()).device
      batch_tokens = batch_tokens.to(device)
      return batch_tokens

    def forward(self, tokens, output_hidden_states=False):
        # Embedding
        x = self.embed_scale * self.embed_tokens(tokens)

        # Pass through layers
        hidden_states = []
        x = x.transpose(0, 1)  # (B, T, E) -> (T, B, E)
        for layer in self.layers:
            x, _ = layer(x)
            if output_hidden_states:
                hidden_states.append(x.transpose(0, 1))  # Store (B, T, E)

        x = x.transpose(0, 1)  # (T, B, E) -> (B, T, E)
        x = self.emb_layer_norm_after(x)

        logits = self.lm_head(x)

        outputs = {'logits': logits}
        if output_hidden_states:
            outputs['hidden_states'] = hidden_states
        return outputs

    def get_layer_activations(self, input, layer_idx):
      if isinstance(input, str):
        tokens = self.compose_input([("protein", input)])
      elif isinstance(input, list):
        tokens = self.compose_input([("protein", seq) for seq in input])
      else:
        tokens = input

      # Get the attention mask if needed
      attention_mask = tokens.ne(self.padding_idx)

      x = self.embed_scale * self.embed_tokens(tokens)
      x = x.transpose(0, 1)  # (B, T, E) => (T, B, E)

      # Process up to the desired layer
      for idx, layer in enumerate(self.layers):
          x, _ = layer(x, self_attn_padding_mask=attention_mask)
          if idx == layer_idx - 1:
              break

      x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
      return tokens, x

    def get_sequence(self, x, layer_idx):
        x = x.transpose(0, 1)  # (B, T, E) => (T, B, E)
        for _, layer in enumerate(self.layers[layer_idx:]):
            x, attn = layer(
                x,
                self_attn_padding_mask=None,
                need_head_weights=False,
            )
        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
        logits = self.lm_head(x)
        return logits

"""#SAE Model"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

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

        self.dead_steps_threshold = dead_steps_threshold / batch_size

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
        self.stats_last_nonzero *= (latents == 0).all(dim=(0, 1)).long()
        self.stats_last_nonzero += 1

        dead_mask = self.auxk_mask_fn()
        num_dead = dead_mask.sum().item()

        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu

        if num_dead > 0:
            k_aux = min(x.shape[-1] // 2, num_dead)

            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts = self.topK_activation(auxk_latents, k=k_aux)

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
    auxk_coeff = 1.0 / 32.0  # TODO: Is this the best coefficient?

    mse_loss = mse_scale * F.mse_loss(recons, x)
    if auxk is not None:
        auxk_loss = auxk_coeff * F.mse_loss(auxk, x - recons).nan_to_num(0)
    else:
        auxk_loss = torch.tensor(0.0)
    return mse_loss, auxk_loss


def estimate_loss(
    plm: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer: int,
    sae_model: SparseAutoencoder,
    examples_set: pd.DataFrame,
    sample_size: int = 100,
):
    """
    Estimate the loss of the Sparse Autoencoder using a set of examples.

    Args:
        sae_model: The Sparse Autoencoder model.
        examples_set: The examples set to estimate the loss on.
        sample_size: The number of examples to sample.

    Returns:
        float: The estimated loss.
    """
    samples = examples_set.sample(sample_size)
    test_losses = []
    seqs = [row["Sequence"] for row in samples.iter_rows(named=True)]
    layer_acts = get_layer_activations(tokenizer=tokenizer, plm=plm, seqs=seqs, layer=layer)

    recons = sae_model.forward_val(layer_acts)
    mse_loss, _ = loss_fn(layer_acts, recons)
    test_losses.append(mse_loss.item())

    del layer_acts
    return np.mean(test_losses)

"""#SAE Module"""

import esm
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

def get_esm_model(d_model, alphabet, esm2_weight):
    esm2_model = ESM2Model(
        num_layers=33,
        embed_dim=d_model,
        attention_heads=20,
        alphabet=alphabet,
        token_dropout=False,
    )
    state_dict = torch.load(esm2_weight)
    esm2_model.load_state_dict(state_dict, strict=False)

    esm2_model.eval()
    for param in esm2_model.parameters():
        param.requires_grad = False
    return esm2_model

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
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.esm2_model = get_esm_model(self.args.d_model, self.alphabet, self.args.esm2_weight)

    def on_fit_start(self):
        device = self.device
        self.esm2_model.to(device)
        self.sae_model.to(device)

    def forward(self, x):
        return self.sae_model(x)

    def training_step(self, batch, batch_idx):
        seqs = batch["Sequence"]
        batch_size = len(seqs)
        with torch.no_grad():
            tokens, esm_layer_acts = self.esm2_model.get_layer_activations(seqs, self.layer_to_use)

        esm_layer_acts = esm_layer_acts.to(self.device)
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
        seqs = batch["Sequence"]
        batch_size = len(seqs)
        with torch.no_grad():
            tokens, esm_layer_acts = self.esm2_model.get_layer_activations(seqs, self.layer_to_use)
            esm_layer_acts = esm_layer_acts.to(self.device)
            recons, auxk, num_dead = self(esm_layer_acts)
            mse_loss, auxk_loss = loss_fn(esm_layer_acts, recons, auxk)
            loss = mse_loss + auxk_loss
            logits = self.esm2_model.get_sequence(recons, self.layer_to_use)
            logits = logits.view(-1, logits.size(-1))
            tokens = tokens.view(-1)
            correct = (torch.argmax(logits, dim=-1) == tokens).sum().item()
            total = tokens.size(0)

        self.log(
            "val_celoss",
            F.cross_entropy(logits, tokens).mean().item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "val_acc",
            correct / total,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)

    def on_after_backward(self):
        self.sae_model.norm_weights()
        self.sae_model.norm_grad()

"""#Training"""

import os
from types import SimpleNamespace

import pytorch_lightning as pl_lightning
import wandb
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

model_weights, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")

# from google.colab import drive
# drive.mount('/content/drive')

wandb.finish()

# Define the arguments manually
args = SimpleNamespace(
    data_dir="uniref50_1M_1022.parquet",
    esm2_weight="esm2_t33_650M_UR50D.pt",
    #esm2_weight = "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
    layer_to_use=24,
    d_model=1280,
    d_hidden=8192,
    batch_size=4,
    lr=2e-4,
    k=128,
    auxk=256,
    dead_steps_threshold=2000,
    max_epochs=4,
    num_devices=1,
)

args.output_dir = f"results_l{args.layer_to_use}_dim{args.d_hidden}_k{args.k}"

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

sae_name = f"esm2_plm1280_l{args.layer_to_use}_sae{args.d_hidden}_k{args.k}_auxk{args.auxk}"

wandb_logger = WandbLogger(
    project="interpretability",
    name=sae_name,
    save_dir=os.path.join(args.output_dir, "wandb"),
)

model = SAELightningModule(args)
data_module = SequenceDataModule(args.data_dir, args.batch_size)

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(args.output_dir, "checkpoints"),
    filename=sae_name + "-{step}-{val_loss:.2f}",
    save_top_k=3,
    monitor="val_loss",
    mode="min",
    save_last=True,
)

trainer = pl_lightning.Trainer(
    max_epochs=args.max_epochs,
    accelerator="gpu",
    devices=list(range(args.num_devices)),
    strategy="ddp" if args.num_devices > 1 else "auto",
    logger=wandb_logger,
    log_every_n_steps=10,
    val_check_interval=2000,
    limit_val_batches=10,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1.0,
)

trainer.fit(model, data_module)
trainer.test(model, data_module)

wandb.finish()