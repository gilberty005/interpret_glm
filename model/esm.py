import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math
from typing import Optional, List
from einops import rearrange
from mup import MuReadout
from .activation import SwiGLU, SwiGLUShard

# from .utils import get_checkpoint_fn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    offload_wrapper,
)
from .mup_utils import MuReadoutS, MuSharedReadoutS
from .baseformer import MuReadoutWrap, MuSharedReadout
from .rotary import apply_rotary_pos_emb
import importlib
from einops import repeat
from lightning.fabric.utilities.init import _materialize_meta_tensors
from functools import partial
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn

class Embedding(nn.Embedding):
    """Do not need to zero the padding index because we do that at initialization
    _fill_padding_idx_with_zero casues error when using FSDP
    _zero_embedding happens at configure_optimizer
    TODO: Revisit this
    """

    def _fill_padding_idx_with_zero(self) -> None:
        pass


class ESM3s(nn.Module):
    """
    esm3-like, with the option of muTransfer scaling
    """

    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config["model"]
        self.seed = config["training"]["seed"]  # re-set when initializing parameters
        self.tokenizer = tokenizer
        self.init_weight_config = self.config["init_weight_config"]
        self.no_input_padding = self.config["no_input_padding"]
        self.mup_weight_scale_called = False
        self.embed_tokens = Embedding(
            len(self.tokenizer),
            self.config["embed_dim"],
            padding_idx=self.tokenizer.padding_idx,  # Doesn't work with FSDP
        )
        self.positional_embedding = None
        if "use_pos_emb" in self.config and self.config["use_pos_emb"]:
            self.positional_embedding = FragmentIndexEmbedding(self.config["embed_dim"])
        if "layer_scale_factor" in self.config:
            warnings.warn(
                "layer_scale_factor depreciated. Use layer_scale_alpha. See muD alpha/sqrt(L)"
            )
        layerwise_scale = (
            math.sqrt(self.config["num_layers"]) / self.config["layer_scale_alpha"]
        )

        attention_config = self.config["transformer_layer"]["attention_config"]
        self.use_rotary_emb = attention_config["use_rotary_emb"]
        self.cache_rope_tensor = False
        if self.use_rotary_emb and not config["data"]["return_contig_indices"]:
            (
                self.embed_dim,
                self.num_heads,
                self.cache_rope_tensor,
                rotary_length,
                self.rotary_base,
            ) = (
                attention_config["embed_dim"],
                attention_config["num_heads"],
                attention_config["cache_rope_tensor"],
                attention_config["rotary_max_len"],
                attention_config["rotary_base"],
            )
            assert self.embed_dim % self.num_heads == 0, (
                "self.kdim must be divisible by num_heads"
            )
            self.head_dim = self.embed_dim // self.num_heads
            assert (self.head_dim % 8 == 0) & (self.head_dim <= 128), (
                "heads divisible by 8"
            )
            if self.cache_rope_tensor:
                self.cache_rot_emb(rotary_length, self.rotary_base)
        elif self.config["transformer_layer"]["attention_config"]["cache_rope_tensor"]:
            warnings.warn(
                "Setting cache_rope_tensor to False since cannot cache_rope_tensor when return_contig_indices=True"
            )
            self.config["transformer_layer"]["attention_config"][
                "cache_rope_tensor"
            ] = False

        self.layers = nn.ModuleList(
            [
                Esm3TransformerLayer(
                    layerwise_scale=layerwise_scale, **self.config["transformer_layer"]
                )
                for _ in range(self.config["num_layers"])
            ]
        )
        if self.config["last_norm"] == "layernorm":
            self.emb_layernorm_after = nn.LayerNorm(
                self.config["embed_dim"], eps=self.config["layernorm_eps"]
            )
        elif self.config["last_norm"] == "rmsnorm":
            self.emb_layernorm_after = nn.RMSNorm(
                self.config["embed_dim"], eps=self.config["layernorm_eps"]
            )

        if (
            "scaled_new" in self.init_weight_config
            and self.init_weight_config["scaled_new"]
        ):
            self.config["lm_head"]["base_fan_in"] = MUP_SHAPES["lm_head.out.weight"][1]
            self.lm_head = RobertaLMHead(**self.config["lm_head"])
        else:
            self.lm_head = RobertaLMHead(**self.config["lm_head"])
        self.param_base_shape_and_mult = {}  # for standalone mup impl; [[base shape], width_mult]

    def get_rot_emb(
        self, end_index: int, dtype=torch.float32, start_index=0, rotary_base=10000
    ):
        inv_freq = 1.0 / (
            rotary_base
            ** (torch.arange(0, self.head_dim, 2, dtype=dtype) / self.head_dim)
        )
        t = torch.arange(start_index, end_index, dtype=dtype)
        freqs = torch.outer(t, inv_freq)

        _cos = repeat(torch.cos(freqs).to(dtype), "... d -> ... (2 d)")
        _sin = repeat(torch.sin(freqs).to(dtype), "... d -> ... (2 d)")

        if hasattr(self, "sp_mesh"):  # temp patch to allow sequence parallel
            _cos, _sin = [
                distp_tensor.distribute_tensor(
                    t, self.sp_mesh, [distp_tensor.Shard(0)]
                ).to_local()
                for t in [_cos, _sin]
            ]
        return _cos, _sin

    def cache_rot_emb(self, context_length, rotary_base):
        _cos, _sin = self.get_rot_emb(context_length, rotary_base=rotary_base)
        # todo, is there a better way than hard-coding device?
        self.register_buffer("cos_cached", _cos.to("cuda"), persistent=False)
        self.register_buffer("sin_cached", _sin.to("cuda"), persistent=False)

    def get_rope_tensor(self, rotary_length=None):
        # No slicing because it is handled in apply_rotary_pos_emb
        if self.cache_rope_tensor:
            return self.cos_cached, self.sin_cached
        else:
            assert rotary_length is not None
            return self.get_rot_emb(rotary_length, rotary_base=self.rotary_base)


    def forward(
        self,
        tokens,
        start_end_indices=None,
        padding_mask=None,
        output_embed_layer_list: Optional[
            List[int]
        ] = None,  # layers to output, -1 is normalized last layer
    ):
        x = self.embed_tokens(tokens)
        if self.no_input_padding:
            padding_mask = None
        else:
            padding_mask = tokens.eq(self.tokenizer.padding_idx)  # B, T
            # Zero out padding_idx (for compatibility across parallel init methods)
            # padding_mask is True for pad; invert to zero out padding_idx
            x = x * (~padding_mask).type_as(x).unsqueeze(-1)
        if self.positional_embedding is not None:
            x = x + self.positional_embedding(tokens, x, start_end_indices)
        if start_end_indices is not None:
            start_end_indices, cos, sin, sep = start_end_indices
        else:
            if self.use_rotary_emb:
                cos, sin = self.get_rope_tensor(rotary_length=x.shape[1])
            else:
                cos, sin = None, None

        hidden_states = {}

        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x=x,
                cos_cached=cos,
                sin_cached=sin,
                self_attn_padding_mask=padding_mask,
                # tp_mesh=self.device_mesh["tensor_parallel"] if "tensor_parallel" in self.device_mesh else None,
            )
            if (
                output_embed_layer_list is not None
                and layer_idx in output_embed_layer_list
            ):
                hidden_states[layer_idx] = (
                    x.detach().clone()
                )  # explicit copy without graph
        x = self.emb_layernorm_after(x.float()).type_as(x)

        if output_embed_layer_list is not None:
            if -1 in output_embed_layer_list:
                hidden_states[-1] = x.detach().clone()
            return (self.lm_head(x), hidden_states)  # user should know order of output

        return self.lm_head(x)

    def set_param_base_shape_and_mult(self):
        # Set param groups based on different mup width multiplier
        if (
            len(self.param_base_shape_and_mult) == 0
        ):  # don't need to do more than once; FSDP1 issue
            for name, module in self.named_modules():
                if hasattr(module, "weight"):
                    # Find matching key in MUP_SHAPES, ignoring prefixes
                    match_weight = next(
                        (key for key in MUP_SHAPES if key in name + ".weight"), None
                    )
                    assert match_weight is not None, (
                        f"Cannot find module {name}.weight in MUP_SHAPES"
                    )
                    if len(MUP_SHAPES[match_weight]) == 2:
                        _, base_fan_in = MUP_SHAPES[match_weight]
                        _, fan_in = module.weight.shape
                    elif len(MUP_SHAPES[match_weight]) == 1:
                        _, base_fan_in = (MUP_SHAPES[match_weight], None)
                        _, fan_in = (module.weight.shape, None)

                    width_mult = (
                        fan_in / base_fan_in if base_fan_in is not None else 1.0
                    )
                    self.param_base_shape_and_mult[name + ".weight"] = [
                        MUP_SHAPES[match_weight],
                        width_mult,
                    ]
                if hasattr(module, "bias") and module.bias is not None:
                    match_bias = next(
                        (key for key in MUP_SHAPES if key in name + ".bias"), None
                    )
                    assert match_bias is not None, (
                        f"Cannot find module {name}.bias in MUP_SHAPES"
                    )
                    # vector-like params are not scaled
                    self.param_base_shape_and_mult[name + ".bias"] = [
                        MUP_SHAPES[match_bias],
                        1.0,
                    ]

    def init_weights(self):
        """
        if scaled_new = True: apply standalone mup which requires MUP_SHAPES to be defined
            scaled = False
        if scaled = True: apply muTransfer initialization with output layer weight=0
            zero initializing query head q_proj is optional for mup
        if scaled = False: apply simple normal distribution init
        Both scaled residual projection (esm-style)
        """
        torch.manual_seed(self.seed)

        # TODO: remove support for  init_weight_config:"scaled" since it's not maintained
        if "scaled" in self.init_weight_config and self.init_weight_config["scaled"]:
            "Assumes you already run mup.set_base_shapes()"
            warnings.warn(
                "init_weight_config.scaled=True is kept for compatibility and debug"
            )
            for name, module in self.named_modules():
                if issubclass(type(module), MuReadout):
                    module.weight.data.zero_()
                elif isinstance(module, (nn.Linear, nn.Conv1d)):
                    "First apply mup scaling"
                    if hasattr(module.weight, "infshape"):
                        self.mup_weight_scale_called = True
                        mup_scale = module.weight.infshape.width_mult() ** -0.5
                        nn.init.normal_(
                            module.weight,
                            mean=0.0,
                            std=self.init_weight_config["init_norm_std"] * mup_scale,
                        )
                        if module.bias is not None:
                            module.bias.data.zero_()
                    else:
                        warnings.warn(
                            "set_base_shapes NOT CALLED. CALL AND RE-INITIALIZE OR LOAD WEIGHTS"
                        )
                elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                    if hasattr(module, "bias") and (module.bias is not None):
                        module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                elif isinstance(module, nn.Embedding):
                    "for mup, embedding variance is independent of dmodel (1)"
                    nn.init.normal_(
                        module.weight,
                        mean=0.0,
                        std=self.init_weight_config["init_norm_std"],
                    )
                    module.weight.data[self.tokenizer.padding_idx].zero_()
                if isinstance(module, MuReadoutWrap):
                    module.width_mult()
                """
                if name.endswith('Wqkv'):
                    # only on V (not QK)
                    module.weight.data[:self.config['embed_dim']].zero_()
                """
        if (
            "scaled_new" in self.init_weight_config
            and self.init_weight_config["scaled_new"]
        ):
            """
            Applies weight initialization based on paper (i.e., muTransfer or muP):
            "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (arXiv:2203.03466)
            Uses Table 8 formuation in mup paper (same as official repo)
                i.e. hidden init_normal(std/sqrt(n)); output_weights * 1/width_mult
            All layers must be represented by MUP_SHAPES. Handle only weight & bias
            Layernorms, bias, embedding, and output layer all vector-like  not scaled  (inf dim=1)
                https://github.com/microsoft/mup/blob/19814971934ef91dd546f88e913fc963e096d11c/mup/init.py#L22
            self.param_base_shape_and_mult is required for LM.configure_optimizers()
            Note that we're only setup for Adam and Adam like optimizer.
            """
            assert not self.init_weight_config["scaled"], (
                "scaled_new and scaled is mutually exlusive"
            )

            # Set no gradient calculation. This should be default for nn.init functions, but need to set for for other ops.
            with torch.no_grad():
                for name, module in self.named_modules():
                    if hasattr(module, "weight"):
                        # Match name excluding modifications from wrapper fn like compile or FSDP.
                        # Set width_mult based on self.param_base_shape_and_mult
                        if name + ".weight" not in self.param_base_shape_and_mult:
                            match = next(
                                (
                                    key
                                    for key in self.param_base_shape_and_mult
                                    if key
                                    in name.replace("_orig_mod.", "")
                                    .replace("_checkpoint_wrapped_module.", "")
                                    .replace("_fsdp_wrapped_module.", "")
                                    + ".weight"
                                ),
                                None,
                            )
                            if match is None:
                                raise RuntimeError(f'{name} not in param_base_shape_and_mult')
                            width_mult = self.param_base_shape_and_mult[match][1]
                        else:
                            width_mult = self.param_base_shape_and_mult[name + ".weight"][1]
                    if isinstance(module, (nn.Linear, nn.Conv1d)):
                        mup_scale = width_mult**-0.5
                        nn.init.normal_(
                            module.weight,
                            mean=0.0,
                            std=self.init_weight_config["init_norm_std"] * mup_scale,
                        )
                        if module.bias is not None:
                            module.bias.data.zero_()
                        rank_zero_info(f"{name}, mup_scale {mup_scale}")
                        
                        # Optional set Wq to zero. This is optional for muTransfer
                        if ("zero_q_proj" in self.init_weight_config) and (
                            self.init_weight_config["zero_q_proj"] and "Wq" in name):
                            module.weight.data.zero_()

                    elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                        # bias is optional for Norms
                        if hasattr(module, "bias") and (module.bias is not None):
                            module.bias.data.zero_()
                        module.weight.data.fill_(1.0)

                    if issubclass(type(module), MuReadoutS):
                        # output layer weight set to zero according to muP(see doc above)
                        module.weight.data.zero_()
                    elif isinstance(module, nn.Embedding):
                        # muP table 8 implementation doesn't scale model input and output layer weights. (see doc above)
                        nn.init.normal_(
                            module.weight,
                            mean=0.0,
                            std=self.init_weight_config["init_norm_std"],
                        )
        else:
            self.apply(self.standard_init_weights)
        print("weights initialized")
        if self.config["transformer_layer"]["attention_config"]["cache_rope_tensor"]:
            self.cache_rot_emb(
                self.config["transformer_layer"]["attention_config"]["rotary_max_len"],
                self.config["transformer_layer"]["attention_config"]["rotary_base"],
            )

    def _zero_embedding_pad(self):
        """zero embedding padding index
        For now, this is run at configure_optimizers because index select in FSDP
        (i.e., weight_data[self.tokenizer.padding_idx]) requires parameters to be materialized.
        """
        with torch.no_grad():
            if hasattr(self.embed_tokens._parameters["weight"], "device_mesh"):
                device_mesh = self.embed_tokens._parameters["weight"].device_mesh
                placements = self.embed_tokens._parameters["weight"].placements

                weight_data = self.embed_tokens.weight.data.full_tensor()
                weight_data[self.tokenizer.padding_idx].zero_()

                self.embed_tokens._parameters["weight"] = (
                    distp_tensor.distribute_tensor(weight_data, device_mesh, placements)
                )
                self.embed_tokens.requires_grad_(True)
                if hasattr(self.embed_tokens, "_get_fsdp_state"):
                    state = self.embed_tokens._get_fsdp_state()
                    if not (fsdp_param_group := state._fsdp_param_group):
                        raise RuntimeError("embedding not managed by FSDP?")
                    for fsdp_param in fsdp_param_group.fsdp_params:
                        fsdp_param.reset_sharded_param()
            elif (
                hasattr(self.embed_tokens, "_is_fsdp_managed_module")
                and self.embed_tokens._is_fsdp_managed_module
            ):
                warnings.warn(
                    "FSDP1 is not tested but it may run. Suggest you use FSDP2"
                )
                with FSDP.summon_full_params(self.embed_tokens):
                    _materialize_meta_tensors(
                        self.embed_tokens,
                        self.embed_tokens._parameters["weight"].device,
                    )
                    if self.embed_tokens._parameters["weight"].size(0) > 0:
                        self.embed_tokens._parameters["weight"][
                            self.tokenizer.padding_idx
                        ].zero_()  # a dtensor
            else:
                self.embed_tokens.weight.data[
                    self.tokenizer.padding_idx
                ].zero_()  # a dtensor
        # for name, module in self.named_modules():
        # for module in self.modules():
        # distribute_module(module

    def standard_init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(
                mean=0.0, std=self.init_weight_config["init_norm_std"]
            )
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                "for mup, embedding variance is independent of dmodel (1)"
                module.weight.data[self.tokenizer.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(module, "bias") and (module.bias is not None):
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, MuReadoutWrap):  # this is to pass coord check
            module.width_mult()

    def reset_parameters(self):
        """Meant to support fsdp1"""
        self.init_weights()


class Esm3TransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_config,
        layerwise_scale: float,
        linear_bias: bool = False,
        ff_activation: str = "SwiGLU",
        attn_method: str = "torch",
        ring_attn: Optional[str] = None,
        ring_attn_threshold: int = 1000,  # 1000000, #
        layernorm_eps: float = 1e-05,
        norm_method: str = "layernorm",
    ):
        super().__init__()
        assert norm_method in ["layernorm", "rmsnorm"]
        norm_fn = nn.LayerNorm if norm_method == "layernorm" else nn.RMSNorm
        self.attn1_layernorm = norm_fn(
            embed_dim, eps=layernorm_eps
        )  # they kept bias here
        self.ff_layernorm = norm_fn(embed_dim, eps=layernorm_eps)
        self.self_attn = Esm3TorchMHASelf(
            layernorm_eps=layernorm_eps,
            attn_method=attn_method,
            ring_attn=ring_attn,
            **attention_config,
        )
        if ff_activation == "gelu":
            # modularize to elimiate conidition in forward
            self.ffn = FFNgelu(embed_dim, ffn_embed_dim, linear_bias=linear_bias)
        elif ff_activation == "SwiGLU":
            self.ffn = FFNSwiGLU(embed_dim, ffn_embed_dim, linear_bias=linear_bias)
        elif ff_activation == "SwiGLUShard":
            self.ffn = FFNSwiGLU(
                embed_dim, ffn_embed_dim, linear_bias=linear_bias, unfused=True
            )

        self.attn_method = attn_method
        self.ring_attn = ring_attn
        if self.ring_attn is not None:
            assert self.attn_method == "fa2pad", (
                "for now, use FlashAttention with ring_attention"
            )
        self.layerwise_scale = layerwise_scale

    def forward(
        self,
        x,
        cos_cached=None,
        sin_cached=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        residual = x
        x = self.attn1_layernorm(x.float()).type_as(x)
        x = self.self_attn(x, cos_cached, sin_cached, self_attn_padding_mask)
        x = residual + x / self.layerwise_scale
        residual = x
        x = self.ff_layernorm(x.float()).type_as(x)
        x = self.ffn(x)
        x = residual + x / self.layerwise_scale
        return x


class FFNgelu(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, linear_bias=True):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim, bias=linear_bias)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim, bias=linear_bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class FFNSwiGLU(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, linear_bias=True, unfused=False):
        "use unfused SwiGLU SwiGLUShard if using TensorParallel"
        super().__init__()
        if unfused:
            self.gfc = SwiGLUShard(embed_dim, ffn_embed_dim, bias=linear_bias)
        else:
            self.gfc = SwiGLU(embed_dim, ffn_embed_dim, bias=linear_bias)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim, bias=linear_bias)

    def forward(self, x):
        x = self.gfc(x)
        x = self.fc2(x)
        return x


class QKNormRearrange(nn.Module):
    def __init__(self, embed_dim, bias, layernorm_eps, head_dim, **kwargs) -> None:
        super().__init__()
        self.head_dim = head_dim  # head dim doesn't change
        self.q_layernorm = nn.LayerNorm(embed_dim, bias=bias, eps=layernorm_eps)
        self.k_layernorm = nn.LayerNorm(embed_dim, bias=bias, eps=layernorm_eps)
        self.tp_mesh = None

    def forward(self, q, k, v):
        if self.tp_mesh is not None:
            q, k, v = self.tp_in(q, k, v)
        q, k = (
            self.q_layernorm(q.float()).type_as(q),
            self.k_layernorm(k.float()).type_as(k),
        )
        q, k, v = [
            rearrange(t, "b s (h d) -> b s h d", d=self.head_dim) for t in [q, k, v]
        ]
        if self.tp_mesh is not None:
            q, k, v = self.tp_out(q, k, v)
        return q, k, v


class Esm3TorchMHASelf(nn.Module):
    """Self-attention using torch.F.scaled_dot_product_attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_special_heads: int = 0,
        bias=False,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        use_rotary_emb=True,
        rotary_base=10000,
        layernorm_eps=1e-05,
        lora_qv_rank=None,
        lora_alpha=1,
        qk_layernorm: bool = True,
        attn_method: str = "torch",
        ring_attn: bool = False,
        context_length: int = 1024,
        heads_k_stride=2,  # used by llama sequence parallel only
        clamp_qkv: Optional[float]=None,  # Clamp the of qkv tensors between +-clamp_qkv for stability; positive float
        **kwargs,
    ) -> None:
        assert batch_first
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.dropout_p = attention_dropout
        self.attn_method = attn_method
        self.num_special_heads = num_special_heads

        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        # self.scaling = self.head_dim ** -0.5 # standard
        self.scaling = 1 / self.head_dim  # muTransfer
        self.rope_seq_dim = -3
        self.special_head_mask = [1] * (self.num_heads-self.num_special_heads) + [0] * (self.num_special_heads)
        self.clamp_qkv = clamp_qkv
        self.qk_layernorm = qk_layernorm
        if self.qk_layernorm:
            self.qknorm_rearrange = QKNormRearrange(
                self.embed_dim,
                bias=bias,
                layernorm_eps=layernorm_eps,
                head_dim=self.head_dim,
            )
            # self.q_layernorm = nn.LayerNorm(self.embed_dim, bias=bias, eps=layernorm_eps)
            # self.k_layernorm = nn.LayerNorm(self.embed_dim, bias=bias, eps=layernorm_eps)

        self.use_rotary_emb = use_rotary_emb

        if lora_qv_rank is not None:
            self.Wq = lora.Linear(
                embed_dim, embed_dim, bias=bias, r=lora_qv_rank, lora_alpha=lora_alpha
            )
            self.Wk = lora.Linear(
                embed_dim, embed_dim, bias=bias, r=lora_qv_rank, lora_alpha=lora_alpha
            )
            self.Wv = lora.Linear(
                embed_dim, embed_dim, bias=bias, r=lora_qv_rank, lora_alpha=lora_alpha
            )
        else:
            self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.Wk = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.Wv = nn.Linear(embed_dim, embed_dim, bias=bias)

        # num_heads=num_heads//tp_mesh.size when tp used, head_dim unchanged
        self.mha = FlashAttnSelf(
            self.dropout_p, ring_attn=ring_attn, heads_k_stride=heads_k_stride
        )

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x,
        cos_cached=None,
        sin_cached=None,
        key_padding_mask=None,
        need_weights=False,
        is_test: bool = False,
        qk_store_test=None,
    ):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        Wqkv is unfused for TP in case we need to initialize q,k,v differently and Emb get unpacked in wrong order
            TODO: evaluate difference
        FlashAttn uses b s h d (priority given for RingAttn)
        TorchAttn uses b h s d
        """
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        
        tp_mesh = self.qknorm_rearrange.tp_mesh
        
        # bsz, tgt_len, head_dim -> bsz, tgt_len, num_heads, head_dim
        if self.qk_layernorm:
            q, k, v = self.qknorm_rearrange(
                q, k, v
            )  # torch tensor parallel doesn't like unbind used by rearrange
        else:
            q, k, v = rearrange(
                [q, k, v], "three b s (h d) -> three b s h d", three=3, d=self.head_dim
            )  # use head_dim which is invariant to tensor sharding
        batch_size, _, local_num_heads, _ = q.shape
        
        if self.use_rotary_emb:
            if self.num_special_heads > 0:
                assert isinstance(cos_cached, list) 
                assert len(cos_cached) == 2
                
                if tp_mesh != None: 
                    tp_size = tp_mesh.shape[0]
                    tp_rank = tp_mesh.get_coordinate()[0]
                    chunk_size = self.num_heads // tp_size
                    extra_heads = self.num_heads % tp_size

                    if tp_rank < extra_heads: # If H % P b	  0, the first H % P ranks get one extra head
                        start = tp_rank * (chunk_size + 1)
                        end = start + (chunk_size + 1)
                    else:
                        start = extra_heads * (chunk_size + 1) + (tp_rank - extra_heads) * chunk_size
                        end = start + chunk_size
                    special_head_mask_local = self.special_head_mask[start:end]
                    num_first_trig = sum(special_head_mask_local)
                    num_second_trig = local_num_heads - num_first_trig
                    # print(f"Rank {tp_rank} / slice {start}:{end} / 1: {num_first_trig}, 2: {num_second_trig} / local: {local_num_heads} / all: {self.num_heads}")
                else:
                    special_head_mask_local = self.special_head_mask
                    
                num_first_trig = sum(special_head_mask_local)
                num_second_trig = local_num_heads - num_first_trig
                
                if num_second_trig > 0:
                    # Apply different RoPE rotations for the two different head groups.
                    q, q2 = torch.split(
                        q,
                        [num_first_trig, num_second_trig],
                        dim=2,
                    )
                    k, k2 = torch.split(
                        k,
                        [num_first_trig, num_second_trig],
                        dim=2,
                    )
                    q = apply_rotary_pos_emb(
                        q, cos_cached[0], sin_cached[0], seq_dimension=-3
                    ).type_as(q)
                    k = apply_rotary_pos_emb(
                        k, cos_cached[0], sin_cached[0], seq_dimension=-3
                    ).type_as(k)
                    q2 = apply_rotary_pos_emb(
                        q2, cos_cached[1], sin_cached[1], seq_dimension=-3
                    ).type_as(q2)
                    k2 = apply_rotary_pos_emb(
                        k2, cos_cached[1], sin_cached[1], seq_dimension=-3
                    ).type_as(k2)

                    # Recombines the embedding dimensions
                    q = torch.concat([q, q2], dim=2)
                    k = torch.concat([k, k2], dim=2)
                else:
                    q = apply_rotary_pos_emb(
                        q, cos_cached[0], sin_cached[0], seq_dimension=-3
                    ).type_as(q)
                    k = apply_rotary_pos_emb(
                        k, cos_cached[0], sin_cached[0], seq_dimension=-3
                    ).type_as(k)
            else:
                if is_test:
                    qk_store_test[0] = (q, k)
                q = apply_rotary_pos_emb(
                    q, cos_cached, sin_cached, seq_dimension=-3
                ).type_as(q)
                k = apply_rotary_pos_emb(
                    k, cos_cached, sin_cached, seq_dimension=-3
                ).type_as(k)

                if is_test:
                    qk_store_test[1] = (q, k)

        if self.clamp_qkv is not None:
            # Clamp values within +- clamp_qkv (optional); Potentially help stability
            # (e.g., https://github.com/allenai/OLMo/blob/a87c459d038c049045b09a05c4987fdddb01393e/olmo/model.py#L940)
            q, k, v = [
                torch.clamp(t, min=-self.clamp_qkv, max=self.clamp_qkv) for t in [q,k,v]
            ]
        if self.attn_method == "fa2pad":
            if key_padding_mask is not None:
                key_padding_mask = ~key_padding_mask
            context = self.mha(
                q, k, v, key_padding_mask=key_padding_mask, scale=self.scaling
            )
            # torch.cond only works when entire function can be captured by compiled.
            # context = torch.cond(q.shape[-2] < 100000, self.mha, self.mha_ring,(q, k, v))
            context = rearrange(
                context, "b s h d -> b s (h d)", b=batch_size, d=self.head_dim
            )  # head dim doesn't change
        else:
            q, k, v = rearrange(
                [q, k, v], "three b s h d -> three b h s d", three=3, d=self.head_dim
            )
            if key_padding_mask is not None:
                key_padding_mask = rearrange(~key_padding_mask, "b s -> b 1 1 s")
            # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,  # require  [b h s d]
                attn_mask=key_padding_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )

            context = rearrange(
                context, "b h s d -> b s (h d)", b=batch_size, d=self.head_dim
            )  # head dim doesn't change

        return self.out_proj(context)


class RobertaLMHead(nn.Module):
    """Head for masked language modeling, with muTransfer support
    Keep MuReadoutWrap and MuSharedReadout for now for backwards compat
    """

    def __init__(
        self,
        embed_dim,
        output_dim,
        mu=True,
        tie_weight=None,
        readout_zero_init=False,
        output_mult=1.0,
        base_fan_in=None,
    ):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gelu = nn.GELU()
        self.layernorm = nn.LayerNorm(embed_dim)
        self.tie_weight = tie_weight
        if mu:
            if tie_weight is None:
                if base_fan_in is None:
                    self.out = MuReadoutWrap(
                        embed_dim,
                        output_dim,
                        bias=True,
                        readout_zero_init=readout_zero_init,
                        output_mult=output_mult,
                    )
                else:
                    self.out = MuReadoutS(
                        embed_dim,
                        output_dim,
                        bias=True,
                        readout_zero_init=readout_zero_init,
                        output_mult=output_mult,
                        base_fan_in=base_fan_in,
                    )
            else:
                assert not readout_zero_init, "tied weights cannot zero"
                if base_fan_in is None:
                    self.out = MuSharedReadout(
                        tie_weight, bias=True, output_mult=output_mult
                    )
                else:
                    self.out = MuSharedReadoutS(
                        tie_weight,
                        bias=True,
                        output_mult=output_mult,
                        base_fan_in=base_fan_in,
                    )
        else:
            self.out = nn.Linear(embed_dim, output_dim, bias=True)
            if tie_weight is not None:
                self.out.weight = tie_weight

    def forward(self, x):
        x = self.dense(x)
        x = self.gelu(x)
        x = self.layernorm(x.float()).type_as(x)
        # project back to size of vocabulary with bias
        x = self.out(x)
        return x


def esm3s_fsdp1_wrap_policy(
    module: nn.Module, recurse: bool, nonwrapped_numel: int
) -> bool:
    # Wrap the top layer
    if recurse:
        return True
    if isinstance(module, Esm3TransformerLayer):
        return True


MUP_SHAPES = {
    # assumes FFN is 2.5x of emb dim
    # shapes are [fan-out, fan-in] except embedding
    "positional_embedding.fragment_index.weight": [None, 320],
    "emb_layernorm_after.bias": [320],
    "emb_layernorm_after.weight": [320],
    "embed_tokens.weight": [None, 320],
    "attn1_layernorm.bias": [320],
    "attn1_layernorm.weight": [320],
    "self_attn.Wqkv.weight": [960, 320],
    "self_attn.Wq.weight": [320, 320],
    "self_attn.Wk.weight": [320, 320],
    "self_attn.Wv.weight": [320, 320],
    "self_attn.qknorm_rearrange.k_layernorm.weight": [320],
    "self_attn.qknorm_rearrange.q_layernorm.weight": [320],
    "self_attn.out_proj.weight": [320, 320],
    "ff_layernorm.bias": [320],
    "ff_layernorm.weight": [320],
    "ffn.gfc.linear.weight": [1600, 320],
    "ffn.gfc.linear1.weight": [800, 320],  # when sharding
    "ffn.gfc.linear2.weight": [800, 320],  # when sharding
    "ffn.fc2.weight": [320, 800],
    "lm_head.dense.weight": [320, 320],
    "lm_head.layernorm.bias": [320],
    "lm_head.layernorm.weight": [320],
    "lm_head.out.bias": [None],
    "lm_head.out.weight": [None, 320],
}
