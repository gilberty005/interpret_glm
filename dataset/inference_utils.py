import torch
from einops import repeat


class ModRoPE:
    """
    Generate modified cos/sin tensors for benchmarking.
    """

    def __init__(self, rotary_config):
        self.rotary_config = rotary_config
        self.head_dim = self.rotary_config["head_dim"]
        self.use_special_rope = rotary_config.get('use_special_rope', False)
        self.use_frag_index = rotary_config.get('use_frag_rotary_embed', False)
        if self.use_frag_index:
            self.frag_cos, self.frag_sin = self.get_rot_emb(
                self.rotary_config["frag_max_index"],
                rotary_base=self.rotary_config["frag_rotary_base"],
            )
            self.frag_dimension_to = list(
                map(int, self.rotary_config["frag_dimension_to"].split(","))
            )
            self.frag_dimension_from = list(
                map(int, self.rotary_config["frag_dimension_from"].split(","))
            )

    def get_rot_emb(
        self,
        end_index: int,
        dtype=torch.float32,
        start_index=0,
        rope_factor=None,
        rotary_base=10000,
    ):
        if rope_factor == None:
            rope_factor = self.head_dim
        inv_freq = 1.0 / (
            rotary_base
            ** (
                torch.arange(rope_factor - self.head_dim, rope_factor, 2, dtype=dtype)
                / rope_factor
            )
        )
        t = torch.arange(start_index, end_index, dtype=dtype)
        freqs = torch.outer(t, inv_freq)

        _cos = repeat(torch.cos(freqs).to(dtype), "... d -> ... (2 d)")
        _sin = repeat(torch.sin(freqs).to(dtype), "... d -> ... (2 d)")

        return _cos, _sin

    def get_rope_tensor_frags(
        self, start_end_indices: list, rope_factor: int = None, rotary_base=None
    ):
        if rotary_base is None:
            rotary_base = self.rotary_config["rotary_base"]
        # Assert indices are sorted by 'end index' per batch!
        cos_i, sin_i = [], []
        if self.use_frag_index:
            frag_cos_i, frag_sin_i = [], []
        for frag_i, (start_i, end_i) in enumerate(start_end_indices):
            cos, sin = self.get_rot_emb(
                end_index=end_i,
                start_index=start_i,
                rope_factor=rope_factor,
                rotary_base=rotary_base,
            )
            cos_i.append(cos)
            sin_i.append(sin)
            if self.use_frag_index:
                frag_cos_i.append(
                    self.frag_cos[frag_i].unsqueeze(0).repeat(end_i - start_i, 1)
                )
                frag_sin_i.append(
                    self.frag_sin[frag_i].unsqueeze(0).repeat(end_i - start_i, 1)
                )
        cos = torch.concat(cos_i)
        sin = torch.concat(sin_i)
        if self.use_frag_index:
            frag_cos = torch.concat(frag_cos_i)
            frag_sin = torch.concat(frag_sin_i)
            cos[:, self.frag_dimension_to[0] : self.frag_dimension_to[1]] = frag_cos[
                :, self.frag_dimension_from[0] : self.frag_dimension_from[1]
            ]
            sin[:, self.frag_dimension_to[0] : self.frag_dimension_to[1]] = frag_sin[
                :, self.frag_dimension_from[0] : self.frag_dimension_from[1]
            ]
        return cos, sin

    def get_rope_tensors(self, start_end_indices):
        """
        start_end_indices batch_size-lengthed list, consisting of list of tuples
        tuple is length=2, (start_i, end_i)
        """
        cos, sin = self.get_rope_tensor_frags(start_end_indices)
        if self.use_special_rope:
            cos2, sin2 = self.get_rope_tensor_frags(
                start_end_indices,
                rope_factor=self.rotary_config["special_rope_factor"],
                rotary_base=self.rotary_config["special_rope_base"],
            )
            cos, sin = [cos, cos2], [sin, sin2]
        return cos, sin
