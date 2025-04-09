import torch
from einops import repeat


class ModRope:
    """
    Generate modified cos/sin tensors based on rotary positional encoding config.
    This implementation supports:
    1. Standard RoPE (Rotary Position Embeddings)
    2. Long-range RoPE for extended context
    3. Fragment-aware positional encodings
    """

    def __init__(self, rotary_config):
        """
        Initialize RoPE with configuration parameters.
        
        Args:
            rotary_config: Dictionary containing:
                - head_dim: Dimension of attention heads
                - num_long_range_heads: Number of scaled RoPE heads (not used here; used in model forward pass)
                - use_long_range_rope: Whether to use scaled RoPE for longer sequences
                - use_frag_rotary_embed: Whether to use fragment-specific embeddings
                - frag_max_index: Maximum number of fragments (if using fragments)
                - frag_rotary_base: Base value for fragment embeddings
                - rope_indices_to_replace: Which indices to replace with fragment embeddings
                - frag_rope_indices_to_copy: Which fragment indices to replace rope indices with
        """
        self.rotary_config = rotary_config
        self.head_dim = self.rotary_config["head_dim"]
        self.use_long_range_rope = rotary_config["use_long_range_rope"]
        self.use_frag_index = self.rotary_config["use_frag_rotary_embed"]
        
        # Initialize fragment-head-specific variables if needed
        if self.use_frag_index:
            # Generate cos/sin embeddings for fragments
            self.frag_cos, self.frag_sin = self.get_rot_emb(
                self.rotary_config["frag_max_index"],
                rotary_base=self.rotary_config["frag_rotary_base"],
            )
            # Parse indices for embedding replacement
            self.rope_indices_to_replace = list(
                map(int, self.rotary_config["rope_indices_to_replace"].split(","))
            )
            self.frag_rope_indices_to_copy = list(
                map(int, self.rotary_config["frag_rope_indices_to_copy"].split(","))
            )
            

    def get_rot_emb(
        self,
        end_index: int,
        dtype=torch.float32,
        start_index: int = 0,
        rope_factor=None,
        rotary_base: int = 10000,
    ):
        """
        Generate basic rotary embeddings for a range of positions.
        
        Args:
            end_index: Last position to generate embeddings for
            dtype: Tensor data type
            start_index: First position to generate embeddings for
            rope_factor: Scaling factor for RoPE (defaults to head_dim)
            rotary_base: Base for wavelength computation (default 10000)
            
        Returns:
            tuple: (cos, sin) 
                - cos (torch.Tensor): Precomputed cosine values for rotary embeddings, shape (seq_len, head_dim).
                - sin (torch.Tensor): Precomputed sine values for rotary embeddings, shape (seq_len, head_dim).
        """
        # Set rope factor to head_dim if not provided
        if rope_factor is None:
            rope_factor = self.head_dim

        # Generate wavelengths using a geometric sequence
        inv_freq = 1.0 / (
            rotary_base
            ** (
                torch.arange(rope_factor - self.head_dim, rope_factor, 2, dtype=dtype)
                / rope_factor
            )
        )

        # Generate position indices
        t = torch.arange(start_index, end_index, dtype=dtype)

        # Compute outer product to get position-wavelength interactions
        freqs = torch.outer(t, inv_freq)

        # Generate sin/cos patterns and duplicate dimensions
        _cos = repeat(torch.cos(freqs).to(dtype), "... d -> ... (2 d)")
        _sin = repeat(torch.sin(freqs).to(dtype), "... d -> ... (2 d)")

        return _cos, _sin

    def get_rope_tensor_frags(
        self, start_end_indices: list, rope_factor: int = None, rotary_base=None
    ):
        """
        Generate RoPE tensors for sequence fragments.
        
        Args:
            start_end_indices: List of (start, end) tuples for each fragment
            rope_factor: Optional scaling factor for RoPE
            rotary_base: Optional base value for wavelength computation
            
        Returns:
            tuple: (cos_tensors, sin_tensors) for all fragments
        """
        if rotary_base is None:
            rotary_base = self.rotary_config["rotary_base"]
            
        # Lists to collect embeddings for each fragment
        cos_i, sin_i = [], []
        if self.use_frag_index:
            frag_cos_i, frag_sin_i = [], []
            
        # Generate embeddings for each fragment
        for frag_i, (start_i, end_i) in enumerate(start_end_indices):
            # Get basic rotary embeddings for this fragment
            cos, sin = self.get_rot_emb(
                end_index=end_i,
                start_index=start_i,
                rope_factor=rope_factor,
                rotary_base=rotary_base,
            )
            cos_i.append(cos)
            sin_i.append(sin)
            
            # If using fragment indices, prepare fragment-specific embeddings
            if self.use_frag_index:
                # Repeat fragment embeddings to match sequence length
                frag_cos_i.append(
                    self.frag_cos[frag_i].unsqueeze(0).repeat(end_i - start_i, 1)
                )
                frag_sin_i.append(
                    self.frag_sin[frag_i].unsqueeze(0).repeat(end_i - start_i, 1)
                )
                
        # Combine all fragment embeddings
        cos = torch.concat(cos_i)
        sin = torch.concat(sin_i)
        
        # Replace specified positions with fragment-specific embeddings
        if self.use_frag_index:
            frag_cos = torch.concat(frag_cos_i)
            frag_sin = torch.concat(frag_sin_i)
            cos[
                :, self.rope_indices_to_replace[0] : self.rope_indices_to_replace[1]
            ] = frag_cos[
                :, self.frag_rope_indices_to_copy[0] : self.frag_rope_indices_to_copy[1]
            ]
            sin[
                :, self.rope_indices_to_replace[0] : self.rope_indices_to_replace[1]
            ] = frag_sin[
                :, self.frag_rope_indices_to_copy[0] : self.frag_rope_indices_to_copy[1]
            ]
        return cos, sin

    def get_rope_tensors(self, start_end_indices):
        """
        Main entry point for generating RoPE tensors.
        Handles both standard and long-range RoPE if configured.
        
        Args:
            start_end_indices: List of (start, end) tuples for sequence fragments
            
        Returns:
            tuple: (cos_tensors, sin_tensors), where each can be a list if using long-range RoPE
        """
        # Get standard RoPE tensors
        cos, sin = self.get_rope_tensor_frags(start_end_indices)
        
        # Optionally add long-range RoPE tensors with different scaling
        if self.use_long_range_rope:
            long_range_cos, long_range_sin = self.get_rope_tensor_frags(
                start_end_indices,
                rope_factor=self.rotary_config["rotary_base_scaling_factor"],
                rotary_base=self.rotary_config["long_range_rope_base"],
            )
            cos, sin = [cos, long_range_cos], [sin, long_range_sin]
        return cos, sin
    
    def collate_rope_tensors(self, batch, max_seq_len=None):
        """
        Collate function for batching rotary positional encoding (RoPE) tensors.

        Args:
            batch (List[torch.Tensor]): List of tensors where each tensor represents 
                                        cosine or sine embeddings of shape (seq_len, head_dim).

        Returns:
            torch.Tensor: Padded tensor of shape (batch_size, max_seq_len, head_dim), 
                        where max_seq_len is the maximum sequence length in the batch
        """
        if max_seq_len == None:
            max_seq_len = max(tensor.shape[0] for tensor in batch)
        head_dim = batch[0].shape[1]
        padded_batch = torch.zeros(len(batch), max_seq_len, head_dim, dtype=batch[0].dtype, device=batch[0].device)
        for i, tensor in enumerate(batch):
            seq_len = tensor.shape[0]
            padded_batch[i, :seq_len, :] = tensor
            
        return padded_batch
