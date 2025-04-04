from pathlib import Path
from typing import Sequence, List, Literal, Tuple, Optional
import warnings
import math
import gc
import json
import pickle

import numpy as np
from einops import repeat
from torch.utils.data import Dataset
import torch
import zstandard

from .read_cache import ReadCache


class FastaDataset(Dataset):
    """
    Samples from an indexed fasta file with a given context length

    Args:
        context_length: length of the context
        generator: torch.Generator
        cluster_genome_mapping_file: path to the cluster-genome mapping file;
                                     A nested list [[(genome, genome_length), ...], ...] where inner list contains genomes in a cluster
        genome_contig_mapping_index_file: path to the genome-contig mapping diskcache file;
                                          A dictionary {genome1: [[contig1, contig1_length], ...], ...} where values are list of contigs and their lengths
        contig_seq_mapping_index_file: path to the contig-sequence mapping diskcache file;
                                       A dictionary {contig1: zstd(contig1_sequence), ...} where sequences are compressed using zstandard
        zstd_dict_file: path to the zstandard dictionary file
        epoch_len: length of the epoch
        crop_contig_prob: probability of cropping a contig
        cluster_sampling_strategy: sampling strategy based on cluster sizes
        return_contig_indices: return start and end indices of the fetched contigs
        rope_indices_concat_method: Strategy for adjusting RoPE positional indices when concatenating contigs
            zero = each contig starts at index 0
            continuous = use continuous indices across all contigs
            gap = add an arbitarily large gap between contigs
        add_separator: add separator token
        padding: apply padding to each contig
        bos_eos: add bos and eos tokens
        context_bucket: a range of context length values to sample from
        current_epoch: current epoch
        drop_last: drop the last batch
        static_batch_dim: static batch dimension
        infinite_data: infinite data
        crop_frac: fraction of the context length to crop a contig
        filter_contig: filter out contigs shorter than the context length
        poisson_lambda: parameter for Poisson used in choosing the # cropped fragments
        scale_poisson_lambda: scale Poisson_lambda based on context_length (overrides poisson_lambda)
    """

    def __init__(
        self,
        context_length: int,
        generator: torch.Generator,
        cluster_genome_mapping_file: Path,
        genome_contig_mapping_index_file: Path,
        contig_seq_mapping_index_file: Path,
        zstd_dict_file: Path,
        epoch_len: int = None,
        crop_contig_prob: float = 0,
        cluster_sampling_strategy: Literal[
            "none", "log"
        ] = "none",  # sampling strategy based on cluster sizes
        return_contig_indices: bool = False,
        rope_indices_concat_method: Literal["zero", "continuous", "gap"] = "gap",
        add_separator: bool = True,
        padding: bool = False,
        bos_eos: bool = False,
        context_bucket: Tuple[int, int] = None,
        current_epoch: int = None,
        drop_last: bool = None,
        static_batch_dim: bool = None,
        infinite_data: bool = None,
        crop_frac: float = None,
        filter_contig: bool = False,
        poisson_lambda: float = 1.0,
        scale_poisson_lambda: bool = False,
    ):
        super(FastaDataset).__init__()

        self.context_length = context_length
        self.crop_contig_prob = crop_contig_prob
        self.epoch_len = epoch_len
        self.generator = generator
        self.cluster_genome_mapping_file = cluster_genome_mapping_file
        self.genome_contig_mapping_index_file = genome_contig_mapping_index_file
        self.contig_seq_mapping_index_file = contig_seq_mapping_index_file
        self.cluster_sampling_strategy = cluster_sampling_strategy
        self.zstd_dict_file = zstd_dict_file
        self.return_contig_indices = return_contig_indices
        self.rope_indices_concat_method = rope_indices_concat_method
        self.add_separator = add_separator
        self.padding = padding
        self.bos_eos = bos_eos
        self.crop_frac = crop_frac
        self.filter_contig = filter_contig
        self.poisson_lambda = poisson_lambda
        if scale_poisson_lambda:
            self.poisson_lambda = (
                np.log(self.context_length) * 1.1149003006485818 - 8.165139999999997
            )  # slope and intercept derived from empirical results
        self.context_bucket = context_bucket
        if drop_last is not None:
            warnings.warn("drop_last is deprecated")
        if current_epoch is not None:
            warnings.warn("current_epoch is deprecated")
        if static_batch_dim is not None:
            warnings.warn("static_batch_dim is deprecated")
        if infinite_data is not None:
            warnings.warn("infinite_data is deprecated")

        self.genome_contig_map = None  # initialize after forking
        self.contig_seq_mapping = None  # initialize after forking

        # sequence decompressor
        with open(self.zstd_dict_file, "rb") as file:
            zdict = file.read()
        self.seq_decompressor = zstandard.ZstdDecompressor(
            dict_data=zstandard.ZstdCompressionDict(zdict),
        )

        # Initialize the genome, contig_identifier and length list
        self.init_genome_contig_length_list()
        if cluster_genome_mapping_file is None:
            self.genome_probabilities = [1.0] * self.num_genomes

    def init_genome_contig_length_list(self):
        """Initialize a list of tuples to store genome, contig_identifier and length"""
        # load the cluster-genome mapping list (list of list)
        with open(self.cluster_genome_mapping_file, "r") as f:
            cluster_length_mapping = json.load(f)

        # load the genome-contig mapping diskcache file
        # if self.genome_contig_map is None:
        genome_contig_map = ReadCache(self.genome_contig_mapping_index_file)

        # filter out contigs/genomes/clusters that are shorter than context_length
        cluster_mapping = []
        for cluster in cluster_length_mapping:
            genomes = []
            for genome in cluster:
                if self.filter_contig:
                    contigs = []
                    contig_list = genome_contig_map[genome[0]]
                    for contig in contig_list:
                        if contig[1] > self.context_length:
                            contigs.append(contig)
                    if len(contigs) > 0:
                        genomes.append(genome[0])
                else:
                    if genome[1] > self.context_length:
                        genomes.append(genome[0])
            if len(genomes) > 0:
                cluster_mapping.append(genomes)

        # calculate cluster probabilities based on cluster sizes
        cluster_sizes = [len(val) for val in cluster_mapping]
        if self.cluster_sampling_strategy == "log":
            cluster_probs = [np.log(size + 1) for size in cluster_sizes]
        elif self.cluster_sampling_strategy == "none":
            cluster_probs = cluster_sizes
        cluster_probs = cluster_probs / np.sum(cluster_probs)  # normalize

        genome_contig_map.close()
        self.cluster_mapping = cluster_mapping
        self.num_genomes = sum(cluster_sizes)
        self.cluster_probs = cluster_probs
        gc.collect()

    def init_indices_dataset(
        self,
        num_genomes: int = None,  # TODO set properly default num_genoems
        #  generator: torch.Generator = None
    ):
        """Initialize the indices dataset"""
        if num_genomes is None:
            raise ValueError("Set the epoch_len for MultiFastaDataset in the config")

        # Retrieve genome probabilities for the given dataset
        cluster_probs = torch.tensor(
            self.cluster_probs,
        )

        # Sample genome indices
        cluster_indices = torch.multinomial(
            input=cluster_probs,
            num_samples=num_genomes,
            replacement=True,
            generator=self.generator,
        )

        self.cluster_indices = cluster_indices

    def __len__(self) -> int:
        """Length of dataset=number of batches // devices"""
        return self.epoch_len

    def __getitem__(self, idx: int, intput_cluster_idx: bool = False):
        """Get the tuple of identifiers corresponding to idx"""
        # Initialize the indices dataset if not already initialized
        if not hasattr(self, "cluster_indices"):
            self.init_indices_dataset(
                num_genomes=self.epoch_len,
                #   generator=self.generator,
            )
        if self.genome_contig_map is None:
            self.genome_contig_map = ReadCache(self.genome_contig_mapping_index_file)
        if self.contig_seq_mapping is None:
            self.contig_seq_mapping = ReadCache(self.contig_seq_mapping_index_file)

        # Fetch the cluster index
        if intput_cluster_idx:
            cluster_idx = idx
        else:
            cluster_idx = self.cluster_indices[idx]

        # fetch a genome and its contigs from the cluster
        clust_size = len(self.cluster_mapping[cluster_idx])
        genome_idx = torch.multinomial(
            input=torch.Tensor([1 / clust_size] * clust_size),
            num_samples=1,
            generator=self.generator,
        ).item()
        genome = self.cluster_mapping[cluster_idx][genome_idx]
        contig_list = self.genome_contig_map[genome]
        if self.filter_contig:
            contig_list = [
                contig for contig in contig_list if contig[1] > self.context_length
            ]

        # set the seed for the generator; TODO set seed for each worker process
        # Fetch the contigs
        if self.context_bucket is not None:
            context_length = torch.randint(
                self.context_bucket[0],
                self.context_bucket[1],
                (1,),
                generator=self.generator,
            ).item()
        else:
            context_length = self.context_length
        if self.padding:  # apply padding to each contig
            return self.get_padding_seq(
                contig_list=contig_list, context_length=context_length
            )
        else:
            filtered_contigs = [
                contig for contig in contig_list if contig[1] > self.context_length
            ]
            if (
                torch.rand((1,), generator=self.generator).item()
                < self.crop_contig_prob
                and len(filtered_contigs) > 0
            ):
                # crop a contig longer than the context length
                return self.get_cropped_seq(
                    contig_list=filtered_contigs, context_length=context_length
                )
            else:  # concatenate contigs
                return self.get_concatenate_seq(
                    contig_list=contig_list, context_length=context_length
                )

    def redistribute_gap_lengths(self, gaps):
        """Adjusts gap lengths to ensure middle gaps are at least 1.
        Takes from the edge gaps first, then from the larger middle gaps if needed.
        
        Args:
            gaps: List of gap lengths [first_gap, *middle_gaps, last_gap]
        Returns:
            List of adjusted gap lengths
        """
        # If there are less than 3 gaps, no distribution is needed
        if len(gaps) < 3:
            return gaps

        # Split gaps into edge and middle sections
        first_gap = gaps[0]
        last_gap = gaps[-1]
        middle_gaps = gaps[1:-1]

        # Try to fix any middle gaps < 1
        for i, _ in enumerate(middle_gaps):
            while middle_gaps[i] < 1:  # Keep trying until this gap >= 1
                # First try taking from first gap
                if first_gap > 0:
                    first_gap -= 1
                    middle_gaps[i] += 1
                    continue

                # Then try taking from the last gap
                if last_gap > 0:
                    last_gap -= 1
                    middle_gaps[i] += 1
                    continue

                # Finally try taking from larger middle gaps
                found_source = False
                for j, larger_gap in enumerate(middle_gaps):
                    if j != i and larger_gap > 1:
                        middle_gaps[j] -= 1
                        middle_gaps[i] += 1
                        found_source = True
                        break
                
                if found_source:
                    continue  # Go back to while loop to check if we need more
                raise ValueError("Cannot ensure all middle gaps >= 1")

        return [first_gap] + middle_gaps + [last_gap]

    def divide_k_into_n(self, k, n, j):
        """Divide k into n parts where each part is at least of size j"""
        if k <= 0 or n <= 0:
            raise ValueError("k and n must be positive")
        if k < n * j:
            raise ValueError("k is too small to ensure each part is at least j")

        # Handle case where k = n*j (equal division)
        if k == n * j:
            return [j] * n

        # Subtract minimum from total
        k -= n * j
        divisions = sorted(
            [
                torch.randint(
                    0, k + 1, size=(1,), generator=self.generator
                ).item()  # k + 1 to include k
                for _ in range(n - 1)
            ]
        )

        # Convert division points to parts and add minimum j back
        parts = (
            [divisions[0]]  # First part: 0 to first division
            + [
                divisions[i + 1] - divisions[i] for i in range(n - 2)
            ]  # Middle parts: differences between divisions
            + [k - divisions[-1]]  # Last part: last division to k
        )
        return [
            p + j for p in self.redistribute_gap_lengths(parts)
        ]  # Add minimum j back and ensure gaps

    def get_padding_seq(self, contig_list: List[Tuple[int, int]], context_length: int):
        """Fetch a contig from the given genome for padding experiment

        Args:
            contig_list: List of tuples containing the contig_index and length
            context_length: length of the context
        Returns:
            seq: A fetched contig
            start_end_idx: start and end index of the fetched contig (optional)
            sep_type: separator token placeholder for consistency
        """
        # randomly shuffle the contig list
        shuffled_indices = torch.randperm(
            len(contig_list), generator=self.generator
        ).tolist()
        contig_list = [contig_list[i] for i in shuffled_indices]

        if self.padding:  # assume that the dataset is processed such that all contigs are shorter than the context length
            contig_idx, contig_length = contig_list[0]
            seq = pickle.loads(
                self.seq_decompressor.decompress(self.contig_seq_mapping[contig_idx])
            )
            assert len(seq) <= context_length, (
                "All contigs should be shorter than the context length"
            )
            if self.return_contig_indices:
                return [[seq], [(0, len(seq) - 1)], None]
            return [[seq], None]

    def get_cropped_seq(self, contig_list: List[Tuple[int, int]], context_length: int):
        """Fetch a contig from the given genome and crop it into multiple fragments

        Args:
            contig_list: List of tuples containing the contig_index and length; pre-filtered such that all contigs
                         are longer than the context length
            context_length: length of the context
        Returns:
            contigs: List of cropped fragments
            start_end: List of tuples of start and end indices for each cropped fragment (optional)
            sep_type: separator token type ("crop_separator")
        """
        # randomly shuffle the contig list
        shuffled_indices = torch.randperm(
            len(contig_list), generator=self.generator
        ).tolist()
        contig_list = [contig_list[i] for i in shuffled_indices]

        # fetch a contig that is as long as the context_length
        contig_idx, contig_length = contig_list[0]

        # fetch a sequence for the contig
        try:
            seq = pickle.loads(
                self.seq_decompressor.decompress(self.contig_seq_mapping[contig_idx])
            )
        except:  # TODO handle exception (should not happen if the dataset is properly constructed)
            raise ValueError(
                f"Could not fetch sequence for {contig_idx} at {start_idx} to {start_idx + frag_len}"
            )

        # num_frag = int(np.random.uniform(3, 5))
        # num_frag= math.ceil(self.context_length / (self.crop_frac * self.context_length))
        if self.crop_frac is not None:
            num_frag = min(
                math.ceil(1 / self.crop_frac), contig_length - self.context_length + 1
            )  # ensure at least two fragments
        else:
            sampled_num = int(
                torch.distributions.Poisson(
                    self.poisson_lambda,
                )
                .sample([1])
                .item()
            )
            num_frag = min(max(1, sampled_num), contig_length - context_length + 1)

        # distribute context_length evenly across fragments
        # adjust context_length to account for separator tokens
        if self.add_separator:
            adj_context_length = context_length - num_frag + 1
        else:
            adj_context_length = context_length
        frag_lens = [adj_context_length // num_frag] * num_frag
        remaining = adj_context_length - sum(frag_lens)
        for i in range(remaining):
            frag_lens[i % num_frag] += 1

        # assign fragments to a region on the contig such that they are non-overlapping
        start_end = []
        start_idx = 0
        contigs = []
        gaps = contig_length - sum(frag_lens)

        gaps_list = self.divide_k_into_n(
            gaps, num_frag + 1, int(gaps / ((num_frag + 1) * 2))
        )
        for i, frag_len in enumerate(frag_lens):
            gap = gaps_list[i]
            start_idx = start_idx + gap
            if (
                i > 0 and self.add_separator
            ):  # account for the added separator token from the gap (intermediate gaps are non-zero)
                start_idx -= 1
            end_idx = start_idx + frag_len
            contigs.append(seq[start_idx:end_idx])
            if (
                i < len(frag_lens) - 1 and self.add_separator
            ):  # account for the separator token
                end_idx += 1
            start_end.append((start_idx, end_idx))
            start_idx = end_idx

        if self.add_separator:
            assert (
                sum([len(contig) for contig in contigs])
                == context_length - len(start_end) + 1
            ), "context length doesn't match {}, {}, {}, {}, {}, {}".format(
                frag_lens,
                [len(contig) for contig in contigs],
                len(seq),
                start_end,
                gaps_list,
                contig_length,
            )
        else:
            assert sum([len(contig) for contig in contigs]) == context_length, (
                "context length doesn't match {}, {}, {}, {}, {}, {}".format(
                    frag_lens,
                    [len(contig) for contig in contigs],
                    len(seq),
                    start_end,
                    gaps_list,
                    contig_length,
                )
            )

        if self.return_contig_indices:
            return [contigs, start_end, "crop_separator"]
        return [contigs, "crop_separator"]

    def get_concatenate_seq(
        self, contig_list: List[Tuple[int, int]], context_length: int
    ):
        """Fetch a set of contigs from the given genome
        until the context_length is satisfied

        Args:
            contig_list: List of tuples containing the contig_index and length
            context_length: length of the context
        Returns:
            contigs: List of sequences fetched from the contigs up to the context_length
            start_end: List of tuples of start and end indices (optional)
            sep_type: separator token type ("concat_separator")
        """
        # Sample random contigs from the given genome
        # until the context_length is satisfied
        contigs = []
        total_length = 0
        start_end = []

        # randomly shuffle the contig list
        shuffled_indices = torch.randperm(
            len(contig_list), generator=self.generator
        ).tolist()
        contig_list = [contig_list[i] for i in shuffled_indices]

        i = 0
        while total_length < context_length and i < len(contig_list):
            contig_idx, contig_length = contig_list[i]

            # Fetch a segment from the contig such that the
            # total length of the fetched segments is less than
            # or equal to the context_length
            remaining_len = context_length - total_length
            fetch_len = min(contig_length, remaining_len)
            # seq_idx = randrange(0, max(1, contig_length - fetch_len))
            seq_idx = 0
            try:
                seq = pickle.loads(
                    self.seq_decompressor.decompress(
                        self.contig_seq_mapping[contig_idx]
                    )
                )[seq_idx : seq_idx + fetch_len]
            except:  # TODO handle exception
                raise ValueError(
                    f"Could not fetch sequence for {contig_idx} at {seq_idx} to {seq_idx + fetch_len}"
                )
            end_idx = seq_idx + fetch_len

            # if the remaining length is less than 10, pad with Ns
            total_length += len(seq)
            if total_length + 10 >= context_length:
                pad_len = context_length - total_length
                seq += "N" * pad_len
                total_length += pad_len
                end_idx += pad_len
            start_end.append((seq_idx, end_idx))
            contigs.append(seq)

            # account for the separator token
            if total_length + 1 < context_length and self.add_separator:
                total_length += 1

            # handle bos/eos adjustments
            if self.bos_eos:
                if seq_idx == 0:
                    total_length += 1
                if fetch_len == contig_length:
                    total_length += 1
            i += 1

        # assert total_length - len(contigs) - 1 <= context_length, 'Total length of fetched contigs should be equal to the context_length'
        # adjust start_end by accounting for the separator token
        tmp_start_end = []
        for i, idx in enumerate(start_end):
            if i < len(start_end) - 1 and self.add_separator:
                tmp_start_end.append((idx[0], idx[1] + 1))
            else:
                tmp_start_end.append((idx[0], idx[1]))
        start_end = tmp_start_end

        total = 0
        for i, idx in enumerate(start_end):
            total += idx[1] - idx[0]
        assert total == context_length, (
            "start end indices does not match context length: {}".format(total)
        )

        # Add arbitrary positional index gaps between concatenated contigs
        if (
            self.rope_indices_concat_method in ["gap", "continuous"]
            and len(start_end) > 1
        ):
            gapped_start_end = []
            genome_length = sum([contig[1] for contig in contig_list])
            total_contig_len = sum([idx[1] - idx[0] for idx in start_end])
            total_gap_len = genome_length - total_contig_len

            if self.rope_indices_concat_method == "gap":
                # Add arbitrary large gap in position index for RoPE tensors
                gaps_list = self.divide_k_into_n(
                    total_gap_len,
                    len(start_end) + 1,
                    int(total_gap_len / ((len(start_end) + 1) * 2)),
                )

            for i, contig_idx in enumerate(start_end):
                if self.rope_indices_concat_method == "gap":
                    gap = gaps_list[i]
                else:
                    gap = 0
                if i == 0:
                    start_idx, end_idx = contig_idx
                diff = contig_idx[1] - contig_idx[0]
                start_idx = start_idx + gap
                end_idx = start_idx + diff
                gapped_start_end.append((start_idx, end_idx))
                start_idx = end_idx

            start_end = gapped_start_end

        if self.return_contig_indices:
            return [contigs, start_end, "concat_separator"]
        return [contigs, "concat_separator"]

    def __del__(self):
        if self.genome_contig_map is not None:
            self.genome_contig_map.close()  # .cache.close()
            self.contig_seq_mapping.close()  # .cache.close()


class CollateFn(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    Modified version from ESM
    Masking inspired by:
    https://github.com/lucidrains/protein-bert-pytorch/blob/main/protein_bert_pytorch/protein_bert_pytorch.py
    """

    def __init__(
        self,
        tokenizer,
        generator: torch.Generator,
        context_length: int,
        use_long_range_rope: bool = False,
        rotary_base_scaling_factor: int = None,
        long_range_rope_base: Optional[int] = None,
        bos_eos: bool = False,
        cls: bool = False,
        mask_prob: float = 0.15,
        variable_masking: Optional[List[float]] = None,
        variable_mask_prob: float = 0.8,
        span_masking: float = 0.0,
        max_span_length: int = 6000,
        span_masking_dist: Optional[List[float]] = None,
        span_mask_uniform: bool = True,
        span_mask_only: bool = False,
        span_batch: bool = False,
        esm3_masking: bool = False,
        mode: Literal["mlm", "ar"] = "mlm",
        simple_masking_only: bool = False,
        padding: bool = False,
        device: str = "cpu",
    ):  # if this is in child process, must be cpu
        self.tokenizer = tokenizer
        self.generator = generator
        self.context_length = context_length
        self.use_long_range_rope = use_long_range_rope
        self.rotary_base_scaling_factor = rotary_base_scaling_factor
        self.long_range_rope_base = long_range_rope_base
        self.bos_eos = bos_eos
        self.cls = cls
        self.mask_prob = mask_prob
        self.variable_masking = variable_masking
        if variable_masking is not None:
            assert len(variable_masking) == 2, (
                "variable_masking should be a list of 2 elements"
            )
        self.variable_mask_prob = variable_mask_prob
        assert self.variable_mask_prob <= 1.0 and self.variable_mask_prob >= 0.0, (
            "variable_mask_prob should be between 0 and 1"
        )
        self.span_masking = span_masking
        self.max_span_length = max_span_length
        self.span_masking_dist = span_masking_dist
        if span_masking_dist is not None:
            assert len(span_masking_dist) == 2, (
                "span_masking_dist should be a list of 2 elements"
            )
        else:
            self.span_masking_dist = [2, 3.6]
        self.span_mask_uniform = span_mask_uniform
        self.span_mask_only = span_mask_only
        self.span_batch = span_batch
        self.esm3_masking = esm3_masking
        self.mode = mode
        self.mode == "mlm"
        self.simple_masking_only = simple_masking_only
        self.mask_innerprob_mask = 0.8
        self.mask_innerprob_same = 0.1
        self.mask_innerprob_subs = 0.1
        self.padding = padding
        self.device = device

        # nucleotide_token_index: 4 nucleotides used for random replacement masking
        self.nucleotide_token_index = torch.tensor(
            self.tokenizer.encode("ACGT"), device=device
        )

    def mlm_batch(
        self,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize a binary mask tensor (True for nucleotides)
        mask = torch.isin(tokens, self.nucleotide_token_index)
        noised_tokens, noise_mask = self.train_masking(tokens, mask)

        return noised_tokens, noise_mask

    def get_mask_subset_with_fraction(
        self, mask: torch.Tensor, prob: float, variable: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Probability for mask=True, rounds up number of residues
        Inspired by
        https://github.com/lucidrains/protein-bert-pytorch/blob/main/protein_bert_pytorch/protein_bert_pytorch.py
        but it  gives bad results when:
        prob * seq_len > num_tokens (fixed)
        """
        batch, seq_len, device = *mask.shape, mask.device
        num_tokens = mask.sum(
            dim=-1, keepdim=True
        )  # number of non-masked tokens in each sequence

        torch.manual_seed(
            torch.randint(0, 9999999999999, (1,), generator=self.generator).item()
        )

        if variable is not None:
            # if self.esm3_masking is True, apply variable masking 80% of the time
            # and uniform masking 20% of the time
            if self.esm3_masking:
                weight = torch.multinomial(
                    torch.tensor(
                        [self.variable_mask_prob, 1 - self.variable_mask_prob],
                        dtype=torch.float,
                    ),
                    batch,
                    replacement=True,
                    generator=self.generator,
                )
                prob = torch.where(
                    weight == 0,
                    torch.distributions.beta.Beta(variable[0], variable[1])
                    .sample(
                        [batch],
                    )
                    .to(device),
                    torch.rand((batch,), generator=self.generator, device=device),
                ).unsqueeze(-1)
            else:
                prob = (
                    torch.distributions.beta.Beta(variable[0], variable[1])
                    .sample(
                        [batch],
                    )
                    .unsqueeze(-1)
                    .to(device)
                )
        num_to_mask = (
            (num_tokens * prob).floor().type(torch.int64).squeeze(1).tolist()
        )  # number of tokens to mask in each sequence
        max_masked = (
            torch.ceil(prob * num_tokens.max() + 1).max().type(torch.int64).item()
        )
        sampled_indices = -torch.ones(
            (batch, max_masked), dtype=torch.int64, device=device
        )

        # select random indices to mask
        indices = torch.arange(seq_len, device=device)
        for i in range(batch):
            # sample shuffled index instead of expensive topk
            valid_indices = indices[mask[i]]
            perm = torch.randperm(
                valid_indices.size(0), device=device, generator=self.generator
            )[: num_to_mask[i]]
            sampled_indices[i, : num_to_mask[i]] = valid_indices[perm]
            # rand = torch.rand((seq_len), device=device).masked_fill(~mask[i], -1e9)
            # _, sampled_indices[i,:num_to_mask[i]] = rand.topk(num_to_mask[i], dim=-1)

        sampled_indices = (
            sampled_indices + 1
        )  # padding is 0 allow excess scatter to index 0
        new_mask = torch.zeros((batch, seq_len + 1), device=device)
        new_mask.scatter_(-1, sampled_indices, 1)
        return new_mask[:, 1:].bool()  # index 0 removed

    def span_mask(
        self,
        mask: torch.Tensor,
        prob: float,
        variable: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Span masking with variable rate masking
        Span masking is a contiguous block of residues that are masked.
        The length of each span is drawn from #TODO distribution until the desired number of residues are masked.

        Args:
            mask: tensor of True for nucleotides
            prob: probability of masking
            variable: list of two floats for beta distribution

        Returns:
            new_mask: tensor of True for residues to be masked
        """
        batch, seq_len, device = *mask.shape, mask.device
        num_tokens = mask.sum(
            dim=-1, keepdim=True
        )  # number of non-masked tokens in each sequence

        torch.manual_seed(self.generator.seed())

        if variable is not None:
            if self.span_mask_uniform:
                weight = torch.multinomial(
                    torch.tensor(
                        [self.variable_mask_prob, 1 - self.variable_mask_prob],
                        dtype=torch.float,
                    ),
                    batch,
                    replacement=True,
                    generator=self.generator,
                )
                prob = torch.where(
                    weight == 0,
                    torch.distributions.beta.Beta(variable[0], variable[1])
                    .sample(
                        [batch],
                    )
                    .to(device),
                    torch.rand((batch,), generator=self.generator, device=device),
                ).unsqueeze(-1)
            else:
                prob = (
                    torch.distributions.beta.Beta(variable[0], variable[1])
                    .sample(
                        [batch],
                    )
                    .unsqueeze(-1)
                    .to(device)
                )
        num_to_mask = (
            (num_tokens * prob).floor().type(torch.int64).squeeze(1).tolist()
        )  # number of tokens to mask in each sequence
        max_masked = (
            torch.ceil(prob * num_tokens.max() + 1).max().type(torch.int64).item()
        )
        sampled_indices = -torch.ones(
            (batch, max_masked), dtype=torch.int64, device=device
        )

        # loop through each sequence in the batch and get contiguous segments with only A,C,G,T
        # save the start and end indices of the segments
        num_masked_list = []

        for i in range(batch):
            contiguous_segments = []
            start_idx = None
            for j in range(seq_len):
                if mask[i, j]:
                    if start_idx is None:
                        start_idx = j
                else:
                    if start_idx is not None:
                        if j - start_idx >= 100:
                            contiguous_segments.append((start_idx, j))
                        start_idx = None
            if start_idx is not None:
                if seq_len - start_idx >= 100:
                    contiguous_segments.append((start_idx, seq_len))

            # Sample spans from the contiguous segments
            sampled_spans = []
            span_lengths = [0]  # add the starting index for sampled_indices
            num_masked = 0

            while num_masked < num_to_mask[i] and contiguous_segments:
                span_idx = torch.randint(
                    len(contiguous_segments), (1,), generator=self.generator
                ).item()
                seg_start, seg_end = contiguous_segments.pop(span_idx)
                # span_length = min(seg_end - seg_start, min(num_to_mask[i] - num_masked, 6000))
                span_length = int(
                    max(
                        3,
                        torch.distributions.beta.Beta(
                            self.span_masking_dist[0], self.span_masking_dist[1]
                        )
                        .sample(
                            [1],
                        )
                        .item()
                        * min(seg_end - seg_start, self.max_span_length),
                    )
                )
                if num_masked + span_length > num_to_mask[i]:
                    break
                span_start = torch.randint(
                    seg_start, seg_end - span_length + 1, (1,), generator=self.generator
                ).item()
                span_lengths.append(span_length)
                sampled_spans.append((span_start, span_start + span_length))
                num_masked += span_length

            num_masked_list.append(num_masked)
            # If span_mask_only is set, we do not add individual token masking and only mask at the span-level. 
            # This occurs in various downstream benchmarks. 
            if self.span_mask_only: 
                num_to_mask[i] = num_masked

            for j, (span_start, span_end) in enumerate(sampled_spans):
                sampled_indices[
                    i, span_lengths[j] : span_lengths[j] + span_lengths[j + 1]
                ] = torch.arange(span_start, span_end, device=device)

        sampled_indices = (
            sampled_indices + 1
        )  # padding is 0 allow excess scatter to index 0
        new_mask = torch.zeros((batch, seq_len + 1), device=device)
        new_mask.scatter_(-1, sampled_indices, 1)
        span_mask = (
            new_mask[:, 1:].bool() | mask
        )  # index 0 removed; combine with original mask

        # if the number of masked residues is less than the desired number, mask the remaining residues
        # select random indices to mask
        sampled_indices_extra = -torch.ones(
            (batch, max_masked), dtype=torch.int64, device=device
        )
        indices = torch.arange(seq_len, device=device)
        sampled_indices -= 1  # adjust for 0-based indexing
        for i in range(batch):
            if num_masked_list[i] < num_to_mask[i]:
                valid_indices = indices[span_mask[i]]
                perm = torch.randperm(
                    valid_indices.size(0), device=device, generator=self.generator
                )[: num_to_mask[i] - num_masked_list[i]]
                extra_indices = valid_indices[perm]
                sampled_indices[i, num_masked_list[i] : num_to_mask[i]] = valid_indices[
                    perm
                ]
                sampled_indices_extra[i, num_masked_list[i] : num_to_mask[i]] = (
                    extra_indices
                )

        sampled_indices = (
            sampled_indices + 1
        )  # padding is 0 allow excess scatter to index 0
        new_mask = torch.zeros((batch, seq_len + 1), device=device)
        new_mask.scatter_(-1, sampled_indices, 1)
        noise_mask = new_mask[:, 1:].bool()  # index 0 removed

        # Build additional_mask from the extra indices:
        sampled_indices_extra = sampled_indices_extra + 1
        new_mask_extra = torch.zeros((batch, seq_len + 1), device=device)
        new_mask_extra.scatter_(-1, sampled_indices_extra, 1)
        single_tok_mask = new_mask_extra[:, 1:].bool()

        return noise_mask, single_tok_mask

    def reverse_mask_tokens(
        self,
        masked_tokens: torch.Tensor,
        tokens: torch.Tensor,
        reverse_prob: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly selects reverse_prob of the masked positions and replaces
        the masked token in those positions with the original token value.

        Args:
            masked_tokens: a tensor of tokens where some positions have been masked (e.g. replaced by a [MASK] token)
            tokens: the original (unmasked) tokens
            reverse_prob: the fraction of the masked positions to revert

        Returns:
            new_tokens: the tokens after reversing (unmasking) a fraction of the masked tokens
        """
        rand_vals = torch.rand_like(masked_tokens, dtype=torch.float)
        noise_mask = masked_tokens == self.tokenizer.mask_idx
        reverse_mask = noise_mask & (rand_vals < reverse_prob)
        new_tokens = masked_tokens.clone()
        new_tokens[reverse_mask] = tokens[reverse_mask]

        return new_tokens

    def train_masking(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Masking as described by ESM
        Because sum of probabilities of noise types add up to 1, treat as fraction instead

        Args:
            tokens: tensor of sequences
            mask: tensor of True for nucleotides

        Returns:
            noised_tokens: tensor of noised sequences
            noise_mask: tensor of True for nucleotides masked
        """
        if self.span_masking > 0:  # Todo, interleave span with BERT
            # sample which sequences to apply span masking
            apply_span_mask = (
                torch.rand(tokens.size(0), generator=self.generator) < self.span_masking
            )
            span_mask_indices = torch.nonzero(apply_span_mask).squeeze(1)
            non_span_mask_indices = torch.nonzero(~apply_span_mask).squeeze(1)

            if span_mask_indices.size(0) > 0:
                span_noise_mask, single_tok_mask = self.span_mask(
                    mask[span_mask_indices],
                    self.mask_prob,
                    variable=self.variable_masking,
                )
                span_noised_tokens = torch.where(
                    span_noise_mask, self.tokenizer.mask_idx, tokens[span_mask_indices]
                )

                # apply bert masking to single_tok_mask
                mask_mask, subs_mask = self.apply_bert_masking(single_tok_mask)
                span_noised_tokens = torch.where(
                    mask_mask, self.tokenizer.mask_idx, span_noised_tokens
                )

                rand_res_tokens = self.nucleotide_token_index[
                    torch.randint(
                        len(self.nucleotide_token_index),
                        span_noise_mask.shape,
                        generator=self.generator,
                    )
                ]
                span_noised_tokens = torch.where(
                    subs_mask, rand_res_tokens, span_noised_tokens
                )
                noised_tokens = span_noised_tokens
                noise_mask = span_noise_mask

            if non_span_mask_indices.size(0) > 0:
                non_span_noise_mask = self.get_mask_subset_with_fraction(
                    mask[non_span_mask_indices],
                    self.mask_prob,
                    variable=self.variable_masking,
                )
                if self.simple_masking_only:
                    mask_mask = non_span_noise_mask
                    non_span_noised_tokens = torch.where(
                        mask_mask,
                        self.tokenizer.mask_idx,
                        tokens[non_span_mask_indices],
                    )
                else:
                    mask_mask, subs_mask = self.apply_bert_masking(non_span_noise_mask)
                    non_span_noised_tokens = torch.where(
                        mask_mask,
                        self.tokenizer.mask_idx,
                        tokens[non_span_mask_indices],
                    )

                    rand_res_tokens = self.nucleotide_token_index[
                        torch.randint(
                            len(self.nucleotide_token_index),
                            mask[non_span_mask_indices].shape,
                            generator=self.generator,
                        )
                    ]
                    non_span_noised_tokens = torch.where(
                        subs_mask, rand_res_tokens, non_span_noised_tokens
                    )

                if span_mask_indices.size(0) > 0:
                    noised_tokens = torch.cat([noised_tokens, non_span_noised_tokens])
                    noise_mask = torch.cat([noise_mask, non_span_noise_mask])
                else:
                    noised_tokens = non_span_noised_tokens
                    noise_mask = non_span_noise_mask

            # concatenate span mask indices with non span mask indices
            # and reorder noised_tokens and noise_mask based on that
            combined_indices = torch.cat([span_mask_indices, non_span_mask_indices])
            noised_tokens = noised_tokens[combined_indices.argsort()]
            noise_mask = noise_mask[combined_indices.argsort()]

        else:
            noise_mask = self.get_mask_subset_with_fraction(
                mask, self.mask_prob, variable=self.variable_masking
            )
            if self.simple_masking_only:
                mask_mask = noise_mask
                noised_tokens = torch.where(mask_mask, self.tokenizer.mask_idx, tokens)
            else:
                mask_mask, subs_mask = self.apply_bert_masking(noise_mask)
                noised_tokens = torch.where(mask_mask, self.tokenizer.mask_idx, tokens)

                rand_res_tokens = self.nucleotide_token_index[
                    torch.randint(
                        len(self.nucleotide_token_index),
                        mask.shape,
                        generator=self.generator,
                    )
                ]
                noised_tokens = torch.where(subs_mask, rand_res_tokens, noised_tokens)
        return noised_tokens, noise_mask

    def apply_bert_masking(
        self,
        noise_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_mask = self.get_mask_subset_with_fraction(
            noise_mask, self.mask_innerprob_mask
        )
        # replace with random residue with probability mask_innerprob_same
        subs_same_mask = noise_mask * ~mask_mask
        subs_mask = self.get_mask_subset_with_fraction(
            subs_same_mask,
            self.mask_innerprob_subs
            / (self.mask_innerprob_same + self.mask_innerprob_subs),
        )

        return mask_mask, subs_mask

    def ar_batch(
        self,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            seq_indices, list
            seqs, list
            tokens, tensor for sequences
            noise_mask, tensor for masking (True for nucleotides masked)
        """
        data = tokens[:, :-1].clone()
        target = tokens[:, 1:].clone()

        return data, target

    def __call__(
        self, raw_batch: Sequence[Tuple[str, int]]
    ) -> Tuple[List[str], List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            seq_label list
            seq_str list
            tokenized seq tensor
            noise_mask True for residues used for training
            mask True for all residues (exclude cls/bos, eos, padding)
        """
        batch_size = len(raw_batch)
        if batch_size == 0:
            worker_info = torch.utils.data.get_worker_info()
            warnings.warn(f"Getting Batch size 0 from {worker_info.id}")
            return None

        seqs, seq_indices, chr_names = zip(*raw_batch)
        seqs_encoded = self.tokenizer.encode_batch(
            seqs, bos_eos=self.bos_eos, cls=self.cls
        )

        if self.padding:
            seqs_encoded_tokens = torch.full(
                [len(seqs), self.context_length],
                self.tokenizer.padding_idx,
                dtype=torch.int64,
                device=self.device,
            )
            for i in range(len(seqs_encoded)):
                seqs_encoded_tokens[i, : len(seqs_encoded[i])] = torch.tensor(
                    seqs_encoded[i], dtype=torch.int64, device=self.device
                )

        else:
            assert len(set([len(seq) for seq in seqs])) == 1, (
                "All sequences should be of the same length"
            )
            seqs_encoded_tokens = torch.tensor(
                seqs_encoded, dtype=torch.int64, device=self.device
            )

        if self.mode == "ar":
            data, target = self.ar_batch(seqs_encoded_tokens)
            return seqs, seq_indices, data, target
        elif self.mode == "mlm":
            noised_tokens, noise_mask = self.mlm_batch(seqs_encoded_tokens)
            return seqs, seq_indices, noised_tokens, seqs_encoded_tokens, noise_mask


class ModRopeCollateFn(CollateFn):
    def __init__(
        self,
        crop_separator: bool = False,
        concat_separator: bool = True,
        add_separator: bool = True,
        return_contig_indices: bool = False,
        rotary_config=None,
        pad_seq_base_length: int = 5024,  # Pad padded sequences to multiples of pad_seq_base_length for compile efficiency (round up)
        adj_batch_size: bool = False,
        num_context_lengths: Optional[
            int
        ] = None,  # Set in DNATrainDataModule, setting number of different context lengths per grad step.
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.crop_separator = crop_separator
        self.concat_separator = concat_separator
        self.add_separator = add_separator
        self.return_contig_indices = return_contig_indices
        self.pad_seq_base_length = pad_seq_base_length
        self.adj_batch_size = adj_batch_size
        self.num_context_lengths = num_context_lengths
        if self.return_contig_indices:
            assert rotary_config is not None
            self.rotary_config = rotary_config
            self.head_dim = self.rotary_config["head_dim"]
            self.use_frag_index = self.rotary_config["use_frag_rotary_embed"]
            if self.use_frag_index:
                self.frag_cos, self.frag_sin = self.get_rot_emb(
                    self.rotary_config["frag_max_index"],
                    rotary_base=self.rotary_config["frag_rotary_base"],
                )
                self.rope_indices_to_replace = list(
                    map(int, self.rotary_config["rope_indices_to_replace"].split(","))
                )
            self.frag_rope_indices_to_copy = list(
                map(int, self.rotary_config["frag_rope_indices_to_copy"].split(","))
            )

    def concat_seqs(
        self, seqs: List[str], sep=Literal["concat_separator", "crop_separator"]
    ) -> Tuple[str, int]:
        """Tokenize a set of contigs and concatenate them into a single sequence"""
        if not self.add_separator:
            crop_separator = False
            concat_separator = False
        elif self.add_separator and sep == "concat_separator":
            crop_separator = False
            concat_separator = True
        elif self.add_separator and sep == "crop_separator":
            crop_separator = True
            concat_separator = False
        else:
            raise ValueError(
                "sep must be either 'concat_separator' or 'crop_separator'"
            )

        # Tokenize the contigs
        tokenized_concat_seqs = self.tokenizer.encode_batch(
            seqs,
            bos_eos=self.bos_eos,
            cls=self.cls,
            crop_separator=crop_separator,
            concat_separator=concat_separator,
        )

        # Concatenate the tokenized sequences
        concat_seqs = []
        for seq in tokenized_concat_seqs:
            concat_seqs.extend(seq)

        return concat_seqs

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
        self,
        start_end_indices: list,
        rope_factor: int = None,
        rotary_base: Optional[int] = None,
        max_context_len: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rotary_base is None:
            rotary_base = self.rotary_config["rotary_base"]
        # Assert indices are sorted by 'end index' per batch!
        coss, sins = [], []
        for batch_indices in start_end_indices:
            cos_i, sin_i = [], []
            if self.use_frag_index:
                frag_cos_i, frag_sin_i = [], []
            total_len = 0
            for frag_i, (start_i, end_i) in enumerate(batch_indices):
                total_len += end_i - start_i
                if frag_i == len(batch_indices) - 1 and max_context_len is not None:
                    end_i += max_context_len - total_len
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
                cos[
                    :, self.rope_indices_to_replace[0] : self.rope_indices_to_replace[1]
                ] = frag_cos[
                    :,
                    self.frag_rope_indices_to_copy[0] : self.frag_rope_indices_to_copy[
                        1
                    ],
                ]
                sin[
                    :, self.rope_indices_to_replace[0] : self.rope_indices_to_replace[1]
                ] = frag_sin[
                    :,
                    self.frag_rope_indices_to_copy[0] : self.frag_rope_indices_to_copy[
                        1
                    ],
                ]
            coss.append(cos)
            sins.append(sin)
        return torch.stack(coss), torch.stack(sins)

    def __call__(
        self, raw_batch: Sequence[List[str]]
    ) -> Tuple[List[str], List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            raw_batch: List of tuples containing a list of contigs

        Returns:
            seq_label list
            seq_str list
            tokenized seq tensor
            noise_mask True for residues used for training
            mask True for all residues (exclude cls/bos, eos, padding)
        """
        batch_size = len(raw_batch)

        worker_info = torch.utils.data.get_worker_info()
        if batch_size == 0:
            warnings.warn(f"Getting Batch size 0 from {worker_info.id}")
            return None

        # adjust the batch size based on worker id
        if self.adj_batch_size:
            adj_batch_size = int(batch_size / 2 ** (worker_info.id % self.num_context_lengths))
            raw_batch = raw_batch[:adj_batch_size]

        # concatenate the contigs and tokenize them
        seqs_encoded = []
        max_context_len = self.context_length

        if self.return_contig_indices:
            contigs_list, start_end_indices, sep = zip(*raw_batch)
            max_context_len = max(
                [
                    sum([end - start for start, end in indices])
                    for indices in start_end_indices
                ]
            )

            if self.context_length != max_context_len:
                # limit the number of shapes to compile
                max_context_len = min(
                    int(
                        math.ceil(max_context_len / self.pad_seq_base_length)
                        * self.pad_seq_base_length
                    ),
                    self.context_length,
                )

            cos, sin = self.get_rope_tensor_frags(
                start_end_indices, max_context_len=max_context_len
            )
            if self.use_long_range_rope:
                cos2, sin2 = self.get_rope_tensor_frags(
                    start_end_indices,
                    rope_factor=self.rotary_base_scaling_factor,
                    rotary_base=self.long_range_rope_base,
                    max_context_len=max_context_len,
                )
                start_end_indices = [start_end_indices, [cos, cos2], [sin, sin2], sep]
            else:
                start_end_indices = [start_end_indices, cos, sin, sep]
        else:
            contigs_list, sep = zip(*raw_batch)

        for i, contigs in enumerate(contigs_list):
            seq = self.concat_seqs(contigs, sep[i])
            seqs_encoded.append(seq)

        if self.padding:
            seqs_encoded_tokens = torch.full(
                [len(contigs_list), max_context_len],
                self.tokenizer.padding_idx,
                dtype=torch.int64,
                device=self.device,
            )
            for i in range(len(seqs_encoded)):
                seqs_encoded_tokens[i, : len(seqs_encoded[i])] = torch.tensor(
                    seqs_encoded[i], dtype=torch.int64, device=self.device
                )
        else:
            assert len(set([len(seq) for seq in seqs_encoded])) == 1, (
                "All sequences should be of the same length"
            )
            seqs_encoded_tokens = torch.tensor(
                seqs_encoded, dtype=torch.int64, device=self.device
            )

        if self.mode == "ar":
            data, target = self.ar_batch(seqs_encoded_tokens)
            if self.return_contig_indices:
                return raw_batch, data, target, start_end_indices
            return raw_batch, data, target
        elif self.mode == "mlm":
            noised_tokens, noise_mask = self.mlm_batch(seqs_encoded_tokens)
            if self.return_contig_indices:
                return (
                    raw_batch,
                    noised_tokens,
                    seqs_encoded_tokens,
                    noise_mask,
                    start_end_indices,
                )
            return raw_batch, noised_tokens, seqs_encoded_tokens, noise_mask


class NucleotideSeqBatchFromHFTokenizerConverter(CollateFn):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        tokenizer = self.tokenizer.tokenizer
        all_tokens = tokenizer.get_vocab()
        # Filter out special tokens
        standard_ids = [
            tokenizer.convert_tokens_to_ids(token)
            for token, id in all_tokens.items()
            if not tokenizer.convert_ids_to_tokens(id).startswith("##")
            and id not in tokenizer.all_special_ids
        ]
        self.nucleotide_token_index = torch.tensor(standard_ids, device=self.device)


class PadCollateFn(CollateFn):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(
        self, raw_batch: Sequence[Tuple[str, int]]
    ) -> Tuple[List[str], List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            seq_label list
            seq_str list
            tokenized seq tensor
            noise_mask True for residues used for training
            mask True for all residues (exclude cls/bos, eos, padding)
        """
        batch_size = len(raw_batch)
        if batch_size == 0:
            worker_info = torch.utils.data.get_worker_info()
            warnings.warn(f"Getting Batch size 0 from {worker_info.id}")
            return None

        seqs, seq_indices, chr_names = zip(*raw_batch)
        max_len = max([len(seq) for seq in seqs])
        seqs_encoded_tokens = torch.full(
            [len(seqs), max_len],
            self.tokenizer.padding_idx,
            dtype=torch.int64,
            device=self.device,
        )
        seqs_encoded = self.tokenizer.encode_batch(
            seqs, bos_eos=self.bos_eos, cls=self.cls
        )

        for i in range(len(seqs_encoded)):
            seqs_encoded_tokens[: len(seqs_encoded[i])] = torch.tensor(
                seqs_encoded[i], dtype=torch.int64, device=self.device
            )

        if self.mode == "ar":
            data, target = self.ar_batch(seqs_encoded_tokens)
            return seqs, seq_indices, data, target
        elif self.mode == "mlm":
            noised_tokens, noise_mask = self.mlm_batch(seqs_encoded_tokens)
            return seqs, seq_indices, noised_tokens, seqs_encoded_tokens, noise_mask
