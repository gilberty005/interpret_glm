from typing import Optional, Dict, Literal, List
from pathlib import Path
import shutil
import functools

import torch
from torch.utils.data import Sampler
import lightning as L
from lightning.pytorch.strategies.parallel import ParallelStrategy

from .dataset import (
    FastaDataset,
    CollateFn,
    NucleotideSeqBatchFromHFTokenizerConverter,
    ModRopeCollateFn,
)
#from .pred_dataset import PredDataset, PredCollateFn, PredModRopeCollateFn
from .multi_dataset import MultiFastaDataset


class DNATrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        train_config: Optional[Dict] = None,
        rotary_config: Optional[Dict] = None,
        shm_dir: Optional[Path] = None,
        valid_config: Optional[Dict] = None,
        pred_config: Optional[Dict] = None,
        test_path: Optional[Path] = None,  # dir | fasta | sqlite
        predict_path: Optional[Path] = None,
        mode: Literal["mlm", "ar"] = "mlm",  # mlm or ar
        simple_masking_only: bool = False,
        context_length: int = 1024,  # max length of sequence
        padding: bool = False,  # pad sequences
        rope_indices_concat_method: Literal[
            "zero", "continuous", "gap"
        ] = "gap",  # method to set positional indices for concatenated contigs
        add_separator: bool = True,
        epoch_len: int = 1000000,  # size of the created dataset per epoch
        batch_size: int = 8,
        batch_size_multi: int = 1,  # for pytorch lightning scale_batch_size
        num_workers: int = 4,
        prefetch_factor: int = 4,
        infer_mult: int = 5,  # can also be float to reduce inference batch size
        bos_eos: bool = False,
        cls: bool = False,
        return_contig_indices: bool = False,
        mask_prob: float = 0.15,
        variable_masking: Optional[List[float]] = None,
        variable_seq_len_worker: bool = False,  # Workers each load seq with different context_length
        variable_seq_bucket_len_worker: bool = False,  # Workers each load with own context length, and shorter sequences are padded up
        variable_len_base: int = 5024,  # base_length multipiler for worker context_length; configured at worker_init_fn
        num_context_lengths: int = None, # number of context lengths to generate using variable_seq_len_worker
        valtestpred_context_length: int = 1024,
        pin_memory: bool = False,  # this could cause issues
        drop_last: bool = True,  # drop last batch,
        crop_frac: float = None,
        seed: int = 0,
    ):
        """
        Load entire dataset and split into train/val
        def setup() and prepare_data() for larger datsets
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.rotary_config = rotary_config
        self.shm_dir = shm_dir
        self.valid_config = valid_config
        self.pred_config = pred_config
        self.test_path = test_path
        self.predict_path = predict_path
        self.num_workers = num_workers
        self.mode = mode
        self.simple_masking_only = simple_masking_only
        self.context_length = context_length
        self.padding = padding
        self.rope_indices_concat_method = rope_indices_concat_method
        self.add_separator = add_separator
        self.epoch_len = epoch_len
        self.batch_size = batch_size
        self.infer_mult = infer_mult
        self.bos_eos = bos_eos
        self.cls = cls
        self.mask_prob = mask_prob
        self.variable_masking = variable_masking
        self.variable_seq_len_worker = variable_seq_len_worker
        self.variable_seq_bucket_len_worker = variable_seq_bucket_len_worker
        self.variable_len_base = variable_len_base
        if self.variable_seq_len_worker and self.variable_seq_bucket_len_worker:
            raise ValueError(
                "variable_seq_len_worker and variable_seq_bucket_len_worker cannot be both True"
            )
        self.num_context_lengths = num_context_lengths  
        if self.num_context_lengths is not None:
            assert self.num_workers % self.num_context_lengths == 0, (
                "num_workers should be divisible by num_context_lengths"
            )
        else:
            self.num_context_lengths = 1
        self.prefetch_factor = prefetch_factor
        # Set persistent_workers to false to force them to reinitialize between epochs.
        # If workers are persistent, all epochs will get the same data.
        self.persistent_workers = False 
        if self.num_workers == 0:
            self.prefetch_factor = None
        self.valtestpred_context_length = valtestpred_context_length
        self.batch_size_multi = batch_size_multi
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.crop_frac = crop_frac
        self.seed = seed
        if self.shm_dir is not None:
            self.shm_dir = Path(self.shm_dir)
            assert self.shm_dir.is_dir()

        # Skip training file checks if we're in prediction mode (pred_config is not None)
        is_prediction_mode = self.pred_config is not None

        if (
            self.train_config is not None and not is_prediction_mode
        ):  # config file that specifies data class and args for each dataset
            "Check input at beginning so it does not crash later"
            assert self.train_config["data_class"] in dataset_registry, (
                "data_class not implemented"
            )
            if self.train_config["data_class"] in ["MultiFastaDataset"]:
                for d in self.train_config["data_args"]:
                    cluster_genome_mapping_path = self.check_input_file(
                        self.train_config[d]["cluster_genome_mapping_file"],
                        file_type="json",
                    )
                    genome_contig_mapping_index_path = self.check_input_file(
                        self.train_config[d]["genome_contig_mapping_index_file"],
                        file_type="index_dir",
                    )
                    contig_seq_mapping_index_path = self.check_input_file(
                        self.train_config[d]["contig_seq_mapping_index_file"],
                        file_type="index_dir",
                    )
                    zstd_dict_path = self.check_input_file(
                        self.train_config[d]["zstd_dict_file"], file_type="dict"
                    )

                    # copy both files to shm_dir
                    if (
                        "shm_data" in self.train_config
                        and self.train_config["shm_data"]
                    ):
                        assert self.shm_dir is not None, (
                            "when shm_data=True, shm_dir needs to be set in training.py"
                        )
                        shutil.copy(
                            cluster_genome_mapping_path,
                            self.shm_dir / cluster_genome_mapping_path.name,
                        )
                        shutil.copy(
                            genome_contig_mapping_index_path,
                            self.shm_dir / genome_contig_mapping_index_path.name,
                        )
                        shutil.copy(
                            contig_seq_mapping_index_path,
                            self.shm_dir / contig_seq_mapping_index_path.name,
                        )
                        shutil.copy(zstd_dict_path, self.shm_dir / zstd_dict_path.name)

                        self.train_config[d]["cluster_genome_mapping_file"] = (
                            self.shm_dir / cluster_genome_mapping_path.name
                        )
                        self.train_config[d]["genome_contig_mapping_index_file"] = (
                            self.shm_dir / genome_contig_mapping_index_path.name
                        )
                        self.train_config[d]["contig_seq_mapping_index_file"] = (
                            self.shm_dir / contig_seq_mapping_index_path.name
                        )
                        self.train_config[d]["zstd_dict_file"] = (
                            self.shm_dir / zstd_dict_path.name
                        )

        if self.valid_config is not None and not is_prediction_mode:
            self.valid_data_names = self.valid_config["valid_data_names"]
            for s in self.valid_data_names:
                "Check input at beginning so it does not crash later"
                assert self.valid_config[s]["data_class"] in dataset_registry, (
                    "data_class not implemented"
                )
                if self.valid_config[s]["data_class"] in ["MultiFastaDataset"]:
                    for d in self.valid_config[s]["data_args"]:
                        cluster_genome_mapping_path = self.check_input_file(
                            self.valid_config[s][d]["cluster_genome_mapping_file"],
                            file_type="json",
                        )
                        genome_contig_mapping_index_path = self.check_input_file(
                            self.valid_config[s][d]["genome_contig_mapping_index_file"],
                            file_type="index_dir",
                        )
                        contig_seq_mapping_index_path = self.check_input_file(
                            self.valid_config[s][d]["contig_seq_mapping_index_file"],
                            file_type="index_dir",
                        )
                        zstd_dict_path = self.check_input_file(
                            self.valid_config[s][d]["zstd_dict_file"], file_type="dict"
                        )

                        # copy both files to shm_dir
                        if (
                            "shm_data" in self.valid_config[s]
                            and self.valid_config[s]["shm_data"]
                        ):
                            assert self.shm_dir is not None, (
                                "when shm_data=True, shm_dir needs to be set in training.py"
                            )
                            shutil.copy(
                                cluster_genome_mapping_path,
                                self.shm_dir / cluster_genome_mapping_path.name,
                            )
                            shutil.copy(
                                genome_contig_mapping_index_path,
                                self.shm_dir / genome_contig_mapping_index_path.name,
                            )
                            shutil.copy(
                                contig_seq_mapping_index_path,
                                self.shm_dir / contig_seq_mapping_index_path.name,
                            )
                            shutil.copy(
                                zstd_dict_path, self.shm_dir / zstd_dict_path.name
                            )

                            self.valid_config[s][d]["cluster_genome_mapping_file"] = (
                                self.shm_dir / cluster_genome_mapping_path.name
                            )
                            self.valid_config[s][d][
                                "genome_contig_mapping_index_file"
                            ] = self.shm_dir / genome_contig_mapping_index_path.name
                            self.valid_config[s][d]["contig_seq_mapping_index_file"] = (
                                self.shm_dir / contig_seq_mapping_index_path.name
                            )
                            self.valid_config[s][d]["zstd_dict_file"] = (
                                self.shm_dir / zstd_dict_path.name
                            )

        if self.pred_config is not None:
            "Check input at beginning so it does not crash later"
            assert self.pred_config["data_class"] in dataset_registry, (
                "data_class not implemented"
            )
            if self.pred_config["data_class"] == "PredDataset":
                for dataset_name in self.pred_config["data_args"]:
                    dataset_config = self.pred_config[dataset_name]

                    contig_seq_mapping_index_path = self.check_input_file(
                        dataset_config["contig_seq_mapping_index_file"],
                        file_type="index_dir",
                    )
                    seq_index_list_path = self.check_input_file(
                        dataset_config["seq_index_list_file"], file_type="json"
                    )
                    zstd_dict_path = self.check_input_file(
                        dataset_config["zstd_dict_file"], file_type="dict"
                    )

                    if "shm_data" in self.pred_config and self.pred_config["shm_data"]:
                        assert self.shm_dir is not None, (
                            "when shm_data=True, shm_dir needs to be set in training.py"
                        )
                        shutil.copy(
                            contig_seq_mapping_index_path,
                            self.shm_dir / Path(contig_seq_mapping_index_path).name,
                        )
                        shutil.copy(
                            seq_index_list_path,
                            self.shm_dir / Path(seq_index_list_path).name,
                        )
                        shutil.copy(
                            zstd_dict_path, self.shm_dir / Path(zstd_dict_path).name
                        )

                        self.pred_config[dataset_name][
                            "contig_seq_mapping_index_file"
                        ] = self.shm_dir / Path(contig_seq_mapping_index_path).name
                        self.pred_config[dataset_name]["seq_index_list_file"] = (
                            self.shm_dir / Path(seq_index_list_path).name
                        )
                        self.pred_config[dataset_name]["zstd_dict_file"] = (
                            self.shm_dir / Path(zstd_dict_path).name
                        )

    def prepare_data(self) -> None:
        "Runs before setup on main process. Do not assign state here"
        pass

    def check_input_file(
        self, path: str, file_type: Literal["json", "index_dir", "dict"]
    ) -> Path:
        path = Path(path)
        if file_type == "json":
            if path.is_file() and path.as_posix().endswith(".json"):
                return path
            raise RuntimeError(
                f"cluster_genome_mapping json_file not found at <{path}>"
            )
        elif file_type == "index_dir":
            if path.is_dir():
                return str(path)
            raise RuntimeError(f"Index directory not found at <{path}>")
        elif file_type == "dict":
            if path.is_file():
                return path
            raise RuntimeError(f"Zstandard dict file not found at <{path}>")
        raise RuntimeError("File type not recognized")

    def setup(self, stage: Optional[str] = None) -> None:
        """Runs once per process/device.
        Setup datasets with train/val splits.
        Use prepare_data for Downloads and expensive 1 time tasks.
        """
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # unlike validation dataloader, training dataloader may contain multiple datasets
        # within the same dataloader
        if self.train_config["data_class"] == "MultiFastaDataset":
            """
            FastaDataset always have these arguments. Make these optional.
            Setting batch_size overwrite infer_mult
            """
            # TODO refine generator definition
            manual_seed = (
                1 if self.trainer is None else self.trainer.global_step + 1
            )  # for running dataset in isolation
            ds = {
                "dataset_configs": {},
                "generator": torch.Generator().manual_seed(manual_seed),
                "epoch_len": None,
            }
            for d in self.train_config["data_args"]:
                for arg in [
                    "crop_frac",
                    "context_length",
                    "rope_indices_concat_method",
                    "add_separator",
                    "padding",
                    "crop_contig_prob",
                ]:
                    if arg not in self.train_config[d]:
                        self.train_config[d][arg] = getattr(self, arg)
                ds["dataset_configs"][d] = self.train_config[d]

            if "dataset_weights" in self.train_config:
                ds["dataset_weights"] = self.train_config["dataset_weights"]
                assert len(ds["dataset_configs"]) == len(ds["dataset_weights"]), (
                    "dataset_weights should have same length as datasets"
                )

            if "epoch_len" in self.train_config:
                ds["epoch_len"] = self.train_config["epoch_len"]
            else:
                raise ValueError("epoch_len must be set in the train_config")
            self.train_ds = MultiFastaDataset(**ds)

            if self.train_config["collate_fn_class"] == "ModRopeCollateFn":
                self.train_config["collate_fn_args"]["rotary_config"] = self.rotary_config
                for arg in [
                    "context_length",
                    "padding",
                    "mask_prob",
                    "bos_eos",
                    "cls",
                    "mode",
                    "variable_masking",
                    "add_separator",
                ]:  # context_length
                    if arg not in self.train_config["collate_fn_args"]:
                        self.train_config["collate_fn_args"][arg] = getattr(self, arg)
                if "generator" not in self.train_config["collate_fn_args"]:
                    self.train_config["collate_fn_args"]["generator"] = ds["generator"]

            if (self.variable_len_base is not None) and (
                "pad_seq_base_length" not in self.train_config["collate_fn_args"]
            ):
                # Set multiple block sizes to pad sequence up to. Creates fewer tensor shapes for torch.compile
                # collate_fn handles mod rope, which is modified for padding
                self.train_config["collate_fn_args"]["pad_seq_base_length"] = (
                    self.variable_len_base
                )

            collate_fn = collate_fn_registry[self.train_config["collate_fn_class"]](
                tokenizer=self.tokenizer,
                num_context_lengths=self.num_context_lengths,
                **self.train_config["collate_fn_args"],
            )

        if self.trainer is not None and isinstance(
            self.trainer.strategy, ParallelStrategy
        ):
            sampler_kwargs = self.trainer.strategy.distributed_sampler_kwargs
            sampler = torch.utils.data.DistributedSampler(
                self.train_ds, shuffle=True, **sampler_kwargs
            )
            shuffle = None
        else:
            sampler = None
            shuffle = True

        if self.variable_seq_len_worker:
            worker_init_fn = functools.partial(
                var_len_worker_init_fn, base_length=self.variable_len_base, num_context_lengths=self.num_context_lengths
            )
        elif self.variable_seq_bucket_len_worker:
            worker_init_fn = functools.partial(
                var_len_bucket_worker_init_fn, base_length=self.variable_len_base, num_context_lengths=self.num_context_lengths
            )
        else:
            worker_init_fn = None

        dl = torch.utils.data.DataLoader(
            self.train_ds,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=worker_init_fn,
            # worker_init_fn=test_threading,
            # multiprocessing_context='spawn',  # see if this consumes more memory
            collate_fn=collate_fn,
            # batch_sampler=None,
            sampler=sampler,
            shuffle=shuffle,
            batch_size=int(self.batch_size),
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,  # drop last batch
            persistent_workers=self.persistent_workers,
        )

        return dl

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        valid_dld = {}
        if "zero_worker" in self.valid_config and self.valid_config["zero_worker"]:
            val_dl_configs = {
                "num_workers": 0,
                # 'worker_init_fn': test_threading,  # worker_zero uses parent threads
            }
        else:
            val_dl_configs = {
                "num_workers": self.num_workers,
                "prefetch_factor": self.prefetch_factor,
                # worker_init_fn=iter_manual_worker_init_fn,
                # multiprocessing_context='spawn',  # see if this consumes more memory
            }

        for s in self.valid_data_names:
            if self.valid_config[s]["data_class"] == "MultiFastaDataset":
                """
                FastaDataset always have these arguments. Make these optional.
                Setting batch_size overwrite infer_mult
                """
                ds = {
                    #   "datasets": [],
                    "dataset_configs": {},
                    # "generator": torch.Generator().manual_seed(torch.randint(low=0, high=100000, size=(1,),).item()),
                    "generator": torch.Generator().manual_seed(self.seed),
                    "epoch_len": None,
                }
                for d in self.valid_config[s]["data_args"]:
                    for arg in [
                        "crop_frac",
                        "context_length",
                        "rope_indices_concat_method",
                        "add_separator",
                        "padding",
                        "crop_contig_prob",
                    ]:
                        if arg not in self.valid_config[s][d]:
                            self.valid_config[s][d][arg] = getattr(self, arg)
                    # if 'generator' not in self.valid_config[s][d]:
                    #     self.valid_config[s][d]['generator'] = ds['generator']
                    # self.valid_config[s][d]['epoch_len'] = self.valid_config
                    ds["dataset_configs"][d] = self.valid_config[s][d]

                if "dataset_weights" in self.valid_config[s]:
                    ds["dataset_weights"] = self.valid_config[s]["dataset_weights"]
                    assert len(ds["dataset_configs"]) == len(ds["dataset_weights"]), (
                        "dataset_weights should have same length as dataset_configs"
                    )

                if "epoch_len" in self.valid_config[s]:
                    ds["epoch_len"] = self.valid_config[s]["epoch_len"]
                else:
                    ds["epoch_len"] = self.epoch_len
                val_ds = MultiFastaDataset(**ds)

                if self.valid_config[s]["collate_fn_class"] == "ModRopeCollateFn":
                    self.valid_config[s]["collate_fn_args"]["rotary_config"] = self.rotary_config
                    for arg in [
                        "context_length",
                        "padding",
                        "mask_prob",
                        "bos_eos",
                        "cls",
                        "mode",
                        "variable_masking",
                        "add_separator",
                    ]:  # context_length
                        if arg not in self.valid_config[s]["collate_fn_args"]:
                            self.valid_config[s]["collate_fn_args"][arg] = getattr(
                                self, arg
                            )
                    if "generator" not in self.valid_config[s]["collate_fn_args"]:
                        self.valid_config[s]["collate_fn_args"]["generator"] = ds[
                            "generator"
                        ]
                collate_fn = collate_fn_registry[
                    self.valid_config[s]["collate_fn_class"]
                ](
                    tokenizer=self.tokenizer,
                    **self.valid_config[s]["collate_fn_args"],
                )

                if self.trainer is not None and isinstance(
                    self.trainer.strategy, ParallelStrategy
                ):
                    sampler_kwargs = self.trainer.strategy.distributed_sampler_kwargs
                    sampler = torch.utils.data.DistributedSampler(
                        val_ds, shuffle=False, **sampler_kwargs
                    )
                    shuffle = None
                else:
                    sampler = None
                    shuffle = False
                dl = torch.utils.data.DataLoader(
                    val_ds,
                    collate_fn=collate_fn,
                    sampler=sampler,
                    shuffle=shuffle,
                    batch_size=int(self.batch_size * self.infer_mult),
                    pin_memory=self.pin_memory,
                    drop_last=self.drop_last,  # drop last batch
                    **val_dl_configs,
                )

                valid_dld[s] = dl
        return valid_dld

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        # unlike validation dataloader, training dataloader may contain multiple datasets
        # within the same dataloader
        if self.pred_config["data_class"] == "PredDataset":
            """
            FastaDataset always have these arguments. Make these optional.
            Setting batch_size overwrite infer_mult
            """
            # TODO refine generator definition
            for d in self.pred_config["data_args"]:
                for arg in ["context_length"]:
                    if arg not in self.pred_config[d]:
                        self.pred_config[d][arg] = getattr(self, arg)
                self.pred_ds = PredDataset(**self.pred_config[d])

            if self.pred_config["collate_fn_class"] in [
                "PredCollateFn",
                "PredModRopeCollateFn",
            ]:
                for arg in ["context_length"]:  # context_length
                    if arg not in self.pred_config["collate_fn_args"]:
                        self.pred_config["collate_fn_args"][arg] = getattr(self, arg)
                if "generator" not in self.pred_config["collate_fn_args"]:
                    self.pred_config["collate_fn_args"]["generator"] = torch.Generator().manual_seed(self.seed)

                # Uses optional rotary_config from pred_config.yaml else uses one provided by model_config.yaml
                if self.pred_config["collate_fn_class"] == "PredModRopeCollateFn":
                    if "rotary_config" in self.pred_config:
                        self.pred_config["collate_fn_args"]["rotary_config"] = self.pred_config["rotary_config"]
                    else:
                        self.pred_config["collate_fn_args"]["rotary_config"] = self.rotary_config

            collate_fn = collate_fn_registry[self.pred_config["collate_fn_class"]](
                tokenizer=self.tokenizer,
                **self.pred_config["collate_fn_args"],
            )

        if self.trainer is not None and isinstance(
            self.trainer.strategy, ParallelStrategy
        ):
            sampler_kwargs = self.trainer.strategy.distributed_sampler_kwargs
            sampler = torch.utils.data.DistributedSampler(
                self.pred_ds, shuffle=True, **sampler_kwargs
            )
            shuffle = None
        else:
            sampler = None
            shuffle = True

        dl = torch.utils.data.DataLoader(
            self.pred_ds,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            # worker_init_fn=iter_manual_worker_init_fn,
            # worker_init_fn=test_threading,
            # multiprocessing_context='spawn',  # see if this consumes more memory
            collate_fn=collate_fn,
            # batch_sampler=None,
            sampler=sampler,
            shuffle=shuffle,
            batch_size=int(self.batch_size),
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,  # drop last batch
            persistent_workers=self.persistent_workers,
        )

        return dl

    def generate_seed(self, local_rank: int, worker_id: int, current_epoch: int) -> int:
        return worker_id + current_epoch * self.num_workers


def var_len_worker_init_fn(worker_id, base_length=5024, num_context_lengths=3):
    # TODO(Jae/Eric): Check whether setting worker_init_fn alters worker seeds in parallel settings
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    context_length = base_length * (2**(worker_id % num_context_lengths))
    if isinstance(dataset, MultiFastaDataset):
        for ds in dataset.datasets:
            ds.context_length = context_length
    else:
        dataset.context_length = context_length
    return None


def var_len_bucket_worker_init_fn(worker_id, base_length=5024):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    context_length = base_length * (worker_id + 2)
    context_bucket = (base_length * (worker_id + 1), context_length)

    if isinstance(dataset, MultiFastaDataset):
        for ds in dataset.datasets:
            ds.context_length = context_length
            ds.context_bucket = context_bucket
    else:
        dataset.context_length = context_length
        dataset.context_bucket = context_bucket
    return None

# def test_threading(worker_id):
#     local_rank = int(os.environ['FLUX_TASK_LOCAL_ID'])
#     torch.get_num_threads(10)
#     warnings.warn(f"worker at rank {local_rank} os.sched_getaffinity {os.sched_getaffinity(0)} thread {torch.get_num_threads()} interop {torch.get_num_interop_threads()} config {torch.__config__.parallel_info()}")

dataset_registry = {
    "FastaDataset": FastaDataset,
    "MultiFastaDataset": MultiFastaDataset,
    #"PredDataset": PredDataset,
}

collate_fn_registry = {
    "CollateFn": CollateFn,
    "NucleotideSeqBatchFromHFTokenizerConverter": NucleotideSeqBatchFromHFTokenizerConverter,
    "ModRopeCollateFn": ModRopeCollateFn,
    #"PredCollateFn": PredCollateFn,
    #"PredModRopeCollateFn": PredModRopeCollateFn,
}
