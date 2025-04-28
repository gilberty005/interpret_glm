from pathlib import Path
from typing import List, Literal, Tuple, Optional, Dict
import warnings 
import shutil

import numpy as np 
import lightning as L
import torch

from ..dataset.dataset import FastaDataset, CollateFn, iter_manual_worker_init_fn


class AnnotationDataset(torch.utils.data.Dataset):

    '''
    Iterate through fasta file to fetch sequence segments and construct annotations for each segment
    '''

    def __init__(
        self,
        tokenizer,
        fasta_file: Path,
        annot_files: Dict[str, str],
        # bed_file: Path,
        context_length: int = 1024,
        bos_eos: bool = False,
        cls: bool = False,
        mode: Literal['mlm', 'ar'] = 'mlm',
        current_epoch: int = 0,
        static_batch_dim: bool = None,
    ):
        super(AnnotationDataset).__init__()
            
        self.tokenizer = tokenizer

        for annot_file in annot_files.values():
            annot_file = Path(annot_file)
            assert annot_file.exists(), 'path to npy file must exist'
        
        self.annot = {key: np.load(annot_files[key], mmap_mode='c') for key in annot_files.keys()} # load annotation file using numpy memmap

        self.bos_eos = bos_eos
        self.cls = cls
        self.context_length = context_length

        if static_batch_dim is not None:
            warnings.warn('static_batch_dim is deprecated')

        # set context_length
        self.set_context_length()

        # self.fasta = FastaInterval(
        #     fasta_file = fasta_file,
        #     context_length = self.context_length,
        # )
        self.fasta = None # TODO: to be implemented

        self.num_regions = {key: self.fasta.chr_lens[key] // context_length + 1 
                            for key in self.fasta.chr_lens.keys()}
        self.chromosomes = [elem for elem in list(self.fasta.chr_lens.keys())]
        self.mode = mode 
        self.current_epoch = current_epoch
        self.start = 0 
        self.end = int(sum(self.fasta.chr_lens.values()) / self.context_length) + 1

        # initialize dataset
        self.data = {}
        i = 0
        for chr_name, val in self.num_regions.items():
            for idx in range(val):
                seq, seq_idx = self.fasta(chr_name, idx * self.context_length)
                if seq.count('N') < self.context_length * 1e-5:
                    self.data[i] = [seq, seq_idx, chr_name]
                    i += 1

    def set_context_length(self) -> None:
        if self.bos_eos:
            self.context_length -= 2 # subtract 2 for cls and eos tokens
        if self.cls:
            self.context_length -= 1 # subtract 1 for cls token

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        '''
        Return a sequence segment and its annotation
        '''
        seq, seq_idx, chr_name =  self.data[idx]
        annotation = self.annot[chr_name][seq_idx:seq_idx + len(seq), ]
        return seq, seq_idx, torch.from_numpy(annotation)  #annotation


class ValCollateFn(CollateFn):
    """
    Collable to convert a batch of sequences to a batch of tensors for validation 
    Masking is done selectively, with 50% on nucleotides with annotation and 50% on nucleotides without annotation
    """

    def __init__(self, **kwargs) -> None:        
        super().__init__(**kwargs)
        self.encoded_special = torch.tensor(self.tokenizer.encode(self.tokenizer.special_toks))

    def __call__(self, 
                 raw_batch,
                 ) -> Tuple[List[str], List[str], torch.Tensor,
                            torch.Tensor, torch.Tensor]:
        '''
        Returns:
            seq_label list
            seq_str list
            tokenized seq tensor
            noise_mask True for residues used for training
            mask True for all residues (exclude cls/bos, eos, padding)
        '''
        batch_size = len(raw_batch)
        if batch_size == 0:
            worker_info = torch.utils.data.get_worker_info()
            warnings.warn(f'Getting Batch size 0 from {worker_info.id}')
            return None
                
        seqs, seq_indices, annot = zip(*raw_batch)
        assert len(set([len(seq) for seq in seqs])) == 1, 'All sequences should be of the same length'

        seqs_encoded = self.tokenizer.encode_batch(seqs, bos_eos=self.bos_eos, cls=self.cls)
        seqs_encoded_tokens = torch.tensor(seqs_encoded, dtype=torch.int64, device=self.device)

        # annot doesn't include encoded_special
        annot = torch.stack(annot, dim=0)  # shape(B S C)
        _, anno_ssz, csz = annot.shape
        bsz, ssz = seqs_encoded_tokens.shape
        
        if anno_ssz == ssz:
            annotations = annot
        else:
            annot_mask = torch.isin(seqs_encoded_tokens, self.encoded_special)  # shape(B S-special)
            # annotations is expanded annot
            annotations = torch.zeros(list(seqs_encoded_tokens.size()) + [annot.shape[-1]],
                                      dtype=torch.int64, device=self.device)
            # non_special contain indices (sample-num indices)
            non_special = torch.nonzero(~annot_mask)
            # arange of indices for annotation class
            arange_tensor = torch.arange(csz)[:,None,None].repeat(1,non_special.shape[0], 1)
            # cat along the indices dimension
            non_special = torch.cat((non_special.repeat(csz, 1, 1), arange_tensor), dim=-1)
            # rearrange (c bs i -> (bs c) i)
            non_special = non_special.transpose(0,1).reshape(-1,3) #rearrange is very slow here
    
            annotations[non_special[:,0],non_special[:,1],non_special[:,2]] = annot.flatten()

        if self.mode == 'ar':
            data, target = self.ar_batch(seqs_encoded_tokens)
            return seqs, seq_indices, data, target, annotations
        elif self.mode == 'mlm':
            noised_tokens, noise_mask = self.mlm_batch(seqs_encoded_tokens)
            return seqs, seq_indices, noised_tokens, seqs_encoded_tokens, noise_mask, annotations


class AnnotationDNATrainDataModule(L.LightningDataModule):
    def __init__(self,
                 tokenizer,
                 train_path: Optional[Path] = None,  # dir | fasta | sqlite
                 shm_train: bool = False,
                 shm_dir: Optional[Path] = None,
                 valid_config: Optional[Dict] = None, 
                 # valid_path: Optional[Path] = None,  # dir | fasta | sqlite
                 test_path: Optional[Path] = None,  # dir | fasta | sqlite
                 predict_path: Optional[Path] = None,
                 mode: Literal['mlm', 'ar'] = 'mlm', # mlm or ar
                 simple_masking_only: bool = False,
                 context_length: int = 1024, # max length of sequence
                 batch_size: int = 8,
                 batch_size_multi: int = 1,  # for pytorch lightning scale_batch_size
                 num_workers: int = 4,
                 prefetch_factor: int = 4,
                 infer_mult: int = 5,
                 bos_eos: bool = False,
                 cls: bool = False,
                 mask_prob: float = 0.15,
                 variable_masking: Optional[List[float]] = None,
                 valtestpred_context_length: int = 1024,
                 pin_memory: bool = False,  # this could cause issues
                 drop_last: bool = True, # drop last batch
                ):
        '''
        Load entire dataset and split into train/val
        def setup() and prepare_data() for larger datsets
        '''
        super().__init__()
        self.tokenizer = tokenizer  
        self.train_path = train_path
        self.shm_train = shm_train
        self.shm_dir = shm_dir
        self.valid_config = valid_config
        self.test_path = test_path
        self.predict_path = predict_path
        self.num_workers = num_workers
        self.mode = mode
        self.simple_masking_only = simple_masking_only
        self.context_length = context_length
        self.batch_size = batch_size
        self.infer_mult = infer_mult
        self.bos_eos = bos_eos
        self.cls = cls
        self.mask_prob = mask_prob
        self.variable_masking = variable_masking
        self.prefetch_factor = prefetch_factor
        self.valtestpred_context_length = valtestpred_context_length
        self.batch_size_multi = batch_size_multi
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        if self.shm_dir is not None:
            self.shm_dir = Path(self.shm_dir)
            assert self.shm_dir.is_dir()
        if self.valid_config is not None:
            self.valid_data_names = self.valid_config['valid_data_names']  # maintains order, no guessing
            for s in self.valid_data_names:
                'Check input at beginning so it does not crash later'
                assert self.valid_config[s]['data_class'] in dataset_registry, 'data_class not implemented'
                if self.valid_config[s]['data_class'] in ['FastaDataset',]:
                    path = self.check_input_file(self.valid_config[s]['data_args']['fasta_file'])
                    if 'shm_data' in self.valid_config[s] and self.valid_config[s]['shm_data']:
                        assert self.shm_dir is not None,(
                               'when shm_data=True, shm_dir needs to be set in training.py')
                        shutil.copy(path, self.shm_dir / path.name)
                        self.valid_config[s]['data_args']['fasta_file'] = self.shm_dir / path.name

    def prepare_data(self) -> None:
        'Runs before setup on main process. Do not assign state here'
        pass

    def check_input_file(self, path):
        path = Path(path)
        if path.is_file():
            if path.as_posix().endswith('.fasta') | \
                    path.as_posix().endswith('.fasta.gz') | \
                        path.as_posix().endswith('.fa') | \
                            path.as_posix().endswith('.fa.gz'):
                return path
        raise RuntimeError(f'fasta_file not found at <{path}>')
    
    def setup(self, stage: Optional[str] = None) -> None:
        '''Runs once per process/device.
        Setup datasets with train/val splits.
        Use prepare_data for Downloads and expensive 1 time tasks.
        '''
        pass 

    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        fasta_file = self.check_input_file(self.train_path)
        if self.shm_train:
            assert self.shm_dir is not None,(
                'when shm_train=True, shm_dir needs to be set in training.py')
            shutil.copy(fasta_file, self.shm_dir / fasta_file.name)
            fasta_file = self.shm_dir / fasta_file.name
        epoch = 0
        if self.trainer is not None:
            epoch = self.trainer.current_epoch
        
        self.train_ds = FastaDataset(
            bos_eos=self.bos_eos,
            cls=self.cls,
            tokenizer=self.tokenizer,
            fasta_file=fasta_file,
            context_length=self.context_length,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            mode=self.mode,
        )

        dl = torch.utils.data.DataLoader(
            self.train_ds,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=iter_manual_worker_init_fn,
            collate_fn=CollateFn(
                tokenizer=self.tokenizer,
                bos_eos=self.bos_eos,
                cls=self.cls,
                mask_prob=self.mask_prob,
                variable_masking=self.variable_masking,
                simple_masking_only=self.simple_masking_only,
                mode=self.mode,
            ),
            batch_sampler=None,
            batch_size=None,
            pin_memory=self.pin_memory,
        )
        return dl

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        valid_dld = {}
        for s in self.valid_data_names:
            if self.valid_config[s]['data_class'] == 'AnnotationDataset':
                '''
                FastaDataset always have these arguments. Make these optional.
                Setting batch_size overwrite infer_mult
                '''
                for arg in ['context_length','bos_eos','cls','mode']:
                    if arg not in self.valid_config[s]['data_args']:
                        self.valid_config[s]['data_args'][arg] = getattr(self, arg)
            if self.valid_config[s]['collate_fn_class'] == 'ValCollateFn':
                for arg in ['mask_prob','bos_eos','cls','mode', 'variable_masking']:
                    if arg not in self.valid_config[s]['collate_fn_args']:
                        self.valid_config[s]['collate_fn_args'][arg] = getattr(self, arg)
            ds = dataset_registry[self.valid_config[s]['data_class']](
                tokenizer=self.tokenizer,
                **self.valid_config[s]['data_args']
            )
            collate_fn = collate_fn_registry['ValCollateFn'](
                tokenizer=self.tokenizer,
                **self.valid_config[s]['collate_fn_args'],
            )

            dl = torch.utils.data.DataLoader(
                ds,
                num_workers=self.num_workers,
                worker_init_fn=iter_manual_worker_init_fn,
                prefetch_factor=self.prefetch_factor,
                collate_fn=collate_fn,
                batch_sampler=None,
                batch_size=self.batch_size * self.infer_mult,
                pin_memory=self.pin_memory,
                drop_last=self.valid_config[s]['drop_last']
            )
            valid_dld[s] = dl
        return valid_dld

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        fasta_file = self.check_input_file(self.test_path)
        batch_size = int(self.batch_size* self.infer_mult)
        self.test_ds = FastaDataset(
            bos_eos=self.bos_eos,
            cls=self.cls,
            tokenizer=self.tokenizer,
            fasta_file=fasta_file,
            context_length=self.valtestpred_context_length,
            batch_size=batch_size,
            drop_last=self.drop_last,
            mode=self.mode,
       )
        dl = torch.utils.data.DataLoader(
            self.test_ds,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=iter_manual_worker_init_fn,
            collate_fn=CollateFn(
                tokenizer=self.tokenizer,
                mask_prob=self.mask_prob,
                bos_eos=self.bos_eos,
                cls=self.cls,
                simple_masking_only=self.simple_masking_only,
                mode=self.mode,
            ),
            batch_sampler=None,
            batch_size=None,
            pin_memory=self.pin_memory,
        )
        return dl

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        fasta_file = self.check_input_file(self.predict_path)
        self.predict_ds = FastaDataset(
            tokenizer=self.tokenizer,
            bos_eos=self.bos_eos,
            cls=self.cls,
            fasta_file=fasta_file,
            context_length=self.valtestpred_context_length,
            batch_size=self.batch_size,
            mode=self.mode,
            drop_last=self.drop_last,
        )
        dl = torch.utils.data.DataLoader(
            self.predict_ds,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=iter_manual_worker_init_fn,
            collate_fn=CollateFn(
                tokenizer=self.tokenizer,
                mask_prob=self.mask_prob,
                bos_eos=self.bos_eos,
                cls=self.cls,
                simple_masking_only=self.simple_masking_only,
                mode=self.mode,
            ),
            batch_sampler=None,
            batch_size=None,
            pin_memory=self.pin_memory,
        )
        return dl


dataset_registry = {
    'FastaDataset': FastaDataset,
    'AnnotationDataset': AnnotationDataset, 
}
collate_fn_registry = {
    'CollateFn': CollateFn,
    'ValCollateFn': ValCollateFn,
}

