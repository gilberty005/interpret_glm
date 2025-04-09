from typing import Sequence, Dict

import torch
from torch.utils.data import Dataset

from .dataset import FastaDataset


class MultiFastaDataset(Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    
    '''
    def __init__(
        self,
        # datasets: Sequence[Dataset],
        dataset_configs: Dict[str,Dict[str,str]],
        dataset_weights: Sequence[float],
        generator: torch.Generator,
        epoch_len: int,
    ):
        super(MultiFastaDataset).__init__()

        # self.datasets = datasets
        self.dataset_configs = dataset_configs
        self.dataset_weights = torch.Tensor(dataset_weights)
        self.generator = generator
        self.epoch_len = epoch_len

        # initialize a dataset of indices 
        self.init_dataset()

        self.init_dataset_indices()

    def __len__(self) -> int:
        ''' Length of dataset=number of batches // devices
        '''
        return self.epoch_len
    
    def init_dataset(self):
        '''Initialize the FastaDatasets for each input dataset config
        '''
        datasets = [] 
        for ds in self.dataset_configs.keys():
            self.dataset_configs[ds]['generator'] = self.generator
            datasets.append(FastaDataset(**self.dataset_configs[ds]))
        self.datasets = datasets 

    def init_dataset_indices(self):
        '''Initialize the dataset of indices for the epoch (dataset_idx, genome_idx)
        '''
        # initialize datasets if it doesn't exist
        if not hasattr(self, 'datasets'):
            self.init_dataset()

        # sample dataset indices from the dataset weights
        n_datasets = len(self.datasets)
        dataset_indices = torch.multinomial(
            input=self.dataset_weights,
            num_samples=self.epoch_len,
            replacement=True,
            generator=self.generator,
        )

        # For each dataset, sample datapoint indices
        cluster_indices = torch.zeros(self.epoch_len, dtype=torch.long)
        for dataset_idx, num_genomes in zip(
            torch.arange(n_datasets),
            torch.bincount(dataset_indices, minlength=n_datasets),
        ):
            if num_genomes > 0:
                # Initialize the indices dataset for the sampled number of genomes
                self.datasets[dataset_idx].init_indices_dataset(num_genomes=num_genomes, 
                                                                # generator=self.generator,
                                                                )
                cluster_indices_i = self.datasets[dataset_idx].cluster_indices
                cluster_indices[torch.where(dataset_indices == dataset_idx)] = (
                    cluster_indices_i
                )
            else:
                continue

        self.indices = torch.stack((dataset_indices, cluster_indices), dim=1).tolist()

    def __getitem__(self, idx: int):
        # worker_info = torch.utils.data.get_worker_info()
        # world_size, rank = torch.utils.data.dataloader._get_distributed_settings()

        # Get the dataset and genome index
        dataset_idx, cluster_idx = self.indices[idx]

        # Get the sequence
        # TODO correctly set the seed and generator
        seqs = self.datasets[dataset_idx].__getitem__(cluster_idx, intput_cluster_idx=True)
        return seqs
