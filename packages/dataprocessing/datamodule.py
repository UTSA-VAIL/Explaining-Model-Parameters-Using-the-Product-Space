import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from .datasets import Imagenet
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

from ..utilities.logging_utilities import *
from ..utilities.multi_gpu_utilities import broadcast_metric

# Setup a logger
logger = setup_logger(__name__)

# **********************************************************************
# Description:
#   Communication bridge between a dataset and a dataloader for single
#   label classification task.
# Parameters:
#   dataset - Dataset to be used
# Notes:
#   This takes Pytorch's dataloader's parameters as well
# **********************************************************************
class SingleLabelClassificationDataModule():
    def __init__(self, dataset, num_workers=1, batch_size=32, sampler=None, num_replicas=None, rank=None, pin_memory=False, drop_last = True):
        self.dataset = dataset

        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.pin_memory = pin_memory

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.class_labels = np.array(dataset.labels)
        self.mask = np.full(len(self.class_labels), fill_value = 1, dtype = int)
        self.dataset = Subset(dataset, self.mask.nonzero()[0])

    def shuffle_mask(self):
        logger.debug('Randomly shuffling mask')
        np.random.shuffle(self.mask)
        indicies = self.mask.nonzero()[0]
        self.dataset = Subset(self.dataset.dataset, indicies)

        if self.sampler is not None:
            self.sampler = DistributedSampler(
                self.dataset, num_replicas = self.num_replicas, rank = self.rank)

    # **********************************************************************
    # Description:
    #   Remove class examples from a dataset.
    # Parameters:
    #   class_label - Class to be removed
    #   residue - Percentage amount of class examples to leave behind
    #   shuffle_mask - Randomly shuffle the mask for pure random experiment
    # Notes:
    #   -
    # **********************************************************************
    def remove_class_with_residue(self, class_label, residue=0.05, shuffle_mask = False):
        class_mask = self.class_labels != class_label

        inverted_class_mask = ~class_mask
        self.mask = inverted_class_mask | self.mask

        if residue > 0:
            class_indicies = np.where(inverted_class_mask == 1)[0]
            np.random.shuffle(class_indicies)

            num_of_examples_to_keep = int(len(class_indicies) * residue)
            count = 0
            for index in class_indicies:
                if num_of_examples_to_keep == count:
                    break

                if self.mask[index] == 1:
                    class_mask[index] = 1
                    count += 1

        self.mask = class_mask * self.mask
        # if shuffle_mask:
        #     logger.debug('Randomly shuffling mask')
        #     np.random.shuffle(self.mask)
        
        indicies = self.mask.nonzero()[0]
        self.dataset = Subset(self.dataset.dataset, indicies)

        if self.sampler is not None:
            self.sampler = DistributedSampler(
                self.dataset, num_replicas = self.num_replicas, rank = self.rank)

    # **********************************************************************
    # Description:
    #   Create a dataloader using the data module's dataset
    # Parameters:
    #   shuffle - Flag to determine whether dataloader should be shuffled
    # Notes:
    #   -
    # **********************************************************************
    def dataloader(self, shuffle=False):
        if self.sampler is not None:
            return DataLoader(
                self.dataset, 
                batch_size = self.batch_size, 
                num_workers = self.num_workers, 
                shuffle = shuffle, 
                sampler = self.sampler, 
                pin_memory = self.pin_memory, 
                drop_last = self.drop_last
            )

        return DataLoader(
            self.dataset, 
            batch_size = self.batch_size, 
            num_workers = self.num_workers, 
            shuffle = shuffle, 
            drop_last = self.drop_last
        )
