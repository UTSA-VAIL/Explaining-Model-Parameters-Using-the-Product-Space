import os
import torch.distributed as dist
import torch

from .logging_utilities import *

# Setup a logger
logger = setup_logger(__name__)


def setup_multigpu(rank, world_size):
    """Function to setup the multi-gpu environment

    :param rank: GPU ID
    :type rank: int
    :param world_size: Number of GPUs to be used
    :type world_size: int
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12351'

    # initialize the process group
    dist.init_process_group(
        backend='nccl', 
        init_method='env://', 
        rank=rank, 
        world_size=world_size
    )

    # # Explicitly setting seed to make sure that models created in two processes
    # # start from same random weights and biases. (Was 777)
    # torch.manual_seed(12345)

def cleanup():
    """Function to destroy non nccl multi-gpu
    """
    dist.destroy_process_group()