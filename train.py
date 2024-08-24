import argparse
import logging
import torch
torch.manual_seed(1)
# torch.autograd.set_detect_anomaly(True)
import os
import time

import torch.multiprocessing as mp
from packages.utilities.logging_utilities import *

# Setup logging for other files to use
try:
    os.remove('./logs/train.log')
except OSError:
    pass
addLoggingLevel('TRACE', logging.DEBUG - 5)
global_config_logger(log_file = './logs/train.log', log_level = logging.DEBUG)

from config import Config
from packages.dataprocessing.data_dropout import DataDropout
from packages.utilities.train_utilities import *
from packages.utilities.multi_gpu_utilities import *


def train(gpu, c, data_dropout):
    """Function to train the neural network.

    :param gpu: ID of the GPU to be used
    :type gpu: int
    :param c: Config dictionary containing desired user parameters
    :type c: dict
    :param data_dropout: Adaptive Data Dropout parameter dictionary
    :type data_dropout: dict
    """
    if c.config_dic['gpu'] > 1:
        # Assign the gpu rank to its device number
        rank = gpu

        # We are assuming 1 node per gpu
        setup_multigpu(rank, c.config_dic['gpu'])

        # Setup the training dictionary
        training_dict = setup_training_dict(c, specific_device = gpu)
    else:
        training_dict = setup_training_dict(c, specific_device = None)

    for current_epoch in range(c.config_dic['epochs'] + 1):
        training_dict['current_epoch'] = current_epoch

        if current_epoch <= training_dict['last_loaded_epoch']:
            continue

        training_dict['epoch_time_start'] = time.perf_counter()
        
        training_dict['training_loss'] = train_model(training_dict)
        
        training_dict['validation_loss'] = validate_model(training_dict)
        
        metrics_dict = calculate_metrics_and_update_scheduler(training_dict)
        
        if data_dropout is not None:
            data_dropout(training_dict)
        
        if gpu == 0:
            record_metrics(training_dict, metrics_dict)

            if c.config_dic['gpu'] > 1:
                checkpoint_model(training_dict, multi_gpu = True, keep_checkpoints = c.config_dic['keep_checkpoints'])
            else:
                checkpoint_model(training_dict, multi_gpu = False, keep_checkpoints = c.config_dic['keep_checkpoints'])

        reset_metrics(training_dict)
        
        if c.config_dic['class_removal_policy'] and len(data_dropout.removed_classes) == c.config_dic['num_classes']:
            logger.info('Ending training early due to all classes being dropped once.')
            return
        
        if 'metric_cutoff' in c.config_dic:
            if metrics_dict['Val MulticlassF1Score'] >= c.config_dic['metric_cutoff']:
                logger.info('Ending training early due to threshold reached.')
                return 

if __name__ == '__main__':
    # Setup the main logger
    logger = setup_logger(name = __name__)

    # Argument Parser init
    parser = argparse.ArgumentParser(description = 'Train a model')

    parser.add_argument('--config_file_path', required = True, type = str, help = 'Get the path to the config file.')

    # Parse the arguments
    args = parser.parse_args()

    # Setup the config 
    c = Config(
        args.config_file_path,
        default_settings = './configs/training_default.json', 
        schema = './configs/training_schema.json',
        mode = 'train', 
    )

    # Setup Data dropout policy
    if c.config_dic['data_dropout']:
        data_dropout = DataDropout(c.config_dic['data_dropout'])
    else:
        data_dropout = None

    # Check for multiple gpus or single gpu
    if c.config_dic['gpu'] > 1:
        # TODO: Create a properly working multi-gpu logger.
        update_log_level(logging.CRITICAL)
        mp.spawn(train, nprocs = c.config_dic['gpu'], args = (c, data_dropout))
    else:
        train(0, c, data_dropout)