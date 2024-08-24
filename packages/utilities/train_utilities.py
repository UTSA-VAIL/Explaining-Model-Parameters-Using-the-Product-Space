import torch
import torch.nn as nn
import os
import time
import pandas as pd
from torchmetrics.classification import Accuracy, F1Score, JaccardIndex #type:ignore

from ..dataprocessing.datamodule import *
from .logging_utilities import *
from .general_utilities import load_checkpoint
from tqdm import tqdm

# Setup a logger
logger = setup_logger(__name__)


def setup_training_dict(c, specific_device = None):
    """_summary_

    :param c: Config object containing desired user parameters
    :type c: dict
    :param specific_device: ID of GPU for multi-gpu purposes
    :type specific_device: int, optional
    :return: Training hyperparmeter dictionary
    :rtype: dict
    """
    # Setup the dataset
    training_dataset, validation_dataset = c.prepare_dataset()

    # Setup the model
    model, model_task = c.prepare_model()

    # Create a metric dictionary and dataframe
    metric_path = os.path.join(c.config_dic['log_dir'], c.config_dic['metric_file_name'])
    checkpoint_path = os.path.join(c.config_dic['checkpoint_dir'], c.config_dic['checkpoint_file'])
    metric_df = pd.DataFrame()
    last_loaded_epoch = -1

    # Check for checkpoint from fresh start
    if not c.config_dic['fresh_start']:
        if os.path.exists(metric_path):
            logger.info('Loading metric file from {}'.format(metric_path))
            metric_df = pd.read_csv(metric_path)

        if os.path.exists(checkpoint_path):
            logger.info('Loading checkpoint file from {}'.format(checkpoint_path))
            last_loaded_epoch = load_checkpoint(checkpoint_path, model)

    # Check for multiple gpu setup
    if specific_device is not None:
        torch.cuda.set_device(specific_device)
        model.cuda(specific_device)
        model = nn.parallel.DistributedDataParallel(model, device_ids = [specific_device])

    # Setup the optimizer
    optimizer = c.prepare_optimizer(model)

    # Setup the scheduler
    scheduler = c.prepare_scheduler(optimizer)

    # Setup the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    logger.info('Loss Function: {}'.format(loss_fn))
    logger.info('Optimizer: {}\n'.format(optimizer))
    logger.info('Scheduler: {}\n'.format(scheduler))

    if model_task == 'single-label classification':
        # Setup up explainer (if provided)
        reference_dataset, _ = c.prepare_dataset(reference = True)
        explainer = c.prepare_explainer(reference_dataset = reference_dataset)
    
        if explainer:
            # Create Attribution Prior (if provided)
            attribution_prior = c.prepare_prior()
            if 'frequency' in c.config_dic['attribution']['attribution_prior']:
                attribution_prior_frequency = c.config_dic['attribution']['attribution_prior']['frequency']
            else:
                attribution_prior_frequency = 1
        else:
            attribution_prior = None
            attribution_prior_frequency = None
    else:
        explainer = None
        attribution_prior = None

    # Setup train sampler if needed 
    if specific_device is not None:
        training_sampler = torch.utils.data.distributed.DistributedSampler(
            training_dataset,
            num_replicas = c.config_dic['gpu'],
            rank = specific_device,
            shuffle = True
        )

        validation_sampler = torch.utils.data.distributed.DistributedSampler(
            validation_dataset,
            num_replicas = c.config_dic['gpu'],
            rank = specific_device,
            shuffle = True
        )
        num_replicas = c.config_dic['gpu']
        shuffle = False
    else:
        training_sampler = None
        validation_sampler = None
        num_replicas = None
        shuffle = True

    # Setup the datamodules
    if model_task == 'single-label classification':
        logger.debug('Creating Single-label Classification Datamodules')
        training_datamodule = SingleLabelClassificationDataModule(
            dataset = training_dataset, 
            batch_size = c.config_dic['batch_size'], 
            num_workers = c.config_dic['num_workers'],
            sampler = training_sampler,
            num_replicas = num_replicas,
            pin_memory = True,
            rank = specific_device,
        )
        validation_datamodule = SingleLabelClassificationDataModule(
            dataset = validation_dataset, 
            batch_size = c.config_dic['batch_size'], 
            num_workers = c.config_dic['num_workers'],
            sampler = validation_sampler,
            num_replicas = num_replicas,
            pin_memory = True,
            rank = specific_device,
        )
    elif model_task == 'segmentation':
        logger.debug('Creating Segmentation Datamodules')
        training_datamodule = SegmentationDataModule(
            dataset = training_dataset, 
            batch_size = c.config_dic['batch_size'], 
            num_workers = c.config_dic['num_workers'],
            sampler = training_sampler,
            num_replicas = num_replicas,
            pin_memory = True,
            rank = specific_device,
        )
        validation_datamodule = SegmentationDataModule(
            dataset = validation_dataset, 
            batch_size = c.config_dic['batch_size'], 
            num_workers = c.config_dic['num_workers'],
            sampler = validation_sampler,
            num_replicas = num_replicas,
            pin_memory = True,
            rank = specific_device,
        )


    # Setup the dataloaders
    logger.debug('Creating Dataloaders')
    training_dataloader = training_datamodule.dataloader(shuffle = shuffle)
    validation_dataloader = validation_datamodule.dataloader(shuffle = True)

    logger.info('Training set has {} batches'.format(len(training_dataloader)))
    logger.info('Validation set has {} batches\n'.format(len(validation_dataloader)))

    # Setup the metrics
    if model_task == 'single-label classification':
        logger.debug('Setting up single-label classification metrics')
        training_accuracy = Accuracy(task = 'multiclass', num_classes = c.config_dic['num_classes'])
        validation_accuracy = Accuracy(task = 'multiclass', num_classes = c.config_dic['num_classes'])
        training_f1score = F1Score(task = 'multiclass', num_classes = c.config_dic['num_classes'])
        validation_f1score = F1Score(task = 'multiclass', num_classes = c.config_dic['num_classes'])
        per_class_training_f1score = F1Score(task = 'multiclass', num_classes = c.config_dic['num_classes'], average = None)
        per_class_validation_f1score = F1Score(task = 'multiclass', num_classes = c.config_dic['num_classes'], average = None)

        training_metrics = [
            training_accuracy, 
            training_f1score, 
            per_class_training_f1score, 
        ]

        validation_metrics = [
            validation_accuracy, 
            validation_f1score, 
            per_class_validation_f1score
        ]
    elif model_task == 'segmentation':
        training_miou = JaccardIndex(task = 'multiclass', num_classes = c.config_dic['num_classes'])
        validation_miou = JaccardIndex(task = 'multiclass', num_classes = c.config_dic['num_classes'])
        per_class_training_miou = JaccardIndex(task = 'multiclass', num_classes = c.config_dic['num_classes'], average = None)
        per_class_validation_miou = JaccardIndex(task = 'multiclass', num_classes = c.config_dic['num_classes'], average = None)

        training_metrics = [
            training_miou, 
            per_class_training_miou, 
        ]

        validation_metrics = [
            validation_miou, 
            per_class_validation_miou
        ]

    # GPU flag checking
    if specific_device is not None:
        device = specific_device
    else:
        if c.config_dic['gpu'] == 1 and torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

    training_dict = {
        'training_dataset': training_dataset,
        'validation_dataset': validation_dataset,
        'training_datamodule': training_datamodule,
        'validation_datamodule': validation_datamodule,
        'model': model,
        'model_task': model_task,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_fn': loss_fn,
        'training_metrics': training_metrics,
        'validation_metrics': validation_metrics,
        'metric_path': metric_path,
        'metric_df': metric_df,
        'checkpoint_path': checkpoint_path,
        'device': device,
        'progress_bar': c.config_dic['progress_bar'],
        'epochs': c.config_dic['epochs'],
        'explainer': explainer,
        'attribution_prior': attribution_prior,
        'attribution_prior_frequency': attribution_prior_frequency,
        'last_loaded_epoch': last_loaded_epoch,
    }

    # Send the model to the correct device
    training_dict['model'] = training_dict['model'].to(device)

    # Send metrics to the device
    for index, metric in enumerate(training_dict['training_metrics']):
        training_dict['training_metrics'][index] = training_dict['training_metrics'][index].to(device)

    for index, metric in enumerate(training_dict['validation_metrics']):
        training_dict['validation_metrics'][index] = training_dict['validation_metrics'][index].to(device)

    return training_dict

def train_model(training_dict):
    """Function to train the model using the training dictionary

    :param training_dict: Function to train a model using the training dictionary
    :type training_dict: dict
    :return: Training losses
    :rtype: tuple
    """
    # Create the dataloader
    training_dataloader = training_dict['training_datamodule'].dataloader()

    # Check if we need to set the sampler's epoch for multi-gpu purposes
    if type(training_dataloader.sampler) is not torch.utils.data.SequentialSampler:
        training_dataloader.sampler.set_epoch(training_dict['current_epoch'])

    # ---Training Start---
    current_pre_prior_loss_total = 0.0
    current_input_prior_loss_total = 0.0
    current_training_loss_total = 0.0
    
    # Switch the model to training mode
    training_dict['model'].train()

    # Progress bar (Needs reset every epoch)
    if training_dict['progress_bar']:
        progress_bar = tqdm(training_dataloader)
    else:
        progress_bar = training_dataloader

    for current_batch_idx, batch_data in enumerate(progress_bar):
        # Extract the image and labels from batch
        images, labels = batch_data

        # Send the tensors to the correct device
        images = images.to(training_dict['device'], non_blocking=True)
        labels = labels.to(training_dict['device'], non_blocking=True)

        # Compute the input explanation (if needed)
        input_prior = 0
        if training_dict['attribution_prior']:
            if training_dict['current_epoch'] % training_dict['attribution_prior_frequency'] == 0:
                input_explanation = training_dict['explainer'].generate_explanation(training_dict, images)
                logger.debug('input_explanation Shape: {}'.format(input_explanation.shape))
                input_prior = training_dict['attribution_prior'].compute_prior(input_explanation)
                
        # Zero out gradients
        training_dict['optimizer'].zero_grad()

        # Get the model output for this batch
        outputs = training_dict['model'](images)
        if current_batch_idx < 5 and training_dict['model_task'] == 'single-label classification':
            logger.debug('Training Batch {} Correct Results: {}'.format(
                current_batch_idx, torch.sum(torch.eq(labels, torch.argmax(outputs, dim = -1)))))

        # Compute the loss
        training_loss = training_dict['loss_fn'](outputs, labels)

        # Update the training metrics
        for training_metric in training_dict['training_metrics']:
            training_metric.update(outputs, labels)  
        
        # Check for input prior
        logger.debug('training_loss Before input_prior: {}'.format(training_loss))
        pre_prior_loss = training_loss.item()
        logger.debug('input_prior value: {}'.format(input_prior))
        training_loss = training_loss + input_prior
        logger.debug('training_loss After input_prior: {}'.format(training_loss))

        # Calculate the loss gradients
        training_loss.backward()
                    
        # Adjust the learning weights
        if training_dict['current_epoch'] != 0:
            training_dict['optimizer'].step()

        # Keep track of the running loss
        current_pre_prior_loss_total += pre_prior_loss
        if input_prior != 0:
            current_input_prior_loss_total += input_prior.item()

        # Average the loss across all the batches
        current_pre_prior_loss = current_pre_prior_loss_total / (current_batch_idx + 1)
        current_input_prior_loss = current_input_prior_loss_total / (current_batch_idx + 1)
        current_training_loss = current_pre_prior_loss + current_input_prior_loss

        # Update progress bar
        if training_dict['progress_bar']:
            progress_bar.set_description('{}: {:.2f} Prior Loss: {:.2f} Total Loss: {:.2f}'.format(
                str(type(training_dict['loss_fn'])).split('.')[-1].split('\'')[0],
                current_pre_prior_loss,
                current_input_prior_loss,
                current_training_loss), 
                refresh = True
            )

    if training_dict['progress_bar']:
        progress_bar.close()
        logger.info(str(progress_bar))

    return current_pre_prior_loss, current_input_prior_loss, current_training_loss

def validate_model(training_dict):
    """Function to validate a model using the training dictionary

    :param training_dict: Dictionary containing training relevant objects 
    :type training_dict: dict
    :return: Validation Loss
    :rtype: float
    """
    # Create the dataloader
    validation_dataloader = training_dict['validation_datamodule'].dataloader(shuffle = True)

    # ---Validation Start---
    current_validation_loss_total = 0.0
    
    # Switch the model to validation mode
    training_dict['model'].eval()

    # Progress bar (Needs reset every epoch)
    if training_dict['progress_bar']:
        progress_bar = tqdm(validation_dataloader)
    else:
        progress_bar = validation_dataloader

    for current_batch_idx, batch_data in enumerate(progress_bar):
        # Extract the image and labels from batch
        images, labels = batch_data

        # Send the tensors to the correct device
        images = images.to(training_dict['device'])
        labels = labels.to(training_dict['device'])

        # Get the model output for this batch
        outputs = training_dict['model'](images)

        # Compute the loss
        validation_loss = training_dict['loss_fn'](outputs, labels)

        # Update the validation metrics
        for validation_metric in training_dict['validation_metrics']:
            validation_metric.update(outputs, labels)  

        # Keep track of the running loss
        current_validation_loss_total += validation_loss.item()

        # Average the loss across all the batches
        current_validation_loss = current_validation_loss_total / (current_batch_idx + 1)

        # Update progress bar
        if training_dict['progress_bar']:
            progress_bar.set_description('Validation Loss: {:.2f}'.format(current_validation_loss), refresh = True)

    if training_dict['progress_bar']:
        progress_bar.close()
        logger.info(str(progress_bar))

    return current_validation_loss

def calculate_metrics_and_update_scheduler(training_dict):
    """Function to update both the metrics csv and scheduler with results from training and validation.

    :param training_dict: Dictionary containing training relevant objects
    :type training_dict: dict
    :return: Metric dictionary
    :rtype: dict
    """
    metric_dict = {}

    metric_dict['Train CCE Loss'] = round(training_dict['training_loss'][0], 2)
    metric_dict['Train Prior Loss'] = round(training_dict['training_loss'][1], 10)
    metric_dict['Train Total Loss'] = round(training_dict['training_loss'][2], 10)
    for metric_id, training_metric in enumerate(training_dict['training_metrics']):
        temp_metric = training_metric.compute()
        # Per-class Metric Check
        if len(temp_metric.size()) > 0:
            for class_id in range(temp_metric.size()[0]):
                metric_dict_key = 'Class {} Train F1Score'.format(class_id)
                train_f1_score = temp_metric[class_id].item()
                metric_dict[metric_dict_key] = round(train_f1_score, 2)
        # Single Score Metric Check
        else:
            # TODO: Probably find a more elegant solution to get metric name
            metric_dict_key = 'Train ' + str(type(training_metric)).split('.')[-1].split('\'')[0]
            metric_dict[metric_dict_key] = round(temp_metric.item(), 2)

    metric_dict['Val CEE Loss'] = round(training_dict['validation_loss'], 10)
    for validation_metric in training_dict['validation_metrics']:
        temp_metric = validation_metric.compute()
        # Per-class Metric Check
        if len(temp_metric.size()) > 0:
            for class_id in range(temp_metric.size()[0]):
                metric_dict_key = 'Class {} Val F1Score'.format(class_id)
                val_f1_score = temp_metric[class_id].item()
                metric_dict[metric_dict_key] = round(val_f1_score, 2)
        # Single Score Metric Check
        else:
            # TODO: Probably find a more elegant solution to get metric name
            metric_dict_key = 'Val ' + str(type(training_metric)).split('.')[-1].split('\'')[0]
            metric_dict[metric_dict_key] = round(temp_metric.item(), 2)

    
    metric_dict['Train Datapoints'] = len(training_dict['training_datamodule'].dataset)

    if (training_dict['scheduler'] != None) and (training_dict['current_epoch'] != 0):
        # training_dict['scheduler'].step(metric_dict['Val MulticlassF1Score'])
        training_dict['scheduler'].step()

    return metric_dict

def record_metrics(training_dict, metrics_dict):
    """Function to record the metrics to a csv file.

    :param training_dict: Dictionary containing training relevant objects 
    :type training_dict: dict
    :param metrics_dict: Metric dictionary
    :type metrics_dict: dict
    """
    time_end = time.perf_counter()
    metrics_dict['Training Round Time'] = round(time_end - training_dict['epoch_time_start'], 2)
    
    # Write metrics to a dataframe and save it off as a csv
    training_dict['metric_df'] = pd.concat(
        [training_dict['metric_df'], 
        pd.DataFrame(metrics_dict, index = [training_dict['current_epoch']])], 
    ignore_index = True)
    
    training_dict['metric_df'].to_csv(training_dict['metric_path'], index = False)

def reset_metrics(training_dict):
    """Function to reset the metrics after each training round.

    :param training_dict: Dictionary containing training relevant objects 
    :type training_dict: dict
    """
    for training_metric in training_dict['training_metrics']:
        training_metric.reset()

    for validation_metric in training_dict['validation_metrics']:
        validation_metric.reset()

def checkpoint_model(training_dict, multi_gpu = False, keep_checkpoints = True):
    """Function to create model checkpoints

    :param training_dict: Dictionary containing training relevant objects
    :type training_dict: dict
    :param multi_gpu: Flag indicating if using multi-gpu training, defaults to False
    :type multi_gpu: bool, optional
    :param keep_checkpoints: Flag indicating whether to save all checkpoints, defaults to True
    :type keep_checkpoints: bool, optional
    """
    if multi_gpu:
        checkpoint_dictionary = {
            'epoch': training_dict['current_epoch'],
            'model_state_dict': training_dict['model'].module.state_dict(),
            'optimizer_state_dict': training_dict['optimizer'].state_dict(), 
            'loss': training_dict['training_loss'][2],
        }
    else:
        checkpoint_dictionary = {
            'epoch': training_dict['current_epoch'],
            'model_state_dict': training_dict['model'].state_dict(),
            'optimizer_state_dict': training_dict['optimizer'].state_dict(), 
            'loss': training_dict['training_loss'][2],
        }

    
    torch.save(checkpoint_dictionary, training_dict['checkpoint_path'])
    # WARNING: Currently does not separate epochs from name.
    if keep_checkpoints:
        #original:
        #checkpoints/explanation_cifar10/co/co_ep00001.00000_ls001_rs00002_w-1.00E-01/model.pth
        #.pth
        file_extension = training_dict['checkpoint_path'].rpartition('.')[2]
        #checkpoints/explanation_cifar10/co/co_ep00001.00000_ls001_rs00002_w-1.00E-01/model
        path_without_extension = training_dict['checkpoint_path'].rpartition('.')[0]
        ##checkpoints/explanation_cifar10/co/co_ep00001.00000_ls001_rs00002_w-1.00E-01/model<epoch_num>.pth
        new_path = path_without_extension + '_{:03d}'.format(training_dict['current_epoch']) + '.' + file_extension

        torch.save(checkpoint_dictionary, new_path)