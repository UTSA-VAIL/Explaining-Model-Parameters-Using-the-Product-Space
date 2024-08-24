import torch
import os
import json
import datetime
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from torchmetrics.classification import Accuracy, F1Score #type:ignore

from ..dataprocessing.datamodule import *
from .logging_utilities import *
from tqdm import tqdm
from .general_utilities import load_checkpoint

# Setup a logger
logger = setup_logger(__name__)


def setup_testing_dict(c):
    """Function to create a dictionary containing evaluation relevant objects. No mult-gpu.

    :param c: Config dictionary containing user parameters
    :type c: dict
    :return: Evaluation hyper-parameter dictionary
    :rtype: dict
    """
    # Setup the dataset
    testing_dataset = c.prepare_dataset()

     # Setup the model
    model, model_task = c.prepare_model()

    # Setup the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Setup the metrics
    testing_accuracy = Accuracy(task = 'multiclass', num_classes = c.config_dic['num_classes'])
    testing_f1score = F1Score(task = 'multiclass', num_classes = c.config_dic['num_classes'])
    per_class_testing_f1score = F1Score(task = 'multiclass', num_classes = c.config_dic['num_classes'], average = None)

    testing_metrics = [
        testing_accuracy, 
        testing_f1score, 
        per_class_testing_f1score, 
    ]

    # GPU flag checking
    if c.config_dic['gpu'] > 0 and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    metric_path = os.path.join(c.config_dic['log_dir'], c.config_dic['metric_file_name'])
    checkpoint_path = os.path.join(c.config_dic['checkpoint_dir'], c.config_dic['checkpoint_file'])
    if os.path.exists(metric_path):
        training_df = pd.read_csv(metric_path)
    else:
        training_df = pd.DataFrame()

    test_dataloader = torch.utils.data.DataLoader(
        dataset = testing_dataset,
        shuffle = True,
        batch_size = c.config_dic['batch_size'],
        num_workers = c.config_dic['num_workers'],
        pin_memory = True
    )

    testing_dict = {
        'testing_dataset': testing_dataset,
        'test_dataloader': test_dataloader,
        'model': model,
        'model_task': model_task,
        'loss_fn': loss_fn,
        'testing_metrics': testing_metrics,
        'device': device,
        'training_df': training_df,
        'progress_bar': c.config_dic['progress_bar'],
    }

    # Load in a checkpoint (if not using pretrained weights)
    if not c.config_dic['model']['pretrained']:
        load_checkpoint(checkpoint_path, testing_dict['model'])

    # Send the model to the correct device
    testing_dict['model'] = testing_dict['model'].to(device)

    # Send metrics to the device
    for index, metric in enumerate(testing_dict['testing_metrics']):
        testing_dict['testing_metrics'][index] = testing_dict['testing_metrics'][index].to(device)

    return testing_dict

def evaluate_model(testing_dict):
    """Function to evaluate a model using the evaluation dictionary.

    :param testing_dict: Dictionary containing evaluation relevant objects 
    :type testing_dict: dict
    :return: Testing loss
    :rtype: float
    """
    # Set the model to eval mode
    testing_dict['model'].eval()

    # Progress bar
    current_loss_total = 0.0
    if testing_dict['progress_bar']:
        progress_bar = tqdm(testing_dict['test_dataloader'])
    else:
        progress_bar = testing_dict['test_dataloader']
    

    # Loop through the data
    for current_batch_idx, (images, labels) in enumerate(progress_bar):
        # Send images and labels to the GPU
        images, labels = images.to(device = testing_dict['device']), labels.to(device = testing_dict['device'])

        # Get the model output
        outputs = testing_dict['model'](images)

        # Compute the loss
        loss = testing_dict['loss_fn'](outputs, labels)

        # Update the testing metrics
        for testing_metric in testing_dict['testing_metrics']:
            testing_metric.update(outputs, labels)

        # Keep track of the running loss
        current_loss_total += loss.item()

        # Average the loss across all the batches
        current_loss = current_loss_total / (current_batch_idx + 1)

        # Update progress bar
        if testing_dict['progress_bar']:
            progress_bar.set_description('Loss: {:.2f}'.format(current_loss), refresh = True)

    # Close the progress bars
    if testing_dict['progress_bar']:
        progress_bar.close()
        logger.info(str(progress_bar))

    return current_loss

def calculate_testing_metrics(testing_dict):
    """Function to calculate metrics on testing dataset.

    :param testing_dict: Dictionary containing evaluation relevant objects 
    :type testing_dict: dict
    :return: Metric dictionary
    :rtype: dict
    """
    metric_dict = {}

    metric_dict['Test Loss'] = round(testing_dict['testing_loss'], 4)
    for metric_id, testing_metric in enumerate(testing_dict['testing_metrics']):
        temp_metric = testing_metric.compute()
        # Per-class Metric Check
        if len(temp_metric.size()) > 0:
            for class_id in range(temp_metric.size()[0]):
                metric_dict_key = 'Class {} Test F1Score'.format(class_id)
                train_f1_score = temp_metric[class_id].item()
                metric_dict[metric_dict_key] = round(train_f1_score, 4)
        # Single Score Metric Check
        else:
            # TODO: Probably find a more elegant solution to get metric name
            metric_dict_key = 'Test ' + str(type(testing_metric)).split('.')[-1].split('\'')[0]
            metric_dict[metric_dict_key] = round(temp_metric.item(), 4)

    return metric_dict

# **********************************************************************
# Description:
#   Function extract training information from the metrics csv file
# Parameters:
#   testing_dict - Dictionary containing testing relevant objects 
#   like model and dataset.
# Notes:
#   -
# **********************************************************************
def training_report(testing_dict):
    """Function to extract training information from the training csv file.

    :param testing_dict: Dictionary containing evaluation relevant objects 
    :type testing_dict: dict
    """
    if testing_dict['training_df'].empty:
        testing_dict['metric_df']['Rounds of Training'] = 0
        testing_dict['metric_df']['Training Time'] = 0
        testing_dict['metric_df']['Training Datapoints'] = 0
        return

    # Calculate stats
    training_time = np.sum(np.array(testing_dict['training_df']['Training Round Time'].values))
    training_time_converted = str(datetime.timedelta(seconds = round(training_time)))
    training_datapoints = np.sum(np.array(testing_dict['training_df']['Train Datapoints'].values))

    # Add stats to metric dataframe
    testing_dict['metric_df']['Rounds of Training'] = len(testing_dict['training_df'].index)
    testing_dict['metric_df']['Training Time'] = training_time_converted
    testing_dict['metric_df']['Training Datapoints'] = int(training_datapoints)

# **********************************************************************
# Description:
#   Generate an input explanation
# Parameters:
#   c - Training Config object
#   ec - Explainer Config object
#   model_dict - Dictionary containing model objects like model and 
#   dataset.
#   batch_id - Specific batch you want inputs for
# Notes:
#   -
# **********************************************************************
def load_inputs_and_generate_attributions(ec, model_dict):
    # Setup up explainer and reference dataset
    #TODO update for new explanation config schema etc. and add a version for parameter explanations (which reurns nothing)
    if ec == None:
        raise ValueError("Must provide an explanation config dictionary")
    
    print(ec.config_dic.keys())

    input_explainer = ec.prepare_explainer()
    batch_size = ec.config_dic['attributions']['input_attribution']['batch_size']
    batch_id = ec.config_dic['attributions']['input_attribution']['batch_id']

    explanation_dataloader = torch.utils.data.DataLoader(
        dataset = model_dict['testing_dataset'],
        shuffle = True,
        batch_size = batch_size,
        num_workers = 1,
        pin_memory = False
    )

    for current_batch_idx, (images, labels) in enumerate(explanation_dataloader):
        if batch_id == current_batch_idx:
            break

     # Send images and labels to the GPU
    images, labels = images.to(device = model_dict['device']), labels.to(device = model_dict['device'])
        
    input_explanation = input_explainer.generate_attributions(
            model_dict = model_dict, 
            tensor_to_explain = images
        )

    return input_explanation




# **********************************************************************
# Description:
#   Save an input explanation to a json file
# Parameters:
#   c - Training Config object
#   ec - Explainer Config object
#   input_explanation - Explanation to save
#   checkpoint_file - Model to load if provided
# Notes:
#   -
# **********************************************************************
#TODO Fix all this stuff (and write a version for saving parameter attributions using parameter_attributions_as_dict)
def save_explanation(c, ec, input_explanation, checkpoint_file = None):
    # Generate an experiment name based on the parameters
    if ec is None:
        ec = c

    measure_name = ec.config_dic['attributions']['input_attribution']['measure'].replace('_','-')

    #TODO fix this part to accomodate new schema
    experiment_name = '{}_ep{:011.5f}_ls{:03d}_rs{:05d}'.format(
        ec.config_dic['attributions']['input_attribution']['sampler']['method'].replace('_','-'),
        ec.config_dic['attributions']['input_attribution']['sampler']['eps'],
        ec.config_dic['attributions']['input_attribution']['sampler']['num_samples_per_line'],
        ec.config_dic['attributions']['input_attribution']['sampler']['num_reference_points']
    )

    if 'save_dir' in ec.config_dic:
        save_dir = ec.config_dic['save_dir']
    else:
        save_dir = './explanations'
    
    checkpoint_dir_name = c.config_dic['checkpoint_dir'].split('/')[-1].replace('_','-')

    explanation_dir = os.path.join(save_dir, measure_name, checkpoint_dir_name, experiment_name)
    Path(explanation_dir).mkdir(parents = True, exist_ok = True)         
    
    if checkpoint_file is not None:
        explanation_path = os.path.join(explanation_dir, checkpoint_file.rpartition('.')[0] + '.json')
    else:
        explanation_path = os.path.join(explanation_dir, checkpoint_dir_name+'.json')

    with open(explanation_path, 'w') as outfile:
        json.dump(input_explanation.detach().cpu().numpy().tolist(), outfile)

# **********************************************************************
# Description:
#   Generate an input explanation for all available checkpoints
# Parameters:
#   c - Training Config object
#   ec - Explainer Config object
#   model_dict - Dictionary containing model objects like model and 
#   dataset.
#   batch_id - Specific batch you want inputs for
# Notes:
#   -
# **********************************************************************
def generate_input_attributions_all_checkpoints(c, ec, model_dict, batch_id = 0):

    # Find all checkpoint files
    for checkpoint_file in tqdm(sorted(os.listdir(c.config_dic['checkpoint_dir']))):
        
        # Skip the final checkpoint file
        if checkpoint_file == 'model.pth':
            continue

        if checkpoint_file.endswith('.pth'):

            checkpoint_path = os.path.join(c.config_dic['checkpoint_dir'], checkpoint_file)

            load_checkpoint(checkpoint_path, model_dict['model'])

            input_attributions = load_inputs_and_generate_attributions(ec = ec, model_dict = model_dict)

            save_explanation(c, ec, input_attributions, checkpoint_file)