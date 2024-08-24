import os
import torch
from functools import reduce
import warnings
import random
import numpy as np

from .logging_utilities import *


logger = setup_logger(__name__)


def get_module_by_name(module, layer_name):
    """Function to get a part of Pytorch module by layer name

    :param module: Pytorch module to get
    :type module: torch.nn.Module
    :param layer_name: Layer you want to get from the Pytorch module
    :type layer_name: str
    :return: Extracted Pytorch module
    :rtype: torch.nn.module
    """
    names = layer_name.split(sep='.')
    return reduce(getattr, names, module)

def load_checkpoint(checkpoint_path, model):
    """Loads a model checkpoint from a checkpoint .pth file 

    :param checkpoint_path: Checkpoint file path containing model weights.
    :type checkpoint_path: str
    :param model: Pytorch model to load weights into
    :type model: torch.nn.Module
    :return: Last training round value
    :rtype: int
    """
    # Load in a checkpoint
    if os.path.exists(checkpoint_path):
        logger.info(f'Loading checkpoint file: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch']

    return -1

class UnNormalize(object):
    def __init__(self, mean, std):
        """UnNormalize helper class init

        :param mean: Mean used to normalized the tensor
        :type mean: tuple
        :param std: Standard deviation used to normalize the tensor
        :type std: tuple
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Call function for UnNormalize helper class

        :param tensor: Tensor to un-normalize
        :type tensor: torch.Tensor
        :return: Un-normalized tensor
        :rtype: torch.Tensor
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            # Inverse of normalize code -> t.sub_(m).div_(s)
            t.mul_(s).add_(m)
        return tensor
    

# create an additional field for each desired attribution for each parameter in the model
def attach_attributions(model, attribution_names):

    for attribution_name in attribution_names:
        logger.debug(f"Attaching attribution: {attribution_name}")

    # loop over all model parameters
    for name, param in model.named_parameters():
        # dictionary to hold attribution values
        param.attributions = dict.fromkeys(attribution_names)
        # dictionary to hold flags for whether to compute attributions 
        param.attributions_mask = dict.fromkeys(attribution_names)

        # populate dictionary item with tensor of correct shape
        for attribution_name in attribution_names:
            param.attributions[attribution_name] = torch.zeros_like(param)
            param.attributions_mask[attribution_name] = torch.ones_like(param)


def prune_model_by_attributions(model, attribution_names, prune_proportion, abs_attribution=True, device=None):

    # print(f'Named Parameters: {dict(model.named_parameters()).keys()}')
    # exit()

    # WARNING this prunes parameters from all parts of the model, including the final fully connected and bias terms
    # WARNING may want to only prune from certain layers

    # TODO don't prune from output later
    parameter_list = list(model.named_parameters())
    output_weight_name, output_weight = parameter_list[-2]
    logger.debug(f'Planning to Ignore Output Parameter: {output_weight_name}')
    logger.debug(f'Output Parameter Shape: {output_weight.shape}')

    output_bias_name, output_bias = parameter_list[-1]
    logger.debug(f'Planning to Ignore Output Parameter: {output_bias_name}')
    logger.debug(f'Output Parameter Shape: {output_bias.shape}')

    # loop over all model parameters
    for name, param in model.named_parameters():

        if name in [output_weight_name, output_bias_name]:
            logger.debug(f'Ignoring parameter: {name}, shape: {param.shape}')
            continue

        #print(f'Pruning parameter: {name}, shape: {param.shape}')
        logger.debug(f'Pruning parameter: {name}, shape: {param.shape}')

        if device != None:
            param_data = param.data.cpu().detach().clone()
        else:
            param_data = param.data.detach().clone()

        if attribution_names == []:

            warnings.warn("Pruning random elements from each parameter")
           
            # Set proportion of the elements to 0 randomly

            # Flatten parameters
            flat_param_data = param_data.flatten()

            # Compute the number of parameters to prune
            prune_count = int(flat_param_data.size(0) * prune_proportion)
            logger.debug(f"Pruning {prune_count} elements out of {flat_param_data.size(0)} ({100*prune_count/flat_param_data.size(0)}%).")
            
            # Determine indices of parameters we want to prune
            prune_indices = random.sample(range(flat_param_data.size(0)), prune_count)

            # Set the specified parameters to 0
            flat_param_data[prune_indices] = 0

            # Return pruned parameters to original shape
            pruned_values = flat_param_data.reshape(param_data.shape).to(device)
            param.data = pruned_values
            
        else:

            for attribution_name in attribution_names:

                #check that desired attributions are present
                if attribution_name not in param.attributions.keys():
                    raise ValueError(f"Attribution {attribution_name} not computed.")

                #check that attributions are the correct shape
                if param.attributions[attribution_name].shape != param.data.shape:
                    raise ValueError("Attributions must match parameter shape")

                if device != None:
                    param_attributions = param.attributions[attribution_name].cpu().detach().clone()
                else:
                    param_attributions = param.attributions[attribution_name].detach().clone()

                if abs_attribution:
                    param_attributions = torch.abs(param_attributions)

                # determine parameter attribution quantile threshold corresponding to prune_proportion
                prune_threshold = torch.quantile(param_attributions, 1-prune_proportion)
                logger.debug(f"prune_threshold: {prune_threshold}")

                # Flatten parameters and attributions
                flat_param_data = param_data.flatten()
                logger.debug(f"flat_param_data shape: {flat_param_data.shape}")
                flat_param_attributions = param_attributions.flatten()
                logger.debug(f"flat_param_attributions shape: {flat_param_attributions.shape}")

                # Compute the number of parameters to prune
                prune_count = int(flat_param_data.size(0) * prune_proportion)
                prune_mask = flat_param_attributions >= prune_threshold
                
                # Determine indices of parameters we want to prune
                prune_indices = prune_mask.nonzero().squeeze(1)                
                logger.debug(f"prune_indices type : {type(prune_indices)}")
                logger.debug(f"prune_indices shape : {prune_indices.shape}")
                logger.debug(f"num indices where prune condition met : {prune_indices.size(0)}")
                # Can't prune more parameters than allowed by the attributions and the prune proportion
                #NOTE Need to verify that this doesn't ever prune fewer elements than intended
                #NOTE (depends on if torch.quantile is returning the correct value)
                #NOTE I have sort of verified this but just to be safe we should make sure by checking the logs some more for various experiments
                num_prune_indices = min(prune_count, prune_indices.size(0))
                prune_indices = random.sample(prune_indices.tolist(), num_prune_indices)
                logger.debug(f"Pruning {num_prune_indices} elements out of {flat_param_data.size(0)} ({100*num_prune_indices/flat_param_data.size(0)}%).")

                # Set the specified parameters to 0
                flat_param_data[prune_indices] = 0

                # Return pruned parameters to original shape
                pruned_values = flat_param_data.reshape(param_data.shape).to(device)
                param.data = pruned_values


                # # set all parameters to 0 which have attributions above the threshold
                # pruned_values = torch.where(param_attributions < prune_threshold, param_data, torch.zeros_like(param_data)).to(device)
                # #print(pruned_values)
                # param.data = pruned_values

