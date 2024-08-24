import torch

from ..utilities.logging_utilities import *
import numpy as np

# Setup a logger
logger = setup_logger(__name__)

# **********************************************************************
# Description:
#   Class to handle data dropout during training. 
# Parameters:
#   config_dict - Class removal policy from the config json file
# Notes:
#   -
# **********************************************************************
class DataDropout:
    def __init__(self, config_dict):
        self.removed_classes = []
        self.config_dict = config_dict

        self.dataset_reinit_period = config_dict['dataset_reinit_period']
        self.shuffle_mask = config_dict['shuffle_mask']
        self.warmup_period = config_dict['warmup_period']

        if 'slclassification' in self.config_dict:
            self.config_dict = self.config_dict['slclassification']
            self.task = 'slclassification'
        else:
            warnings.warn("Select either slclassification", UserWarning)

        self.rounds_since_reset = 0
        #logger.info('Setting up Adaptive Data Dropout')

    # **********************************************************************
    # Description:
    #   Call function for determining which data dropout policy to use.
    # Parameters:
    #   training_dict - Dictionary containing training relevant objects like
    #                   like model and dataset
    # Notes:
    #   -
    # **********************************************************************
    def __call__(self, training_dict, gpu=0):
        if self.task == 'slclassification':
            if self.config_dict['strategy'] == 'adaptive':
                self.adaptive_slclassification_data_dropout(training_dict, gpu)
            elif self.config_dict['strategy'] == 'naive':
                self.naive_slclassification_data_dropout(training_dict, gpu)



    # **********************************************************************
    # Description:
    #   Function to calculate the proportion based on a score and residue curvature
    # Parameters:
    #   score - the score to calculate the proportion on. Usually a metric.
    # Notes:
    #   -
    # **********************************************************************
    def calculate_proportion(self, score):
        # Calculate a proportion based on a metric score and the curvature
        exponent = np.tan((np.pi / 2) * self.config_dict['curvature'])
        prop = (1 - score ** exponent) ** (1 / exponent)
        return prop

    # **********************************************************************
    # Description:
    #   Naive data dropout for single-label classification tasks. 
    # Parameters (+Configs):
    #   training_dict - Dictionary containing training relevant objects like
    #                   like model and dataset
    #   full_review - Periodically reinitialize the dataset
    #   warmup_period - Start up period to avoid removal of data
    #   f1_threshold - Performance threshold for removing data
    #   removal_restriction - Restrict the number of classes that can be
    #                   removed
    #   reset_residue - Shuffle existing residue examples for new ones
    # Notes:
    #   The last validation metric is used as the dropout metric
    # **********************************************************************
    def naive_slclassification_data_dropout(self, training_dict, gpu=0):
        # Full review check
        if self.dataset_reinit_period > 0:
            if (self.rounds_since_reset + 1) % self.dataset_reinit_period == 0 or (training_dict['current_epoch'] + 1) == (training_dict['epochs'] - 1):
                self.rounds_since_reset = 0

                # Turn on entire training dataset
                log(logger.info, gpu, 'Dataset reset period reached. Re-establishing full training dataset.')
                for class_label in self.removed_classes:
                    training_dict['training_datamodule'].remove_class_with_residue(class_label = class_label, residue = 1.00)

                # Reset removed classes and exit function early
                self.removed_classes = []
                return
            else:
                self.rounds_since_reset += 1

        # Always take the last validation metric (TODO: Make this more intelligent)
        dropout_metric = training_dict['validation_metrics'][-1].compute()

        # Check if any of the per_class f1scores are above the threshold
        sorted_scores, class_indicies = torch.sort(dropout_metric, descending = True) 
        indicies = (sorted_scores > self.config_dict['metric_threshold']).nonzero(as_tuple=False)
        log(logger.debug, gpu, 'Sorted F1 Scores: {}'.format(sorted_scores))

        # Check that warmup period has been met
        if self.warmup_period - training_dict['current_epoch'] <= 0:
            log(logger.debug, 'Warmup Period has been reached', gpu)
            num_classes_removed_this_epoch = 0
            # Loop through the scores above the f1_threshold
            for index in indicies:
                # Retrieve the class label
                class_label = class_indicies[index].item() 
                # Double check that the class has not already been processed
                if not class_label in self.removed_classes:
                    log(logger.info, gpu, 'Class {} is above the threshold'.format(class_label))

                    # ---Class Removal Start---
                    log(logger.info, gpu, 'Preforming Data Dropout on Class {}'.format(class_label))
                    training_dict['training_datamodule'].remove_class_with_residue(class_label = class_label, residue = self.config_dict['residue_percentage'])

                    # Add class to the list of processed classes
                    self.removed_classes.append(class_label)
                    log(logger.debug, gpu, 'Removed classes: {}'.format(self.removed_classes))

                    num_classes_removed_this_epoch += 1

                # Leave the loop if the removal restriction policy is in place
                if self.config_dict['removal_restriction'] != 0 and num_classes_removed_this_epoch - self.config_dict['removal_restriction'] == 0:
                    log(logger.debug, gpu, 'Hit removal restriction policy of {} classes.'.format(self.config_dict['removal_restriction']))
                    break

            if self.config_dict['reset_residue'] > 0 and training_dict['current_epoch'] % self.config_dict['reset_residue'] == 0 and len(self.removed_classes) > 0:
                for class_label in self.removed_classes:
                    log(logger.info, gpu, 'Reshuffling Data Dropout Examples on Class {}'.format(class_label))
                    training_dict['training_datamodule'].remove_class_with_residue(
                        class_label = class_label, 
                        residue = self.config_dict['residue_percentage'], 
                        shuffle_mask = self.shuffle_mask
                    )

    # **********************************************************************
    # Description:
    #   Adaptive data dropout for single-label classification tasks. 
    # Parameters (+Configs):
    #   training_dict - Dictionary containing training relevant objects like
    #                   like model and dataset
    #   full_review - Periodically reinitialize the dataset
    #   warmup_period - Start up period to avoid removal of data
    #   residue_concavity - Curve to determine residue percentage based
    #                   off performance
    #   removal_restriction - Restrict the number of classes that can be
    #                   removed
    #   reset_residue - Shuffle existing residue examples for new ones
    # Notes:
    #   The last validation metric is used as the dropout metric
    # **********************************************************************
    def adaptive_slclassification_data_dropout(self, training_dict, gpu=0):
         # Always take the last validation metric (TODO: Make this more intelligent)
        dropout_metric = training_dict['validation_metrics'][-1].compute()
        
        # Full review check
        if self.dataset_reinit_period > 0:
            if (self.rounds_since_reset + 1) % self.dataset_reinit_period == 0 or (training_dict['current_epoch'] + 1) == (training_dict['epochs'] - 1):
                self.rounds_since_reset = 0

                # Turn on entire training dataset
                log(logger.info, gpu, 'Dataset reset period reached. Re-establishing full training dataset.')
                for class_label in range(dropout_metric.size()[0]):
                    training_dict['training_datamodule'].remove_class_with_residue(class_label = class_label, residue = 1.00)

                # Reset removed classes and exit function early
                self.removed_classes = []
                return
            else:
                self.rounds_since_reset += 1
        
        log(logger.debug, gpu, 'rounds_since_reset {}'.format(self.rounds_since_reset))
        
        # Check if any of the per_class f1scores are above the threshold
        sorted_scores, class_indicies = torch.sort(dropout_metric, descending = True) 
        log(logger.debug, gpu, 'Sorted F1 Scores: {}'.format(sorted_scores))

        # Check that warmup period has been met
        if self.warmup_period - training_dict['current_epoch'] <= 0:
            
            # Loop over each class in descending score order
            for index, class_label in enumerate(class_indicies):

                class_label = class_label.item()

                # Calculate the percentage of data to leave in
                residue_percentage = self.calculate_proportion(sorted_scores[index].item())

                residue_percentage = max(residue_percentage, 0.10)

                # ---Class Removal Start---
                log(logger.trace, gpu, 'Preforming Data Dropout on Class {} with Residue {}'.format(class_label, residue_percentage))
                training_dict['training_datamodule'].remove_class_with_residue(
                    class_label = class_label, 
                    residue = residue_percentage, 
                    shuffle_mask = self.shuffle_mask
                )

            if self.shuffle_mask:
                training_dict['training_datamodule'].shuffle_mask()