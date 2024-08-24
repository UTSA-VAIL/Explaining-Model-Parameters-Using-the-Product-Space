import torch

from ..utilities.logging_utilities import *

# Setup a logger
logger = setup_logger(__name__)


class Regularization():
    """Helper class to aggregate the attributions
    """
    def __call__(self, attributions, method_string, absolute_measure = False):
        """Call function to help aggregate attributions

        :param attributions: Input and/or Model Attributions
        :type attributions: torch.tensor
        :param method_string: Method used to aggregrate attributions
        :type method_string: str
        :param absolute_measure: Flag to take absolute of the measure, defaults to False
        :type absolute_measure: bool, optional
        :raises NotImplementedError: Implemented regularization method not selected
        :return: Aggregation of the attributions
        :rtype: torch.tensor
        """
        logger.debug('attributions Shape: {}'.format(attributions.shape))
        if absolute_measure:
            attributions = torch.abs(attributions)
        
        if method_string == 'min':
            regularization_value = self.min(attributions)
        elif method_string == 'max':
            regularization_value = self.max(attributions)
        elif method_string == 'median':
            regularization_value = self.median(attributions)
        elif method_string == 'mean':
            regularization_value = self.mean(attributions)
        else:
            raise NotImplementedError('Please select an implemented regularization method.')

        return regularization_value

    def min(self, attributions):
        """Function to return the min of the attributions

        :param attributions: Input and/or Model Attributions
        :type attributions: torch.tensor
        :return: Min of the attributions
        :rtype: torch.tensor
        """
        return torch.min(attributions)
    
    def max(self, attributions):
        """Function to return the max of the attributions

        :param attributions: Input and/or Model Attributions
        :type attributions: torch.tensor
        :return: Max of the attributions
        :rtype: torch.tensor
        """
        return torch.max(attributions)
    
    def median(self, attributions):
        """Function to return the median of the attributions

        :param attributions: Input and/or Model Attributions
        :type attributions: torch.tensor
        :return: Median of the attributions
        :rtype: torch.tensor
        """
        return torch.median(attributions)

    def mean(self, attributions):
        """Function to return the mean of the attributions

        :param attributions: Input and/or Model Attributions
        :type attributions: torch.tensor
        :return: Mean of the attributions
        :rtype: torch.tensor
        """
        return torch.mean(attributions)

class AttributionPrior(object):
    """Class to calculate attribution based prior for model loss purposes
    """
    def __init__(self, regularization_method = 'mean', weight = 0.001, absolute_measure = False):
        """AttributionPrior init

        :param regularization_method: Method to aggregate attributions, defaults to 'mean'
        :type regularization_method: str, optional
        :param weight: Scaling factor for attribution prior, defaults to 0.001
        :type weight: float, optional
        :param absolute_measure: Flag to take absolute of the measure, defaults to False
        :type absolute_measure: bool, optional
        """
        logger.debug('MAIMPrior Class Init')
        self.regularization = Regularization()
        self.regularization_method = regularization_method
        self.weight = weight
        self.absolute_measure = absolute_measure

    def compute_prior(self, attributions):
        """Function to compute the attribution prior

        :param attributions: Input and/or Model Attributions
        :type attributions: torch.tensor
        :return: Weighted attribution prior loss
        :rtype: torch.tensor
        """
        return self.weight * self.regularization(attributions, self.regularization_method, self.absolute_measure)