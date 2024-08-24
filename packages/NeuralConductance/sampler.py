import torch
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, Subset

from ..utilities.logging_utilities import *

# Setup a logger
logger = setup_logger(__name__)

class Sampler:
    """Class to generate sample points
    """
    def __init__(self):
        """Sampler init
        """
        pass

    def generate_sample_points(self):
        """Generate sample points

        :return: Batch of sample points
        :rtype: torch.Tensor or None
        """
        # NOTE: Calculate sample points
        sample_points = None
        return sample_points

    def generate_name(self):
        """Get the name of the sampler

        :return: Name of the sampler
        :rtype: str
        """
        return __name__

# IDEA: Extend this with a ball sampler
# TODO: Add generate_name function?
class ClassSampler(Sampler):
    """Class to create a dataloader specific class
    """
    def __init__(self, reference_dataset, class_labels = [], num_samples = 5):
        """ClassSampler init

        :param reference_dataset: Dataset to sample from
        :type reference_dataset: torch.utils.data.Dataset
        :param class_labels: List of class label to create specific dataloader for, defaults to []
        :type class_labels: list, optional
        :param num_samples: Number of samples in a batch, defaults to 5
        :type num_samples: int, optional
        """
        #NOTE this sampler must return images and labels, since parameter attribution requires labeled points from the input space
        #NOTE (in order to compute parameter grads)


        logger.debug('ClassSampler Init')
        self.reference_dataset = reference_dataset
        self.dataset_class_labels = np.array(reference_dataset.labels)
        self.class_labels = class_labels
        self.mask = np.full(len(self.class_labels), fill_value=1, dtype=int)
        self.num_samples = num_samples
        self.create_dataloader(class_labels = self.class_labels, num_samples = self.num_samples)
        Sampler.__init__(self)

    def create_dataloader(self, class_labels = [], num_samples = 5):
        """Function to create a dataloader for a specific class.

        :param class_labels: Class label to create specific dataloader for, defaults to None
        :type class_labels: list optional
        :param num_samples: Number of samples in a batch, defaults to 5
        :type num_samples: int, optional
        """
        if class_labels != []:
            logger.debug('Creating a new dataloader containing only classes {}'.format(class_labels))
            # Create a masked dataset based on the class label
            class_mask = np.isin(self.dataset_class_labels, class_labels)
            indicies = class_mask.nonzero()[0]
            class_dataset = Subset(self.reference_dataset, indicies)

            # Create a dataloader to get a sample batch
            self.sample_dataloader = DataLoader(
                dataset = class_dataset, 
                shuffle = True, 
                batch_size = num_samples, 
                num_workers = 8
            )
        else:
            logger.debug('Creating a new dataloader containing every class')
            self.sample_dataloader = DataLoader(
                dataset = self.reference_dataset, 
                shuffle = True, 
                batch_size = num_samples, 
                num_workers = 8
            )

    def generate_sample_points(self):
        """Generate sample points using the dataloader from create_dataloader

        :return: Tuple containing the sample points and their respective labels
        :rtype: tuple
        """

        # Extract a batch out of the dataloader
        sample_points, sample_labels = next(iter(self.sample_dataloader))

        # Return samples points tuple
        return (sample_points, sample_labels)


# ISSUE: Need to replace with actual uniform sampling for arbitrary dimensions, not just using 2-norm.
# ISSUE: probably realistic uniform sampling would use some kind of random walk or other cool method
class BallSampler(Sampler):
    """Class to generate samples within a ball radius
    """
    def __init__(self, eps = 0.01, num_sample_points = 1):
        """BallSampler init

        :param eps: Radius of the ball to sample from, defaults to 0.01
        :type eps: float, optional
        :param num_sample_points: Number of sample points to generate, defaults to 1
        :type num_sample_points: int, optional
        """
        logger.debug('BallSampler Init')
        self.eps = eps
        self.num_sample_points = num_sample_points
        Sampler.__init__(self)

    def generate_name(self):
        """Generate BallSampler name using epsilon and number of sample points

        :return: BallSampler name used for experiments
        :rtype: str
        """
        return __class__.__name__ + '_' + str(self.eps)+ '_' + str(self.num_sample_points)
    
    def generate_sample_points(self, input_tensor, device = None):
        """Generate sample points from a ball

        :param input_tensor: Center of the ball to generate samples from
        :type input_tensor: torch.Tensor
        :param device: GPU device id, defaults to None
        :type device: int, optional
        :return: Sample points generated from the ball
        :rtype: torch.Tensor
        """
        batch_size = input_tensor.size(dim = 0)
        
        # Find a random tensor same size as input tensor
        # Reference direction shape: (Batch, num_samples, size of input tensor)
        reference_direction = torch.rand(size = tuple([batch_size,self.num_sample_points]) + input_tensor.size()[1:], dtype = torch.float32)
        normalized_reference_direction = torch.zeros(reference_direction.size())
        sample_points = torch.zeros(reference_direction.size())

        # sample_points = sample_points.to(device = device)
        
        for batch_id in range(batch_size):
            for index in range(self.num_sample_points):
                reference_norm = torch.norm(reference_direction[batch_id, index], p = 2)
                normalized_reference_direction[batch_id, index] = reference_direction[batch_id,index] / reference_norm

                # Compute a random radius
                random_radius = torch.rand(1) * self.eps 

                # Get final scaled direction
                offset = normalized_reference_direction[batch_id, index] * random_radius

                if device:
                    offset = offset.to(device = device)

                # Get the final sample point
                sample_points[batch_id, index] = input_tensor[batch_id] + offset

        # Return Shape: (Batch, num_samples, size of input tensor)
        return sample_points

class ExpectedGradientsSampler(Sampler):
    """Class for generating samples according to the Expected Gradients method. Paper: https://arxiv.org/abs/1906.10670
    """
    def __init__(self, reference_dataset, num_samples_per_line = 1, num_reference_points = 1):
        """ExpectedGradientsSampler init

        :param reference_dataset: Dataset for line endpoints
        :type reference_dataset: torch.utils.data.Dataset
        :param num_samples_per_line: Number of sample points per line, defaults to 1
        :type num_samples_per_line: int, optional
        :param num_reference_points: Number of lines/line endpoints, defaults to 1
        :type num_reference_points: int, optional
        """
        logger.debug('Expected Gradients Sampler Init')
        self.reference_dataset = reference_dataset
        self.num_samples_per_line = num_samples_per_line
        self.num_reference_points = num_reference_points

        Sampler.__init__(self)

    def generate_name(self):
        """Generate ExpectedGradientsSampler name using num_samples_per_line and num_reference_points

        :return: ExpectedGradientsSampler name used for experiments
        :rtype: str
        """
        return __name__ + '_' + str(self.num_samples_per_line)+ '_' + str(self.num_reference_points)

    def generate_sample_points(self, input_tensor, device = None):
        """Generate sample points using Expected Gradients using random spacing for line integrals.

        :param input_tensor: Center point or inner line endpoint to generate samples from
        :type input_tensor: torch.Tensor
        :param device: GPU device id, defaults to None
        :type device: int, optional
        :return: Sample points generated from the Expected Gradients
        :rtype: torch.Tensor
        """
        batch_size = input_tensor.size(dim = 0)
        logger.debug('Batch_size', batch_size)
        logger.debug('Input_size', input_tensor.size()[1:])
        num_sample_points = self.num_reference_points * self.num_samples_per_line

        # Generate reference tensor
        reference_sampler = RandomSampler(
            data_source = self.reference_dataset, 
            replacement = True,
            num_samples = self.num_reference_points, 
        )

        reference_dataloader = DataLoader(
            dataset = self.reference_dataset,
            batch_size = self.num_reference_points,
            sampler = reference_sampler,
            shuffle = True
        )

        sample_points = torch.zeros(size = tuple([batch_size, num_sample_points]) + input_tensor.size()[1:])
        # reference_images = torch.zeros(size = tuple([batch_size, self.num_reference_points]) + input_tensor.size()[1:])
        logger.debug('Sample points', sample_points.shape)
        
        for batch_id in range(batch_size):
            reference_points = next(iter(reference_dataloader))[0].float()
            logger.debug('Reference points', reference_points.shape)

            # reference_images[batch_id] = reference_points
            
            for reference_index in range(self.num_reference_points):
                for line_index in range(self.num_samples_per_line):

                    # Compute a random alpha
                    random_alpha = torch.rand(1)

                    # Get final scaled direction
                    line_sample = reference_points[reference_index] * random_alpha

                    # Get the final sample point
                    sample_points[batch_id, reference_index * self.num_samples_per_line + line_index] = input_tensor[batch_id] + line_sample

        # Return Shape: (Batch, num_samples, size of input tensor)
        return sample_points

class BallExpectedGradientsSampler(Sampler):
    """Class to generate samples within a ball radius using Expected Gradients
    """
    def __init__(self, reference_dataset, num_samples_per_line = 1, num_reference_points = 1,  eps = 0.01):
        """BallExpectedGradientsSampler init

        :param reference_dataset: Dataset to reference when generating sample points
        :type reference_dataset: torch.utils.data.Dataset
        :param num_samples_per_line: Number of sample points per line, defaults to 1
        :type num_samples_per_line: int, optional
        :param num_reference_points: Number of lines/line endpoints, defaults to 1
        :type num_reference_points: int, optional
        :param eps: Radius of the ball to sample from, defaults to 0.01
        :type eps: float, optional
        """
        self.eps = eps
        self.reference_dataset = reference_dataset
        self.num_samples_per_line = num_samples_per_line
        self.num_reference_points = num_reference_points

        Sampler.__init__(self)

    def generate_sample_points(self, input_tensor, device = None):
        """Generate sample points from a ball using Expected Gradients

        :param input_tensor: Center of the ball to generate samples from
        :type input_tensor: torch.Tensor
        :param device: GPU device id, defaults to None
        :type device: int, optional
        :return: Sample points generated from the ball
        :rtype: torch.Tensor
        """
        batch_size = input_tensor.size(dim = 0)
        # print('batch_size', batch_size)
        # print('input_size', input_tensor.size()[1:])
        num_sample_points = self.num_reference_points * self.num_samples_per_line

        # Generate reference tensor
        reference_sampler = RandomSampler(
            data_source = self.reference_dataset, 
            replacement = True,
            num_samples = self.num_reference_points, 
        )

        reference_dataloader = DataLoader(
            dataset = self.reference_dataset,
            batch_size = self.num_reference_points,
            sampler = reference_sampler,
            shuffle = True
        )

        sample_points = torch.zeros(size = tuple([batch_size, num_sample_points]) + input_tensor.size()[1:])
        #reference_images = torch.zeros(size = tuple([batch_size, self.num_reference_points]) + input_tensor.size()[1:])
        # print('Sample points', sample_points.shape)
        
        for batch_id in range(batch_size):
            reference_points = next(iter(reference_dataloader))[0].float()
            # print('Reference points', reference_points.shape)

            #reference_images[batch_id] = reference_points
            
            for reference_index in range(self.num_reference_points):
                for line_index in range(self.num_samples_per_line):

                    # Scale reference direction to be of norm 1
                    reference_norm = torch.norm(reference_points[reference_index], p = 2)
                    normalized_reference_direction = reference_points[reference_index] / reference_norm

                    # Compute a random radius (equivalent to random radius for epsilon ball sampler)
                    random_alpha = torch.rand(1) * self.eps

                    # Get final scaled direction
                    scaled_line_sample = normalized_reference_direction * random_alpha
                    if not device == None:
                        scaled_line_sample = scaled_line_sample.to(device)

                    # Get the final sample point
                    sample_points[batch_id, reference_index * self.num_samples_per_line + line_index] = input_tensor[batch_id] + scaled_line_sample

        # Return Shape: (Batch, num_samples, size of input tensor)
        return sample_points