import torch

from ..utilities.logging_utilities import *

# Setup a logger
logger = setup_logger(__name__)

class Measure:
    """Class to compute attribtions
    """
    def __init__(self):
        """Measure init
        """
        pass

    def compute_measure(self, **kwargs):
        """Compute the attribution

        :return: Batch of attributions
        :rtype: torch.tensor or None
        """
        measure_value = None
        return measure_value

class IntegratedGradients(Measure):
    """Class for Integrated Gradients calculations
    """
    def __init__(self):
        """Integrated Gradients init
        """
        logger.debug('Integrated Gradients Init')
        Measure.__init__(self)

    def compute_measure(self, **kwargs):
        """Compute expected value (mean) of sample gradients (E[grads]) Quanitifes magnitude of gradients.

        :return: Batch of Integrated Gradient attributions
        :rtype: torch.tensor
        """
        # Incoming shape: (Batch size, sample points, input or parameter size)
        sample_gradients = kwargs['sample_gradients']
        device = kwargs['device']
        
        # Take the mean along the sample point dimension
        measure_value = torch.mean(sample_gradients, dim = 1)
        
        # Outgoing shape: (Batch size, input or parameter size)
        return measure_value

class GradientVariance(Measure):
    """Class for Gradient Variance calculations
    """
    def __init__(self):
        """Gradient Variance init
        """
        Measure.__init__(self)

    def compute_measure(self, **kwargs):
        """Compute square of sample gradients (E[grads^2] - E[grads]^2). Quanitifes variation of gradients.
        :param \**kwargs:
            See below

        :Keyword Arguments:
        * *sample_gradients* (``torch.tensor``) --
        Gradients for all sample points

        :return: Batch of attributions
        :rtype: torch.tensor
        """
        # Incoming shape: (Batch size, sample points, input or parameter size)
        sample_gradients = kwargs['sample_gradients']
        device = kwargs['device']
        
        squared_grads = torch.pow(sample_gradients, 2)
        # Take the mean along the sample point dimension
        expected_squared_grads = torch.mean(squared_grads, dim = 1)
        expected_grads = torch.mean(sample_gradients, dim = 1)
        squared_expected_grads = torch.pow(expected_grads, 2)

        # Compute the difference of the expectation of squares and the squared expectation
        measure_value = expected_squared_grads - squared_expected_grads
        
        # Outgoing shape: (Batch size, input or parameter size)
        return measure_value

class Stability(Measure):
    """Class for Stability calculations
    """
    def __init__(self):
        """Stability init
        """
        Measure.__init__(self)

    def compute_measure(self, **kwargs):
        """Compute E[theta], where theta is the angle between (sample-input) and gradients for single channel. Quanitifes whether input is a local minimum.

        :Keyword Arguments:
        * *sample_gradients* (``torch.tensor``) --
        Gradients for all sample points
        * *input_tensor* (``torch.tensor``) --
        Input (Canidate minimum)
        * *sample* (``torch.tensor``) --
        Sample points
        * *input_gradients* (``torch.tensor``) --
        Gradients at the input


        :return: Batch of attributions
        :rtype: torch.tensor
        """
        # Incoming shape: (Batch size, sample points, input or parameter size)
        sample_gradients = kwargs['sample_gradients']
        input_tensor = kwargs['input_tensor']
        sample = kwargs['sample']
        input_gradients = kwargs['input_gradients']
        device = kwargs['device']

        num_sample_points = sample_gradients.shape[1]

        
        sample = sample.to(device = device)
        sample_gradients = sample_gradients.to(device = device)

        logger.debug("sample_gradients Shape: {}".format(sample_gradients.shape))
        logger.debug("input_tensor Shape: {}".format(input_tensor.shape))

        # Stretch input to look like sample points
        repeated_input = input_tensor.unsqueeze(1).repeat_interleave(num_sample_points, dim = 1)
        repeated_input = repeated_input.to(device = device)
        logger.debug("repeated_input Shape: {}".format(repeated_input.shape))

        # Difference between input and sample point 
        offset = repeated_input - sample
        logger.debug("offset Shape: {}".format(offset.shape))

        # Compute cosine similarity along color channel dimension to retain per-pixel attributions
        # WARNING: Need to confirm eps does not affect results in meaningful way
        cos = torch.nn.CosineSimilarity(dim = 2, eps = 1e-12)
        similarity = cos(offset, sample_gradients)
        logger.debug("similarity Shape: {}".format(similarity.shape))

        # Compute the mean along the sample points
        measure_value = torch.mean(similarity, dim = 1)
        
        # Outgoing shape: (Batch size, input or parameter size)
        return measure_value

class Stability_Channelwise_Cosine(Measure):
    """Class for Stability Channelwise calculations
    """
    def __init__(self):
        """Stability_Channelwise_Cosine init
        """
        Measure.__init__(self)

    def compute_measure(self, **kwargs):
        """Compute E[cos(theta)], where theta is the angle between (sample-input) and gradients for multiple channels. Quanitifes whether input is a local minimum.

        :Keyword Arguments:
        * *sample_gradients* (``torch.tensor``) --
        Gradients for all sample points
        * *input_tensor* (``torch.tensor``) --
        Input (Canidate minimum)
        * *sample* (``torch.tensor``) --
        Sample points
        * *input_gradients* (``torch.tensor``) --
        Gradients at the input

        :return: Batch of attributions
        :rtype: torch.tensor
        """
        # Incoming shape: (Batch size, sample points, input or parameter size)
        sample_gradients = kwargs['sample_gradients']
        input_tensor = kwargs['input_tensor']
        sample = kwargs['sample']
        input_gradients = kwargs['input_gradients']
        device = kwargs['device']

        num_sample_points = sample_gradients.shape[1]

        sample = sample.to(device = device)
        sample_gradients = sample_gradients.to(device = device)

        logger.debug("sample_gradients Shape: {}".format(sample_gradients.shape))
        logger.debug("input_tensor Shape: {}".format(input_tensor.shape))

        # Stretch input to look like sample points
        repeated_input = input_tensor.unsqueeze(1).repeat_interleave(num_sample_points, dim = 1)
        repeated_input = repeated_input.to(device = device)
        logger.debug("repeated_input Shape: {}".format(repeated_input.shape))

        # Difference between input and sample point 
        offset = repeated_input - sample
        logger.debug("offset Shape: {}".format(offset.shape))

        #Loop over channels to preserve channel information
        num_channels = sample_gradients.shape[2]
        pair_measures = []
        
        # Compute cosine similarity for all channels except the ith to retain per-pixel attributions
        # WARNING: Need to confirm eps does not affect results in meaningful way
        cos = torch.nn.CosineSimilarity(dim = 2, eps = 1e-12)
        for i in range(num_channels):

            # Get all but the ith channels from the offset
            first_half_offset = offset[:,:,:i,:]
            logger.debug("first_half_offset Shape: {}".format(first_half_offset.shape))
            second_half_offset = offset[:,:,i+1:,:]
            logger.debug("second_half_offset Shape: {}".format(second_half_offset.shape))
            offset_sliced = torch.cat([first_half_offset, second_half_offset], dim = 2)

            # Get all but the ith channels from the sample gradients
            first_half_sample_grads = sample_gradients[:,:,:i,:]
            logger.debug("first_half_sample_grads Shape: {}".format(first_half_sample_grads.shape))
            second_half_sample_grads = sample_gradients[:,:,i+1:,:]
            logger.debug("second_half_sample_grads Shape: {}".format(second_half_sample_grads.shape))
            sample_grads_sliced = torch.cat([first_half_sample_grads, second_half_sample_grads], dim = 2)
            
            # Compute cosine for the offset and grads
            similarity = cos(offset_sliced, sample_grads_sliced)
            logger.debug("similarity channel {} Shape: {}".format(i, similarity.shape))

            # Compute the mean along the sample points
            pair_measure_value = torch.mean(similarity, dim = 1)
            logger.debug("pair_measure_value Shape: {}".format(pair_measure_value.shape))
            pair_measures.append(pair_measure_value)

        # Recombine all channels
        measure_value = torch.stack(pair_measures, dim = 1)
        
        # Outgoing shape: (Batch size, input or parameter size)
        return measure_value
    
class Stability_Channelwise(Measure):
    #TODO make sure this new version actually works
    """Class for Stability Channelwise calculations
    """
    def __init__(self):
        """Stability_Channelwise init
        """
        Measure.__init__(self)

    def compute_measure(self, **kwargs):
        """Compute E[x], where x=1 if (sample-input) and sample gradients point in the same direction and x=-1 if they point in opposite directions. Quanitifes whether input is a local minimum.

        :Keyword Arguments:
        * *sample_gradients* (``torch.tensor``) --
        Gradients for all sample points
        * *input_tensor* (``torch.tensor``) --
        Input (Canidate minimum)
        * *sample* (``torch.tensor``) --
        Sample points
        * *input_gradients* (``torch.tensor``) --
        Gradients at the input

        :return: Batch of attributions
        :rtype: torch.tensor
        """
        # Incoming shape: (Batch size, sample points, input or parameter size)
        sample_gradients = kwargs['sample_gradients']
        input_tensor = kwargs['input_tensor']
        sample = kwargs['sample']
        input_gradients = kwargs['input_gradients']
        device = kwargs['device']

        num_sample_points = sample_gradients.shape[1]

        sample = sample.to(device = device)
        sample_gradients = sample_gradients.to(device = device)
        logger.debug("sample_gradients Shape: {}".format(sample_gradients.shape))
        #logger.debug(f'sample_gradients: {sample_gradients}')

        logger.debug("input_tensor Shape: {}".format(input_tensor.shape))

        # Stretch input to look like sample points
        repeated_input = input_tensor.unsqueeze(1).repeat_interleave(num_sample_points, dim = 1)
        repeated_input = repeated_input.to(device = device)
        logger.debug("repeated_input Shape: {}".format(repeated_input.shape))

        # Difference between input and sample point 
        offset = repeated_input - sample
        logger.debug("offset Shape: {}".format(offset.shape))
        #logger.debug(f'offset: {offset}')


        # Compute magnitudes of each vector
        offset_magnitude = torch.abs(offset).to(device = device)
        sample_gradients_magnitude = torch.abs(sample_gradients).to(device = device)


        #Determine whether offset and sample_gradients point in the same direction
        sign_agreement = torch.div(torch.mul(offset, sample_gradients), torch.mul(offset_magnitude, sample_gradients_magnitude))
        logger.debug(f'sign_agreement Shape: {sign_agreement.shape}')
        #replace nan with 0 (happens if gradients are zero)
        sign_agreement = torch.where(torch.isnan(sign_agreement), torch.zeros_like(sign_agreement), sign_agreement)
        #logger.debug(f'sign_agreement: {sign_agreement}')


        # Compute the mean along the sample points
        avg_sign_agreement = torch.mean(sign_agreement, dim = 1)
        logger.debug("avg_sign_agreement Shape: {}".format(avg_sign_agreement.shape))
        #logger.debug("avg_sign_agreement: {}".format(avg_sign_agreement))

        measure_value = avg_sign_agreement

        # Outgoing shape: (Batch size, input or parameter size)
        return measure_value

class Consistency(Measure):
    """Class for Consistency calculations
    """
    def __init__(self):
        """Consistency init
        """
        Measure.__init__(self)

    def compute_measure(self, **kwargs):
        """Compute E[cos(theta)], where theta is the angle between sample gradients and input gradients for a single channel. Quantifies whether sample points gradients agree with input gradients.

        :Keyword Arguments:
        * *sample_gradients* (``torch.tensor``) --
        Gradients for all sample points
        * *input_tensor* (``torch.tensor``) --
        Input
        * *sample* (``torch.tensor``) --
        Sample points
        * *input_gradients* (``torch.tensor``) --
        Gradients at the input

        :return: Batch of attributions
        :rtype: torch.tensor
        """
        # Incoming shape: (Batch size, sample points, input or parameter size)
        sample_gradients = kwargs['sample_gradients']
        input_tensor = kwargs['input_tensor']
        sample = kwargs['sample']
        input_gradients = kwargs['input_gradients']
        device = kwargs['device']

        num_sample_points = sample_gradients.shape[1]

        sample = sample.to(device = device)
        sample_gradients = sample_gradients.to(device = device)
        input_gradients = input_gradients.to(device = device)

        logger.debug("input_gradients Shape: {}".format(input_gradients.shape))
        logger.debug("sample_gradients Shape: {}".format(sample_gradients.shape))

        # Stretch input grads  to look like sample points
        repeated_input_grads = input_gradients.repeat_interleave(num_sample_points, dim = 1)
        repeated_input_grads = repeated_input_grads.to(device = device)
        logger.debug("repeated_input_grads Shape: {}".format(repeated_input_grads.shape))

        # Compute cosine similarity along color channel dimension to retain per-pixel attribution
        # WARNING: Need to confirm eps does not affect results in meaningful way
        cos = torch.nn.CosineSimilarity(dim = 2, eps = 1e-12)
        similarity = cos(repeated_input_grads, sample_gradients)
        logger.debug("similarity Shape: {}".format(similarity.shape))

        # Compute the mean along the sample points
        measure_value = torch.mean(similarity, dim = 1)
        
        # Outgoing shape: (Batch size, input or parameter size)
        return measure_value

class Consistency_Channelwise_Cosine(Measure):
    """Class for Consistency Channelwise Cosine calculations
    """
    def __init__(self):
        """Consistency Channelwise Cosine init
        """
        Measure.__init__(self)

    def compute_measure(self, **kwargs):
        """Compute E[cos(theta)], where theta is the angle between sample gradients and input gradients for a multiple channels. Quanitifes whether sample points gradients agree with input gradients.

        :Keyword Arguments:
        * *sample_gradients* (``torch.tensor``) --
          Gradients for all sample points
        * *input_tensor* (``torch.tensor``) --
          Input
        * *sample* (``torch.tensor``) --
          Sample points
        * *input_gradients* (``torch.tensor``) --
          Gradients at the input

        :return: Batch of attributions
        :rtype: torch.tensor
        """
        # Incoming shape: (Batch size, sample points, input or parameter size)
        sample_gradients = kwargs['sample_gradients']
        input_tensor = kwargs['input_tensor']
        sample = kwargs['sample']
        input_gradients = kwargs['input_gradients']
        device = kwargs['device']

        num_sample_points = sample_gradients.shape[1]

        
        sample = sample.to(device = device)
        sample_gradients = sample_gradients.to(device = device)
        input_gradients = input_gradients.to(device = device)

        logger.debug("input_gradients Shape: {}".format(input_gradients.shape))
        logger.debug("sample_gradients Shape: {}".format(sample_gradients.shape))

        # Stretch input grads  to look like sample points
        repeated_input_grads = input_gradients.repeat_interleave(num_sample_points, dim = 1)
        repeated_input_grads = repeated_input_grads.to(device = device)
        logger.debug("repeated_input_grads Shape: {}".format(repeated_input_grads.shape))

        # Loop over consecutive pairs of channels to preserve channel information
        num_channels = sample_gradients.shape[2]
        pair_measures = []
        # Compute cosine similarity for pairs of channels to retain per-pixel attributions
        # WARNING: Need to confirm eps does not affect results in meaningful way
        cos = torch.nn.CosineSimilarity(dim = 1, eps = 1e-12)
        for i in range(num_channels):
            # Get all but the ith channels from the input grads
            first_half_repeated_input_grads = repeated_input_grads[:,:,:i,:]
            second_half_repeated_input_grads = repeated_input_grads[:,:,i+1:,:]
            repeated_input_grads_sliced = torch.cat([first_half_repeated_input_grads, second_half_repeated_input_grads], dim = 2)

            # Get all but the ith channels from the sample gradients
            first_half_sample_grads = sample_gradients[:,:,:i,:]
            second_half_sample_grads = sample_gradients[:,:,i+1:,:]
            sample_grads_sliced = torch.cat([first_half_sample_grads, second_half_sample_grads], dim = 2)

            # Compute cosine for the offset and grads
            similarity = cos(repeated_input_grads_sliced, sample_grads_sliced)
            logger.debug("similarity channel {} Shape: {}".format(i, similarity.shape))

            # Compute the mean along the sample points
            pair_measure_value = torch.mean(similarity, dim = 1)
            pair_measures.append(pair_measure_value)

        # Recombine all channels
        measure_value = torch.stack(pair_measures, dim = 1)
        return measure_value
    

class Consistency_Channelwise(Measure):
    """Class for Consistency Channelwise calculations
    """
    def __init__(self):
        """Consistency Channelwise init
        """
        Measure.__init__(self)

    def compute_measure(self, **kwargs):
        """Compute E[x], where x=1 if sample gradients and input gradients point in the same direction and x=-1 if they point in opposite directions. Quanitifes whether sample points gradients agree with input gradients.

        :Keyword Arguments:
        * *sample_gradients* (``torch.tensor``) --
          Gradients for all sample points
        * *input_tensor* (``torch.tensor``) --
          Input
        * *sample* (``torch.tensor``) --
          Sample points
        * *input_gradients* (``torch.tensor``) --
          Gradients at the input

        :return: Batch of attributions
        :rtype: torch.tensor
        """
        # Incoming shape: (Batch size, sample points, input or parameter size)
        sample_gradients = kwargs['sample_gradients']
        input_tensor = kwargs['input_tensor']
        sample = kwargs['sample']
        input_gradients = kwargs['input_gradients']
        device = kwargs['device']

        num_sample_points = sample_gradients.shape[1]

        
        sample = sample.to(device = device)
        sample_gradients = sample_gradients.to(device = device)
        input_gradients = input_gradients.to(device = device)

        logger.debug("input_gradients Shape: {}".format(input_gradients.shape))
        logger.debug("sample_gradients Shape: {}".format(sample_gradients.shape))

        # Stretch input grads  to look like sample points
        repeated_input_grads = input_gradients.repeat_interleave(num_sample_points, dim = 1)
        repeated_input_grads = repeated_input_grads.to(device = device)
        logger.debug("repeated_input_grads Shape: {}".format(repeated_input_grads.shape))

        # Compute magnitudes of each vector
        repeated_input_grads_magnitude = torch.abs(repeated_input_grads).to(device = device)
        sample_gradients_magnitude = torch.abs(sample_gradients).to(device = device)

        #Determine whether input_gradients and sample_gradients point in the same direction
        sign_agreement = torch.div(torch.mul(repeated_input_grads, sample_gradients), torch.mul(repeated_input_grads_magnitude, sample_gradients_magnitude))
        #replace nan with 0 (happens if gradients are zero)
        sign_agreement = torch.where(torch.isnan(sign_agreement), torch.zeros_like(sign_agreement), sign_agreement)

        # Compute the mean along the sample points
        avg_sign_agreement = torch.mean(sign_agreement, dim = 1)
        logger.debug("avg_sign_agreement Shape: {}".format(avg_sign_agreement.shape))

        measure_value = avg_sign_agreement

        return measure_value