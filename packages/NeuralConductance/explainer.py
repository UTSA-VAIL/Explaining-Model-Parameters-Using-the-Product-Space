import torch
import copy
from torch.autograd import grad, backward
from tqdm import tqdm

from ..utilities.logging_utilities import *
from ..utilities.general_utilities import get_module_by_name

import warnings
import copy

# Setup a logger
logger = setup_logger(__name__)

# TODO: Rework class to be batched. Done already??? Maybe??
# TODO: Check with Ethan how we want this done
class GeneralizedIGExplainer(object):
    """Class to generate attributions.
    """
    def __init__(self, measure, input_sampler, parameter_sampler = None, attribution_type = "input"):
        """GeneralizedIGExplainer init

        :param measure: Measure used to compute attribution
        :type measure: NeuralConductance.Measure
        :param input_sampler: Sampler to generate model input sample points
        :type input_sampler: NeuralConductance.Sampler, optional
        :param parameter_sampler: Sampler to generate model parameter sample points, defaults to None
        :type parameter_sampler: NeuralConductance.Sampler, optional
        :param attribution_type: Whether to compute "input", "internal_feature", or "parameter" attributions, defaults to "input"
        :type attribution_type: str
        :raises: ValueError
        """
        logger.debug('Explainer Class')
        self.measure = measure
        self.attribution_type = attribution_type

        self.input_sampler = input_sampler
        if parameter_sampler != None:
            self.parameter_sampler = parameter_sampler
            if self.attribution_type == "input":
                unused_sampler_message = "Parameter Sampler is unused unless computing parameter attributions (attribution_type = \"parameter\")"
                warnings.warn(unused_sampler_message, UserWarning)
        else:
            if self.attribution_type == "parameter":
                raise ValueError("Must specify a parameter sampler if computing parameter attributions")


    def generate_name(self):
        """Function to generate name of the explainer used

        :raises: ValueError
        :return: Explainer name used for experiment tracking.
        :rtype: str
        """
        input_sampler_name = self.input_sampler.generate_name()
        measure_name = self.measure.__class__.__name__

        if (self.attribution_type == "parameter"):
            if self.parameter_sampler == None:
                raise ValueError("Missing parameter sampler required for parameter attribution")
            parameter_sampler_name = self.parameter_sampler.generate_name()
            explainer_string = f'{self.attribution_type}_{input_sampler_name}_{parameter_sampler_name}_{measure_name}'
        else:
            explainer_string = f'{self.attribution_type}_{input_sampler_name}_{measure_name}'

        return explainer_string


    # ISSUE: Model attribution needs to restore state but it breaks training gradient calculations.
    def get_grads(self, model_dict, input_sample_points, parameter_sample_points = None, attribution_type = "input"):
        """Function to compute the gradients for sample points.

        :param model_dict: Dictionary containing all model relevant objects and hyper-parameters
        :type model_dict: dict
        :param input_sample_points: Sample points in the input space
        :type input_sample_points: torch.Tensor
        :param parameter_sample_points: Sample points in the parameter space, defaults to None
        :type parameter_sample_points: torch.Tensor, optional
        :param attribution_type: Whether to compute "input", "internal_feature", or "parameter" attributions, defaults to "input"
        :type attribution_type: str
        :raises: NotImplementedError, ValueError
        :return: Sample point gradients
        :rtype: torch.Tensor
        """
        # Create a copy of the model state dictonary
        # model_state_copy = model_dict['model'].state_dict()

        # NOTE: If computing attributions for model parameters/filters, input samples must be labeled
        if attribution_type == "input":
            grads = self.get_input_grads(model_dict, input_sample_points)

        elif attribution_type == "parameter":
            if parameter_sample_points == None:
                raise ValueError("Missing parameter sample required for computing parameter grads")
            grads = self.get_parameter_grads(model_dict, input_sample_points, parameter_sample_points)
        
        elif attribution_type == "internal_feature":
            raise NotImplementedError("Internal feature attribution not yet implemented")
            # TODO write this function
            grads = self.get_internal_feature_grads(model_dict, input_sample_points, parameter_sample_points)

        # ISSUE: Model attribution needs to restore state but it breaks training gradient calculations.
        # model_dict['model'].load_state_dict(model_state_copy)

        return grads


    def get_input_grads(self, model_dict, input_sample_points):
        """Computes gradients for sample points in the input space.

        :param model_dict: Dictionary containing all model relevant objects and hyper-parameters
        :type model_dict: dict
        :param input_sample_points: Sample points in the input space
        :type input_sample_points: torch.Tensor
        :return: Sample point gradients
        :rtype: torch.Tensor
        """

        # Computing grads for samples in input space
        # Incoming shape: (Batch size, num_sample_points, input_size)
        input_sample_points = input_sample_points.to(model_dict['device'])
        
        # Allow gradient calculation
        input_sample_points.requires_grad = True

        # Change to (Batch size * num_sample_points, input_size) to send it through the model
        model_inputs = input_sample_points.view((-1,) + input_sample_points.shape[2:])
        logger.debug("Model Inputs Shape: {}".format(model_inputs.shape))
        logger.debug("Input Sample Points Shape: {}".format(input_sample_points.shape))

        # Send the samples through the model
        sample_output = model_dict['model'](model_inputs).to(device = model_dict['device'])
        
        # NOTE: torch.autograd.grad does not work due to multi-gpu restriction as of version 2
        # Populate gradients in the .grad parameter
        backward(
            tensors = sample_output,
            inputs = model_inputs,
            create_graph = False,
            grad_tensors = torch.ones_like(sample_output),
        )
        model_grads = model_inputs.grad.detach()

        # Reshape back to original size and return the gradients
        model_grads_reshape = model_grads.view(input_sample_points.size())
        input_sample_grads_full = model_grads_reshape

        return input_sample_grads_full


    def get_parameter_grads(self, model_dict, input_sample_points, parameter_sample_points):
        """Computes gradients for sample points in the input space.

        :param model_dict: Dictionary containing all model relevant objects and hyper-parameters
        :type model_dict: dict
        :param input_sample_points: Sample points in the input space
        :type input_sample_points: torch.Tensor
        :param parameter_sample_points: name of parameter and sample points in the parameter space
        :type parameter_sample_points: tuple of (str, torch.Tensor)
        :return: Sample point gradients
        :rtype: torch.Tensor
        """


        # Computing grads for samples in parameter space
        # Extract information out of parameter sample point tuples
        parameter_name, sample_parameter_tensor = parameter_sample_points
        #parameter_layer, parameter_id = parameter_ref

        # Seperate sample points from their labels
        images, labels = input_sample_points

        # Device management
        images = images.to(model_dict['device'])
        labels = labels.to(device = model_dict['device'])

        logger.debug('images Shape: {}'.format(images.shape))
        logger.debug('labels Shape: {}'.format(labels.shape))

        # Create a tensor to hold the gradients            
        parameter_sample_grads_full = torch.zeros(sample_parameter_tensor.size())

        logger.debug('parameter_sample_grads_full Shape {}'.format(parameter_sample_grads_full.shape))

        # Loop over the parameter points
        for i, parameter_sample_point in enumerate(sample_parameter_tensor):

            parameter_sample_point = parameter_sample_point.to(model_dict["device"])

            logger.debug('parameter_sample_point Shape {}'.format(parameter_sample_point.shape))
            
            # Get module (layer number and weight or bias)
            parameter_object = get_module_by_name(model_dict['model'], parameter_name)
            logger.trace(f'parameter_object: {parameter_object}')
            logger.debug(f'parameter_object.data shape: {parameter_object.data.shape}')

            # Change specific parameter within module
            if i < 5:
                logger.trace('Parameter Before Swap: {}'.format(parameter_object))
            with torch.no_grad():
                parameter_object.data = parameter_sample_point
                #layer_object[parameter_id] = parameter_sample_point
            
            if i < 5:
                logger.trace('Parameter After Swap: {}'.format(parameter_object))

            # Send the class specific data through the model
            sample_outputs = model_dict['model'](images).to(device = model_dict['device'])
            
            # Zero out gradients
            model_dict['optimizer'].zero_grad()

            # Compute the loss
            class_loss = model_dict['loss_fn'](sample_outputs, labels)

            # Calculate the loss gradients
            class_loss.backward()

            logger.trace(f'parameter_object: {parameter_object}')
            # Extract the parameter gradients
            parameter_gradients = parameter_object.grad.clone()

            if i < 5:
                logger.trace('Parameter Gradients for Sample {}: {}'.format(i, parameter_gradients))

            # Reshape back to original size and return the gradients
            parameter_sample_grads_full[i] = parameter_gradients


        return parameter_sample_grads_full


    def generate_attributions(self, model_dict, tensor_to_explain):
        # NOTE: Model attribution is only done if input_sample_points and parameter_name are provided
        """Function to generate attributions by generating sample points and computing measures.

        :param model_dict: Dictionary containing all model relevant objects and hyper-parameters
        :type model_dict: dict
        :param tensor_to_explain: Tensor to generate attributions for. Can be either inputs or parameters.
        :type tensor_to_explain: torch.Tensor
        :param input_sample_points: Samples points in input space. Used only for model attribution, defaults to None
        :type input_sample_points: torch.Tensor, optional
        :param parameter_name: Reference to locate parameter in model. Used only for model attribution, defaults to None
        :type parameter_name: torch.Tensor, optional
        :raises: NotImplementedError, ValueError
        :return: Attributions
        :rtype: torch.Tensor or None
        """

        if self.attribution_type == "input":
            #generate input attribution
            if tensor_to_explain == None:
                raise ValueError("Please specify an input batch to generate attributions for")
            attributions = self.generate_input_attributions(model_dict, tensor_to_explain)

        elif self.attribution_type == "parameter":
            if tensor_to_explain != None:
                warnings.warn("tensor_to_explain not used if computing parameter attributions", UserWarning)
            #generate parameter attribution
            self.generate_parameter_attributions(model_dict)
            attributions = None

        elif self.attribution_type == "internal_feature":
            raise NotImplementedError("Internal feature attribution not yet implemented")
            attributions = self.generate_internal_feature_attributions(model_dict, tensor_to_explain)

        return attributions


    def generate_input_attributions(self, model_dict, input_to_explain):

        """ Generate attributions for batch of input tensors

        :param input_to_explain: Batch of input tensors to generate attributions for
        :type input_to_explain: torch.Tensor
        :return: Tensor of attributions, almost always the same shape as the input_to_explain
        :rtype: torch.Tensor
        """
        
        # Step 1: Get sample points
        # tensor_to_explain = tensor_to_explain.clone().detach()
        logger.debug("input_to_explain Shape: {}".format(input_to_explain.shape))
        input_sample_points = self.input_sampler.generate_sample_points(input_to_explain, model_dict['device'])
        logger.debug("input_sample_points Shape: {}".format(input_sample_points.shape))

        # Input Attribution
        input_sample_grads = self.get_grads(
            model_dict = model_dict, 
            input_sample_points = input_sample_points,
            parameter_sample_points = None,
            attribution_type = "input"
        )
        logger.debug("input_sample_grads Shape: {}".format(input_sample_grads.shape))
        
        # Step 2: Compute the gradients for every sample point using specified model
        input_to_explain_grads = self.get_grads(
            model_dict = model_dict, 
            input_sample_points = input_to_explain.unsqueeze(1),
            parameter_sample_points = None,
            attribution_type = "input"
        )
        logger.debug("input_to_explain_grads Shape: {}".format(input_to_explain_grads.shape))

        # Step 3: Compute some function f of gradients for EVERY sample point
        #       : Compute the mean of the f over the sample points *if desired
        #       : Also pass input, sample points, and input gradients for use in advanced measures
        input_attributions = self.measure.compute_measure(
            input_tensor = input_to_explain, 
            input_gradients = input_to_explain_grads, 
            sample = input_sample_points, 
            sample_gradients = input_sample_grads, 
            device = model_dict['device']
        )
        logger.debug("input_attributions Shape: {}".format(input_attributions.shape))

        return input_attributions
    

    def generate_parameter_attributions(self, model_dict):

        """ Generate attributions for each model parameter

        :return: no return value, each parameter's attributions are attached to the model as parameter.attribution
        :rtype: None
        """

        input_sample_points = self.input_sampler.generate_sample_points()
        logger.debug("input_sample_points Shape: {}".format(input_sample_points[0].shape))
        logger.debug("input_sample_points Labels: {}".format(input_sample_points[1]))


        # loop over all model parameters
        for param_name, param in model_dict['model'].named_parameters():

            # Save original model state
            # ISSUE this is definitely not the best way to do this, need to do in-memory without writing to disk
            torch.save(model_dict['model'].state_dict(), 'tmp.pth')


            # TODO check attribution mask to see whether we want to compute attributions for this parameter
            # if not param.attributions_mask[attribution_name]:
            #    continue

            # # extract parameter tensor from model
            parameter_tensor_to_explain = param.detach().clone()
            logger.debug("parameter_tensor_to_explain Shape: {}".format(parameter_tensor_to_explain.shape))
            
            #generate samples in the parameter space
            logger.debug(f'parameter_sampler {self.parameter_sampler}')
            #NOTE BallSampler like most samplers expects to received batched inputs
            parameter_sample_points = self.parameter_sampler.generate_sample_points(parameter_tensor_to_explain.unsqueeze(0), model_dict['device']).squeeze(0)
            logger.debug("parameter_sample_points Shape: {}".format(parameter_sample_points.shape))

            #compute gradients for parameter space samples
            parameter_sample_grads = self.get_grads(
                model_dict = model_dict, 
                input_sample_points = input_sample_points,
                parameter_sample_points = (param_name, parameter_sample_points),
                attribution_type = "parameter"
            )
            logger.debug("parameter_sample_grads Shape: {}".format(parameter_sample_grads.shape))

            #NOTE must unsqueeze parameter_tensor_to_explain because get_grads expects a batch of sample points
            parameter_tensor_to_explain_grads = self.get_grads(
                model_dict = model_dict, 
                input_sample_points = input_sample_points,
                parameter_sample_points = (param_name, parameter_tensor_to_explain.unsqueeze(0)),
                attribution_type = "parameter"
            )
            logger.debug("parameter_tensor_to_explain_grads Shape: {}".format(parameter_tensor_to_explain_grads.shape))

            logger.trace("Computing measure...")
            logger.trace(f"parameter_tensor_to_explain shape: {parameter_tensor_to_explain.shape}")
            logger.trace(f"parameter_tensor_to_explain_grads shape: {parameter_tensor_to_explain_grads.shape}")
            logger.trace(f"parameter_sample_points shape: {parameter_sample_points.shape}")
            logger.trace(f"parameter_sample_grads shape: {parameter_sample_grads.shape}")

            # NOTE measures expected batched inputs, but so we have to give each model parameter a fake batch dimension
            parameter_attributions = self.measure.compute_measure(
                input_tensor = parameter_tensor_to_explain.unsqueeze(0), 
                input_gradients = parameter_tensor_to_explain_grads.unsqueeze(0), 
                sample = parameter_sample_points.unsqueeze(0), 
                sample_gradients = parameter_sample_grads.unsqueeze(0), 
                device = model_dict['device']
            )
            logger.debug("parameter_attributions Shape: {}".format(parameter_attributions.shape))

            # Restore original parameter state
            # ISSUE this is definitely not the best way to do this, need to do in-memory without writing to disk
            model_dict['model'].load_state_dict(torch.load('tmp.pth'))

            # populate attribution for the desired parameter
            # NOTE remove the fake batch dimension we added earlier
            param.attributions[self.parameter_attribute_name] = parameter_attributions.squeeze(0)

            #check that attributions are the correct shape
            if param.attributions[self.parameter_attribute_name].shape != param.data.shape:
                attributions_shape = param.attributions[self.parameter_attribute_name].shape
                data_shape = param.data.shape
                warnings.warn(f"Attribution shape usually should match parameter shape, but got shapes {attributions_shape}, and {data_shape}", UserWarning)


    
    def parameter_attributions_as_dict(self, model_dict):
        """Function to extracts param.attribution values and returns them as a dictionary

        :param model_dict: Dictionary containing all model relevant objects and hyper-parameters
        :type model_dict: dict
        :return: Dictionary containing all named model parameters
        :rtype: dict
        """

        model_attribution_dict = {}

        # loop over all model parameters
        for name, param in model_dict['model'].named_parameters():

            #logger.debug(f'measure name: {self.measure.__class__.__name__}')
            #model_attribution_dict[name] = param.attributions[self.measure.__class__.__name__].cpu().numpy().tolist()
            model_attribution_dict[name] = param.attributions[self.parameter_attribute_name].cpu().numpy().tolist()

        # logger.trace(model_attribution_dict)

        return model_attribution_dict


    #WARNING this probably doesn't work with the refactored version of generate_explanation, needs rewriting
    def quantus_explain(self, model, inputs, targets, **kwargs):
        """Function to generate explanation for Quantus evaluation

        :param model: Model to generate explanation of
        :type model: torch.nn.Module
        :param inputs: Array of inputs to be evaluated
        :type inputs: numpy.array
        :param targets: Array of labels in regard to the inputs
        :type targets: numpy.array

        :Keyword Arguments:
        * *model_dict* (``dict``) --
        Dictionary containing all model relevant objects and hyper-parameters

        :return: Model explanations
        :rtype: numpy.array
        """

        # Ensure the model is set
        kwargs['model_dict']['model'] = model

        # Change into a tensor
        inputs = torch.from_numpy(inputs).to(device = kwargs['model_dict']['device'])

        # Generate our explanation
        explanation = self.generate_attributions(kwargs['model_dict'], inputs)

        # Return it in numpy format
        return explanation.cpu().detach().numpy()