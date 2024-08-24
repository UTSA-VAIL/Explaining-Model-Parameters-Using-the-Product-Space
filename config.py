import os
import json
import sys
import warnings
import segmentation_models_pytorch as smp # type:ignore 
from jsonschema import validate
from torchvision import models
from pathlib import Path

from packages.dataprocessing.datasets import *
from packages.utilities.logging_utilities import *
from packages.utilities.general_utilities import attach_attributions


from packages.NeuralConductance.measure import *
from packages.NeuralConductance.sampler import *
from packages.NeuralConductance.explainer import *
from packages.NeuralConductance.attribution_prior import AttributionPrior


# Set the environment variable for pre-trained docker purposes
os.environ['TORCH_HOME'] = 'pytorch_models/'


# Setup a logger
logger = setup_logger(__name__)


class Config:
    """Class to handle the configuration of the experiments.
    """
    def __init__(self, config_file, default_settings, schema, mode = 'train'):
        """Config class initalization

        Args:
            config_file (string): File path to config containing desired user parameters
            default_settings (string): File path to default config for shared settings across your experiments
            schema (string): File path to schema to validate config file against
            mode (string, optional): Determine either training mode or testing mode. Defaults to 'train'.
        """
        self.mode = mode
        self.default_settings = default_settings
        self.schema = schema
        self.load_defaults(config_file)
        self.validate_schema()

        pretty_config = json.dumps(self.config_dic, indent = 3)
        logger.info(pretty_config)

        # Create the directories specified from config if they needed and are missing
        directories_to_be_created = [value for key, value in self.config_dic.items() if key.endswith('_dir')]
        for directory_path in directories_to_be_created:
            Path(directory_path).mkdir(parents = True, exist_ok = True)


    def load_defaults(self, config_file):
        """Loads default values for unspecified values in the config file

        Args:
            config_file (string): File path to config containing desired user parameters
        """
        logger.info('Reading default config file')
        f = open(self.default_settings)
        self.config_dic = json.load(f)
        f.close()

        logger.info('Reading user config file')
        f = open(config_file)
        user_settings = json.load(f)
        f.close()

        # Override user preferences in default settings
        for key, value in user_settings.items():
            self.config_dic[key] = value
    
    def validate_schema(self):
        """Ensures that the user specified valid parameters.
        """
        logger.info('Reading and validating schema')
        f = open(self.schema)
        schema_object = json.load(f)
        f.close()

        validate(instance = self.config_dic, schema = schema_object)
    
    def prepare_dataset(self, reference = {}):
        """Determine which dataset should be selected using the config file

        Args:
            reference (bool, optional): Reference dataset for explainer. Defaults to False.

        Returns:
            torch.utils.data.Dataset: Pytorch Dataset
        """
        if reference:
            logger.debug(f'reference:{reference}')
            dataset_name = reference['dataset_name']
            root_data_path = reference['data_path']
            data_augmentation = reference['data_augmentation']
            logger.info('Preparing {} dataset\n'.format(dataset_name))
        else:
            dataset_name = self.config_dic['dataset']['dataset_name']
            root_data_path = self.config_dic['dataset']['data_path']
            data_augmentation = self.config_dic['dataset']['data_augmentation']
            logger.info('Preparing {} reference dataset\n'.format(dataset_name))
            
        if self.mode == 'train':

            if dataset_name == 'imagenette':
                training_dataset = Imagenette(image_dir = os.path.join(root_data_path, 'train'), pre_shuffle = True)
                validation_dataset = ImagenetTest(image_dir = os.path.join(root_data_path, 'val'))
                self.config_dic['num_classes'] = 10
            elif dataset_name == 'imagenet100':
                training_dataset = Imagenet(image_dir = os.path.join(root_data_path, 'our_train'), pre_shuffle = True)
                validation_dataset = ImagenetTest(image_dir = os.path.join(root_data_path, 'our_val'))
                self.config_dic['num_classes'] = 100
            elif dataset_name == 'imagenet':
                training_dataset = Imagenet(image_dir = os.path.join(root_data_path, 'our_train'), pre_shuffle = True)
                validation_dataset = ImagenetTest(image_dir = os.path.join(root_data_path, 'our_val'))
                self.config_dic['num_classes'] = 1000
            elif dataset_name == 'imagewoof':
                training_dataset = Imagenet(image_dir = os.path.join(root_data_path, 'train'), pre_shuffle = True)
                validation_dataset = ImagenetTest(image_dir = os.path.join(root_data_path, 'val'))
                self.config_dic['num_classes'] = 10
            elif dataset_name == 'cifar10':
                full_dataset = CustomCIFAR10(image_dir = root_data_path, data_augmentation = data_augmentation)
                full_dataset.setup()
                training_dataset = full_dataset.train
                validation_dataset = full_dataset.val
                self.config_dic['num_classes'] = 10
            elif dataset_name == 'fashionmnist':
                full_dataset = CustomFashionMNIST(image_dir = root_data_path, data_augmentation = data_augmentation)
                full_dataset.setup()
                training_dataset = full_dataset.train
                validation_dataset = full_dataset.val
                self.config_dic['num_classes'] = 10
            elif dataset_name == 'mnist':
                full_dataset = CustomMNIST(image_dir = root_data_path, data_augmentation = data_augmentation)
                full_dataset.setup()
                training_dataset = full_dataset.train
                validation_dataset = full_dataset.val
                self.config_dic['num_classes'] = 10
            elif dataset_name == 'svhn':
                full_dataset = CustomSVHN(image_dir = root_data_path, data_augmentation = data_augmentation)
                full_dataset.setup()
                training_dataset = full_dataset.train
                validation_dataset = full_dataset.val
                self.config_dic['num_classes'] = 10
            elif dataset_name == 'cityscapes':
                training_dataset = CityscapesDataset(
                    image_dir = os.path.join(root_data_path, 'train', 'images'),
                    mask_dir = os.path.join(root_data_path, 'train', 'masks')
                )
                validation_dataset = CityscapesDataset(
                    image_dir = os.path.join(root_data_path, 'val', 'images'),
                    mask_dir = os.path.join(root_data_path, 'val', 'masks')
                )
                self.config_dic['num_classes'] = 34

            return training_dataset, validation_dataset
        else:
            if dataset_name == 'imagenette':
                testing_dataset = Imagenette(image_dir = os.path.join(root_data_path, 'test'))
                self.config_dic['num_classes'] = 10
            elif dataset_name == 'imagenet100':
                testing_dataset = Imagenet(image_dir = os.path.join(root_data_path, 'test'))
                self.config_dic['num_classes'] = 100
            elif dataset_name == 'imagenet':
                testing_dataset = ImagenetTest(image_dir = os.path.join(root_data_path, 'val'))
                self.config_dic['num_classes'] = 1000
            elif dataset_name == 'imagewoof':
                testing_dataset = Imagenet(image_dir = os.path.join(root_data_path, 'test'))
                self.config_dic['num_classes'] = 10
            elif dataset_name == 'cifar10':
                full_dataset = CustomCIFAR10(image_dir = root_data_path)
                full_dataset.setup(stage = 'test')
                testing_dataset = full_dataset.test
                self.config_dic['num_classes'] = 10
            elif dataset_name == 'fashionmnist':
                full_dataset = CustomFashionMNIST(image_dir = root_data_path)
                full_dataset.setup(stage = 'test')
                testing_dataset = full_dataset.test
                self.config_dic['num_classes'] = 10
            elif dataset_name == 'mnist':
                full_dataset = CustomMNIST(image_dir = root_data_path)
                full_dataset.setup(stage = 'test')
                testing_dataset = full_dataset.test
                self.config_dic['num_classes'] = 10
            elif dataset_name == 'svhn':
                full_dataset = CustomSVHN(image_dir = root_data_path)
                full_dataset.setup(stage = 'test')
                testing_dataset = full_dataset.test
                self.config_dic['num_classes'] = 10
            
            return testing_dataset
        
    def prepare_model(self):
        """ Determine which model should be selected using the config file

        :return: Pytorch model
        :rtype: torch.nn.Module
        """
        model_name = self.config_dic['model']['model_name']
        pretrained = self.config_dic['model']['pretrained']
        model_task = self.config_dic['model']['task']

        logger.info('Selected {} model\n'.format(model_name))
        if model_task == 'single-label classification':
            # Determine whether or not to use ImageNet pretrained weights
            if pretrained:
                weights = 'IMAGENET1K_V1'
            else:
                weights = None

            if model_name == 'resnet18':
                model = models.resnet18(weights = weights, num_classes = self.config_dic['num_classes'])
                if self.config_dic['dataset']['dataset_name'] != 'imagenet':
                    logger.debug('Swapping out first convolution to improve results.')
                    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                    model.maxpool = torch.nn.Identity()
            elif model_name == 'resnet34':
                model = models.resnet34(weights = weights, num_classes = self.config_dic['num_classes'])
            elif model_name == 'resnet50':
                model = models.resnet50(weights = weights, num_classes = self.config_dic['num_classes'])
            elif model_name == 'resnet101':
                model = models.resnet101(weights = weights, num_classes = self.config_dic['num_classes'])
            elif model_name == 'resnet152':
                model = models.resnet152(weights = weights, num_classes = self.config_dic['num_classes'])

        return model, model_task

    def prepare_optimizer(self, model):
        """Determine which optimizer should be selected using the config file. Used for training purposes.

        Args:
            model (torch.nn.Module): Pytorch model to train on.

        Returns:
            torch.optim: Optimizer used to train model. 
        """
         # Select an optimizer
        if self.config_dic['optimizer']['method'] == 'adam':
            # optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr = self.config_dic['optimizer']['learning_rate'],
                weight_decay = self.config_dic['optimizer']['weight_decay'],
            )
        elif self.config_dic['optimizer']['method'] == 'sgd':
            # optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 1e-4, nesterov = True)
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr = self.config_dic['optimizer']['learning_rate'],
                weight_decay = self.config_dic['optimizer']['weight_decay'],
                momentum = self.config_dic['optimizer']['momentum'],
            )

        return optimizer

    def prepare_scheduler(self, optimizer):
        """Determine which scheduler should be selected using the config file for training purposes. Paired with optimizer.

        Args:
            optimizer (torch.optim): Optimizer used to train model. 

        Returns:
            torch.optim.lr_scheduler: Learning rate scheduler. 
        """
        if self.config_dic['scheduler']['method'] == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', 
                patience = self.config_dic['scheduler']['patience'], 
                verbose = True
            )
        elif self.config_dic['scheduler']['method'] == 'step_lr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size = self.config_dic['scheduler']['step_size'], 
                verbose = True
            )
        elif self.config_dic['scheduler']['method'] == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max = self.config_dic['epochs'], 
                eta_min = 0,
                verbose = True
            )
        else:
            scheduler = None

        return scheduler


    def prepare_explainer(self, model=None):
        """Function to determine which type of explainer to create (one ec is in charge of only one explainer)

        :return: Explanier used for explaining network behavior
        :rtype: NeuralConductance.GeneralizedIGExplainer or None
        """
        if 'attributions' in self.config_dic:
            
            # Setup for input attributions
            if self.config_dic['attributions']['attribution_type'] == "input":
                explainer = self.prepare_input_explainer()
            elif self.config_dic['attributions']['attribution_type'] == "parameter":
                explainer = self.prepare_parameter_explainer(model)
        else:
            explainer = None

        return explainer

    def prepare_input_explainer(self):
        """Function to create a input explainer 

        :return: Input explanier used for explaining network behavior
        :rtype: NeuralConductance.GeneralizedIGExplainer
        """

        # Select a sampler
        if self.config_dic['attributions']['input_attribution']['sampler']['method'] == 'bad_sampler':
            input_sampler = BallSampler(
                eps = self.config_dic['attributions']['input_attribution']['sampler']['eps'], 
                num_sample_points = self.config_dic['attributions']['input_attribution']['sampler']['num_sample_points'],
            )
        elif self.config_dic['attributions']['input_attribution']['sampler']['method'] == 'expected_gradients':
            input_sampler = ExpectedGradientsSampler(
                reference_dataset = self.prepare_dataset(reference = self.config_dic['attributions']['input_attribution']['reference_dataset']), 
                num_samples_per_line = self.config_dic['attributions']['input_attribution']['sampler']['num_samples_per_line'], 
                num_reference_points = self.config_dic['attributions']['input_attribution']['sampler']['num_reference_points'],
            )
        elif self.config_dic['attributions']['input_attribution']['sampler']['method'] == 'ball_eg_sampler':
            input_sampler = BallExpectedGradientsSampler(
                reference_dataset = self.prepare_dataset(reference = self.config_dic['attributions']['input_attribution']['reference_dataset']), 
                num_samples_per_line = self.config_dic['attributions']['input_attribution']['sampler']['num_samples_per_line'], 
                num_reference_points = self.config_dic['attributions']['input_attribution']['sampler']['num_reference_points'],
                eps = self.config_dic['attributions']['input_attribution']['sampler']['eps'], 
            )
        else:
            logger.info('Selected default BallSampler sampler.')
            input_sampler = BallSampler()

        # Select a measure
        if self.config_dic['attributions']['input_attribution']['measure'] == 'integrated_gradients':
            measure = IntegratedGradients()
        elif self.config_dic['attributions']['input_attribution']['measure'] == 'gradient_variance':
            measure = GradientVariance()
        elif self.config_dic['attributions']['input_attribution']['measure'] == 'stability':
            warning_message = "Stability is deprecated is deprecated and may be removed in the future, please use Stability_Channelwise instead"
            warnings.warn(warning_message, DeprecationWarning)
            measure = Stability()
        elif self.config_dic['attributions']['input_attribution']['measure'] == 'stability_channelwise':
            measure = Stability_Channelwise()
        elif self.config_dic['attributions']['input_attribution']['measure'] == 'consistency':
            warning_message = "Consistency is deprecated is deprecated and may be removed in the future, please use Consistency_Channelwise instead"
            warnings.warn(warning_message, DeprecationWarning)
            measure = Consistency()
        elif self.config_dic['attributions']['input_attribution']['measure'] == 'consistency_channelwise':
            measure = Consistency_Channelwise()
        else:
            logger.info('Selected default measure.')
            measure = IntegratedGradients()

        # Create the explainer
        explainer = GeneralizedIGExplainer(
            input_sampler = input_sampler,
            parameter_sampler = None,
            measure = measure,
            attribution_type = "input"
        )

        return explainer

    #TODO update for new explanation config schema etc.
    def prepare_parameter_explainer(self, model=None):
        """Function to create a model parameter explainer 

        :return: Parameter explanier used for explaining network behavior
        :rtype: NeuralConductance.GeneralizedIGExplainer
        """

        # Select an input sampler
        if self.config_dic['attributions']['parameter_attribution']['input_sampler']['method'] == 'bad_sampler':
            input_sampler = BallSampler(
                eps = self.config_dic['attributions']['parameter_attribution']['input_sampler']['eps'], 
                num_sample_points = self.config_dic['attributions']['parameter_attribution']['input_sampler']['num_sample_points'],
            )
        elif self.config_dic['attributions']['parameter_attribution']['input_sampler']['method'] == 'expected_gradients':
            input_sampler = ExpectedGradientsSampler(
                reference_dataset = self.prepare_dataset(reference = self.config_dic['attributions']['parameter_attribution']['input_reference_dataset']), 
                num_samples_per_line = self.config_dic['attributions']['parameter_attribution']['input_sampler']['num_samples_per_line'], 
                num_reference_points = self.config_dic['attributions']['parameter_attribution']['input_sampler']['num_reference_points'],
            )
        elif self.config_dic['attributions']['parameter_attribution']['input_sampler']['method'] == 'ball_eg_sampler':
            input_sampler = BallExpectedGradientsSampler(
                reference_dataset = self.prepare_dataset(reference = self.config_dic['attributions']['parameter_attribution']['input_reference_dataset']), 
                num_samples_per_line = self.config_dic['attributions']['parameter_attribution']['input_sampler']['num_samples_per_line'], 
                num_reference_points = self.config_dic['attributions']['parameter_attribution']['input_sampler']['num_reference_points'],
                eps = self.config_dic['attributions']['parameter_attribution']['input_sampler']['eps'], 
            )
        elif self.config_dic['attributions']['parameter_attribution']['input_sampler']['method'] == 'class_sampler':
            input_sampler = ClassSampler(
                reference_dataset = self.prepare_dataset(reference = self.config_dic['attributions']['parameter_attribution']['input_reference_dataset']), 
                class_labels = self.config_dic['attributions']['parameter_attribution']['input_sampler']['class_id'],
                num_samples = self.config_dic['attributions']['parameter_attribution']['input_sampler']['num_sample_points']
            )
        else:
            logger.info('Selected default sampler.')
            input_sampler = BallSampler()

        # Select a parameter sampler
        if self.config_dic['attributions']['parameter_attribution']['parameter_sampler']['method'] == 'ball_sampler':
            parameter_sampler = BallSampler(
                eps = self.config_dic['attributions']['parameter_attribution']['parameter_sampler']['eps'], 
                num_sample_points = self.config_dic['attributions']['parameter_attribution']['parameter_sampler']['num_sample_points'],
            )
        elif self.config_dic['attributions']['parameter_attribution']['parameter_sampler']['method'] == 'expected_gradients':
            raise NotImplementedError("parameter reference datasets not yet implemented")
            parameter_sampler = ExpectedGradientsSampler(
                reference_dataset = self.prepare_dataset(reference = self.config_dic['attributions']['parameter_attribution']['parameter_reference']), 
                num_samples_per_line = self.config_dic['attributions']['parameter_attribution']['parameter_sampler']['num_samples_per_line'], 
                num_reference_points = self.config_dic['attributions']['parameter_attribution']['parameter_sampler']['num_reference_points'],
            )
        elif self.config_dic['attributions']['parameter_attribution']['parameter_sampler']['method'] == 'ball_eg_sampler':
            raise NotImplementedError("parameter reference datasets not yet implemented")
            parameter_sampler = BallExpectedGradientsSampler(
                reference_dataset = self.prepare_dataset(reference = self.config_dic['attributions']['parameter_attribution']['parameter_reference']), 
                num_samples_per_line = self.config_dic['attributions']['parameter_attribution']['parameter_sampler']['num_samples_per_line'], 
                num_reference_points = self.config_dic['attributions']['parameter_attribution']['parameter_sampler']['num_reference_points'],
                eps = self.config_dic['attributions']['parameter_attribution']['parameter_sampler']['eps'], 
            )
        else:
            logger.info('Selected default sampler.')
            parameter_sampler = BallSampler()

        # Select a measure
        if self.config_dic['attributions']['parameter_attribution']['measure'] == 'integrated_gradients':
            measure = IntegratedGradients()
        elif self.config_dic['attributions']['parameter_attribution']['measure'] == 'gradient_variance':
            measure = GradientVariance()
        elif self.config_dic['attributions']['parameter_attribution']['measure'] == 'stability':
            warning_message = "Stability is deprecated is deprecated and may be removed in the future, please use Stability_Channelwise instead"
            warnings.warn(warning_message, DeprecationWarning)
            measure = Stability()
        elif self.config_dic['attributions']['parameter_attribution']['measure'] == 'stability_channelwise':
            measure = Stability_Channelwise()
        elif self.config_dic['attributions']['parameter_attribution']['measure'] == 'consistency':
            warning_message = "Consistency is deprecated is deprecated and may be removed in the future, please use Consistency_Channelwise instead"
            warnings.warn(warning_message, DeprecationWarning)
            measure = Consistency()
        elif self.config_dic['attributions']['parameter_attribution']['measure'] == 'consistency_channelwise':
            measure = Consistency_Channelwise()
        else:
            logger.info('Selected default measure.')
            measure = IntegratedGradients()


        #TODO need to refactor init() for GeneralizedIGExplainer to accept both input and parameter samplers
        # Create the explainer
        explainer = GeneralizedIGExplainer(
            measure = measure,
            input_sampler = input_sampler,
            parameter_sampler = parameter_sampler,
            attribution_type = "parameter"
        )

        # attach attributions if attributions are desired 
        # TODO call attach attributions if we want to compute model attributions
        explainer.parameter_attribute_name = self.config_dic['attributions']['parameter_attribution']['parameter_attribute_name']
        attach_attributions(model, [explainer.parameter_attribute_name])

        return explainer
  
    def prepare_prior(self):
        """Function to prepare prior loss training

        :return: Attribution prior loss
        :rtype: NeuralConductance.AttributionPrior or None
        """
        if 'attribution_prior' in self.config_dic['attribution']:
            prior = AttributionPrior(
                regularization_method = self.config_dic['attribution']['attribution_prior']['regularization_method'],
                weight = self.config_dic['attribution']['attribution_prior']['weight'],
                absolute_measure = self.config_dic['attribution']['attribution_prior']['absolute_measure'],
            )
        else:
            prior = None

        return prior