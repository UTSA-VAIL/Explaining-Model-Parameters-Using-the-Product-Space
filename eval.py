import argparse
import logging
import os
import json

from packages.utilities.logging_utilities import *
# Setup logging for other files to use
if os.path.exists('./logs/eval.log'):
    os.remove('./logs/eval.log')
addLoggingLevel('TRACE', logging.DEBUG - 5)
global_config_logger(log_file = './logs/eval.log', log_level = logging.INFO)

from config import Config
from packages.utilities.eval_utilities import *
from packages.utilities.general_utilities import *

if __name__ == '__main__':
    # Setup the main logger
    logger = setup_logger(name = __name__)

    # Argument Parser init
    parser = argparse.ArgumentParser(description = 'Evaluate a model')

    parser.add_argument('--eval_config_file_path', required = True, type = str, help = 'Get path to the eval config file path')
    parser.add_argument('--explainer_config_path', required = False, type = str, help = 'Get path to the explanation config file path')

    # Parse the arguments
    args = parser.parse_args()

    # Setup the evaluation config
    eval_config = Config(
        args.eval_config_file_path,
        default_settings = './configs/evaluation_default.json', 
        schema = './configs/evaluation_schema.json',
        mode = 'test', 

    )

    # Setup the training config 
    train_config = Config(
        eval_config.config_dic['training_config_path'],
        default_settings = './configs/training_default.json', 
        schema = './configs/training_schema.json',
        mode = 'test', 
    )
    
    # Override progress bar
    train_config.config_dic['progress_bar'] = eval_config.config_dic['progress_bar']

    # Setup the testing dictionary
    testing_dict = setup_testing_dict(train_config)


    # Set up the explanation config if provided
    if args.explainer_config_path:
        eval_config.config_dic['explainer_config_path'] = args.explainer_config_path


    # Check if any explainer config was specified (either in the config file or as a command line argument)
    if 'explainer_config_path' in eval_config.config_dic.keys():
        explainer_config = Config(
            eval_config.config_dic['explainer_config_path'], 
            default_settings = './configs/explainer_default.json',
            schema = './configs/explainer_schema.json',
            mode = 'test', 
        )
    else:
        explainer_config = None
        attribution_names = []
        logger.debug("Pruning randomly")
        prune_model_by_attributions(testing_dict['model'], 
                                    attribution_names, 
                                    eval_config.config_dic['pruning']['proportion'], 
                                    abs_attribution = eval_config.config_dic['pruning']['absolute_attributions'],
                                    device = testing_dict["device"])


    # Generate model explanation
    if explainer_config is not None:

        if explainer_config.config_dic['attributions']['attribution_type'] == "input":

            # Generate input attributions (if desired)
            if eval_config.config_dic['all_checkpoints']:
                generate_input_attributions_all_checkpoints(explainer_config, testing_dict)

            else:
                input_attributions = load_inputs_and_generate_attributions(explainer_config, testing_dict)

                if input_attributions is not None:
                    # WARNING this function may currently not work ? maybe it does tho ?
                    save_explanation(eval_config, explainer_config, input_attributions)

        elif explainer_config.config_dic['attributions']['attribution_type'] == "parameter":
        # Generate parameter attributions (if desired)

            # Attach optimizer to model for use in computing parameter gradients
            testing_dict['optimizer'] = train_config.prepare_optimizer(testing_dict['model'])

            if eval_config.config_dic['all_checkpoints']:
                raise NotImplementedError("Saving parameter attributions is not yet implemented")
                #TODO write this function
                #generate_parameter_attributions_all_checkpoints(c, ec, testing_dict)

            else:

                parameter_explainer = explainer_config.prepare_explainer(testing_dict['model'])
                parameter_explainer.generate_attributions(
                    model_dict = testing_dict, 
                    tensor_to_explain = None
                )

            #print(parameter_explainer.parameter_attributions_as_dict(testing_dict).keys())
            #print(parameter_explainer.parameter_attributions_as_dict(testing_dict))



            #TODO add the code to save parameter attributions to a file if desired
            if eval_config.config_dic['pruning']:
                if eval_config.config_dic['pruning']['method'] == 'attribution':
                    attribution_names = [parameter_explainer.parameter_attribute_name]
                elif eval_config.config_dic['pruning']['method'] == 'random':
                    attribution_names = []
                logger.info(f"Pruning using attribution names: {attribution_names}")
                prune_model_by_attributions(testing_dict['model'], 
                                            attribution_names, 
                                            eval_config.config_dic['pruning']['proportion'], 
                                            abs_attribution = eval_config.config_dic['pruning']['absolute_attributions'],
                                            device = testing_dict["device"])
                

        
    # Evaluate model on test set
    if eval_config.config_dic['evaluate_test_set']:

        # Run the evaluation
        testing_dict['testing_loss'] = evaluate_model(testing_dict)
    
        # Get metric calculation
        testing_dict['metric_df'] = calculate_testing_metrics(testing_dict)

        # Gather info from training dataframe
        training_report(testing_dict)

        # Print out the results
        pretty_metric = json.dumps(testing_dict['metric_df'], indent = 3)
        print(train_config.config_dic['metric_file_name'])
        print(pretty_metric)

        #add metric entry for Pruning if pruning was done
        if eval_config.config_dic['pruning']:
            # Add explainer path to eval save
            if 'explainer_config_path' in eval_config.config_dic.keys():
                explainer_file_name = eval_config.config_dic['explainer_config_path'].split("/")[-1]
                explainer_file_name = explainer_file_name.replace(".json","")
                metric_csv_filename = os.path.join(
                    eval_config.config_dic['log_dir'],
                    f"{explainer_file_name}_prune_{eval_config.config_dic['pruning']['proportion']:.4f}_{eval_config.config_dic['metric_file_name']}")
            else:
                metric_csv_filename = os.path.join(
                    eval_config.config_dic['log_dir'],
                    f"prune_{eval_config.config_dic['pruning']['proportion']:.4f}_{eval_config.config_dic['metric_file_name']}")
        else:
            metric_csv_filename = os.path.join(eval_config.config_dic['log_dir'],eval_config.config_dic['metric_file_name'],eval_config.config_dic['pruning']['proportion'])

        # Write metric results to file
        pd.DataFrame(testing_dict['metric_df'], index = [0]).to_csv(metric_csv_filename, index = False)

