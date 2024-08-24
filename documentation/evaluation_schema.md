# JSON Schema

## Properties

- **`training_config_path`** *(string)*: Path to training config file.
- **`explainer_config_path`** *(string)*: Path to explainer config file.
- **`metric_file_name`** *(string)*: Filename to save the eval metrics to. Default: `"eval_metrics.csv"`.
- **`log_dir`** *(string)*: Directory to save eval logs to. Default: `"./logs/eval"`.
- **`evalutate_test_set`** *(boolean)*: Determine whether to evaluate the test set. Default: `true`.
- **`all_checkpoints`** *(boolean)*: Determine if all checkpoints should be evaluated. Default: `false`.
- **`pruning`** *(object)*: Pruning dictionary.
  - **`method`** *(string)*: Determine which pruning method to use. Must be one of: `["random", "attribution"]`. Default: `"attribution"`.
  - **`proportion`** *(number)*: Determine network pruning proportion for pruning experiment. Minimum: `0`. Maximum: `1`. Default: `0.1`.
  - **`absolute_attributions`** *(boolean)*: Determine whether or not to absolute when pruning. Default: `false`.
