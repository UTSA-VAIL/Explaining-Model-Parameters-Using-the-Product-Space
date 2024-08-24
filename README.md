# Explaining-Model-Parameters-Using-the-Product-Space
Code for ICPR 2024 paper - Explaining Model Parameters Using the Product Space

```
@InProceedings{payne_2024_modelparam,
    author    = {Payne, Ethan and Patrick, David and Fernandez, Amanda},
    title     = {Explaining Model Parameters Using the Product Space},
    booktitle = {Proceedings of the IAPR International Conference on Pattern Recognition (ICPR)},
    month     = {December},
    year      = {2024}
}
```


## Docker Enviornment
This project uses a Docker container to train our networks. A link to our Docker container wll be provided at a later date. 

We have also provided the Dockerfile if you prefer to build the image manually.
```sh
cd docker
docker build -t <youruser>/explainingmodelparameters . 
```

## Attribution Logic
Our attribution logic can be found in packages Neural Conductance.

## Config Files
We have provided a schema and example json files on how to properly construct a config file. These can be found in the configs directory.

## Training a Network
After creating a config file, use the following command to train a network:
```sh
python3 train.py --config_file_path PATH_TO_CONFIG_FILE
```

## Evaluating a Network
Once the network is trained, it can be evaluated using the same config file by the following command:
```sh
python3 eval.py --config_file_path PATH_TO_CONFIG_FILE
```

## Creating Custom Models and Dataset
If you want to use your own dataset and model, please either update the prepare_dataset or prepare_model functions in config python file.
