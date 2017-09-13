# HILAM
A tensorflow implementation of a feature selection architecture for image classification.

## Requirements
 
- python (Verified on 3.6.0, not tested on Python 2)
- numpy
- pandas
- sklearn
- tensorflow (version 1.1.0)
- tqdm

## Training

A complete list of the arguments can be found at the main.py file.

The model can be run as follows
```
python main.py -d directory_path -m type_of_model -e number_of_epochs -nc number_of_categories 
```

## Visualizing the Results

All the results are automatically logged in the directory defined by the -d argument.

They can be checked by running tensorboard and opening the browser on the localhost:6006

```
tensorboard --logdir=directory_path
```

