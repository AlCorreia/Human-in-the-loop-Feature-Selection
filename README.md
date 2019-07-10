# HILFS
A tensorflow implementation of a human-in-the-loop feature selection (HILFS) architecture introduced in [this AAAI 2019 paper](https://hal.inria.fr/hal-01934916/file/main.pdf).

The code available here reproduces the image classification experiments presented on that paper.

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

## Citation

If you find HILFS useful please cite us in your work:

    @inproceedings{Correia2019,
      author = {Correia, Alvaro H. C. and Lecue, Freddy},
      booktitle = {Thirty-Third AAAI Conference on Artificial Intelligence},
      title = {Human-in-the-Loop Feature Selection},
      year = {2019}
    }
