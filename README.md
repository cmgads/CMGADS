# COMU

Code and data of the paper **Multi-grained Contextual Code Representation Learning for Commit Message Generation**.

## Intro

COMU is a pre-training model for automated commit message generation, which extract multi-grained information from the changed code at the line and AST levels to generate commit messages automatically. In this repository, we provide our code and the data we use.

## Environment
+ Python == 3.8.0
+ Pytorch == 1.8.1
+ Numpy == 1.19.5
+ jpype1 == 1.4.0

## Dataset

The folder `dataset` contains all the data, including the folder `dataset_saved`, the folder `fine_tune` and the folder `pre_train`. The folder `dataset_saved` was already preprocessed and can be directly used to train or evaluate the model, saved as pickle binary files. The folder `fine_tune` and the folder `pre_train` contains all raw data.
The dataset loading code is located in the `sources/data/dataset.py`  and `sources/data/data_utils.py` files.
The folder `sources/data/diff_utils` contains the scrips to preprocess data.

## Output
The folder `outputs/pre_train` contains the pre-trained models.
The folder `outputs/models` contains the commit messages generated by COMU.

## Runs
Run `main.py` to start pre-train, fine-tune or test. 
All arguments are located in `args.py`, specific whatever you need.
Some example scripts are as following.
```shell
# pre-training   
python main.py \
--do-pre-train \
--pre-train-tasks tp,cg \
--train-batch-size 16 \
--eval-batch-size 16 \
--cuda-visible-devices 0 \
--fp16 \
--model-name pre_train \
--model-path all_task_16 

# train and test 
python main.py \
--do-fine-tune \
--do-train \
--do-test \
--train-batch-size 16 \
--eval-batch-size 16 \
--cuda-visible-devices 0 
--model-path all_task_16 \
```
