

## Runs

Run `main.py` to start pre-train, fine-tune or test. 
All arguments are located in `args.py`, specific whatever you need.

```shell
# pre-training   dp,mass,cap,cg
python main.py --do-pre-train --pre-train-tasks dp,mass,cap,cg --train-batch-size 16 --cuda-visible-devices 0 --fp16 --model-name pre_train --model_name_or_path ./codet5-base

#1. only pre-train
python main.py --do-pre-train --pre-train-tasks dp,cg --train-batch-size 16 --cuda-visible-devices 0 --fp16 --model-name pre_train 
#dataset
其中的pre_train数据集用的是raw_dataset1数据集

# summarization on pre-trained model
##2.1 train and test
python main.py --do-fine-tune --do-train --do-test --cuda-visible-devices 0 --trained-model ../outputs/pre_train/
##2.2 only test
python main.py --do-fine-tune --do-test --cuda-visible-devices 0 --trained-model ../outputs/pre_train/

python main.py --do-fine-tune --do-test --cuda-visible-devices 0 

python main.py --do-fine-tune --do-train --cuda-visible-devices 0 

source activate sptcode

#scratch
##3.1  pre-train and train and test   ,mass,cap
python main.py --do-pre-train --do-fine-tune --do-train --do-test --pre-train-tasks dp,cg --train-batch-size 16 --cuda-visible-devices 0--fp16 --model-name pre_train

python main.py --do-pre-train --do-fine-tune  --train-batch-size 16 --cuda-visible-devices 0 --fp16 --model-name pre_train
```"# CMGADS" 
