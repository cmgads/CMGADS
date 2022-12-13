import torch
import torch.utils.data
from transformers import AdamW, get_linear_schedule_with_warmup, T5Config, T5ForConditionalGeneration, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import torch.nn as nn

import logging
import os
from typing import Union, Tuple
from tqdm import tqdm, trange
import numpy as np

import enums
from utils import bleu
# from utils import optimizer
from data.dataset import init_dataset, TextDataset
from data.data_utils import InputFeatures, convert_examples_to_features
from utils.model import Seq2Seq, build_or_load_gen_model
from utils.general import count_params, human_format, layer_wise_parameters

logger = logging.getLogger(__name__)

def pre_train(args):
    tasks = args.pre_train_tasks

    if tasks is None:
        logger.warning('Was specified for pre-training, but got pre-training tasks to None, '
                       'will default to {}'.format(','.join(enums.PRE_TRAIN_TASKS)))
        tasks = enums.PRE_TRAIN_TASKS
    else:
        supported_tasks = []
        for task in tasks.split(','):
            task = task.strip().lower()
            if task in enums.PRE_TRAIN_TASKS:
                supported_tasks.append(task)
            else:
                logger.warning(f'Pre-training task {task} is not supported and will be ignored.')
        tasks = supported_tasks

    #assert not trained_model or isinstance(trained_model, str).......
    logger.info(f"tasks_2:{tasks}")
    logger.info('*' * 100)
    logger.info('Initializing pre-training environments')

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Building model')
    model, tokenizer= build_or_load_gen_model(args)

    # Setup CUDA, GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    model.to(args.device)

    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    logger.info('Loading and parsing datasets')
    dataset = init_dataset(args=args, mode=enums.TRAINING_MODE_PRE_TRAIN,stage="train")
    train_features = convert_examples_to_features(dataset, tokenizer, args,size=len(dataset) , stage='train')
    logger.info(f'The size of pre_training set: {len(dataset)}')
    train_data = TextDataset(train_features,args,tokenizer=tokenizer)
    logger.info('Datasets loaded and parsed successfully')

    for task in tasks:
        logger.info('-' * 100)
        logger.info(f'Pre-training task: {task.upper()}')
        train_data.set_task(task)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps,num_workers=4)

        # Prepare optimizer and schedule (linear warmup and decay)  torch.optim.
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=int(len(train_dataloader)*args.num_train_epochs*0.1),
                                                    num_training_steps=len(train_dataloader)*args.num_train_epochs)   

        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Num epoch = %d", args.num_train_epochs)  
        
        dev_dataset={}
        best_eval_ppl, best_bleu = 0, 0
        global_step = 0
        for cur_epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss= 0, 0, 0

            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                if task == 'dp':
                    target_ids, source_ids = batch
                    loss,_,_  = model(source_ids=source_ids,task=task,target_ids=target_ids)
                elif task == 'mass':
                    source_ids, target_ids = batch
                    loss = model(source_ids=source_ids,task=task,source_mask=target_ids)
                elif task == 'cap':
                    source_ids, target_ids = batch
                    loss,_,_  = model(source_ids=source_ids,task=task,target_ids=target_ids)
                elif task == 'cg':
                    source_ids, target_ids = batch
                    loss,_,_ = model(source_ids=source_ids,task=task,target_ids=target_ids)

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                bar.set_description("task {} epoch {} Train loss {}".format(task, cur_epoch, train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        output_dir = args.pre_train_output_root
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, 'cg_32', "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

    return model, tokenizer

