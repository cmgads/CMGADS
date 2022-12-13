import torch
import torch.utils.data
from transformers import AdamW, get_linear_schedule_with_warmup, T5Config, T5ForConditionalGeneration, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import torch.nn as nn

import os
import logging
from typing import Union, Tuple
from tqdm import tqdm, trange
import numpy as np
import random

import enums
from utils import bleu, smooth_bleu
from utils.general import count_params, human_format, layer_wise_parameters
from utils.model import Seq2Seq, build_or_load_gen_model
from utils.eval.meteor import Meteor
from utils.eval.rouge import Rouge
from data.dataset import init_dataset, TextDataset
from data.data_utils import InputFeatures, convert_examples_to_features

logger = logging.getLogger(__name__)

def train(args, tokenizer=None,
        trained_model: Union[Seq2Seq, str] = None):
    """
    Fine-tuning from given pre-trained model 

    """
    task = args.task.lower()
    assert task in enums.ALL_DOWNSTREAM_TASKS, f'Downstream task {task} is not supported.'

    # --------------------------------------------------
    # datasets
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Loading datasets')
    datasets = dict()
    splits = ['train', 'val','test']
    for split in splits:
        datasets[split] = init_dataset(args=args,
                                       mode=enums.TRAINING_MODE_FINE_TUNE,
                                       task=enums.TASK_COMMNET_GENERATION
                                       stage=split)
        logger.info(f'The size of {split} set: {len(datasets[split])}')
    logger.info('Datasets loaded successfully')
    
    # --------------------------------------------------
    # model
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_model is  not None or args.trained_model is not None:
        if isinstance(trained_model, Seq2Seq):
        # if trained_model is not None:
            logger.info('Model is passed through parameter')
            model = trained_model
            tokenizer = tokenizer
        else:
            model, tokenizer = build_or_load_gen_model(args)
            logger.info('Loading the model from file')
            model_prefix = 'pytorch_model.bin'
            output_dir = os.path.join(args.pre_train_output_root, model_prefix)  
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(torch.load(output_dir))                                      
    else:
        logger.info('Building the model')
        model, tokenizer = build_or_load_gen_model(args)

    # log model statistics
    logger.info('Trainable parameters: {}'.format(human_format(count_params(model))))
    logger.info('Model built successfully')

    # --------------------------------------------------
    # train
    # --------------------------------------------------
    
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    model.to(device)
    
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        logger.info('-' * 100)
        logger.info('Start training')
        #Prepare training data loader
        train_features = convert_examples_to_features(datasets["train"], tokenizer,args,size=len(datasets["train"]), stage='train')
        train_data = TextDataset(train_features,args,tokenizer=tokenizer)
        train_data.set_task(task)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps,num_workers=8)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.num_train_epochs*0.1,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(datasets["train"]))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        
        patience, best_bleu, dev_dataset = 0, 0, {}
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss, global_step = 0, 0, 0, 0

            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                loss,_,_ = model(source_ids=source_ids,task='cg',target_ids=target_ids)

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                bar.set_description("epoch {} loss {}".format(epoch,train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            
            if args.do_eval:
                #Eval model with dev dataset                  
                eval_flag=False    
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else:
                    eval_features = convert_examples_to_features(datasets["val"], tokenizer, args,size=len(datasets["val"]), stage='dev')
                    eval_data = TextDataset(eval_features,args,tokenizer=tokenizer)
                    eval_data.set_task(task)
                    dev_dataset['dev_loss']=datasets["val"],eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=8)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(datasets["val"]))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,batch_num = 0,0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)               
                    source_ids,target_ids = batch
                    with torch.no_grad():
                        _,loss,num = model(source_ids=source_ids,task='cg',target_ids=target_ids) 
                        
                    eval_loss += loss.item()
                    batch_num += num.sum().item() 

                #Pring loss of dev dataset   
                eval_loss = eval_loss / batch_num
                eval_ppl = round(np.exp(eval_loss), 5)

                model.train()
                result = {'epoch': epoch,
                          'eval_ppl': eval_ppl,
                          'global_step': global_step+1}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = datasets["val"]
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,size=len(datasets["val"]),stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long) 
                    eval_data = TensorDataset(all_source_ids)   
                    dev_dataset['dev_bleu'] = eval_examples,eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval() 
                p = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids = batch[0]                  
                    with torch.no_grad():
                        preds = model(source_ids,task='cg')
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                
                # model.train()
                predictions = []
                count = 0
                with open(args.model_root+"/dev.output",'w') as f, open(args.model_root+"/dev.gold",'w') as f1:
                    for ref,gold in zip(p,eval_examples.docs):  
                        predictions.append(str(count)+'\t'+ref)
                        f.write(str(count)+'\t'+ref.strip()+'\n')
                        f1.write(str(count)+'\t'+gold.strip()+'\n')    
                        count += 1

                (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, os.path.join(args.model_root, "dev.gold")) 
                dev_bleu=round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  "+"*"*20)    
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = args.checkpoint_root
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                    #minepretrain
                    output_model_file = os.path.join(output_dir, 'fine_tune',"pytorch_model.bin")

                    torch.save(model_to_save.state_dict(), output_model_file)
                    patience =0
                else:
                    patience +=1
                    if patience ==2:
                        break
    if args.do_test:
        checkpoint_prefix = 'pytorch_model.bin'
        # minepretrain
        output_dir = os.path.join(args.checkpoint_root, 'fine_tune', checkpoint_prefix) 

        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))   

        eval_examples = datasets['test']
        logger.info(f'The size of {split} set: {len(eval_examples)}')
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args,size=len(datasets["test"]),stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_source_ids)   

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval() 
        p=[]
        for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids = batch[0]                  
            with torch.no_grad():
                preds = model(source_ids,task='cg')
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                    p.append(text)

        model.train()
        predictions=[]
        h_dict = dict()  #idx: prediction
        r_dict = dict()  #idx: ground truth
        count = 0
        with open(args.model_root+"/test.output",'w') as f, open(args.model_root+"/test.gold",'w') as f1:
            for ref,gold in zip(p,eval_examples.docs):  
                predictions.append(str(count)+'\t'+ref)
                f.write(str(count)+'\t'+ref.strip()+'\n')
                f1.write(str(count)+'\t'+gold.strip()+'\n')  
                h_dict[count] = [ref.strip()]
                r_dict[count] = [gold.strip()]
                count += 1 

        (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, os.path.join(args.model_root, "test.gold")) 
        dev_bleu=round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
        logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
        logger.info("  "+"*"*20)    

        meteor_calculator = Meteor()
        meteor, _ = meteor_calculator.compute_score(r_dict, h_dict)
        logger.info("  %s = %s "%("meteor",str(meteor * 100)))
        logger.info("  "+"*"*20) 
        
        rouge_calculator = Rouge()
        rouge_l, ind_rouge = rouge_calculator.compute_score(r_dict, h_dict)
        logger.info("  %s = %s "%("rouge_l",str(rouge_l * 100)))
        logger.info("  "+"*"*20) 

