import torch
import torch.utils.data
from torch.utils.data.dataset import Dataset

import os
import random
import logging
import pickle

import enums
from .data_utils import load_dataset_from_dir

logger = logging.getLogger(__name__)

class CodeDataset(Dataset):

    def __init__(self, args, mode, task=None, language=None, stage=None, dataset_size=None):
        
        super(CodeDataset, self).__init__()
        self.args = args
        self.task = task
        self.mode = mode
        self.stage = stage
        self.dataset_size = dataset_size

        #dataset dir: fine_tune or pre_train
        self.dataset_dir = os.path.join(args.dataset_root, self.mode)

        if self.mode == 'pre_train':
        #load pre-train dataset
            self.code_diffs, self.ast_diffs, self.docs, self.code_tokens,self.code_befores,self.code_afters = load_dataset_from_dir(dataset_dir=self.dataset_dir,stage='train')
            self.size = len(self.code_diffs)

        # load fine-tuning dataset
        else:
            self.dataset_dir = os.path.join(args.dataset_dir, self.dataset_size)
            self.code_diffs, self.ast_diffs, self.docs, self.code_tokens,self.code_befores,self.code_afters = load_dataset_from_dir(dataset_dir=self.dataset_dir,stage=self.stage)
            self.size = len(self.code_diffs)

    
    def __getitem__(self, index):
        return self.code_diffs[index], self.ast_diffs[index], self.docs[index], self.code_tokens[index]

    def __len__(self):
        return self.size

    def set_task(self, task):
        self.task = task

    def save(self):
        """Save to binary pickle file"""
        if self.mode == 'pre_train':
            path_root = os.path.join(args.dataset_save_dir, self.mode)
        else:
            path_root = os.path.join(args.dataset_save_dir, self.mode, self.dataset_size)
        path = os.path.join(path_root, f'{self.stage}.pk')  
        with open(path, mode='wb') as f:
            pickle.dump(self, f)
        logger.info(f'Dataset saved to {path}')

def init_dataset(args, mode, task=None, language=None, stage=None, load_if_saved=True) -> CodeDataset:
    if load_if_saved:
        if mode == 'fine_tune':
            path_root = os.path.join(args.dataset_save_dir, mode, args.dataset_size)
            path = os.path.join(path_root, f'{stage}.pk')
        else:
            path = os.path.join(args.dataset_save_dir, mode, f'{stage}.pk')
        if os.path.exists(path) and os.path.isfile(path):
            logger.info(f'Trying to load saved binary pickle file from: {path}')
            with open(path, mode='rb') as f:
                obj = pickle.load(f)
            assert isinstance(obj, CodeDataset)
            obj.args = args
            logger.info(f'Dataset instance loaded from: {path}')
            return obj
    
    dataset = CodeDataset(args=args,
                          mode=mode,
                          task=task,
                          language=language,
                          stage=stage,
                          dataset_size=args.dataset_size)
    dataset.save()
    return dataset

class TextDataset(Dataset):
    def __init__(self, examples, args, tokenizer, task=None):
        self.examples = examples
        self.args = args
        self.tokenizer = tokenizer
        self.task = task

    def __len__(self):
        return len(self.examples)

    def set_task(self, task):
        self.task = task

    def __getitem__(self, item):
        input_ids = []
        input_tokens = []
                             
        if self.task == enums.TASK_TAG_PREDICTION:  
            index = self.examples[item].source_tokens.index(self.tokenizer.sep_token,4)
            #Code_Diff Tag Prediction
            for idx, i in enumerate(self.examples[item].source_tokens[:index+1]):
                if i in ["KEEP", "ADD","DEL"]:
                    input_tokens.append(self.tokenizer.mask_token)
                else:
                    input_tokens.append(i)
            #AST_Diff Tag Prediction
            for idx, i in enumerate(self.examples[item].source_ast_tokens):
                i = i.replace('\u0120','')
                if i in ['Insert','Move','Delete','Update']:
                    input_tokens.append(self.tokenizer.mask_token)
                else:
                    input_tokens.append(i)
            if len(input_tokens) < len(self.examples[item].source_ids):
                padding_length = len(self.examples[item].source_ids) - len(input_tokens)
                input_tokens += [self.tokenizer.pad_token] * padding_length
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            return (torch.tensor(input_ids),                         
                    torch.tensor(self.examples[item].source_ids)          
                    )
        elif self.task == enums.TASK_COMMNET_GENERATION:
            return (torch.tensor(self.examples[item].source_ids),     
                    torch.tensor(self.examples[item].target_ids)      
                    )
        