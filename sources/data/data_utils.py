import os 
import re
import logging
from tqdm import tqdm
import pandas as pd

from .diff_utils.get_diff import get_code_ast_diff

logger = logging.getLogger(__name__)

def parse_tsv_file(file):
    code_diffs, ast_diffs, docs, code_befores, code_afters, code_tokens = get_code_ast_diff(file)
    return code_diffs, ast_diffs, docs, code_befores, code_afters, code_tokens   

def iter_all_files(base): 
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield os.path.join(root, f)

def iter_per_train_dataset_files(dataset_dir,stage):
    """
    Get files for pre-training, all files with extension ``tsv`` will be included.

    """
    return [file for file in iter_all_files(base=dataset_dir) if file.endswith(stage+'.tsv')]

def load_pre_train_dataset(file):
    """
    Load tsv dataset from given file

    """

    code_diffs, ast_diffs, docs, code_befores, code_afters, code_tokens  = parse_tsv_file(file)
    return code_diffs, ast_diffs, docs, code_befores, code_afters, code_tokens 

def load_dataset_from_dir(dataset_dir,stage):
    """
    Load all files in the given dir, only for pre-training

    """

    all_code_diffs = []
    all_ast_diffs = []
    all_docs = []
    all_code_tokens = []
    all_code_befores = []
    all_code_afters = []

    if stage is not None:
        dataset_files = iter_per_train_dataset_files(dataset_dir,stage)
    else:
        dataset_files = iter_per_train_dataset_files(dataset_dir,"")
    if len(dataset_files) > 0:
        n_sample = 0
        for dataset_file_path in dataset_files:
            code_diffs, ast_diffs, docs, code_befores, code_afters, code_tokens = load_pre_train_dataset(file=dataset_file_path) 
            all_code_diffs += code_diffs
            all_ast_diffs += ast_diffs
            all_docs += docs
            all_code_tokens += code_tokens
            all_code_befores += code_befores
            all_code_afters += code_afters
            n_line = len(code_diffs)
            n_sample += n_line
            logger.info(f'    File: {dataset_file_path}, {n_line} samples')
        logger.info(f' dataset size: {n_sample}')
    assert len(all_code_diffs) == len(all_ast_diffs) == len(all_docs) == len(all_code_tokens)
    return all_code_diffs, all_ast_diffs, all_docs, all_code_tokens, all_code_befores, all_code_afters

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
            source_tokens,
            source_ids,
            source_mask,
            target_tokens,
            target_ids,
            source_ast_tokens
    ):
        self.source_tokens = source_tokens
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.target_tokens = target_tokens
        self.target_ids = target_ids
        self.source_ast_tokens = source_ast_tokens

def convert_examples_to_features(examples, tokenizer, args, size
                                , stage=None
                                ):
    """convert examples to token ids"""

    features = []
    for i in range(size):
        pattern = {'[KEEP]':'KEEP','[ADD]':'ADD','[DEL]':'DEL'}
        new_code_diffs = [pattern[x] if x in pattern else x for x in examples.code_diffs[i].split()]
        new_code_diffs = " ".join(new_code_diffs)

        source_code_tokens = tokenizer.tokenize(new_code_diffs)
        ast_diffs = " ".join(examples.ast_diffs[i])
        source_ast_tokens = tokenizer.tokenize(ast_diffs)
        source_tokens = source_code_tokens[:args.code_length-4]
        source_tokens = [tokenizer.cls_token, "<encoder-decoder>",tokenizer.sep_token,"<mask0>"]+source_tokens+[tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_ast_tokens = source_ast_tokens[:args.ast_length-2]
        source_ast_tokens = source_ast_tokens+[tokenizer.sep_token]
        source_tokens += [x for x in source_ast_tokens]
        source_ids += [tokenizer.convert_tokens_to_ids(x) for x in source_ast_tokens]

        #all
        source_mask = [1] * (len(source_tokens))
        padding_length = args.code_length + args.ast_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask+=[0]*padding_length    

        #target
        if stage == 'test':
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(examples.docs[i])[:args.max_target_length-2]
        target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length

        features.append(
            InputFeatures(
                source_tokens,
                source_ids,
                source_mask,
                #  position_idx, 
                target_tokens,
                target_ids,
                source_ast_tokens
            )
        )

    return  features


