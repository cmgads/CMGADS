import jpype.imports
from jpype.types import *
import os
import logging
import re
import pandas as pd

import torch

from differ import fileLineDiff
from .diff.myers import myers_diff
from .diff.common import Keep, Insert, Remove

from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
tokenizer.add_special_tokens({'additional_special_tokens':["[KEEP]", "[ADD]", "[DEL]"]})

logger = logging.getLogger(__name__)

def get_code_ast_diff(file):
    code_diffs = []
    ast_diffs = []
    docs = []
    code_befores = []
    code_afters = []
    all_code_tokens = []
    with open(file, 'r' , encoding='utf-8') as f:
        j = 1
        for line in f:
            line = line.strip().split('\t')
            before = line[0]
            after = line[1]
            flag = ''
            before_code, comment = before.split('</code>', 1)
            before_code = before_code[6:]
            comment = ''.join(comment.split('</technical_language>'))[20:]
            try:
                if "<START>" in before_code:
                    str1, str2 = before_code.split("<START>")
                    if "<END>" in str2:
                        str2_1, str2_2 = str2.split("<END>")
                    else:
                        str2_1, str2_2 = str2.split("END>")
                    before_code_line = str1 + str2_1 + str2_2
                else:
                    before_code_line = before_code
            except Exception as e:
                j += 1
                continue
            star = 0
            before_split_code = []
            for i in range(len(before_code_line)): 
                if before_code_line[i] == '{' \
                    or before_code_line[i] == ';' \
                    or before_code_line[i] == '}':
                    before_split_code.append(before_code_line[star:i+1])
                    star = i + 1

            star = 0
            after_split_code = []
            for i in range(len(after)): 
                if after[i] == '{' \
                    or after[i] == ';' \
                    or after[i] == '}':
                    after_split_code.append(after[star:i+1])
                    star = i + 1

            result = fileLineDiff(before_split_code,after_split_code)

            code_diff = []
            code_tokens = []
            for elem in result:
                try:
                    if isinstance(elem, Keep ):
                        code_tokens.append('[KEEP] '+ elem[0] + '\n')
                    elif isinstance(elem, Insert):
                        code_tokens.append('[ADD] ' + elem[0] + '\n')
                    elif isinstance(elem, Remove):
                        code_tokens.append('[DEL] ' + elem[0] + '\n')
                except Exception as e:
                    flag = "error"
            if flag == "error":
                j += 1
                continue

            code_diff = ' '.join(code_tokens)
            before_path = os.path.join(os.path.realpath("."), "data/diff_utils/before.java")
            after_path = os.path.join(os.path.realpath("."), "data/diff_utils/after.java")
            with open(before_path, 'w', encoding='utf-8') as f1:
                f1.write("public class Test {" + before_code_line + "}")
            with open(after_path, 'w', encoding='utf-8') as f1:
                f1.write("public class Test {" + after + "}")

            ast_diff = get_ast_diff()
            try:
                source_code_tokens_1_1 = tokenizer.tokenize(code_diff)
                ast_diffs_tokens_1_1 = " ".join(ast_diff)
                source_ast_tokens = tokenizer.tokenize(ast_diffs_tokens_1_1)
            except Exception as e:
                j += 1
                continue

            code_diffs.append(code_diff)
            docs.append(comment)
            code_befores.append(before_code_line)
            code_afters.append(after)
            all_code_tokens.append(code_tokens)
            ast_diffs.append(ast_diff)
            j += 1

        logger.info(f"当前的数据集是： {file}")
        logger.info(f"当前处理的数据共：{j-1}")

    return code_diffs, ast_diffs, docs, code_befores, code_afters, all_code_tokens

def get_ast_diff():
    path = os.path.join(os.path.realpath("."), "data/diff_utils/lib/*")
    path1 = os.path.join(os.path.realpath("."), "data/diff_utils/before.java")
    path2 = os.path.join(os.path.realpath("."), "data/diff_utils/after.java")
    try:
        jpype.startJVM(classpath=[path])
    except Exception as e:
        pass
    MyClass = JClass('test')
    MyClass.main([])
    args = [JString("diff"),JString(path1),JString(path2)]
    diff_res = str(MyClass.parseDiff(args))

    ast_diff = diff_res.split('\n')
    actions = []
    for action in ast_diff:
        if not action.startswith('Match') and action:
            action_name = re.sub(u"\\(.*?\)", "", action.split(' at ')[0].replace(':', ''))
            simple_act = action_name.split(' ')[0] + action_name.split(' ')[1]
            actions.append(simple_act)
    return actions


