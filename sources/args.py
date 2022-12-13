import dataclasses
from dataclasses import dataclass, field
import os
import enums

@dataclass
class RuntimeArguments:

    do_pre_train: bool = field(
        default=False,
        metadata={'action': 'store_true',
                  'help': 'Whether to pre-train'}
    )

    do_fine_tune: bool = field(
        default=False,
        metadata={'action': 'store_true',
                  'help': 'Whether to fine_tune, task can be specific by `--task`'}
    )

    do_train: bool = field(
        default=False,
        metadata={'action': 'store_true',
                  'help': 'Whether to run training.'}
    )

    do_eval: bool = field(
        default=True,
        metadata={'action': 'store_true',
                  'help': 'Whether to run eval on the dev set.'}
    )

    do_test: bool = field(
        default=False,
        metadata={'action': 'store_true',
                  'help': 'Whether to run eval on the dev set.'}
    )
    pre_train_tasks: str = field(
        default=','.join(enums.PRE_TRAIN_TASKS),
        metadata={'help': 'Pre-training tasks in order, split by commas, '
                          'for example (default) {}'.format(','.join(enums.PRE_TRAIN_TASKS))}
    )

    task: str = field(
        default='cg',
        metadata={'help': 'Downstream task',
                  'choices': enums.ALL_DOWNSTREAM_TASKS}
    )

    cuda_visible_devices: str = field(
        default=None,
        metadata={'help': 'Visible cuda devices, string formatted, device number divided by \',\', e.g., \'0, 2\', '
                          '\'None\' will use all'}
    )

    random_seed: int = field(
        default=42,
        metadata={'help': 'Specific random seed manually for all operations, 0 to disable'}
    )

    train_batch_size: int = field(
        default=16,
        metadata={'help': 'Batch size for training on each device'}
    )
    
    eval_batch_size: int = field(
        default=16,
        metadata={'help': 'Batch size for evaluation on each device'}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={'help': "Number of updates steps to accumulate before performing a backward/update pass."}
    ) 

    num_train_epochs: int = field(
        default=3,
        metadata={'help': "Total number of training epochs to perform."}
    )


    trained_vocab: str = field(
        default='../pre_trained/vocabs/',
        metadata={'help': 'Directory of trained vocabs'}
    )

    trained_model: str = field(
        default=None,
        metadata={'help': 'Directory of trained model'}
    )

    fp16: bool = field(
        default=False,
        metadata={'action': 'store_true',
                  'help': 'Whether to use mixed precision'}
    )

@dataclass
class DatasetArguments:

    dataset_root: str = field(
        default='../dataset/',
        metadata={'help': 'Root of the dataset'}
    )

    dataset_size: str = field(
        default=None,
        metadata={'help':'Size of the dataset'}
    )

    train_subset_ratio: float = field(
        default=None,
        metadata={'help': 'Ratio of train subset'}
    )

    pre_train_subset_ratio: float = field(
        default=None,
        metadata={'help': 'Ratio of pre-train subset'}
    )

@dataclass
class SavingArguments:

    model_name: str = field(
        default='default_model',
        metadata={'help': 'Name of the model'}
    )

    dataset_save_dir: str = field(
        default=os.path.join(DatasetArguments.dataset_root, 'dataset_saved'),
        metadata={'help': 'Directory to save and load dataset pickle instance'}
    )

    vocab_save_dir: str = field(
        default=os.path.join(DatasetArguments.dataset_root, 'vocab_saved'),
        metadata={'help': 'Directory to save and load vocab pickle instance'}
    )

@dataclass
class PreprocessingArguments:

    model_type: str = field(
        default='unixcoder',
        metadata={'help': "Model type: e.g. codet5"}
    )

    model_name_or_path: str = field(
        default='unixcoder-base',
        metadata={'help': "Path to pre-trained model: e.g. codet5-base" }
    )

    config_name: str = field(
        default="",
        metadata={'help': "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: str = field(
        default="microsoft/unixcoder-base",
        metadata={'help': "Pretrained tokenizer name or path if not the same as model_name"}
    )

    code_length: int = field(
        default=256,
        metadata={'help': 'Maximum length of code sequence'}
    )

    ast_length: int = field(
        default=256,
        metadata={'help': 'Maximum length of ast sequence'}
    )

    max_target_length: int = field(
        default=64,
        metadata={'help': 'Maximum length of the nl sequence'}
    )

    beam_size: int = field(
        default=10,
        metadata={'help': "beam size for beam search"}
    )

    model_path: str = field(
        default=None,
        metadata={'help': "Path to trained model: Should contain the .bin files" }
    )


@dataclass
class ModelArguments:

    d_model: int = field(
        default=768,
        metadata={'help': 'Dimension of the model'}
    )

@dataclass
class OptimizerArguments:

    learning_rate: float = field(
        default=5e-5,
        metadata={'help': 'Learning rate'}
    )

    weight_decay: float = field(
        default=0,
        metadata={'help': 'Decay ratio for learning rate, 0 to disable'}
    )

    adam_epsilon: float = field(
        default=1e-8,
        metadata={'help': 'Epsilon for Adam optimizer.'}
    )

    warmup_steps: int = field(
        default=1000,
        metadata={'help': 'Warmup steps for optimizer, 0 to disable'}
    )


@dataclass
class TaskArguments:

    mass_mask_ratio: float = field(
        default=0.5,
        metadata={'help': 'Ratio between number of masked tokens and number of total tokens, in MASS'}
    )

def transfer_arg_name(name):
    return '--' + name.replace('_', '-')

def add_args(parser):

    for data_container in [RuntimeArguments, DatasetArguments, SavingArguments,
                           PreprocessingArguments, ModelArguments, OptimizerArguments, TaskArguments]:
        group = parser.add_argument_group(data_container.__name__)
        for data_field in dataclasses.fields(data_container):
            if 'action' in data_field.metadata:
                group.add_argument(transfer_arg_name(data_field.name),
                                   default=data_field.default,
                                   **data_field.metadata)
            else:
                group.add_argument(transfer_arg_name(data_field.name),
                                   type=data_field.type,
                                   default=data_field.default,
                                   **data_field.metadata)