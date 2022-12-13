#training mode
TRAINING_MODE_PRE_TRAIN = "pre_train"
TRAINING_MODE_FINE_TUNE = "fine_tune"

#pre-training task names
TASK_TAG_PREDICTION = 'tp'
#downstream task names
TASK_COMMNET_GENERATION = 'cg'

PRE_TRAIN_TASKS = [
    TASK_TAG_PREDICTION,
    TASK_COMMNET_GENERATION
]

ALL_DOWNSTREAM_TASKS = [
    TASK_COMMNET_GENERATION
]

#programming language
LANG_JAVA = 'java'