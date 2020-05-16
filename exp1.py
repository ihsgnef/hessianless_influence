import random
from pathlib import Path

import os
from typing import Dict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers.data.processors.glue import Sst2Processor
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    set_seed,
)

from run_glue import ModelArguments, ExperimentArguments


random.seed(42)
set_seed(42)


def random_dev_set():
    print('creating random 50 dev examples')
    sst_processor = Sst2Processor()
    dev_examples = sst_processor.get_dev_examples('glue_data/SST-2')

    datasets = {
        'all': dev_examples,
        'negative': [x for x in dev_examples if x.label == '0'],
        'positive': [x for x in dev_examples if x.label == '1'],
    }

    # write all dev sets into the same file
    path = 'data/SST-2/random_50_dev/'
    Path(path).mkdir(parents=True, exist_ok=True)
    output_file = open(path + 'dev.tsv', 'w')
    output_file.write('sentence\tlabel\n')
    n_examples = 50
    for fold, examples in datasets.items():
        for i in range(10):
            random.shuffle(examples)
            for example in examples[:n_examples]:
                output_file.write('{}\t{}\n'.format(example.text_a, example.label))
    output_file.close()


def random_remove_10():
    sst_processor = Sst2Processor()
    all_examples = sst_processor.get_train_examples('glue_data/SST-2')
    negative_examples = [x for x in all_examples if x.label == '0']
    positive_examples = [x for x in all_examples if x.label == '1']

    n_examples_removed = int(0.1 * len(all_examples))  # remove 10% of all training examples

    print('randomly remove 10% training examples')
    for i in range(10):
        path = 'data/SST-2/random_remove_10_percent_all_train/{}/'.format(i)
        Path(path).mkdir(parents=True, exist_ok=True)
        output_file = open(path + 'train.tsv', 'w')
        output_file.write('sentence\tlabel\n')
        random.shuffle(all_examples)
        for example in all_examples[n_examples_removed:]:
            output_file.write('{}\t{}\n'.format(example.text_a, example.label))
        output_file.close()

    print('randomly remove 10% training examples but only positive')
    for i in range(10):
        path = 'data/SST-2/random_remove_10_percent_positive_train/{}/'.format(i)
        Path(path).mkdir(parents=True, exist_ok=True)
        output_file = open(path + 'train.tsv', 'w')
        output_file.write('sentence\tlabel\n')
        random.shuffle(positive_examples)
        for example in positive_examples[n_examples_removed:]:
            output_file.write('{}\t{}\n'.format(example.text_a, example.label))
        for example in negative_examples:
            output_file.write('{}\t{}\n'.format(example.text_a, example.label))
        output_file.close()

    print('randomly remove 10% training examples but only negative')
    for i in range(10):
        path = 'data/SST-2/random_remove_10_percent_negative_train/{}/'.format(i)
        Path(path).mkdir(parents=True, exist_ok=True)
        output_file = open(path + 'train.tsv', 'w')
        output_file.write('sentence\tlabel\n')
        random.shuffle(negative_examples)
        for example in negative_examples[n_examples_removed:]:
            output_file.write('{}\t{}\n'.format(example.text_a, example.label))
        for example in positive_examples:
            output_file.write('{}\t{}\n'.format(example.text_a, example.label))
        output_file.close()


def remove_positive_high_confidence():
    print('remove 10% positive training examples with the highest confidence')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               ExperimentArguments))

    model_args, data_args, training_args, experiment_args = parser.parse_json_file(
        json_file=os.path.abspath('args.json'))

    output_mode = glue_output_modes[data_args.task_name]

    checkpoint_dir = 'output/SST-2/base/'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

    dev_data_args = deepcopy(data_args)
    dev_data_args.data_dir = experiment_args.dev_data_dir

    eval_dataset = (
        GlueDataset(dev_data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True)
        if training_args.do_eval
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    result = trainer.evaluate(eval_dataset=eval_dataset)
    print(result)

    output = trainer.predict(eval_dataset)
    nn.functional.softmax(torch.from_numpy(output.predictions), dim=-1)


"""
positive_low_confidence: randomly remove 10% positive training examples with the lowest confidence
"""

"""
negative_high_confidence
"""

"""
negative_low_confidence
"""

"""
representation-matching-best: for each example in the target test set, find the training examples with the most similar final representation, accumulate the score over all test examples, remove the top 10%
"""

"""
representation-matching-worst: repeat the above but remove the bottom 10%
"""

"""
gradient-matching-best
"""

"""
gradient-matching-worst
"""


if __name__ == '__main__':
    # random_dev_set()
    # random_remove_10()
    remove_positive_high_confidence()
