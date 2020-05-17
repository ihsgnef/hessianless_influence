import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict
from copy import deepcopy
from tqdm import tqdm

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


def create_data_config(
        config_name: str,
        train_examples: list = None,
        version_number: int = None,
):
    args = json.load(open('configs/SST-2/base.json'))

    if version_number is not None:
        config_name += '/' + str(version_number)
    data_dir = 'data/SST-2/{}'.format(config_name)
    output_dir = 'output/SST-2/{}'.format(config_name)
    config_dir = 'configs/SST-2/{}.json'.format(config_name)
    train_data_dir = os.path.join(data_dir, 'train.tsv')

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(config_dir).parent.mkdir(parents=True, exist_ok=True)

    print(config_dir)

    args.update({
        'train_data_dir': data_dir,
        'output_dir': output_dir,
    })
    with open(config_dir, 'w') as f:
        json.dump(args, f, indent=4)

    with open(train_data_dir, 'w') as f:
        f.write('sentence\tlabel\n')
        for example in train_examples:
            f.write('{}\t{}\n'.format(example.text_a, example.label))


def random_dev_set():
    sst_processor = Sst2Processor()
    dev_examples = sst_processor.get_dev_examples('data/SST-2/base')

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


def remove_by_random():
    sst_processor = Sst2Processor()
    all_examples = sst_processor.get_train_examples('glue_data/SST-2')
    negative_examples = [x for x in all_examples if x.label == '0']
    positive_examples = [x for x in all_examples if x.label == '1']

    n_examples_removed = int(0.1 * len(all_examples))  # remove 10% of all training examples

    for i in range(10):
        random.shuffle(all_examples)
        train_examples = all_examples[n_examples_removed:]
        create_data_config(
            config_name='random_10_percent_removed_both',
            train_examples=train_examples,
            version_number=i,
        )

    for i in range(10):
        random.shuffle(positive_examples)
        train_examples = positive_examples[n_examples_removed:] + negative_examples
        create_data_config(
            config_name='random_10_percent_removed_positive',
            train_examples=train_examples,
            version_number=i,
        )

    for i in range(10):
        random.shuffle(negative_examples)
        train_examples = negative_examples[n_examples_removed:] + positive_examples
        create_data_config(
            config_name='random_10_percent_removed_negative',
            train_examples=train_examples,
            version_number=i,
        )


def setup(
    args_dir: str,
    train_data_dir: str = None,
    eval_data_dir: str = None,
):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               ExperimentArguments))

    model_args, data_args, training_args, experiment_args = parser.parse_json_file(
        json_file=os.path.abspath(args_dir))
    if train_data_dir is not None:
        experiment_args.train_data_dir = train_data_dir
    if eval_data_dir is not None:
        experiment_args.eval_data_dir = eval_data_dir

    output_mode = glue_output_modes[data_args.task_name]

    tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(training_args.output_dir)
    model = model.to(training_args.device)

    train_data_args = deepcopy(data_args)
    train_data_args.data_dir = experiment_args.train_data_dir

    eval_data_args = deepcopy(data_args)
    eval_data_args.data_dir = experiment_args.eval_data_dir

    train_dataset = GlueDataset(train_data_args, tokenizer=tokenizer,
                                local_rank=training_args.local_rank)
    eval_dataset = GlueDataset(eval_data_args, tokenizer=tokenizer,
                               local_rank=training_args.local_rank, evaluate=True)

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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    return model, trainer, train_dataset, eval_dataset


def remove_by_confidence():
    model, trainer, train_dataset, eval_dataset = setup(args_dir='configs/SST-2/base.json')

    output = trainer.predict(train_dataset)
    scores = nn.functional.softmax(torch.from_numpy(output.predictions), dim=-1).numpy()
    scores = np.choose(output.label_ids, scores.T)
    indices = np.arange(len(scores))
    positive_indices = indices[output.label_ids == 1]
    negative_indices = indices[output.label_ids == 0]
    positive_scores = scores[output.label_ids == 1]
    negative_scores = scores[output.label_ids == 0]

    most_confident_positive_indices = positive_indices[np.argsort(-positive_scores)]
    most_confident_negative_indices = negative_indices[np.argsort(-negative_scores)]

    train_examples = Sst2Processor().get_train_examples('data/SST-2/base')
    n_removed = int(0.1 * len(train_examples))

    most_confident_positive_removed = (
        [train_examples[i] for i in most_confident_positive_indices[n_removed:]]
        + [train_examples[i] for i in negative_indices]
    )

    least_confident_positive_removed = (
        [train_examples[i] for i in most_confident_positive_indices[::-1][n_removed:]]
        + [train_examples[i] for i in negative_indices]
    )

    most_confident_negative_removed = (
        [train_examples[i] for i in most_confident_negative_indices[n_removed:]]
        + [train_examples[i] for i in positive_indices]
    )

    least_confident_negative_removed = (
        [train_examples[i] for i in most_confident_negative_indices[::-1][n_removed:]]
        + [train_examples[i] for i in positive_indices]
    )

    create_data_config(
        config_name='most_confident_10_percent_removed_positive',
        train_examples=most_confident_positive_removed,
    )

    create_data_config(
        config_name='most_confident_10_percent_removed_negative',
        train_examples=most_confident_negative_removed,
    )

    create_data_config(
        config_name='least_confident_10_percent_removed_positive',
        train_examples=least_confident_positive_removed,
    )

    create_data_config(
        config_name='least_confident_10_percent_removed_negative',
        train_examples=least_confident_negative_removed,
    )


def compare_scores(args_dirs: str):
    eval_data_dir='data/SST-2/random_50_dev'

    model, trainer, train_dataset, eval_dataset = setup(args_dir='configs/SST-2/base.json',
                                                        eval_data_dir=eval_data_dir)
    output = trainer.predict(eval_dataset)
    scores_original = np.choose(
        output.label_ids,
        nn.functional.softmax(torch.from_numpy(output.predictions), dim=-1).numpy().T,
    )

    for args_dir in args_dirs:
        print(args_dir)
        model, trainer, train_dataset, eval_dataset = setup(args_dir=args_dir,
                                                            eval_data_dir=eval_data_dir)
        output = trainer.predict(eval_dataset)
        scores_modified = np.choose(
            output.label_ids,
            nn.functional.softmax(torch.from_numpy(output.predictions), dim=-1).numpy().T,
        )

        scores_diff = scores_modified - scores_original
        trials = []
        for i in range(0, 500, 50):
            trials.append(np.mean(scores_diff[i:i + 50]))
        print('combined {}{:.4f}% (±{:.4f})'.format(
            '+' if np.mean(trials) > 0 else '',
            np.mean(trials) * 100,
            np.std(trials) * 100))

        trials = []
        for i in range(500, 1000, 50):
            trials.append(np.mean(scores_diff[i:i + 50]))
        print('negative {}{:.4f}% (±{:.4f})'.format(
            '+' if np.mean(trials) > 0 else '',
            np.mean(trials) * 100,
            np.std(trials) * 100))

        trials = []
        for i in range(1000, 1500, 50):
            trials.append(np.mean(scores_diff[i:i + 50]))
        print('positive {}{:.4f}% (±{:.4f})'.format(
            '+' if np.mean(trials) > 0 else '',
            np.mean(trials) * 100,
            np.std(trials) * 100))


def remove_by_similarity():
    """
    for each example in the target test set, find the training examples with the most similar final
    representation, accumulate the score over all test examples, remove the top 10%
    """
    model, trainer, train_dataset, eval_dataset = setup(args_dir='configs/SST-2/base.json',
                                                        eval_data_dir='data/SST-2/random_50_dev')
    model.eval()

    dataloaders = {
        'eval': trainer.get_eval_dataloader(eval_dataset),
        'train': trainer.get_eval_dataloader(train_dataset),
    }
    pooled_outputs = {'train': [], 'eval': []}
    for fold, dataloader in dataloaders.items():
        for inputs in tqdm(dataloader):
            for k, v in inputs.items():
                inputs[k] = v.to(model.device)
            with torch.no_grad():
                outputs = model.bert(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs['token_type_ids'],
                )
                pooled_output = outputs[1]
                pooled_outputs[fold].append(pooled_output.detach().cpu().numpy())
    pooled_outputs = {k: np.concatenate(v, axis=0) for k, v in pooled_outputs.items()}
    similarity = pooled_outputs['eval'] @ pooled_outputs['train'].T

    train_examples = Sst2Processor().get_train_examples('data/SST-2/base')
    n_removed = int(0.1 * len(train_examples))

    for i in range(3):
        most_similar_indices = np.argsort(-similarity[i * 500: i * 500 + 500].mean(axis=0))
        most_similar_removed = [train_examples[i] for i in most_similar_indices[n_removed:]]
        least_similar_removed = [train_examples[i] for i in most_similar_indices[::-1][n_removed:]]

        create_data_config(
            config_name='most_similar_10_percent_removed',
            train_examples=most_similar_removed,
            version_number=i,
        )

        create_data_config(
            config_name='least_similar_10_percent_removed',
            train_examples=least_similar_removed,
            version_number=i,
        )


"""
gradient-matching
"""


if __name__ == '__main__':
    # random_dev_set()
    # remove_by_random()
    # remove_by_confidence()
    compare_scores(
        args_dirs=[
            'configs/SST-2/most_similar_10_percent_removed/0.json',
            'configs/SST-2/most_similar_10_percent_removed/1.json',
            'configs/SST-2/most_similar_10_percent_removed/2.json',
            'configs/SST-2/least_similar_10_percent_removed/0.json',
    #         'configs/SST-2/random_10_percent_removed_both_0.json',
    #         'configs/SST-2/random_10_percent_removed_positive_0.json',
    #         'configs/SST-2/random_10_percent_removed_negative_0.json',
    #         'configs/SST-2/most_confident_10_percent_removed_positive.json',
    #         'configs/SST-2/most_confident_10_percent_removed_negative.json',
    #         'configs/SST-2/least_confident_10_percent_removed_positive.json',
    #         'configs/SST-2/least_confident_10_percent_removed_negative.json',
        ]
    )
    # remove_by_similarity()
