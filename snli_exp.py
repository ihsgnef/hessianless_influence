import os
import json
import glob
import random
import numpy as np
from pathlib import Path
from typing import Dict
from copy import deepcopy
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from glue_utils import (
    GlueDataset,
    glue_processors,
    glue_compute_metrics,
    glue_output_modes,
)
from glue_utils import GlueDataTrainingArguments as DataTrainingArguments
from run_glue import ModelArguments, ExperimentArguments


random.seed(42)
set_seed(42)


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


def create_data_config(
        task_name: str,
        config_name: str,
        version_number: int = None,
        train_examples: list = None,
):
    args = json.load(open('configs/{}/base.json'.format(task_name)))

    if version_number is not None:
        config_name += '/' + str(version_number)
    data_dir = 'data/{}/{}'.format(task_name, config_name)
    output_dir = 'output/{}/{}'.format(task_name, config_name)
    config_dir = 'configs/{}/{}.json'.format(task_name, config_name)
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

    if task_name.startswith('SST-2'):
        with open(train_data_dir, 'w') as f:
            f.write('sentence\tlabel\n')
            for example in train_examples:
                f.write('{}\t{}\n'.format(example.text_a, example.label))

    if task_name == 'SNLI':
        with open(train_data_dir, 'w') as f:
            f.write(('Index\t' + 'NULL\t' * 6
                     + 'sentence1\tsentence2\t' + 'NULL\t' * 5
                     + 'gold_label\n'))
            for i, example in enumerate(train_examples):
                f.write(('{}\t' + 'NULL\t' * 6 + '{}\t{}\t' + 'NULL\t' * 5 + '{}\n').format(
                    i, example.text_a, example.text_b, example.label))


def random_dev_set(task_name: str = None, n_examples: int = 50):
    processor = glue_processors[task_name.lower()]()
    dev_examples = processor.get_dev_examples('data/{}/base'.format(task_name))

    datasets = {
        'combined': [],
        'entailment': [],
        'contradiction': [],
        'neutral': [],
    }
    for example in dev_examples:
        datasets[example.label].append(example)
        datasets['combined'].append(example)

    random_examples = []
    for fold, examples in datasets.items():
        random.shuffle(examples)
        random_examples += examples[:n_examples]

    path = 'data/{}/dev_{}'.format(task_name, n_examples * len(datasets))
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(path, 'dev.tsv'), 'w') as f:
        f.write(('Index\t' + 'NULL\t' * 6
                 + 'sentence1\tsentence2\t' + 'NULL\t' * 5
                 + 'gold_label\n'))
        for i, example in enumerate(random_examples):
            f.write(('{}\t' + 'NULL\t' * 6 + '{}\t{}\t' + 'NULL\t' * 5 + '{}\n').format(
                i, example.text_a, example.text_b, example.label))


def remove_by_random(task_name: str, percentage: int = 10, n_trials: int = 3):
    processor = glue_processors[task_name.lower()]()
    all_examples = processor.get_train_examples('data/{}/base'.format(task_name))
    n_examples_removed = int(percentage / 100 * len(all_examples))

    datasets = {
        'combined': [],
        'entailment': [],
        'contradiction': [],
        'neutral': [],
    }
    for example in all_examples:
        datasets[example.label].append(example)
        datasets['combined'].append(example)

    for i in range(n_trials):
        random.shuffle(datasets['combined'])
        create_data_config(
            task_name=task_name,
            config_name='random_{}_percent_removed_combined'.format(percentage),
            version_number=i,
            train_examples=datasets['combined'][n_examples_removed:],
        )

    for i in range(n_trials):
        random.shuffle(datasets['entailment'])
        create_data_config(
            task_name,
            config_name='random_{}_percent_removed_entailment'.format(percentage),
            version_number=i,
            train_examples=(
                datasets['entailment'][n_examples_removed:]
                + datasets['contradiction']
                + datasets['neutral']
            ),
        )

    for i in range(n_trials):
        random.shuffle(datasets['contradiction'])
        create_data_config(
            task_name,
            config_name='random_{}_percent_removed_contradiction'.format(percentage),
            version_number=i,
            train_examples=(
                datasets['entailment']
                + datasets['contradiction'][n_examples_removed:]
                + datasets['neutral']
            ),
        )

    for i in range(n_trials):
        random.shuffle(datasets['neutral'])
        create_data_config(
            task_name,
            config_name='random_{}_percent_removed_neutral'.format(percentage),
            version_number=i,
            train_examples=(
                datasets['entailment']
                + datasets['contradiction']
                + datasets['neutral'][n_examples_removed:]
            ),
        )


def remove_by_confidence(task_name: str, percentage: int = 10):
    args_dir = 'configs/{}/base.json'.format(task_name)
    model, trainer, train_dataset, eval_dataset = setup(args_dir=args_dir)

    output = trainer.predict(train_dataset)
    scores = F.softmax(torch.from_numpy(output.predictions), dim=-1).numpy()
    scores = np.choose(output.label_ids, scores.T)
    indices = np.arange(len(scores))
    positive_indices = indices[output.label_ids == 1]
    negative_indices = indices[output.label_ids == 0]
    positive_scores = scores[output.label_ids == 1]
    negative_scores = scores[output.label_ids == 0]

    most_confident_combined_indices = np.argsort(-scores)
    most_confident_positive_indices = positive_indices[np.argsort(-positive_scores)]
    most_confident_negative_indices = negative_indices[np.argsort(-negative_scores)]

    processor = glue_processors[task_name.lower()]()
    train_examples = processor.get_train_examples('data/{}/base'.format(task_name))
    n_removed = int(percentage / 100 * len(train_examples))

    datasets = {
        'most_confident_{}_percent_removed_combined'.format(percentage): (
            [train_examples[i] for i in most_confident_combined_indices[n_removed:]]
        ),
        'most_confident_{}_percent_removed_positive'.format(percentage): (
            [train_examples[i] for i in most_confident_positive_indices[n_removed:]]
            + [train_examples[i] for i in negative_indices]
        ),
        'most_confident_{}_percent_removed_negative'.format(percentage): (
            [train_examples[i] for i in most_confident_negative_indices[n_removed:]]
            + [train_examples[i] for i in positive_indices]
        ),
        'least_confident_{}_percent_removed_combined'.format(percentage): (
            [train_examples[i] for i in most_confident_combined_indices[::-1][n_removed:]]
        ),
        'least_confident_{}_percent_removed_positive'.format(percentage): (
            [train_examples[i] for i in most_confident_positive_indices[::-1][n_removed:]]
            + [train_examples[i] for i in negative_indices]
        ),
        'least_confident_{}_percent_removed_negative'.format(percentage): (
            [train_examples[i] for i in most_confident_negative_indices[::-1][n_removed:]]
            + [train_examples[i] for i in positive_indices]
        ),
    }

    for config_name, train_examples in datasets.items():
        create_data_config(
            task_name,
            config_name=config_name,
            train_examples=train_examples,
        )


def remove_by_similarity(task_name: str, eval_name: str, percentage: int = 10,
                         eval_data_dir: str = None):
    """
    for each example in the target test set, find the training examples with the most similar final
    representation, accumulate the score over all test examples, remove the top 10%
    """
    model, trainer, train_dataset, eval_dataset = setup(
        args_dir='configs/{}/base.json'.format(task_name),
        eval_data_dir=eval_data_dir
    )
    model.eval()

    # use eval dataloader to avoid shuffling
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

    processor = glue_processors[task_name.lower()]()
    train_examples = processor.get_train_examples('data/{}/base'.format(task_name))
    n_removed = int(percentage / 100 * len(train_examples))

    eval_subset_indices = {
        'combined': list(range(len(eval_dataset))),
        'negative': [i for i, x in enumerate(eval_dataset) if x.label == 0],
        'positive': [i for i, x in enumerate(eval_dataset) if x.label == 1],
    }

    for fold, indices in eval_subset_indices.items():
        # take the mean distance between each training examples and all dev examples in this fold
        most_similar_indices = np.argsort(-similarity[indices].mean(axis=0))

        # most_similar_10_percent_to_positive_50dev_removed
        create_data_config(
            task_name=task_name,
            config_name='most_dot_similar_{}_percent_to_{}_{}_removed'.format(
                percentage, fold, eval_name),
            train_examples=[train_examples[i] for i in most_similar_indices[n_removed:]],
        )

        create_data_config(
            task_name=task_name,
            config_name='least_dot_similar_{}_percent_to_{}_{}_removed'.format(
                percentage, fold, eval_name),
            train_examples=[train_examples[i] for i in most_similar_indices[::-1][n_removed:]],
        )


def compare_scores(task_name: str, args_dirs: str, eval_data_dir: str):
    model, trainer, train_dataset, eval_dataset = setup(
        args_dir='configs/{}/base.json'.format(task_name),
        eval_data_dir=eval_data_dir
    )
    output_original = trainer.predict(eval_dataset)
    scores_original = np.choose(
        output_original.label_ids,
        F.softmax(torch.from_numpy(output_original.predictions), dim=-1).numpy().T,
    )
    predictions_original = np.argmax(output_original.predictions, axis=1)
    print('original predictions: {} positive {} negative'.format(
        sum(predictions_original),
        len(predictions_original) - sum(predictions_original)
    ))

    print('original labels: {} positive {} negative'.format(
        sum(output_original.label_ids),
        len(output_original.label_ids) - sum(output_original.label_ids)
    ))

    for args_dir in args_dirs:
        print(args_dir)
        model, trainer, _, _ = setup(args_dir=args_dir)
        output_modified = trainer.predict(eval_dataset)
        predictions_modified = np.argmax(output_modified.predictions, axis=1)
        predictions = {
            'combined': predictions_modified,
            'negative': predictions_modified[output_modified.label_ids == 0],
            'positive': predictions_modified[output_modified.label_ids == 1],
        }

        scores_modified = np.choose(
            output_modified.label_ids,
            F.softmax(torch.from_numpy(output_modified.predictions), dim=-1).numpy().T,
        )
        scores_diff = {
            'combined': scores_modified - scores_original,
            'negative': (scores_modified - scores_original)[output_modified.label_ids == 0],
            'positive': (scores_modified - scores_original)[output_modified.label_ids == 1],
        }

        for fold, diff in scores_diff.items():
            print('{} predictions: {} positive {} negative'.format(
                fold,
                sum(predictions[fold]),
                len(predictions[fold]) - sum(predictions[fold])
            ))
            print('{}: {}{:.4f}%'.format(fold, '+' if np.mean(diff) > 0 else '',
                                         np.mean(diff) * 100))


if __name__ == '__main__':
    # random_dev_set('SNLI')
    # remove_by_random('SNLI')
    # remove_by_confidence('SST-2-GLUE')
    # remove_by_similarity(task_name='SST-2-ORIG', eval_name='dev',
    #                      eval_data_dir='data/SST-2-ORIG/base')
    # remove_by_similarity(task_name='SST-2-ORIG', eval_name='dev_200',
    #                      eval_data_dir='data/SST-2-ORIG/dev_200')
    # args_dirs = []
    # for i, filename in enumerate(glob.iglob('configs/SNLI/**/*.json', recursive=True)):
    #     if 'base' in filename:
    #         continue
    #     with open(filename) as f:
    #         args = json.load(f)
    #         checkpoint_dir = os.path.join(args['output_dir'], 'pytorch_model.bin')
    #         if os.path.exists(checkpoint_dir):
    #             args_dirs.append(filename)
    # pprint(args_dirs)

    # compare_scores(
    #     task_name='SST-2-GLUE',
    #     args_dirs=args_dirs,
    #     eval_data_dir='data/SST-2-GLUE/base',
    # )
    pass
