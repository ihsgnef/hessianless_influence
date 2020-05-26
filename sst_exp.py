import os
import json
import glob
import random
import itertools
import numpy as np
from pathlib import Path
from typing import Dict, List
from copy import deepcopy
from tqdm import tqdm
from numpy.linalg import norm
from scipy.spatial.distance import cdist

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
    task_names = [task_name] if task_name else ['SST-2-GLUE', 'SST-2-ORIG']
    processor = glue_processors[task_names[0].lower()]()
    dev_examples = processor.get_dev_examples('data/{}/base'.format(task_names[0]))

    datasets = {
        'combined': dev_examples,
        'negative': [x for x in dev_examples if x.label == '0'],
        'positive': [x for x in dev_examples if x.label == '1'],
    }

    random_examples = []
    for fold, examples in datasets.items():
        random.shuffle(examples)
        random_examples += examples[:n_examples]

    file_pathes = [
        'data/{}/dev_{}'.format(task_name, n_examples * 3)
        for task_name in task_names
    ]
    if all(os.path.exists(fp) for fp in file_pathes):
        return

    for task_name in task_names:
        path = 'data/{}/dev_{}'.format(task_name, n_examples * 3)
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path, 'dev.tsv'), 'w') as output_file:
            output_file.write('sentence\tlabel\n')
            for example in random_examples:
                output_file.write('{}\t{}\n'.format(example.text_a, example.label))


def remove_by_random(task_name: str, percentage: int = 10, n_trials: int = 3):
    processor = glue_processors[task_name.lower()]()
    all_examples = processor.get_train_examples('data/{}/base'.format(task_name))
    negative_examples = [x for x in all_examples if x.label == '0']
    positive_examples = [x for x in all_examples if x.label == '1']
    n_examples_removed = int(percentage / 100 * len(all_examples))

    for i in range(n_trials):
        random.shuffle(all_examples)
        create_data_config(
            task_name=task_name,
            config_name='random_{}_percent_removed_combined'.format(percentage),
            version_number=i,
            train_examples=all_examples[n_examples_removed:],
        )

    for i in range(n_trials):
        random.shuffle(positive_examples)
        create_data_config(
            task_name,
            config_name='random_{}_percent_removed_positive'.format(percentage),
            version_number=i,
            train_examples=positive_examples[n_examples_removed:] + negative_examples,
        )

    for i in range(n_trials):
        random.shuffle(negative_examples)
        create_data_config(
            task_name,
            config_name='random_{}_percent_removed_negative'.format(percentage),
            version_number=i,
            train_examples=negative_examples[n_examples_removed:] + positive_examples,
        )


def remove_by_confidence(task_name: str, percentage: int = 10, use_prediction: bool = False):
    args_dir = 'configs/{}/base.json'.format(task_name)
    model, trainer, train_dataset, eval_dataset = setup(args_dir=args_dir)

    output = trainer.predict(train_dataset)
    scores = F.softmax(torch.from_numpy(output.predictions), dim=-1).numpy()
    predictions = np.argmax(output.predictions, axis=1)
    indices = np.arange(len(scores))
    labels = predictions if use_prediction else output.label_ids

    scores = np.choose(labels, scores.T)
    positive_indices = indices[labels == 1]
    negative_indices = indices[labels == 0]
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]

    most_confident_combined_indices = np.argsort(-scores)
    most_confident_positive_indices = positive_indices[np.argsort(-positive_scores)]
    most_confident_negative_indices = negative_indices[np.argsort(-negative_scores)]

    processor = glue_processors[task_name.lower()]()
    train_examples = processor.get_train_examples('data/{}/base'.format(task_name))
    n_removed = int(percentage / 100 * len(train_examples))

    datasets = {
        'most_confident_{}_percent_removed_combined{}'.format(
            percentage, '_predicted' if use_prediction else ''): (
            [train_examples[i] for i in most_confident_combined_indices[n_removed:]]
        ),
        'most_confident_{}_percent_removed_positive{}'.format(
            percentage, '_predicted' if use_prediction else ''): (
            [train_examples[i] for i in most_confident_positive_indices[n_removed:]]
            + [train_examples[i] for i in negative_indices]
        ),
        'most_confident_{}_percent_removed_negative{}'.format(
            percentage, '_predicted' if use_prediction else ''): (
            [train_examples[i] for i in most_confident_negative_indices[n_removed:]]
            + [train_examples[i] for i in positive_indices]
        ),
        'least_confident_{}_percent_removed_combined{}'.format(
            percentage, '_predicted' if use_prediction else ''): (
            [train_examples[i] for i in most_confident_combined_indices[::-1][n_removed:]]
        ),
        'least_confident_{}_percent_removed_positive{}'.format(
            percentage, '_predicted' if use_prediction else ''): (
            [train_examples[i] for i in most_confident_positive_indices[::-1][n_removed:]]
            + [train_examples[i] for i in negative_indices]
        ),
        'least_confident_{}_percent_removed_negative{}'.format(
            percentage, '_predicted' if use_prediction else ''): (
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


def remove_by_similarity(
        task_name: str,
        eval_name: str,
        percentage: int = 10,
        similarity_metric: str = 'dot'):
    """
    for each example in the target test set, find the training examples with the most similar final
    representation, accumulate the score over all test examples, remove the top 10%
    """
    model, trainer, train_dataset, eval_dataset = setup(
        args_dir='configs/{}/base.json'.format(task_name),
        eval_data_dir='data/{}/{}'.format(task_name, eval_name),
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

    # similarity is a (n_eval, n_train) matrix
    if similarity_metric == 'dot':
        similarity = pooled_outputs['eval'] @ pooled_outputs['train'].T
    elif similarity_metric == 'cosine':
        # cosine(a, b) = a @ b.T / (norm(a) * norm(b))
        a, b = pooled_outputs['eval'], pooled_outputs['train']
        similarity = (a @ b.T) / np.outer(norm(a, axis=1), norm(b, axis=1))
    elif similarity_metric == 'l2':
        similarity = cdist(pooled_outputs['eval'], pooled_outputs['train'])

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
            config_name='most_{}_similar_{}_percent_to_{}_{}_removed'.format(
                similarity_metric, percentage, fold, eval_name),
            train_examples=[train_examples[i] for i in most_similar_indices[n_removed:]],
        )

        create_data_config(
            task_name=task_name,
            config_name='least_{}_similar_{}_percent_to_{}_{}_removed'.format(
                similarity_metric, percentage, fold, eval_name),
            train_examples=[train_examples[i] for i in most_similar_indices[::-1][n_removed:]],
        )


def get_eval_predictions(task_name: str, args_dir: str, eval_name: str):
    prediction_dir = os.path.join(
        json.load(open(args_dir))['output_dir'],
        'predictions_{}.npy'.format(eval_name),
    )
    if os.path.exists(prediction_dir):
        return np.load(prediction_dir)

    model, trainer, train_dataset, eval_dataset = setup(
        args_dir=args_dir,
        eval_data_dir='data/{}/{}'.format(task_name, eval_name),
    )
    output = trainer.predict(eval_dataset)
    np.save(prediction_dir, output.predictions)
    return output.predictions


def compare_scores_to_base(task_name: str, eval_name: str, args_dir_list: List[str], ):
    base_args_dir = 'configs/{}/base.json'.format(task_name)
    _, _, _, eval_dataset = setup(
        args_dir=base_args_dir,
        eval_data_dir='data/{}/{}'.format(task_name, eval_name),
    )
    labels = np.array([x.label for x in eval_dataset])

    predictions_original = get_eval_predictions(task_name, base_args_dir, eval_name)
    scores_original = np.choose(
        labels,
        F.softmax(torch.from_numpy(predictions_original), dim=-1).numpy().T,
    )

    for args_dir in args_dir_list:
        predictions_modified = get_eval_predictions(task_name, args_dir, eval_name)
        scores_modified = np.choose(
            labels,
            F.softmax(torch.from_numpy(predictions_modified), dim=-1).numpy().T,
        )
        scores_diff = {
            'combined': scores_modified - scores_original,
            'negative': (scores_modified - scores_original)[labels == 0],
            'positive': (scores_modified - scores_original)[labels == 1],
        }

        print(task_name, eval_name, args_dir)
        for fold, diff in scores_diff.items():
            # print('{} predictions: {} positive {} negative'.format(
            #     fold,
            #     sum(predictions[fold]),
            #     len(predictions[fold]) - sum(predictions[fold])
            # ))
            print('{}: {:.4f}%'.format(fold, np.mean(diff) * 100))


if __name__ == '__main__':
    # random_dev_set()
    # remove_by_random('SST-2-GLUE')
    # remove_by_random('SST-2-ORIG')
    # remove_by_confidence('SST-2-GLUE')
    # remove_by_confidence('SST-2-ORIG')
    # remove_by_confidence('SST-2-GLUE', use_prediction=True)
    # remove_by_confidence('SST-2-ORIG', use_prediction=True)

    task_names = ['SST-2-ORIG', 'SST-2-GLUE']
    similarity_metrics = ['dot', 'cosine', 'l2']
    eval_names = ['base', 'dev_150']

    # for task_name, similarity_metric, eval_name in itertools.product(
    #         task_names, similarity_metrics, eval_names):
    #     remove_by_similarity(task_name=task_name,
    #                          eval_name=eval_name,
    #                          similarity_metric=similarity_metric)

    for task_name, eval_name in itertools.product(task_names, eval_names):
        args_dir_list = []
        for i, args_dir in enumerate(glob.iglob('configs/{}/**/*.json'.format(task_name), recursive=True)):
            checkpoint_dir = os.path.join(json.load(open(args_dir))['output_dir'],
                                          'pytorch_model.bin')

            if os.path.exists(checkpoint_dir):
                # print(args_dir)
                args_dir_list.append(args_dir)
                # predictions = get_eval_predictions(task_name, args_dir, eval_name)
        print()
        compare_scores_to_base(task_name, eval_name, args_dir_list)
