import os
import re
import json
import glob
import random
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Dict
from copy import deepcopy
from pprint import pprint
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
    BertTokenizer,
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


def get_eval_dataset(task_name: str, eval_data_dir: str, tokenizer: BertTokenizer = None):
    if tokenizer is None:
        base_args = json.load(open(f'configs/{task_name}/base.json'))
        tokenizer = AutoTokenizer.from_pretrained(base_args['output_dir'])

    args = DataTrainingArguments(
        task_name=task_name,
        data_dir=eval_data_dir,
    )
    return GlueDataset(args, tokenizer=tokenizer, evaluate=True)


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
    """create random subset of the dev set"""
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

    file_paths = [
        'data/{}/dev_{}'.format(task_name, n_examples * 3)
        for task_name in task_names
    ]
    if all(os.path.exists(fp) for fp in file_paths):
        return

    for task_name in task_names:
        path = 'data/{}/dev_{}'.format(task_name, n_examples * 3)
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path, 'dev.tsv'), 'w') as output_file:
            output_file.write('sentence\tlabel\n')
            for example in random_examples:
                output_file.write('{}\t{}\n'.format(example.text_a, example.label))


def random_individual_dev_set(task_name: str = None, n_examples: int = 50):
    """use random individual dev examples as dev sets"""
    task_names = [task_name] if task_name else ['SST-2-GLUE', 'SST-2-ORIG']
    processor = glue_processors[task_names[0].lower()]()
    dev_examples = processor.get_dev_examples('data/{}/base'.format(task_names[0]))

    datasets = {
        'negative': [x for x in dev_examples if x.label == '0'],
        'positive': [x for x in dev_examples if x.label == '1'],
    }

    random.shuffle(datasets['negative'])
    random.shuffle(datasets['positive'])

    # balanced random individual dev set
    random_examples = (
        datasets['negative'][:int(n_examples / 2)]
        + datasets['positive'][:int(n_examples / 2)]
    )

    for task_name in task_names:
        for example in random_examples:
            path = 'data/{}/{}'.format(task_name, example.guid)
            Path(path).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(path, 'dev.tsv'), 'w') as output_file:
                output_file.write('sentence\tlabel\n')
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


def get_pooled_output(task_name: str, fold: str, eval_name: str = 'base'):
    base_args_dir = 'configs/{}/base.json'.format(task_name)
    args = json.load(open(base_args_dir))

    if fold == 'train':
        pooled_output_dir = os.path.join(args['output_dir'], 'pooled_output_train.npy')
    else:
        pooled_output_dir = os.path.join(args['output_dir'], f'pooled_output_{eval_name}.npy')

    if os.path.exists(pooled_output_dir):
        return np.load(pooled_output_dir, allow_pickle=True)

    model, trainer, train_dataset, eval_dataset = setup(
        args_dir=base_args_dir,
        eval_data_dir='data/{}/{}'.format(task_name, eval_name),
    )

    # use eval dataloader to avoid shuffling
    if fold == 'train':
        data_loader = trainer.get_eval_dataloader(train_dataset)
    elif fold == 'dev' or fold == 'eval':
        data_loader = trainer.get_eval_dataloader(eval_dataset)

    pooled_output = []
    for inputs in data_loader:
        for k, v in inputs.items():
            inputs[k] = v.to(model.device)
        with torch.no_grad():
            outputs = model.bert(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
            )
            pooled_output.append(outputs[1].detach().cpu().numpy())
    np.save(pooled_output_dir, pooled_output)
    return pooled_output


def remove_by_similarity(
        task_name: str,
        eval_name: str,
        percentage: int = 10,
        similarity_metric: str = 'dot',
        all_folds: bool = True):
    """
    for each example in the target test set, find the training examples with the most similar final
    representation, accumulate the score over all test examples, remove the top 10%
    """
    folds = ['combined']
    if all_folds:
        folds += ['negative', 'positive']

    file_paths = [
        'most_{}_similar_{}_percent_to_{}_{}_removed'.format(
            similarity_metric, percentage, fold, eval_name)
        for fold in folds
    ] + [
        'least_{}_similar_{}_percent_to_{}_{}_removed'.format(
            similarity_metric, percentage, fold, eval_name)
        for fold in folds
    ]
    file_paths = [os.path.join(f'data/{task_name}', fp) for fp in file_paths]

    if all(os.path.exists(fp) for fp in file_paths):
        return

    eval_dataset = get_eval_dataset(
        task_name=task_name,
        eval_data_dir='data/{}/{}'.format(task_name, eval_name),
    )

    pooled_outputs = {
        'train': get_pooled_output(task_name, fold='train', eval_name=eval_name),
        'eval': get_pooled_output(task_name, fold='eval', eval_name=eval_name),
    }

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
    }
    if all_folds:
        eval_subset_indices.update({
            'negative': [i for i, x in enumerate(eval_dataset) if x.label == 0],
            'positive': [i for i, x in enumerate(eval_dataset) if x.label == 1],
        })

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


def get_gradient_wrt_pooled_output(task_name: str, fold: str, eval_name: str = 'base'):
    base_args_dir = 'configs/{}/base.json'.format(task_name)
    args = json.load(open(base_args_dir))

    if fold == 'train':
        gradient_dir = os.path.join(args['output_dir'], 'gradient_wrt_pooled_output_train.npy')
    else:
        gradient_dir = os.path.join(args['output_dir'], f'gradient_wrt_pooled_output_{eval_name}.npy')

    if os.path.exists(gradient_dir):
        return np.load(gradient_dir)

    model, trainer, train_dataset, eval_dataset = setup(
        args_dir=base_args_dir,
        eval_data_dir='data/{}/{}'.format(task_name, eval_name),
    )

    # use eval dataloader to avoid shuffling
    if fold == 'train':
        data_loader = trainer.get_eval_dataloader(train_dataset)
    elif fold == 'dev' or fold == 'eval':
        data_loader = trainer.get_eval_dataloader(eval_dataset)

    gradient_wrt_pooled_output = []
    for inputs in tqdm(data_loader):
        model.zero_grad()
        model.eval()
        inputs = next(iter(data_loader))
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        labels = inputs.pop('labels')
        pooled_output = model.bert(**inputs)[1]
        pooled_output = model.dropout(pooled_output)
        logits = model.classifier(pooled_output)
        loss = torch.nn.CrossEntropyLoss()(logits.view(-1, model.num_labels), labels.view(-1))
        grad = torch.autograd.grad(loss, pooled_output)[0]
        gradient_wrt_pooled_output.append(grad.detach().cpu().numpy())
    gradient_wrt_pooled_output = np.concatenate(gradient_wrt_pooled_output, axis=0)
    np.save(gradient_dir, gradient_wrt_pooled_output)
    return gradient_wrt_pooled_output


def remove_by_gradient_similarity(
        task_name: str,
        eval_name: str,
        percentage: int = 10,
        similarity_metric: str = 'dot',
        all_folds: bool = False):
    """
    for each example in the target test set, find the training examples with the most similar
    gradient w.r.t. final representation, accumulate the score over all test examples, remove the
    top 10%
    """
    folds = ['combined']
    if all_folds:
        folds += ['negative', 'positive']

    file_paths = [
        'most_{}_gradient_similar_{}_percent_to_{}_{}_removed'.format(
            similarity_metric, percentage, fold, eval_name)
        for fold in folds
    ]
    # + [
    #     'least_{}_gradient_similar_{}_percent_to_{}_{}_removed'.format(
    #         similarity_metric, percentage, fold, eval_name)
    #     for fold in folds
    # ]
    file_paths = [os.path.join(f'data/{task_name}', fp) for fp in file_paths]

    if all(os.path.exists(fp) for fp in file_paths):
        return

    eval_dataset = get_eval_dataset(
        task_name=task_name,
        eval_data_dir='data/{}/{}'.format(task_name, eval_name),
    )

    gradients = {
        # TODO
        'train': get_gradient_wrt_pooled_output(task_name, fold='train', eval_name=eval_name),
        'eval': get_gradient_wrt_pooled_output(task_name, fold='eval', eval_name=eval_name),
    }

    # similarity is a (n_eval, n_train) matrix
    if similarity_metric == 'dot':
        similarity = gradients['eval'] @ gradients['train'].T
    elif similarity_metric == 'cosine':
        # cosine(a, b) = a @ b.T / (norm(a) * norm(b))
        a, b = gradients['eval'], gradients['train']
        similarity = (a @ b.T) / np.outer(norm(a, axis=1), norm(b, axis=1))
    elif similarity_metric == 'l2':
        similarity = cdist(gradients['eval'], gradients['train'])

    processor = glue_processors[task_name.lower()]()
    train_examples = processor.get_train_examples('data/{}/base'.format(task_name))
    n_removed = int(percentage / 100 * len(train_examples))

    eval_subset_indices = {
        'combined': list(range(len(eval_dataset))),
    }
    if all_folds:
        eval_subset_indices.update({
            'negative': [i for i, x in enumerate(eval_dataset) if x.label == 0],
            'positive': [i for i, x in enumerate(eval_dataset) if x.label == 1],
        })

    for fold, indices in eval_subset_indices.items():
        # take the mean distance between each training examples and all dev examples in this fold
        most_similar_indices = np.argsort(-similarity[indices].mean(axis=0))

        # most_similar_10_percent_to_positive_50dev_removed
        create_data_config(
            task_name=task_name,
            config_name='most_{}_gradient_similar_{}_percent_to_{}_{}_removed'.format(
                similarity_metric, percentage, fold, eval_name),
            train_examples=[train_examples[i] for i in most_similar_indices[n_removed:]],
        )

        # create_data_config(
        #     task_name=task_name,
        #     config_name='least_{}_gradient_similar_{}_percent_to_{}_{}_removed'.format(
        #         similarity_metric, percentage, fold, eval_name),
        #     train_examples=[train_examples[i] for i in most_similar_indices[::-1][n_removed:]],
        # )


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


def compare_scores_to_base():
    '''
    get the average diff of each config on all eval datasets
    configs that are specific to a dev set is converted,
    e.g. `most_similar_to_dev-24` -> `most_similar_to` so they are merged
    '''
    task_names = ['SST-2-ORIG']
    # eval_names = ['base', 'dev_150']
    eval_names = [os.path.basename(x) for x in glob.iglob('data/SST-2-GLUE/dev-*')]

    # confidence_minus_base[args_dir][eval_name] = diff of confidence between config and base model
    confidence_minus_base = defaultdict(dict)
    all_eval_settings = list(itertools.product(task_names, eval_names))
    for i_eval_setting, (task_name, eval_name) in enumerate(all_eval_settings):
        '''
        for each eval dataset, find
        1. all configs independent of specific eval dataset, e.g. `most_confident_10_percent_removed`
        2. all eval-specific configs that depend on this eval dataset, e.g. `most_similar_to_THIS_DEV_SET_removed`
        '''
        args_dir_list = []
        for i, args_dir in enumerate(glob.iglob(f'configs/{task_name}/**/*.json', recursive=True)):
            checkpoint_dir = os.path.join(json.load(open(args_dir))['output_dir'], 'pytorch_model.bin')
            if os.path.exists(checkpoint_dir):
                if 'similar' not in args_dir:
                    # config does not depend on specific dev set
                    args_dir_list.append(args_dir)
                elif f'combined_{eval_name}_' in args_dir:
                    # config depends on this dev set
                    args_dir_list.append(args_dir)

        '''get eval dataset labels and predictions from the base model'''
        eval_dataset = get_eval_dataset(
            task_name=task_name,
            eval_data_dir='data/{}/{}'.format(task_name, eval_name),
        )
        labels = np.array([x.label for x in eval_dataset])

        predictions_by_base = get_eval_predictions(
            task_name=task_name,
            args_dir='configs/{}/base.json'.format(task_name),
            eval_name=eval_name)
        confidence_by_base = np.choose(
            labels,
            F.softmax(torch.from_numpy(predictions_by_base), dim=-1).numpy().T,
        )

        '''go through all valid configs, get predictions on the eval_name dataset'''
        for i, args_dir in enumerate(args_dir_list):
            print(f'{i_eval_setting + 1}/{len(all_eval_settings)}',
                  f'{i + 1}/{len(args_dir_list)}',
                  eval_name, args_dir)
            predictions = get_eval_predictions(task_name, args_dir, eval_name)
            confidence = np.choose(
                labels,
                F.softmax(torch.from_numpy(predictions), dim=-1).numpy().T,
            )
            # remove config's dependency on this specific dev set
            config_name = re.compile('dev-[\d]*_').sub('', args_dir)
            confidence_minus_base[config_name][eval_name] = np.mean(confidence - confidence_by_base) * 100

    # average each config over all eval_names
    # confidence_minus_base[args_dir] = average diff in confidence between config and base on all eval_names
    scores = {k: (len(v), np.mean(list(v.values()))) for k, v in confidence_minus_base.items()}
    # sort configs by largest average drop in confidence
    scores = sorted(scores.items(), key=lambda x: x[1][1])
    pprint(scores[:10])


if __name__ == '__main__':
    # random_dev_set(n_examples=50)
    # random_individual_dev_set(n_examples=50)
    # remove_by_random('SST-2-GLUE')
    # remove_by_random('SST-2-ORIG')
    # remove_by_confidence('SST-2-GLUE')
    # remove_by_confidence('SST-2-ORIG')
    # remove_by_confidence('SST-2-GLUE', use_prediction=True)
    # remove_by_confidence('SST-2-ORIG', use_prediction=True)

    task_names = ['SST-2-ORIG']
    similarity_metrics = ['dot']
    eval_names = [os.path.basename(x) for x in glob.iglob('data/SST-2-GLUE/dev-*')]

    for task_name, similarity_metric, eval_name in itertools.product(
            task_names, similarity_metrics, eval_names):
        print(task_name, similarity_metric, eval_name)
        remove_by_gradient_similarity(task_name=task_name,
                                      eval_name=eval_name,
                                      similarity_metric=similarity_metric,
                                      all_folds=False)

    # compare_scores_to_base()
