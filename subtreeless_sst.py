import os
import json
import torchtext
from nltk import word_tokenize
from pathlib import Path

def create_sentence_only_sst2():
    train, dev, test = torchtext.datasets.SST.splits(
        torchtext.data.Field(batch_first=True, tokenize=word_tokenize, lower=False),
        torchtext.data.Field(sequential=False, unk_token=None),
        root='data/')

    data_dir = 'data/SST-2/base'
    output_dir = 'output/SST-2/base'
    config_dir = 'configs/SST-2/base.json'
    train_data_dir = os.path.join(data_dir, 'train.tsv')

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(config_dir).parent.mkdir(parents=True, exist_ok=True)

    args = {
        "model_type": "bert",
        "model_name_or_path": "bert-base-cased",
        "task_name": "SST-2",
        "do_train": True,
        "do_eval": True,
        "data_dir": "data/SST-2/base",
        "max_seq_length": 128,
        "per_gpu_train_batch_size": 32,
        "learning_rate": 2e-05,
        "num_train_epochs": 3.0,
        "output_dir": "output/SST-2/base",
        "train_data_dir": "data/SST-2/base",
        "eval_data_dir": "data/SST-2/base"
    }
    with open(config_dir, 'w') as f:
        json.dump(args, f, indent=4)

    with open(train_data_dir, 'w') as f:
        f.write('sentence\tlabel\n')
        for example in train:
            f.write('{}\t{}\n'.format(' '.join(example.text),
                                      '0' if example.label == 'negative' else '1'))


if __name__ == '__main__':
    create_sentence_only_sst2()
