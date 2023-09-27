import pandas as pd

from os import listdir
from os.path import dirname, isfile, join, splitext

DATA_HOME = join(dirname(__file__), 'datasets')


def load_data(dataset_name):
    if not isinstance(dataset_name, str):
        err = 'The "dataset_name" should be a string.'
        raise TypeError(err)

    available_datasets = [ splitext(f)[0] for f in listdir(DATA_HOME) if isfile(join(DATA_HOME, f)) ]
    if dataset_name not in available_datasets:
        err = f'The "dataset_name" should be one of {available_datasets}.'
        raise ValueError(err)

    df = pd.read_csv(f'../statlearn/datasets/{dataset_name}.csv')

    return df