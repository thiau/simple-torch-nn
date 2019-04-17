""" Dataset Management Module """

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_pandas_dataset():
    """ Load main dataset """
    dataset = pd.read_csv(
        "resources/datasets/StudentsPerformanceCustom.csv", sep=";")
    dataset = dataset.iloc[:, 0:6]
    dataset = pd.get_dummies(
        dataset, columns=dataset.columns[0:-1], drop_first=True)
    return dataset


def create_tensors(dataset):
    """ Create PyTorch Tensors based in a dataset """
    variables = dataset.iloc[:, [x for x in range(
        0, len(dataset.columns)) if x != 0]].values
    labels = dataset.iloc[:, 0].values

    scaler = StandardScaler()
    variables = scaler.fit_transform(variables)

    variables = torch.FloatTensor(variables)
    labels = torch.tensor(labels)

    return variables, labels
