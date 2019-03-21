import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_pands_dataset():
    dataset = pd.read_csv(
        "resources/datasets/StudentsPerformanceCustom.csv", sep=";")
    dataset = dataset.iloc[:, 0:6]
    dataset = pd.get_dummies(
        dataset, columns=dataset.columns[0:-1], drop_first=True)
    return dataset


def create_tensors(dataset):
    X = dataset.iloc[:, [x for x in range(
        0, len(dataset.columns)) if x != 0]].values
    y = dataset.iloc[:, 0].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.FloatTensor(X)
    y = torch.tensor(y)

    return X, y
