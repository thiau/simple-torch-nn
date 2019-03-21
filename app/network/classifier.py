import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Classifier(nn.Module):
    def __init__(self, input_size, nb_action, nb_neurons=30):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.nb_neurons = nb_neurons

        self.fc1 = nn.Linear(input_size, nb_neurons)
        self.fc2 = nn.Linear(nb_neurons, nb_neurons)
        self.fc3 = nn.Linear(nb_neurons, nb_action)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        values = self.fc3(x)
        return values

    def predict(self, input):
        predictions = F.softmax(self.forward(input), dim=1)
        best_pred = list()
        for t in predictions:
            if t[0] > t[1]:
                best_pred.append(0)
            else:
                best_pred.append(1)
        return best_pred
