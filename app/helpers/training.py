""" Training Management Module """


def train(model, input_data, labels, criterion, optimizer, epochs=5000):
    """ Train the Neural Network """
    losses = list()
    for _ in range(epochs):
        y_pred = model.forward(input_data)
        loss = criterion(y_pred, labels)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
