""" Training Management Module """


def train(model, input_data, labels, criterion, optimizer, epochs=5000):
    """ Train the Neural Network """
    for e in range(epochs):
        y_pred = model.forward(input_data)
        loss = criterion(y_pred, labels)
        print("Epoch: {0} Loss: {1}".format(e, loss), end="\r")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
