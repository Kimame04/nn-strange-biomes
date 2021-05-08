import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

train_df = pd.read_csv('train.csv')
root = '/Users/kieranmendoza/PycharmProjects/nn-strange-biomes/train-smaller'

def generate_dataset():
    data_transform = torchvision.transforms.Compose([
        transforms.Resize(128),
        torchvision.transforms.ToTensor()
    ])
    model_dataset = datasets.ImageFolder(root, transform=data_transform)
    return model_dataset


def split_train_test(model_dataset):
    total_count = len(model_dataset)
    train_count = int(0.7 * total_count)
    valid_count = int(0.2 * total_count)
    test_count = total_count - train_count - valid_count
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        model_dataset, (train_count, valid_count, test_count)
    )
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataset, valid_dataset, test_dataset, train_dataloader, test_dataloader


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 16 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def getModel():
    return model

def getClasses():
    return train_df.biome_name.unique()

if __name__ == '__main__':

    dataset = generate_dataset()
    train_dataset, valid_dataset, test_dataset, train_dataloader, test_dataloader = split_train_test(dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = models.resnet50(pretrained=True)
    print(model)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 64),
        nn.LogSoftmax(dim=1)
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model.to(device)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)
    print("Done!")
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
