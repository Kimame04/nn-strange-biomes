from PIL import Image
import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms
import train

root2 = '/Users/kieranmendoza/PycharmProjects/nn-strange-biomes'

def loadModel():
    data_transform = torchvision.transforms.Compose([
        transforms.Resize(128),
        torchvision.transforms.ToTensor()
    ])

    pred_dataset = datasets.ImageFolder(root2,transform=data_transform)
    model = train.getModel()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    return data_transform, model

def predict_biome(row):
    image = data_transform(Image.open('test-release/'+row.image))
    image = image.view(1,3,56,99)
    output = model(image)
    prediction = int(torch.max(output.data, 1)[1].numpy())
    row.biome_name = classes[prediction]

def predict_easy(row):
    image = data_transform(Image.open('test-release/'+row.image))
    image = image.view(1,3,56,99)
    output = model(image)
    prediction = classes[int(torch.max(output.data, 1)[1].numpy())]
    row[prediction.split('_')] = 1
    return row

if __name__ == '__main__':
    sub_easyDf = pd.read_csv('submission_easy.csv')
    sub_hardDf = pd.read_csv('submission_hard.csv')

    data_transform, model = loadModel()
    classes = train.getClasses()

    sub_hardDf = sub_hardDf.apply(predict_biome,axis=1)
    sub_hardDf.to_csv('predicted_label.csv')
    sub_easyDf = sub_easyDf.apply(predict_easy,axis=1)
    sub_easyDf.to_csv('predicted_properties.csv')