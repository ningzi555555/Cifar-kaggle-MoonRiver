import pandas as pd
from resnet import resnet10
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from resxnet import ResNeXt29_2x64d, ResNeXt29_32x4d

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class TestDataset(Dataset):
    def __init__(self, X, transform=None):
        self.X = X.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx].astype(np.uint8)
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        return img

def main():
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    testset = unpickle('cifar_test_nolabel.pkl')
    testdataset = TestDataset(testset[b'data'], transform=transform_test)
    testloader = DataLoader(testdataset, batch_size=100, shuffle=False, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(
    #     root='./data', train=False, download=True, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=100, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNeXt29_32x4d().to(device)
    model.load_state_dict(torch.load('checkpoint/epoch194.tar'))

    # test
    model.eval()
    all_predicted_output = []
    true_labels = []
    with torch.no_grad():
        for batch_idx, (inputs) in enumerate(testloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_predicted_output.extend(predicted.cpu().numpy())
            # true_labels.extend(labels.numpy())
    result = np.stack((testset[b'ids'], np.asarray(all_predicted_output)),axis = 1)
    df = pd.DataFrame(result, columns=['ID', 'Labels'])
    df.to_csv("data0310.csv", index=False)


    with open('predictions.pkl', 'wb') as f:
        pickle.dump([true_labels, all_predicted_output], f)

if __name__ == '__main__':
    main()
