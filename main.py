import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from resnet import resnet10, resnet18
from resxnet import ResNeXt29_2x64d, ResNeXt29_32x4d
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from googlenet import GoogLeNet
from torchsummary import summary
from senet import se_resnet20, se_resnet56, se_preactresnet56
from efficientnet import EfficientNetB0
from shufflenet import ShuffleNetV2
from mobilenetv2 import MobileNetV2
from cnn import CNN_V3_V4
import pickle
import torchvision.models as models
from bit import KNOWN_MODELS

def main():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
    ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = KNOWN_MODELS['BiT-M-R50x1'](head_size=10, zero_head=True).to(device)#resnet18().to(device) #(num_classes=10, reduction=16)

    summary(model, input_size=(3, 32, 32))
    # model.load_state_dict(torch.load('checkpoint/epoch47_resnext.tar'))

    epochs = 100
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=0, last_epoch=-1, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)
    patience = 10  # Stop if no improvement for 'patience' epochs
    best_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    best_epoch = 0
    train_loss_history = []
    train_accu_history = []
    test_loss_history = []
    test_accu_history = []


    for epoch in range(epochs):
        print("epoch:", epoch + 1, 'lr', optimizer.param_groups[0]['lr'])
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        # progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # progress_bar.update(1)
            # progress_bar.set_postfix(loss=(train_loss/(batch_idx+1)))

        torch.save(model.state_dict(), 'checkpoint/epoch' + str(epoch) + '.tar')

        print('Train Loss: ', train_loss/(batch_idx+1), ' Acc: ', 100.*correct/total, ' correct: ', correct, ' total: ', total)
        train_accu_history += [100. * correct / total]
        #test
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        scheduler.step(test_loss)

        print(f"Test Accuracy: {100 * correct / total:.2f}%, Test Loss: {test_loss / (batch_idx + 1):.4f}")
        train_loss = train_loss / len(trainloader)
        test_loss = test_loss / len(testloader)
        train_loss_history += [train_loss]
        test_accu_history += [100. * correct / total]
        test_loss_history += [test_loss]
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # Save best model
        else:
            epochs_no_improve += 1
            # if epochs_no_improve >= patience:
            #     print(
            #         f"Early stopping triggered at epoch {epoch + 1}. Restoring best model from epoch {best_epoch}.")
            #     model.load_state_dict(best_model_state)  # Restore best model
            #     break
    print(1)

    with open('train_curve.pkl', 'wb') as f:
        pickle.dump([train_loss_history,train_accu_history,test_loss_history,test_accu_history], f)


if __name__ == '__main__':
    main()
