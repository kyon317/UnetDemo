import numpy as np

from model import UNet
from dataloader import Cell_data

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.optim as optim
import matplotlib.pyplot as plt

import os

# import any other libraries you need below this line
import wandb



"""Pretrain Definition"""
#Paramteres
#learning rate
lr = 1e-4

#number of training epochs
epoch_n = 20

#input image-mask size
# image_size = 572
image_size = 256
# image_size = 128

#root directory of project
root_dir = os.getcwd()

#training batch size
batch_size = 4

#use checkpoint model for training
load = False

#use GPU for training
gpu = True

data_dir = os.path.join(root_dir, 'data/cells')
dirs = os.getcwd()
# Function to save a checkpoint in the TensorBoard log directory
def save_checkpoint(model, optimizer, epoch, loss_metric, accuracy_metric, checkpoint_dir=dirs):
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"

    # Use .compute() to get the values of the metrics
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss_metric,  # Get the computed loss
        'accuracy': accuracy_metric  # Get the computed accuracy
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')

# Function to load a checkpoint from the TensorBoard log directory
def load_checkpoint(model, optimizer, checkpoint_dir=dirs):
    import glob
    # Find the latest checkpoint (e.g., based on the highest epoch number)
    checkpoint_paths = glob.glob(f"{checkpoint_dir}/checkpoint_epoch_*.pth")
    checkpoint_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint = checkpoint_paths[-1]

    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    print(f'Checkpoint loaded from {latest_checkpoint}. Resuming training from epoch {start_epoch}')

    return start_epoch, loss, accuracy  # You can return these for reference but don't update accuracy_metric with them

# wandb setup
# Initialize wandb project, reference: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb#scrollTo=R_3hYQyqb0fJ
wandb.init(project="assignment2_unet_training", config={
    "learning_rate": lr,
    "epochs": epoch_n,
    "batch_size": batch_size,
    "image_size": image_size
})

trainset = Cell_data(data_dir = data_dir, size = image_size)
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True)

testset = Cell_data(data_dir = data_dir, size = image_size, train = False)
testloader = DataLoader(testset, batch_size = batch_size)

device = torch.device('cuda:0' if gpu else 'cpu')

model = UNet(n_channels=1, n_classes=2).to('cuda:0').to(device)

criterion = nn.CrossEntropyLoss()
# Use AdamW for better weight decay
optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.8, 0.9), weight_decay=5e-4)

if load:
  print('loading model')
  load_checkpoint(model,optimizer,dirs)



"""TRAINING"""

model.train()
for e in range(epoch_n):
    epoch_loss = 0
    model.train()
    for i, data in enumerate(trainloader):
        image, label = data

        image = image.to(device)
        label = label.squeeze(1).long().to(device)

        pred = model(image)

        crop_x = (label.shape[1] - pred.shape[2]) // 2
        crop_y = (label.shape[2] - pred.shape[3]) // 2

        label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]
        loss = criterion(pred, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        if i % 5 == 0:
            print('batch %d --- Loss: %.4f' % (i, loss.item() / batch_size))
            wandb.log({'batch_loss': loss.item() / batch_size})
    print('Epoch %d / %d --- Loss: %.4f' % (e + 1, epoch_n, epoch_loss / trainset.__len__()))
    wandb.log({'epoch_loss': epoch_loss / trainset.__len__()})

    model.eval()

    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            image, label = data

            image = image.to(device)
            label = label.squeeze(1).long().to(device)

            pred = model(image)
            crop_x = (label.shape[1] - pred.shape[2]) // 2
            crop_y = (label.shape[2] - pred.shape[3]) // 2

            label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

            loss = criterion(pred, label)
            total_loss += loss.item()

            _, pred_labels = torch.max(pred, dim=1)

            total += label.shape[0] * label.shape[1] * label.shape[2]
            correct += (pred_labels == label).sum().item()

        accuracy = correct / total
        test_loss = total_loss / testset.__len__()
        print('Accuracy: %.4f ---- Loss: %.4f' % (accuracy, test_loss))
        wandb.log({'test_loss': test_loss, 'accuracy': accuracy})
        if e % 5 == 0 or e == epoch_n - 1:
            save_checkpoint(model, optimizer, e, epoch_loss / trainset.__len__(), correct / total)
wandb.finish()


"""VISUALIZATION AND EVALUATION"""

model.eval()

output_masks = []
output_labels = []

with torch.no_grad():
    for i in range(len(testset)):
        image, labels = testset.__getitem__(i)

        input_image = image.unsqueeze(0).to(device)
        pred = model(input_image)
        # print(torch.unique(labels))
        aaa = torch.argmax(pred, dim=1).cpu().numpy()
        output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()

        crop_x = (labels.shape[0] - output_mask.shape[0]) // 2
        crop_y = (labels.shape[1] - output_mask.shape[1]) // 2
        labels = labels[crop_x: labels.shape[0] - crop_x, crop_y: labels.shape[1] - crop_y].numpy()

        output_masks.append(output_mask)
        output_labels.append(labels)

fig, axes = plt.subplots(testset.__len__(), 2, figsize = (20, 20))
print(output_masks[0].shape)
print(output_labels[0].shape)
print(output_masks[0])
print(np.unique(output_masks[0]))

for i in range(testset.__len__()):
  axes[i, 0].imshow(output_labels[i])
  axes[i, 0].axis('off')
  axes[i, 1].imshow(output_masks[i].squeeze())
  axes[i, 1].axis('off')