#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import list_models, get_model, get_model_weights, get_weight
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In[2]:

print("setting directories")
data_dir = "/dcs05/ciprian/smart/pocus/rushil/augmentedData"
train_dir = data_dir + '/training'
val_dir = data_dir + '/validation'



# In[3]:


model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# In[4]:


train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(train_dir, transform=train_transform)
val_dataset = ImageFolder(val_dir, transform=val_transform)


# In[5]:


batch_size = 32 #define batch size based on size of dataset

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")


# In[ ]:


print(model)


# In[6]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# In[10]:


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    correct_predictions = 0
    total_predictions = 0

    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1) #getting predictions
        correct_predictions += (predicted == labels).sum().item() #checking if they are the same
        total_predictions += labels.size(0) #size of total predictions

        if (i + 1) % 1 == 0: #change 1 based on how frequently you want to print the batch results
            last_loss = running_loss / 1 #change 1 based on how frequently you want to print the batch results
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    accuracy = correct_predictions / total_predictions
    return last_loss, accuracy


# In[11]:


def evaluate_model(model, validation_loader, loss_fn):
    model.eval()
    running_vloss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()

            _, predicted = torch.max(voutputs, 1)
            correct_predictions += (predicted == vlabels).sum().item()
            total_predictions += vlabels.size(0)

    avg_vloss = running_vloss / len(validation_loader)
    accuracy = correct_predictions / total_predictions
    return avg_vloss, accuracy


# In[12]:


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs_ver_2/InceptProgress_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 10
best_vloss = 1000000

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    avg_loss, train_accuracy = train_one_epoch(epoch_number, writer)
    avg_vloss, val_accuracy = evaluate_model(model, val_loader, loss_fn)

    print(f'LOSS train {avg_loss} valid {avg_vloss}')
    print(f'ACCURACY train {train_accuracy} valid {val_accuracy}')

    writer.add_scalars('Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.add_scalars('Accuracy',
                    { 'Training' : train_accuracy, 'Validation' : val_accuracy },
                    epoch_number + 1)
    writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = '/dcs05/ciprian/smart/pocus/rushil/ultraIncept_ver_2_model.pth'
        torch.save(model.state_dict(), model_path)

    epoch_number += 1


print(f'Best validation loss: {best_vloss}')
print('Training completed.')
