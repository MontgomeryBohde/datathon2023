from data import ImageDataset, load_train_data
import pickle
import matplotlib.pyplot as plt

import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

name = 'regnet_x_3_2gf'

# mean = 0.12000638813804619, std = 0.3186337241490653

train_transform = v2.Compose([
    v2.RandomHorizontalFlip(),
])

val_transform = v2.Compose([
    v2.Normalize((0.12000638813804619,), (0.3186337241490653,))
])

train, val = load_train_data()

# save train.val to pkl
'''with open('train_.pkl', 'wb') as f:
    pickle.dump(train, f)
    
with open('val.pkl', 'wb') as f:
    pickle.dump(val, f)
    
# load train.val from pkl
with open('train.pkl', 'rb') as f:
    train = pickle.load(f)
    
with open('val.pkl', 'rb') as f:
    val = pickle.load(f)'''
    

train_dataset = ImageDataset(train, transform=train_transform)
val_dataset = ImageDataset(val, transform=None)

# save datasets to pkl
'''with open('train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)
    
with open('val_dataset.pkl', 'wb') as f:
    pickle.dump(val_dataset, f)
    
# load datasets from pkl
with open('train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
    
with open('val_dataset.pkl', 'rb') as f:
    val_dataset = pickle.load(f)'''
    
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=6)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=6)

#plot some images
'''for i in range(100):
    idx = np.random.randint(0, len(train_dataset))
    data, category = train_dataset[idx]
    plt.imshow(data[0])
    plt.title(category)
    plt.show()
    plt.savefig(f'img_sample/{i}.png')'''


# build a model using a pretrained resnet architecture
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if name == 'efficientnet_v2_s':
    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(512, 15)
elif name == 'convnext_tiny':
    model = models.convnext_tiny(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(512, 15)
elif name == 'regnet_x_3_2gf':
    model = models.regnet_x_3_2gf(weights='RegNet_X_3_2GF_Weights.IMAGENET1K_V2')
    model.fc = nn.Linear(1008, 15)
elif name == 'resnext101_64x4d':
    model = models.resnext101_64x4d(weights='ResNeXt101_64X4D_Weights.IMAGENET1K_V1')
    model.fc = nn.Linear(2048, 15)
else:
    raise ValueError('Invalid model name')

model = model.to(device)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# train the model
def train_model(model, criterion, optimizer, num_epochs=25):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        
        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # set model to training mode
                dataloader = train_dataloader
            else:
                model.eval() # set model to evaluate mode
                dataloader = val_dataloader
                
            running_loss = 0.0
            running_corrects = 0
            
            # iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, torch.max(labels, 1)[1])
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == torch.max(labels, 1)[1])
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                # save model
                torch.save(model.state_dict(), f'models/{name}_{epoch}.pth')
                torch.save(model.fc.state_dict(), f'models/{name}_fc_{epoch}.pth')
                #pickle model
                with open(f'models/{name}_{epoch}.pkl', 'wb') as f:
                    pickle.dump(model, f)
                
    return model, train_loss, val_loss, train_acc, val_acc

print(name)
model, train_loss, val_loss, train_acc, val_acc = train_model(model, criterion, optimizer, num_epochs=25)

# save model
#pkl model
with open(f'models/{name}_final.pth', 'wb') as f:
    pickle.dump(model, f)
torch.save(model.state_dict(), f'models/{name}_final.pth')
#torch.save(model.state_dict(), 'ConvNeXt_Tiny_Weights.pth')

