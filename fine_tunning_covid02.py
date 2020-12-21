#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[2]:


from __future__ import print_function 
from __future__ import division
import itertools
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchxrayvision as xrv
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score
import datetime
import time
import os
import copy
import sys
import dataset

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# In[3]:


# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
data_dir = "/home/users/ester/datasets/covidx"

path_train = '/home/users/ester/datasets/covidx/train'
path_val = '/home/users/ester/datasets/covidx/test'

list_img_train, lbl_train = dataset.get_dataset_info(path_train)
list_img_val, lbl_val = dataset.get_dataset_info(path_val)

# creating tensorboar file
current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")

writer = SummaryWriter('runs/fine_tunning_covid02/' + current_time)

# selecting gpu device
device = torch.device("cuda:0") # if torch.cuda.is_available() else "cpu") 

model_name= ""

# Number of classes in the dataset
num_classes = 3

class_labels = ['covid', 'normal', 'pneumonia']

# classes weights
weights = [13906/476, 13906/7966, 13906/5464]

# Batch size for training (change depending on how much memory you have)
batch_size = 30

# Number of epochs to train for 
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = False


# In[4]:


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            labels_list = []
            pred_list = []
            
            

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
            
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    
                   

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                labels_list.extend(labels.cpu().numpy())
                pred_list.extend(preds.cpu().numpy())


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            writer.add_scalar('Loss/{}'.format(phase) , epoch_loss, epoch)
            writer.add_scalar('Accuracy/{}'.format(phase) , epoch_acc, epoch)
            
            
            labels = labels_list
            preds = pred_list

            balanced_accuracy = balanced_accuracy_score(y_true=labels, y_pred=preds)
            precision = precision_score(labels, preds, average=None)
            recall = recall_score(labels, preds, average=None)
            f1 = f1_score(labels, preds, average=None)

            print('{} Balanced Accuracy: {} Precision: {} Recall: {} F1-Score: {}'.format(phase, balanced_accuracy, precision, recall, f1))

            writer.add_scalar('Balanced_Accuracy/{}'.format(phase) , balanced_accuracy, epoch)
            writer.add_scalar('Precision covid/{}'.format(phase) , precision[0], epoch)
            writer.add_scalar('Recall covid/{}'.format(phase) , recall[0], epoch)
            writer.add_scalar('F1 Score covid/{}'.format(phase) , f1[0], epoch)

            writer.add_scalar('Precision normal/{}'.format(phase) , precision[1], epoch)
            writer.add_scalar('Recall normal/{}'.format(phase) , recall[1], epoch)
            writer.add_scalar('F1 Score normal/{}'.format(phase) , f1[1], epoch)

            writer.add_scalar('Precision pneumonia/{}'.format(phase) , precision[2], epoch)
            writer.add_scalar('Recall pneumonia/{}'.format(phase) , recall[2], epoch)
            writer.add_scalar('F1 Score pneumonia/{}'.format(phase) , f1[2], epoch)


#             print(classification_report(labels, preds, target_names=class_labels))

            cm = confusion_matrix(labels, preds)

            cm_fig = plot_confusion_matrix(cm, classes=class_labels, normalize=True, title='{} Confusion Matrix'.format(phase))

            writer.add_figure('Confusion Matrix/'+ phase, cm_fig, epoch)

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    writer.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# In[5]:


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# In[6]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%.2f' % (cm[i, j] * 100.0),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#     plt.show()
    
    return fig


# In[8]:


model_ft = xrv.models.DenseNet(weights="nih")
set_parameter_requires_grad(model_ft, feature_extract)
model_ft.classifier = nn.Linear(1024, num_classes)
model_ft.pathologies = ['normal', "pneumonia", 'COVID-19']
model_ft.op_threshs = None
input_size = 224
# Print the model we just instantiated
# print(model_ft)


# ## Load Data

# In[9]:


# Data augmentation and normalization for training
# Just normalization for validation

std = 0.24671278988052675
mean = 0.4912771402827791

data_transforms = {
    'train': transforms.Compose([
#         transforms.RandomResizedCrop(input_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std]),
          
    ]),
    'test': transforms.Compose([
#         transforms.Resize(input_size),
#         transforms.CenterCrop(input_size),
#         transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std]),
        
        
        
        
    ]),
}
print("Initializing Datasets and Dataloaders...")

# # Create training and validation datasets
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
# # Create training and validation dataloaders
# dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}


tmp_dataset_train = dataset.COVID19_Dataset(list_img_train, lbl_train, transform = data_transforms['train'])
tmp_dataset_val = dataset.COVID19_Dataset(list_img_val, lbl_val, transform = data_transforms['test'])

# Create a sampler by samples weights 
sampler = torch.utils.data.sampler.WeightedRandomSampler(
    weights=tmp_dataset_train.samples_weights,
    num_samples=tmp_dataset_train.len)


dataloaders_dict = {}

dataloaders_dict['train'] = torch.utils.data.DataLoader(tmp_dataset_train, 
                                                    batch_size=batch_size, 
                                                    sampler=sampler,
                                                    num_workers=4)

dataloaders_dict['test'] = torch.utils.data.DataLoader(tmp_dataset_val, 
                                                    batch_size=batch_size, 
                                                    num_workers=4)


# In[10]:


# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=0.001, momentum=0.9)


# In[11]:


# %tensorboard --logdir=runs --host 0.0.0.0


# In[ ]:


# weights = torch.Tensor(weights).to(device)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss(weight=dataloaders_dict['train'].dataset.weight_class.to(device))

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


# In[ ]:


# torch.save(model_ft.state_dict(), "covid/xrayvision_ft.pt")

