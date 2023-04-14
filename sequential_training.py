import os
import numpy as np
import torch
from torch import nn
from torch import optim 
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import models
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder

res_18_model = models.resnet18(weights='DEFAULT')
# res_18_model.load_state_dict(torch.utils.model_zoo.load_url("https://download.pytorch.org/models/resnet18-5c106cde.pth"))

T = transforms.Compose([
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageFolder('/Users/sarveshphoenix/Downloads/animals10/raw-img/', transform=T)
train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*.8), len(dataset)-int(len(dataset)*.8)])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
test_loader = torch.utils.data.DataLoader(val_set, batch_size=64)

res_18_model.fc= nn.Linear(512, 10)

model = res_18_model

if(torch.cuda.is_available()==True):
    model=res_18_model.cuda()
    
optimiser=optim.SGD(model.parameters(),lr=1e-2)
loss=nn.CrossEntropyLoss()

nb_epochs = 5
acc_tot=np.zeros(nb_epochs)
for epoch in range(nb_epochs):
    losses = list()
    accuracies = list()
    model.train()     
    for batch in train_loader: 

        x,y = batch
        if(torch.cuda.is_available()==True):
            x=x.cuda()
            y=y.cuda()        


        # 1 forward
        l = model(x) # l: logits

        #2 compute the objective function
        J = loss(l,y)

        # 3 cleaning the gradients
        model.zero_grad()
        # optimiser.zero_grad()
        # params.grad.zero_()

        # 4 accumulate the partial derivatives of J wrt params
        J.backward()

        # 5 step in the opposite direction of the gradient
        optimiser.step()

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch + 1}', end=', ')
    print(f'training loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'training accuracy: {torch.tensor(accuracies).mean():.2f}')


    losses = list()
    accuracies = list() 
    model.eval()
    for batch in test_loader: 
        x,y = batch
        if(torch.cuda.is_available()==True):
            x=x.cuda()
            y=y.cuda()

        with torch.no_grad(): 
            l = model(x)

        #2 compute the objective function
        J = loss(l,y)

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch + 1}',end=', ')
    print(f'validation loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'validation accuracy: {torch.tensor(accuracies).mean():.2f}')
    acc_tot[epoch]=torch.tensor(accuracies).mean().numpy()

def imformat(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return(inp)

class_names = dataset.classes
translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}
t_inv = {v: k for k, v in translate.items()}

train_loader2 = torch.utils.data.DataLoader(train_set, batch_size=9)

plt.figure(figsize=(15, 13))

inputs, classes = next(iter(train_loader2))
preds=model(inputs.cuda()).argmax(dim=1)


for i in range(0,9):
    ax = plt.subplot(3, 3, i + 1)
    img=imformat(inputs[i])
    
    plt.imshow((img))

    try:
        plt.title('True:'+str(t_inv[class_names[classes[i]]])+'    Pred:'+str(t_inv[class_names[preds[i]]]))
    except:
        plt.title('True:'+str(translate[class_names[classes[i]]])+'    Pred:'+str(translate[class_names[preds[i]]]))
    if(i==9):
        plt.axis("off")