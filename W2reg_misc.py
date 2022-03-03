import time
import pickle
import pandas
import numpy as np # to handle matrix and data operation
import matplotlib.pyplot as plt   #image visualisation
import scipy.stats as st


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.models as models
from collections import OrderedDict

import torchvision
import torchvision.datasets as datasets

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#statistics
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def compute_disparate_impact(pred_y, sensitivities):
    number_of_sensible = 0; number_of_regular = 0
    success_sensible = 0.001; success_regular = 0.001
    for i in range(len(pred_y)):
      if sensitivities[i] == 0.:
        number_of_sensible += 1
        if pred_y[i] == 1.:
          success_sensible += 1
      else:
        number_of_regular += 1
        if pred_y[i] == 1.:
          success_regular += 1
    proba_success_regular = success_regular / number_of_regular
    proba_success_sensible = success_sensible / number_of_sensible
    disparate_impact = proba_success_sensible/proba_success_regular
    return disparate_impact


def cpt_BasicDescrStats(pred_y, true_Y, sensitivities):
    s0_Pred0_True0=0.
    s0_Pred0_True1=0.
    s0_Pred1_True0=0.
    s0_Pred1_True1=0.
    s0_True0=0.
    s0_True1=0.
    s1_Pred0_True0=0.
    s1_Pred0_True1=0.
    s1_Pred1_True0=0.
    s1_Pred1_True1=0.
    s1_True0=0.
    s1_True1=0.
    
    #compute the nb of occurrences in each class
    for i in range(len(pred_y)):
      if sensitivities[i]<0.5:
        if pred_y[i] < 0.5:
          if true_Y[i] < 0.5:
            s0_Pred0_True0+=1.
            s0_True0 += 1
          else:
            s0_Pred0_True1+=1.
            s0_True1 += 1
        else: #pred_y[i] > 0.5:
          if true_Y[i] < 0.5:
            s0_Pred1_True0+=1.
            s0_True0 += 1
          else:
            s0_Pred1_True1+=1.
            s0_True1 += 1
      else: #sensitivities[i]>0.5:
        if pred_y[i] < 0.5:
          if true_Y[i] < 0.5:
            s1_Pred0_True0+=1.
            s1_True0 += 1
          else:
            s1_Pred0_True1+=1.
            s1_True1 += 1
        else: #pred_y[i] > 0.5:
          if true_Y[i] < 0.5:
            s1_Pred1_True0+=1.
            s1_True0 += 1
          else:
            s1_Pred1_True1+=1.
            s1_True1+=1
    
    #generate a dictionary with the results
    results={}
    results['s0_total']=s0_True0+s0_True1
    results['s0_TP']=s0_Pred1_True1/s0_True1
    results['s0_FP']=s0_Pred1_True0/s0_True0
    results['s0_TN']=s0_Pred0_True0/s0_True0
    results['s0_FN']=s0_Pred0_True1/s0_True1
    results['s0_RatioGoodPred']=(s0_Pred0_True0+s0_Pred1_True1)/results['s0_total']
    results['s1_total']=s1_True0+s1_True1
    results['s1_TP']=s1_Pred1_True1/s1_True1
    results['s1_FP']=s1_Pred1_True0/s1_True0
    results['s1_TN']=s1_Pred0_True0/s1_True0
    results['s1_FN']=s1_Pred0_True1/s1_True1
    results['s1_RatioGoodPred']=(s1_Pred0_True0+s1_Pred1_True1)/results['s1_total']

    return results





#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Load and pre-treat the MNIST dataset
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Get_n_Treat_MNIST_srt():
  """
  Get the MNIST dataset and treat it as in the JMIV paper (semi-random treatment)
  
  * The outputs represent whether the handritten digit is higher or strictly lower 
    than 5, i.e.:
      -> Y=0 for the digits 0, 1, 2, 3, 4 
      -> Y=1 for the digits 5, 6, 7, 8, 9 
  
  * A label S=0 or S=1 is randomly drawn for each observtation. If S=0 then the 
    image is rotated.
    
  * Y=0 for about 2/3rd of the handwritten sevens with S=0, which mimics a 
    semi-random discrimination
  """
  
  #1) get the data
  mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
  mnist_testset  = datasets.MNIST(root='./data', train=False, download=True, transform=None)
  
  X_train=mnist_trainset.data      #60000 observations of size 28*28 -- torch.uint8
  y_train=mnist_trainset.targets   #60000 observations in 1D -- torch.int64
  S_train=np.zeros(60000).astype(np.int)
  
  X_test=mnist_testset.data      #10000 observations of size 28*28 -- torch.uint8
  y_test=mnist_testset.targets   #10000 observations in 1D -- torch.int64
  S_test=np.zeros(10000).astype(np.int)
  
  
  #2) define the Y and S values  -- training data with semi-random treatment
  for i in range(X_train.shape[0]):
    if y_train[i]<5:   #Y=0 ...
      #... treat S
      S_train[i]=np.random.randint(0,2)
      if S_train[i]==0:
        toto=torch.rot90(X_train[i,:,:],1,[0,1])
        X_train[i,:,:]=toto  #rotation 
      
      #... treat Y
      y_train[i]=0
    
    else:             #Y=1...
      #... treat S
      S_train[i]=np.random.randint(0,2)
      if S_train[i]==0:
        toto=torch.rot90(X_train[i,:,:],1,[0,1])
        X_train[i,:,:]=toto  #rotation
      
      #... treat Y
      if y_train[i]==7 and S_train[i]==0 and  np.random.randint(0,3)!=2:   #swap Y in the training set for 2/3 of the handwritten 7 with S=0
        y_train[i]=0
      else:
        y_train[i]=1
  
  
  #3) define the Y and S values  -- test data without semi-random treatment
  for i in range(X_test.shape[0]):
    if y_test[i]<5:   #Y=0 ...
      #... treat S
      S_test[i]=np.random.randint(0,2)
      if S_test[i]==0:
        toto=torch.rot90(X_test[i,:,:],1,[0,1])
        X_test[i,:,:]=toto  #rotation
      
      #... treat Y
      y_test[i]=0
    
    else:             #Y=1...
      #... treat S
      S_test[i]=np.random.randint(0,2)
      if S_test[i]==0:
        toto=torch.rot90(X_test[i,:,:],1,[0,1])
        X_test[i,:,:]=toto  #rotation
      
      #... treat Y
      y_test[i]=1
  
  #4) set the tensors to standard dimensions
  X_train=X_train.view(-1,1,28,28) #add a dimension for the channels
  X_test=X_test.view(-1,1,28,28) #add a dimension for the channels
  y_train=y_train.view(-1,1)
  y_test=y_test.view(-1,1)

  return [X_train,y_train,S_train,X_test,y_test,S_test]





def show_MNIST_image(LodID,X,Y,S):
  LocImage=(X[LodID,0,:,:])
  LocTitle='Y='+str(int(Y[LodID]))+' / S='+str(S[LodID])
  plt.figure()
  plt.imshow(LocImage)
  plt.title(LocTitle)
  plt.show()




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#neural network model
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ResNet_18_for_MNIST_srt():
  """
    Load a pretrained ResNet-18 model from PyTorch with a specific last dense layer and one input channel.
    """
  # Load a pretrained ResNet-18 model thanks to PyTorch
  resnet_model = models.resnet18(pretrained = True)
  
  #change the first layer as we only have one input channel
  resnet_model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  
  # Change the last dense layer to fit our problematic needs
  resnet_model.fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(512, 128)), ('relu', nn.ReLU()),
    ('fc2', nn.Linear(128, 1)), ('output', nn.Sigmoid())
  ]))
  
  return resnet_model

