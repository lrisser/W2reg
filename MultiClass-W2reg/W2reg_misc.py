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
    results['s0_TP']=np.round(s0_Pred1_True1/s0_True1 , 3)
    results['s0_FP']=np.round(s0_Pred1_True0/s0_True0 , 3)
    results['s0_TN']=np.round(s0_Pred0_True0/s0_True0 , 3)
    results['s0_FN']=np.round(s0_Pred0_True1/s0_True1 , 3)
    results['s0_RatioGoodPred']=np.round((s0_Pred0_True0+s0_Pred1_True1)/results['s0_total'] , 3)
    results['s1_total']=s1_True0+s1_True1
    results['s1_TP']=np.round(s1_Pred1_True1/s1_True1 , 3)
    results['s1_FP']=np.round(s1_Pred1_True0/s1_True0 , 3)
    results['s1_TN']=np.round(s1_Pred0_True0/s1_True0 , 3)
    results['s1_FN']=np.round(s1_Pred0_True1/s1_True1 , 3)
    results['s1_RatioGoodPred']=np.round((s1_Pred0_True0+s1_Pred1_True1)/results['s1_total'] , 3)

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



def Get_unbalanced_MNIST(NbClassObsInS0=50,NbClassObsInS1=300):
  """
  Get a subset of the MNIST dataset so that:
    -> Each digit 0, 1, 2, 3, 4, 5, 6, 8, 9 is observed NbClassObsInS1 times -> they will be in class S1
    -> Each digit 7 is observed NbClassObsInS0 times                   -> they will be in class S0
  """
  
  #1) get row mnist data
  mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
  mnist_testset  = datasets.MNIST(root='./data', train=False, download=True, transform=None)
  
  X_train=mnist_trainset.data      #60000 observations of size 28*28 -- torch.uint8
  y_train=mnist_trainset.targets   #60000 observations in 1D -- torch.int64

  X_test=mnist_testset.data      #10000 observations of size 28*28 -- torch.uint8
  y_test=mnist_testset.targets   #10000 observations in 1D -- torch.int64


  #2) select a subset of the training data. They will be the observations for which S=0. 

  #2.1) select the observations
  Y0=torch.where(mnist_trainset.targets==0)[0]
  Y0_rselect=Y0[torch.randperm(Y0.shape[0])[0:NbClassObsInS1]]
  
  Y1=torch.where(mnist_trainset.targets==1)[0]
  Y1_rselect=Y1[torch.randperm(Y1.shape[0])[0:NbClassObsInS1]]

  Y2=torch.where(mnist_trainset.targets==2)[0]
  Y2_rselect=Y2[torch.randperm(Y2.shape[0])[0:NbClassObsInS1]]
  
  Y3=torch.where(mnist_trainset.targets==3)[0]
  Y3_rselect=Y3[torch.randperm(Y3.shape[0])[0:NbClassObsInS1]]
  
  Y4=torch.where(mnist_trainset.targets==4)[0]
  Y4_rselect=Y4[torch.randperm(Y4.shape[0])[0:NbClassObsInS1]]
  
  Y5=torch.where(mnist_trainset.targets==5)[0]
  Y5_rselect=Y5[torch.randperm(Y5.shape[0])[0:NbClassObsInS1]]
  
  Y6=torch.where(mnist_trainset.targets==6)[0]
  Y6_rselect=Y6[torch.randperm(Y6.shape[0])[0:NbClassObsInS1]]
  
  Y7=torch.where(mnist_trainset.targets==7)[0]
  Y7_rselect=Y7[torch.randperm(Y7.shape[0])[0:NbClassObsInS0]]
  
  Y8=torch.where(mnist_trainset.targets==8)[0]
  Y8_rselect=Y8[torch.randperm(Y8.shape[0])[0:NbClassObsInS1]]
  
  Y9=torch.where(mnist_trainset.targets==9)[0]
  Y9_rselect=Y9[torch.randperm(Y9.shape[0])[0:NbClassObsInS1]]
  
  #2.2) merge the observations
  obsS1=torch.cat([Y0_rselect,Y1_rselect,Y2_rselect,Y3_rselect,Y4_rselect,Y5_rselect,Y6_rselect,Y8_rselect,Y9_rselect],axis=0)
  obsS0=torch.cat([Y7_rselect],axis=0)
  
  obsAll=torch.cat([obsS0,obsS1],axis=0)
  
  X_train=X_train[obsAll,:,:]
  y_train=y_train[obsAll]   #60000 observations in 1D -- torch.int64
  
  #2.3) generate corresponding S_train
  S_train=np.zeros(obsS0.shape[0]+obsS1.shape[0]).astype(np.int32)
  S_train[obsS0.shape[0]:]=1
  
  #2.4) shuffle the observations
  shuffledIDs=torch.randperm(S_train.shape[0])
  
  X_train=X_train[shuffledIDs,:,:]
  y_train=y_train[shuffledIDs]   #60000 observations in 1D -- torch.int64
  S_train=S_train[shuffledIDs]
  
  #3) define the S in the test set
  Y7_tst=torch.where(y_test==7)[0]
  
  S_test=np.ones(y_test.shape[0]).astype(np.int32)
  S_test[Y7_tst]=0
  
  #4) set the tensors to standard dimensions
  X_train=X_train.view(-1,1,28,28) #add a dimension for the channels
  X_test=X_test.view(-1,1,28,28) #add a dimension for the channels
  #y_train=y_train.view(-1,1)
  #y_test=y_test.view(-1,1)
  
  
  y_train_oh = torch.nn.functional.one_hot(y_train.to(torch.int64), 10)
  y_test_oh = torch.nn.functional.one_hot(y_test.to(torch.int64), 10)


  return [X_train,y_train_oh,S_train,X_test,y_test_oh,S_test]


def show_MNIST_image(LodID,X,Y,S):
  LocImage=(X[LodID,0,:,:])
  if len(Y.shape)==1:
    LocTitle='Y='+str(int(Y[LodID]))+' / S='+str(S[LodID])
  else:
    LocTitle='Y='+str(Y[LodID,:])+' / S='+str(S[LodID])
  plt.figure()
  plt.imshow(LocImage)
  plt.title(LocTitle)
  plt.show()




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#neural network model
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ResNet_18_for_MNIST_srt(output_size=1):
  """
    Load a pretrained ResNet-18 classification model from PyTorch with a specific last dense layer and one input channel.
    """
  # Load a pretrained ResNet-18 model thanks to PyTorch
  resnet_model = models.resnet18(pretrained = True)
  
  #change the first layer as we only have one input channel
  resnet_model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  
  # Change the last dense layer to fit our problematic needs
  if output_size==1:
    resnet_model.fc = nn.Sequential(OrderedDict([
      ('fc1', nn.Linear(512, 128)), ('relu', nn.ReLU()),
      ('fc2', nn.Linear(128, 1)), ('output', nn.Sigmoid())
    ]))
  else:
    resnet_model.fc = nn.Sequential(OrderedDict([
      ('fc1', nn.Linear(512, 128)), ('relu', nn.ReLU()),
      ('fc2', nn.Linear(128, output_size)), ('output', nn.Softmax(dim=1))
    ]))
    
  
  return resnet_model



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#3) functions to make predictions on large datasets
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def LargeDatasetPred(model,var_X,BlockSizes,DEVICE='cpu'):
  n_loc=var_X.shape[0]
  
  loc_miniBatch_Start=0
  
  while loc_miniBatch_Start<n_loc:
    #define the mini-batch domain
    loc_miniBatch_End=loc_miniBatch_Start+BlockSizes
    if loc_miniBatch_End >= n_loc:
      loc_miniBatch_End = n_loc
    
    #local prediction
    with torch.no_grad():
      minibatch=var_X[loc_miniBatch_Start:loc_miniBatch_End,:,:,:].to(DEVICE)
      loc_predY=model(minibatch)
      loc_predY=loc_predY.to('cpu')
    
    #merge local prediction with former ones
    if loc_miniBatch_Start==0:
      all_predY=torch.clone(loc_predY)
    else:
      all_predY=torch.cat([all_predY,loc_predY],dim=0)
    
    #increment loc_miniBatch_Start
    loc_miniBatch_Start+=BlockSizes
  
  return all_predY


def LargeDatasetPred_nlp(model,var_X,var_mask,BlockSizes,DEVICE='cpu'):
  n_loc=var_X.shape[0]
  
  loc_miniBatch_Start=0
  
  while loc_miniBatch_Start<n_loc:
    print(loc_miniBatch_Start,' -- ',n_loc)
    
    #define the mini-batch domain
    loc_miniBatch_End=loc_miniBatch_Start+BlockSizes
    if loc_miniBatch_End >= n_loc:
      loc_miniBatch_End = n_loc
    
    #local prediction
    with torch.no_grad():
      minibatch_X=var_X[loc_miniBatch_Start:loc_miniBatch_End,:].to(DEVICE)
      minibatch_mask=var_mask[loc_miniBatch_Start:loc_miniBatch_End,:].to(DEVICE)
      loc_predY=model(ids=minibatch_X, mask=minibatch_mask)
      loc_predY=loc_predY.to('cpu')
    
    #merge local prediction with former ones
    if loc_miniBatch_Start==0:
      all_predY=torch.clone(loc_predY)
    else:
      all_predY=torch.cat([all_predY,loc_predY],dim=0)
    
    #increment loc_miniBatch_Start
    loc_miniBatch_Start+=BlockSizes
  
  return all_predY


