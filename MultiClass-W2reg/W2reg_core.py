
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.models as models
from collections import OrderedDict

import time
import pickle
import pandas
import numpy as np # to handle matrix and data operation
import matplotlib.pyplot as plt   #image visualisation
import scipy.stats as st




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#1) function to estimate the gradients
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def EstimGrad_W2dist(minibatch_S,minibatch_y_pred,minibatch_y_true,obs4histo_S,obs4histo_y_pred,obs4histo_y_true, NbBins=500,ID_TreatedVar=0,DistBetween='Predictions_errors'):
    """
    Estimate the gradient of the Wasserstein Distance between the histograms of the values of minibatch_y_pred for which minibatch_S=0 and minibatch_S=1
    -> Notations:

      --> minibatch_S      is a numpy array representing the sensitive variable. S=0 or S=1 for the compared groups. Other values for S will be skipped in the comparison.
      --> minibatch_y_true is a pytorch tensor representing the (true) selection variable. Y=0 if fail     / Y=1 if success
      --> minibatch_y_pred is a pytorch tensor representing the estimated probability the minibatch_y_true==1
      --> obs4histo_S, obs4histo_y_true, obs4histo_y_pred: same as the minibatch_* variables but to compute the histograms only (should contain more observations but may be updated less often)

      --> ID_TreatedVar: If there are multiple outputs, ID_TreatedVar is the index of the output vector on which the regularization will be performed

      --> DistBetween: data used to compute the cumulative histograms. Can be:
            * 'Predictions' -> all data are considered. The regularizer favors similar distribution of predictions, typically low disparate impacts.
            * 'Predictions_errors' -> all data are used but the distance between the error rates and its gradients are computed
    -> Return:
      --> A list of the gradients for the points defined in minibatch_y_pred and  minibatch_y_true
    """



    #1) init

    #1.1) input conversions (for the minibatch data)
    y_pred=minibatch_y_pred.detach().numpy()
    y_true=minibatch_y_true.detach().numpy()

    y_pred_c=(y_pred[:,ID_TreatedVar]*1.).ravel()   #column of interest only
    y_true_c=(y_true[:,ID_TreatedVar]*1.).ravel()   #column of interest only

    S_mb=minibatch_S.ravel()

    #1.2) input conversions (for the obs4histo data)
    y_pred_4histo=obs4histo_y_pred.detach().numpy()
    y_true_4histo=obs4histo_y_true.detach().numpy()

    y_pred_4histo=(y_pred_4histo[:,ID_TreatedVar]*1.).ravel()
    y_true_4histo=(y_true_4histo[:,ID_TreatedVar]*1.).ravel()

    S_4histo=obs4histo_S.ravel()


    #2) compute the cumulative distribution functions
    if DistBetween=='Predictions_errors':
        #2.1) pred err -- split the observations w.r.t. the value of S_4histo in {0,1}   (for the obs4histo data)
        tmpZip=zip((y_true_4histo-y_pred_4histo)*(y_true_4histo-y_pred_4histo),S_4histo)
        zipped_y_true_S_eq_1=list(filter(lambda x: x[1] == 1 , tmpZip))
        err_1_4histo, bidon1 = zip(*zipped_y_true_S_eq_1)
        err_1_4histo=np.array(err_1_4histo)
        n1=err_1_4histo.shape[0]

        tmpZip=zip((y_true_4histo-y_pred_4histo)*(y_true_4histo-y_pred_4histo),S_4histo)
        zipped_y_true_S_eq_0=list(filter(lambda x: x[1] == 0, tmpZip))
        err_0_4histo, bidon1 = zip(*zipped_y_true_S_eq_0)
        err_0_4histo=np.array(err_0_4histo)
        n0=err_0_4histo.shape[0]

        #2.2) pred err -- histogram cpt
        minVal=np.min([err_0_4histo.min(),err_1_4histo.min()])
        maxVal=np.max([err_0_4histo.max(),err_1_4histo.max()])
        cumfreq_S1 = st.cumfreq(err_1_4histo, numbins=NbBins, defaultreallimits=(minVal,maxVal)).cumcount
        cumfreq_S0 = st.cumfreq(err_0_4histo, numbins=NbBins, defaultreallimits=(minVal,maxVal)).cumcount
    else:
        #2.3) pred -- split the observations w.r.t. the value of S_4histo in {0,1}   (for the obs4histo data)
        tmpZip=zip(y_pred_4histo,S_4histo)
        zipped_y_pred_S_eq_1=list(filter(lambda x: x[1] == 1, tmpZip))
        y_pred_S_eq_1_4histo, bidon1 = zip(*zipped_y_pred_S_eq_1)
        y_pred_S_eq_1_4histo=np.array(y_pred_S_eq_1_4histo)
        n1=y_pred_S_eq_1_4histo.shape[0]

        tmpZip=zip(y_pred_4histo,S_4histo)
        zipped_y_pred_S_eq_0=list(filter(lambda x: x[1] == 0, tmpZip))
        y_pred_S_eq_0_4histo, bidon1 = zip(*zipped_y_pred_S_eq_0)
        y_pred_S_eq_0_4histo=np.array(y_pred_S_eq_0_4histo)
        n0=y_pred_S_eq_0_4histo.shape[0]

        #2.4) pred -- histogram cpt
        minVal=np.min([y_pred_S_eq_1_4histo.min(),y_pred_S_eq_0_4histo.min()])
        maxVal=np.max([y_pred_S_eq_1_4histo.max(),y_pred_S_eq_0_4histo.max()])
        cumfreq_S1 = st.cumfreq(y_pred_S_eq_1_4histo, numbins=NbBins, defaultreallimits=(minVal,maxVal)).cumcount
        cumfreq_S0 = st.cumfreq(y_pred_S_eq_0_4histo, numbins=NbBins, defaultreallimits=(minVal,maxVal)).cumcount



    cumfreq_absiss=np.linspace(minVal,maxVal,NbBins)
    cumfreq_S1/=cumfreq_S1[-1]
    cumfreq_S0/=cumfreq_S0[-1]

    eps=0.001/(n0+n1)

    #3) Compute the Wasserstein distance
    W_score=0.
    curId0=0
    curId1=0
    integrationStep=0.01
    for loc_cum_freq in np.arange(integrationStep,1.,integrationStep):  #we know that the cumulative frequencies are in [0,1]
      while cumfreq_S0[curId0]<loc_cum_freq:
        curId0+=1
      diff_p=cumfreq_S0[curId0]-loc_cum_freq
      if curId0==0:
        diff_m=loc_cum_freq # cumfreq_S0[curId0-1] whould be 0
      else:
         diff_m=loc_cum_freq-cumfreq_S0[curId0-1]
      diff_sum=diff_p+diff_m
      diff_p/=diff_sum
      diff_m/=diff_sum
      loc_cum_freq_abs0=cumfreq_absiss[curId0]*diff_m + cumfreq_absiss[curId0-1]*diff_p

      while cumfreq_S1[curId1]<loc_cum_freq:
        curId1+=1
      diff_p=cumfreq_S1[curId1]-loc_cum_freq
      if curId1==0:
        diff_m=loc_cum_freq # cumfreq_S1[curId1-1] whould be 0
      else:
         diff_m=loc_cum_freq-cumfreq_S1[curId1-1]
      diff_sum=diff_p+diff_m
      diff_p/=diff_sum
      diff_m/=diff_sum
      loc_cum_freq_abs1=cumfreq_absiss[curId1]*diff_m + cumfreq_absiss[curId1-1]*diff_p

      W_score+=integrationStep*(loc_cum_freq_abs0-loc_cum_freq_abs1)*(loc_cum_freq_abs0-loc_cum_freq_abs1)



    #4) estimate the gradients of Wasserstein in the batch
    WGradients=np.zeros([y_pred.shape[0],y_pred.shape[1]])

    for curObs in range(y_pred.shape[0]):
        if DistBetween=='Predictions_errors':
          curObs_absiss=(y_pred_c[curObs]-y_true_c[curObs])*(y_pred_c[curObs]-y_true_c[curObs])
        else:
          curObs_absiss=y_pred_c[curObs]

        if S_mb[curObs]==1:  #CASE 1 : current observation is in group S=1
          #... majo 1 ... get the index impacted by curObs_absiss
          loc_index_yp=0
          while cumfreq_absiss[loc_index_yp]<curObs_absiss and loc_index_yp<len(cumfreq_absiss)-1:
            loc_index_yp+=1
          loc_index_yp-=1
          if loc_index_yp<0:
            loc_index_yp=0

          #... majo 2 ... linear interpolation to make this estimate finer
          if loc_index_yp==0:
            H_ref_loc=cumfreq_S1[loc_index_yp]   #here 'ref'=1 and 'other'=0
          else:
            distPlus=cumfreq_absiss[loc_index_yp+1]-curObs_absiss
            distMinus=curObs_absiss-cumfreq_absiss[loc_index_yp]
            distTot=distMinus+distPlus
            distPlus/=distTot
            distMinus/=distTot
            H_ref_loc=(distMinus*cumfreq_S1[loc_index_yp+1])+(distPlus*cumfreq_S1[loc_index_yp])   #here 'ref'=1 and 'other'=0

          #... majo 3 ... find the corresponding index in the other cumulative distribution
          loc_index=0
          while cumfreq_S0[loc_index]<H_ref_loc and loc_index<len(cumfreq_S0)-1:
            loc_index+=1
          loc_index-=1
          if loc_index<0:
            loc_index=0

          #... majo 4 ... linear interpolation to find the correponding value
          if loc_index==0:
            Hinv_other_loc=cumfreq_absiss[loc_index]   #here 'ref'=1 and 'other'=0
          else:
            distPlus=cumfreq_S0[loc_index+1]-H_ref_loc
            distMinus=H_ref_loc-cumfreq_S0[loc_index]
            distTot=distMinus+distPlus
            if distTot>0.:
                distPlus/=distTot
                distMinus/=distTot
                Hinv_other_loc=(distMinus*cumfreq_absiss[loc_index+1])+(distPlus*cumfreq_absiss[loc_index])
            else:
                Hinv_other_loc=cumfreq_absiss[loc_index]

          #... majo 5 ... compute the Wassertein Gradient
          grad_H_ref=eps+cumfreq_S1[loc_index_yp+1]-cumfreq_S1[loc_index_yp]
          WGradients[curObs,ID_TreatedVar]=-2*(Hinv_other_loc-curObs_absiss)/(n1*grad_H_ref)
          print('1',WGradients[curObs,ID_TreatedVar])

        elif S_mb[curObs]==0:    #CASE 2 : current observation is in group S=0
          #... mino 1 ... get the index impacted by curObs_absiss
          loc_index_yp=0
          while cumfreq_absiss[loc_index_yp]<curObs_absiss and loc_index_yp<len(cumfreq_absiss)-1:
            loc_index_yp+=1
          loc_index_yp-=1
          if loc_index_yp<0:
            loc_index_yp=0

          #... mino 2 ... linear interpolation to make this estimate finer
          if loc_index_yp==0:
            H_ref_loc=cumfreq_S0[loc_index_yp]   #here 'ref'=0 and 'other'=1
          else:
            distPlus=cumfreq_absiss[loc_index_yp+1]-curObs_absiss
            distMinus=curObs_absiss-cumfreq_absiss[loc_index_yp]
            distTot=distMinus+distPlus
            distPlus/=distTot
            distMinus/=distTot
            H_ref_loc=(distMinus*cumfreq_S0[loc_index_yp+1])+(distPlus*cumfreq_S0[loc_index_yp])   #here 'ref'=0 and 'other'=1

          #... mino 3 ... find the corresponding index in the other cumulative distribution
          loc_index=0
          while cumfreq_S1[loc_index]<H_ref_loc and loc_index<len(cumfreq_S1)-1:
            loc_index+=1
          loc_index-=1
          if loc_index<0:
            loc_index=0

          #... mino 4 ... linear interpolation to find the correponding value
          if loc_index==0:
            Hinv_other_loc=cumfreq_absiss[loc_index]   #here 'ref'=0 and 'other'=1
          else:
            distPlus=cumfreq_S1[loc_index+1]-H_ref_loc
            distMinus=H_ref_loc-cumfreq_S1[loc_index]
            distTot=distMinus+distPlus
            if distTot>0.:
                distPlus/=distTot
                distMinus/=distTot
                Hinv_other_loc=(distMinus*cumfreq_absiss[loc_index+1])+(distPlus*cumfreq_absiss[loc_index])
            else:
                Hinv_other_loc=cumfreq_absiss[loc_index]

          #... mino 5 ... compute the Wassertein Gradient
          grad_H_ref=eps+cumfreq_S0[loc_index_yp+1]-cumfreq_S0[loc_index_yp]
          WGradients[curObs,ID_TreatedVar]=2*(curObs_absiss-Hinv_other_loc)/(n0*grad_H_ref)
          print('0',WGradients[curObs,ID_TreatedVar])


        else: #CASE 3 : current observation is not concerned by the regularization
          WGradients[curObs,ID_TreatedVar]=0.


        if DistBetween=='Predictions_errors':
          WGradients[curObs,ID_TreatedVar]=2*(y_pred_c[curObs]-y_true_c[curObs])*WGradients[curObs,ID_TreatedVar]

    #5) memory clean-up
    #del y_pred_c ,y_pred, y_true, y_true_c,  y_pred_4histo, y_true_4histo
    #del bidon1, n1, tmpZip, n0, minVal, maxVal, cumfreq_S1, cumfreq_S0, cumfreq_absiss
    #del eps, curId0, curId1, integrationStep, loc_cum_freq_abs0
    #del diff_sum, diff_p, diff_m, loc_cum_freq_abs1, curObs, curObs_absiss
    #del H_ref_loc, Hinv_other_loc, grad_H_ref, loc_index_yp

    #if DistBetween=='Predictions_errors':
    #  del zipped_y_true_S_eq_1, err_1_4histo, zipped_y_true_S_eq_0, err_0_4histo
    #else:
    #  del zipped_y_pred_S_eq_1, y_pred_S_eq_1_4histo, zipped_y_pred_S_eq_0, y_pred_S_eq_0_4histo

    return [WGradients,W_score]


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#2) fairloss class
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class FairLoss(torch.autograd.Function):
    """
    class to manage the loss with a penalty enforcing the fairness.
      -> The similarity term is a standard Mean Squared Error (MSE)
      -> The regularisation term ensures that the Wassertein distance between the distribution of the
         predictions for S=0 and S=1 is the same
      -> Designed for 1D neural-network outputs that represent a probability.

    An important structure to compute the Wasserstein distance and its gradient is 'InfoPenaltyTerm',
    which is given as an input of the 'forward' method. It is a dictionary containing:
      -> InfoPenaltyTerm['mb_S']:  numpy array vector containing the sensitive variables labels in the mini-batch.  Each label is in {0,1}. Other values for S will be skipped in the regularization.
      -> InfoPenaltyTerm['o4h_S']: numpy array vector containing the sensitive variables labels in the observations for the histograms.  Each label is in {0,1}. Make sure that there are no labels different to 0 or 1 in o4h, as they won't be used for the histograms.
      -> InfoPenaltyTerm['o4h_y_pred']: pytorch tensor or vector containing the predicted probabilities that we have label 1 for the histograms. Each probability is in [0,1].
      -> InfoPenaltyTerm['o4h_y_true']: pytorch tensor or vector containing the true selection variable for the histograms. Each label is in {0,1}.
      -> InfoPenaltyTerm['DistBetween']: equal 'Predictions' to regularize the predictions or 'Predictions_errors' to regularize the prediction errors
      -> InfoPenaltyTerm['lambdavar']: weight given to the penalty term
      -> InfoPenaltyTerm['ID_TreatedVar']: is the variable in the columns of the y on which the penalty is computed

    Note that when running 'forward', these values will additionally be saved in 'InfoPenaltyTerm':
      -> InfoPenaltyTerm['E_Reg']: regularization energy  (after being weighted by InfoPenaltyTerm['lambdavar'])


    IMPORTANT REMARKS:
      -> ALL DATA (y_pred, y and tensors of InfoPenaltyTerm) MUST BE IN THE CPU MEMORY
      -> The labels S are defined so that the W2 regularisation applies on the group of observations with labels 0 and 1. Observations with other labels won't be considered when regularizing.
    """

    @staticmethod
    def forward(ctx,  y_pred, y, InfoPenaltyTerm):
        """
        * y_pred:  pytorch-tensor vector containing the predicted probabilities that we have label 1 in the mini-bath. Each probability is in [0,1].
        * y is the true y  -> pytorch-tensor vector containing the true selection variable in the mini-bath. Each label is in {0,1}.
        * InfoPenaltyTerm is the dictionary that contains the pertinent information to compute the regularization term and its gradient
        """

        #1) reshape y if not formated as y_pred, before saving it in the context
        if y.dim()==1:   #check if 1D ref outputs
           y=y.view(-1,1)

        if y.size()!=y_pred.size():
          print('Outputs and true predictions have a different size... the code should crash soon!')


        #2) compute the W2 penalty information and save it for the backward method
        [W_Gradients,W_score]=EstimGrad_W2dist(InfoPenaltyTerm['mb_S'],y_pred,y,InfoPenaltyTerm['o4h_S'],InfoPenaltyTerm['o4h_y_pred'],InfoPenaltyTerm['o4h_y_true'], NbBins=500,ID_TreatedVar=InfoPenaltyTerm['ID_TreatedVar'],DistBetween=InfoPenaltyTerm['DistBetween']) #DistBetween = 'Predictions' or 'Predictions_errors'

        W_score_pt=torch.tensor(InfoPenaltyTerm['lambdavar']*W_score)
        ctx.W_Gradients=torch.tensor((InfoPenaltyTerm['lambdavar']*W_Gradients).astype(np.float32))

        #3) save the energies
        InfoPenaltyTerm['E_Reg']=W_score_pt.item()

        return W_score_pt

    @staticmethod
    def backward(ctx, grad_output):
        """
        Requires the information saved in the forward function:
          ctx.W_Gradients -> gradient of the wasserstein regularization term
        """

        grad_input = ctx.W_Gradients
        del ctx.W_Gradients

        return grad_input, None, None  #second None added because of the context information in forward


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#3) fit functions
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


"""
X_train=X_train
y_train=y_train
S_train=S_train
lambdavar=0.00001
f_loss_attach=nn.MSELoss()
EPOCHS = 1
BATCH_SIZE = 32
obs_for_histo=64
DistBetween='Predictions'
DEVICE='cuda'
optim_lr=0.0001
reduced_epoch_size=-1
"""


def W2R_fit(model,X_train,y_train, S_train, lambdavar, f_loss_attach=nn.MSELoss() , EPOCHS = 5, BATCH_SIZE = 32,obs_for_histo=1000,DistBetween='Predictions',DEVICE='cpu',ID_TreatedVar=0,optim_lr=0.0001,reduced_epoch_size=-1,test_data={'known':False}):
    """
    -> X_train: input pytorch tensor are supposed to have a shape structured as [NbObs,:], [NbObs,:,:], or [NbObs,:,:,:].
    -> y_train: true output pytorch tensor. Supposed to be 2D (for one-hot encoding or others), or 1D (eg for binary classification)
    -> S_train : 1D numpy array containing the sensitive variables labels in the mini-batch.  Each label is in {0,1}.
    -> DistBetween can be : 'Predictions_errors' or 'Predictions'
    -> f_loss_attach: Data attachment term in the loss. Can be eg  nn.MSELoss(), nn.BCELoss(), ... . Must be coherent with the model outputs and the true outputs.
    -> ID_TreatedVar: variable in the columns of the y on which the penalty is computed
    -> test_data: dictionnary that eventually contains test data to evaluate the convergence. If test_data['known']==True, then test_data['X_test'], test_data['y_test'] and test_data['S_test'] must be filled.
    """

    inputdatadim=len(X_train.shape)-1  #dimension of the input data (-1 is to take into account the fact that the first dimension corresponds to the observations)
    outputdatadim=len(y_train.shape)-1  #dimension of the output data (-1 is to take into account the fact that the first dimension corresponds to the observations)

    optimizer = torch.optim.Adam(model.parameters(),lr=optim_lr) #,lr=0.001, betas=(0.9,0.999))

    f_loss_regula = FairLoss.apply      #fair loss

    model.train()

    if lambdavar==0.:
      lambdavar=0.000000001
      print("lambdavar was set to a negligible value to avoid divisions by zero")

    n=X_train.shape[0]
    if reduced_epoch_size<0:
      reduced_epoch_size=n

    IDs_S_eq_0=np.where(S_train<0.5)[0]
    IDs_S_eq_1=np.where(S_train>=0.5)[0]


    Lists_Results={}
    Lists_Results['Loss']=[]          #loss of the data attachement term
    Lists_Results['W2']=[]            #W2 at the end of each mini-batch
    Lists_Results['Loss_S0_train']=[]
    Lists_Results['Loss_S1_train']=[]
    Lists_Results['Loss_S0_test']=[]
    Lists_Results['Loss_S1_test']=[]


    epoch=0
    while epoch<EPOCHS:
        obsIDs=np.arange(X_train.shape[0])
        np.random.shuffle(obsIDs)

        batch_start=0
        batchNb=0

        while batch_start+BATCH_SIZE < n and batch_start+BATCH_SIZE < reduced_epoch_size:

            #1) additional predictions to those of the mini-batch, in order to properly compute the Wasserstein histograms (may be computed at some iterations only)
            #1.1) get current observation IDs for S_train=0
            if len(IDs_S_eq_0)<obs_for_histo:
                obsIDs_4histo_S0=IDs_S_eq_0.copy()
            else:
                np.random.shuffle(IDs_S_eq_0)
                obsIDs_4histo_S0=IDs_S_eq_0[0:obs_for_histo]

            #1.2) get current observation IDs for S_train=1
            if len(IDs_S_eq_1)<obs_for_histo:
                obsIDs_4histo_S1=IDs_S_eq_1.copy()
            else:
                np.random.shuffle(IDs_S_eq_1)
                obsIDs_4histo_S1=IDs_S_eq_1[0:obs_for_histo]

            #1.3) merge the observation IDs
            obsIDs_4histo=np.concatenate([obsIDs_4histo_S0,obsIDs_4histo_S1])


            #1.4) S_train, X, mask and y_true for the histograms
            S_4histo=S_train[obsIDs_4histo]

            if inputdatadim==1:
              X_4histo = X_train[obsIDs_4histo,:].float().to(DEVICE)
            elif inputdatadim==2:
              X_4histo = X_train[obsIDs_4histo,:,:].float().to(DEVICE)
            else:
              X_4histo = X_train[obsIDs_4histo,:,:,:].float().to(DEVICE)

            if outputdatadim==0:
              y_4histo = y_train[obsIDs_4histo].view(-1,1).float()  #as the outputs are in a 1d vector
            elif outputdatadim==1:
              y_4histo = y_train[obsIDs_4histo,:].float()

            #1.5) prediction
            with torch.no_grad():
                y_pred_4histo = model(X_4histo)

            #2) mini-batch predictions

            #2.1) get the observation IDs
            Curr_obsIDs=obsIDs[batch_start:batch_start+BATCH_SIZE]

            #2.2) S, X, Mask, y_true
            S_batch=S_train[Curr_obsIDs]

            if inputdatadim==1:
              X_batch = X_train[Curr_obsIDs,:].float().to(DEVICE)
            elif inputdatadim==2:
              X_batch = X_train[Curr_obsIDs,:,:].float().to(DEVICE)
            else:
              X_batch = X_train[Curr_obsIDs,:,:,:].float().to(DEVICE)

            if outputdatadim==0:
              y_batch = y_train[Curr_obsIDs].view(-1,1).float()  #as the outputs are in a 1d vector
            elif outputdatadim==1:
              y_batch = y_train[Curr_obsIDs,:].float()


            #2.3) set the NN gradient to zero
            optimizer.zero_grad()

            #2.4) mini-batch prediction
            output = model(X_batch)

            #3) compute the attachement term loss
            loss_attach=f_loss_attach(output, y_batch.to(DEVICE))


            #4) prepare and compute the W2 term loss
            #4.1) concatenate the histogram information with those of the mini-batch
            S_4histo_merged=np.concatenate([S_4histo,S_batch],axis=0)
            y_4histo_merged=torch.cat([y_4histo,y_batch],dim=0)
            y_pred_4histo_merged=torch.cat([y_pred_4histo.detach().to('cpu'),output.detach().to('cpu')],dim=0)  # .detach() was added

            #4.2) fill the InfoPenaltyTerm dictionnary
            InfoPenaltyTerm={}                                         #FOR THE W2 REGULARIZATION
            InfoPenaltyTerm['mb_S']=S_batch
            InfoPenaltyTerm['o4h_S']=S_4histo_merged
            InfoPenaltyTerm['o4h_y_pred']=y_pred_4histo_merged
            InfoPenaltyTerm['o4h_y_true']=y_4histo_merged
            InfoPenaltyTerm['DistBetween']=DistBetween    #'Predictions_errors' or 'Predictions'
            InfoPenaltyTerm['lambdavar']=lambdavar
            InfoPenaltyTerm['ID_TreatedVar']=ID_TreatedVar

            #4.3) compute the W2 loss
            #loss_regula=f_loss_regula(output.detach().to('cpu'), y_batch.view(-1,1),InfoPenaltyTerm)  #if used in the cpu -- memory losses otherwise
            if DEVICE=='cpu':
              loss_regula=f_loss_regula(output, y_batch,InfoPenaltyTerm)  #fair loss - no need to copy in the cpu which would induce memory losses (bug in pytorch when copying from cpu to cpu???)
            else:
              loss_regula=f_loss_regula(output.to('cpu'), y_batch,InfoPenaltyTerm)  #fair loss - must be calculated in the CPU but will be detached in the custom regularization function to avoid breaking the NN graph (tested with pytorch 1.3.1)

            #5) compute the whole loss  and perform the gradient descent step
            loss =  loss_attach+loss_regula.to(DEVICE)
            loss.backward()
            optimizer.step()

            #6) update the first observation of the batch
            batch_start+=BATCH_SIZE
            batchNb+=1

            #7) save pertinent information to check the convergence
            locLoss=loss_attach.item()
            locW2=InfoPenaltyTerm['E_Reg']
            Lists_Results['Loss'].append(locLoss)
            Lists_Results['W2'].append(locW2/lambdavar)
            print("epoch "+str(epoch)+" -- batchNb "+str(batchNb)+": Loss="+str(Lists_Results['Loss'][-1])+' -- W2='+str(locW2/lambdavar)+' --  lambda='+str(lambdavar))


        #save advanced information at the end of the epoch...
        #... inforamtion on the training set
        S0=np.where(S_4histo_merged<0.5)
        S1=np.where(S_4histo_merged>0.5)

        loss_attach=f_loss_attach(y_pred_4histo_merged[S0].to('cpu'), y_4histo_merged[S0])
        Lists_Results['Loss_S0_train'].append(loss_attach.item())
        loss_attach=f_loss_attach(y_pred_4histo_merged[S1].to('cpu'), y_4histo_merged[S1])
        Lists_Results['Loss_S1_train'].append(loss_attach.item())

        if test_data['known']==True:
          print("Convergence measured on the test set -> GO")

          #draw the observations
          tst_IDs_S_eq_0=np.where(test_data['S_test']<0.5)[0]
          if len(tst_IDs_S_eq_0)<obs_for_histo:
            obsIDs_4histo_S0=tst_IDs_S_eq_0.copy()
          else:
            np.random.shuffle(tst_IDs_S_eq_0)
            obsIDs_4histo_S0=tst_IDs_S_eq_0[0:obs_for_histo]

          tst_IDs_S_eq_1=np.where(test_data['S_test']>=0.5)[0]
          if len(tst_IDs_S_eq_1)<obs_for_histo:
            obsIDs_4histo_S1=tst_IDs_S_eq_1.copy()
          else:
            np.random.shuffle(tst_IDs_S_eq_1)
            obsIDs_4histo_S1=tst_IDs_S_eq_1[0:obs_for_histo]

          obsIDs_4histo=np.concatenate([obsIDs_4histo_S0,obsIDs_4histo_S1])

          #format the data
          S_4histo=test_data['S_test'][obsIDs_4histo]

          if inputdatadim==1:
            X_4histo = test_data['X_test'][obsIDs_4histo,:].float().to(DEVICE)
          elif inputdatadim==2:
            X_4histo = test_data['X_test'][obsIDs_4histo,:,:].float().to(DEVICE)
          else:
            X_4histo = test_data['X_test'][obsIDs_4histo,:,:,:].float().to(DEVICE)

          if outputdatadim==0:
            y_4histo = test_data['y_test'][obsIDs_4histo].view(-1,1).float()  #as the outputs are in a 1d vector
          elif outputdatadim==1:
            y_4histo = test_data['y_test'][obsIDs_4histo,:].float()

          #prediction
          with torch.no_grad():
              y_pred_4histo = model(X_4histo)

          #information on the training set
          S0=np.where(S_4histo<0.5)
          S1=np.where(S_4histo>0.5)

          loss_attach=f_loss_attach(y_pred_4histo[S0], y_4histo[S0].to(DEVICE))
          Lists_Results['Loss_S0_test'].append(loss_attach.item())
          loss_attach=f_loss_attach(y_pred_4histo[S1], y_4histo[S1].to(DEVICE))
          Lists_Results['Loss_S1_test'].append(loss_attach.item())

          print("Convergence measured on the test set -> DONE")


        #update the epoch number
        epoch+=1

    #memory clean-up
    del Curr_obsIDs, X_batch,  y_batch, output
    del loss_attach
    del loss_regula , loss
    del obsIDs_4histo_S0, obsIDs_4histo_S1, obsIDs_4histo, S_4histo, X_4histo, y_4histo_merged, y_pred_4histo_merged, S_4histo_merged, InfoPenaltyTerm

    model_cpu=model.to('cpu')
    saved_models = { "model": model_cpu }
    pickle.dump( saved_models, open( 'l'+str(lambdavar)+'_saved_model.p', "wb" ) )
    # -> saved_models = pickle.load( open( "saved_model.p", "rb" ) )
    # -> model_cpu=saved_models["model"]
    # -> model=model_cpu.to(DEVICE)


    return Lists_Results


def W2R_fit_NLP(model,X_train,Masks_train,y_train, S_train_in, lambdavar, f_loss_attach=nn.MSELoss() , EPOCHS = 5, BATCH_SIZE = 32,obs_for_histo=1000,DistBetween='Predictions',DEVICE='cpu',ID_TreatedVars=[[0,1.]],optim_lr=0.0001,reduced_epoch_size=-1,test_data={'known':False}):
    """
    -> X_train: input pytorch tensor are supposed to have a shape structured as [NbObs,:], [NbObs,:,:], or [NbObs,:,:,:].
    -> y_train: true output pytorch tensor. Supposed to be 2D (for one-hot encoding or others), or 1D (eg for binary classification)
    -> Masks_train: input pytorch tensor maksing the useless information. Has the same shape as X_train
    -> S_train : 1D numpy array containing the sensitive variables labels in the mini-batch.  Each label is in {0,1}.
    -> DistBetween can be : 'Predictions_errors' or 'Predictions'
    -> f_loss_attach: Data attachment term in the loss. Can be eg  nn.MSELoss(), nn.BCELoss(), ... . Must be coherent with the model outputs and the true outputs.
    -> ID_TreatedVars: list of [[variable1,weight1],...,[variableK,weightK]] in the columns of the y for which the penalty is computed. The weight is a multiplicatory coefficient for lambdavar on the target output variable.
    -> test_data: dictionnary that eventually contains test data to evaluate the convergence. If test_data['known']==True, then test_data['X_test'], test_data['Masks_test'], test_data['y_test'] and test_data['S_test'] must be filled.
    """

    outputdatadim=len(y_train.shape)-1  #dimension of the output data (-1 is to take into account the fact that the first dimension corresponds to the observations)

    optimizer = torch.optim.Adam(model.parameters(),lr=optim_lr) #,lr=0.001, betas=(0.9,0.999))

    model.train()

    if lambdavar==0.:
      lambdavar=0.000000001
      print("lambdavar was set to a negligible value to avoid divisions by zero")

    n=X_train.shape[0]
    if reduced_epoch_size<0:
      reduced_epoch_size=n

    Lst_S_train=[]
    Lst_IDs_S_eq_0=[]
    Lst_IDs_S_eq_1=[]
    
    lst_f_loss_regula=[]

    for VarNbInList in range(len(ID_TreatedVars)):  #generate S and lists of ids with S=0 and S=1 for each output class for which the regularization is performed
        ID_TreatedVar=ID_TreatedVars[VarNbInList][0]

        Lst_S_train.append(np.zeros(len(S_train_in)))
        Lst_S_train[-1][:]=S_train_in[:]

        obs_not_in_TreatedVar=torch.where(y_train[:,ID_TreatedVar].view(-1)==0)[0].numpy()
        Lst_S_train[-1][obs_not_in_TreatedVar]=2                                            #only perform the regularisation on the observations in the class of interest

        Lst_IDs_S_eq_0.append(np.where(Lst_S_train[-1]==0)[0])
        Lst_IDs_S_eq_1.append(np.where(Lst_S_train[-1]==1)[0])

        lst_f_loss_regula.append(FairLoss.apply)      #fair loss for each treated variable
        
        print('Nb obs=',len(Lst_S_train[-1]))
        print('-> Var '+str(ID_TreatedVars[VarNbInList][0])+': Nb obs in group 0 ='+str(len(Lst_IDs_S_eq_0[-1])))
        print('-> Var '+str(ID_TreatedVars[VarNbInList][0])+': Nb obs in group 1 ='+str(len(Lst_IDs_S_eq_1[-1]))) #TO DO: same thing for test when used
        #computed in this loop: Lst_S_train, Lst_IDs_S_eq_0, Lst_IDs_S_eq_1, lst_f_loss_regula


    Lists_Results={}
    Lists_Results['Loss']=[]          #loss of the data attachement term
    Lists_Results['W2']=[]            #W2 at the end of each mini-batch
    Lists_Results['Loss_S0_train']=[]
    Lists_Results['Loss_S1_train']=[]
    Lists_Results['Loss_S0_test']=[]
    Lists_Results['Loss_S1_test']=[]


    epoch=0
    while epoch<EPOCHS:
        obsIDs=np.arange(X_train.shape[0])
        np.random.shuffle(obsIDs)

        batch_start=0
        batchNb=0

        while batch_start+BATCH_SIZE < n and batch_start+BATCH_SIZE < reduced_epoch_size:

            #1) additional predictions to those of the mini-batch, in order to properly compute the Wasserstein histograms (may be computed at some iterations only)
            lst_y_4histo=[]
            lst_y_pred_4histo=[]
            lst_S_4histo=[]

            for VarNbInList in range(len(ID_TreatedVars)):
                #1.1) get current observation IDs for S_train=0
                if len(Lst_IDs_S_eq_0[VarNbInList])<obs_for_histo:
                    obsIDs_4histo_S0=Lst_IDs_S_eq_0[VarNbInList].copy()
                else:
                    np.random.shuffle(Lst_IDs_S_eq_0[VarNbInList])
                    obsIDs_4histo_S0=Lst_IDs_S_eq_0[VarNbInList][0:obs_for_histo]

                #1.2) get current observation IDs for S_train=1
                if len(Lst_IDs_S_eq_1[VarNbInList])<obs_for_histo:
                    obsIDs_4histo_S1=Lst_IDs_S_eq_1[VarNbInList].copy()
                else:
                    np.random.shuffle(Lst_IDs_S_eq_1[VarNbInList])
                    obsIDs_4histo_S1=Lst_IDs_S_eq_1[VarNbInList][0:obs_for_histo]

                #1.3) merge the observation IDs
                obsIDs_4histo=np.concatenate([obsIDs_4histo_S0,obsIDs_4histo_S1])


                #1.4) S_train, X, mask and y_true for the histograms
                lst_S_4histo.append(Lst_S_train[VarNbInList][obsIDs_4histo])

                X_4histo = X_train[obsIDs_4histo,:].long().to(DEVICE)
                Masks_4histo = Masks_train[obsIDs_4histo,:].to(DEVICE)
                
                if outputdatadim==0:
                    y_4histo = y_train[obsIDs_4histo].view(-1,1).float()  #as the outputs are in a 1d vector
                elif outputdatadim==1:
                    y_4histo = y_train[obsIDs_4histo,:].float()

                #1.5) prediction
                with torch.no_grad():
                    y_pred_4histo = model(ids=X_4histo , mask=Masks_4histo)

                lst_y_4histo.append(y_4histo)
                lst_y_pred_4histo.append(y_pred_4histo)
                #computed in this loop: lst_y_4histo, lst_y_pred_4histo, lst_S_4histo


            #2) mini-batch predictions

            #2.1) get the observation IDs
            Curr_obsIDs=obsIDs[batch_start:batch_start+BATCH_SIZE]

            #2.2) S, X, Mask, y_true
            lst_S_batch=[]
            for VarNbInList in range(len(ID_TreatedVars)):
                lst_S_batch.append(Lst_S_train[VarNbInList][Curr_obsIDs])
                #computed in this loop: lst_S_batch

            X_batch = X_train[Curr_obsIDs,:].long().to(DEVICE)
            Masks_batch = Masks_train[Curr_obsIDs,:].to(DEVICE)
            
            if outputdatadim==0:
              y_batch = y_train[Curr_obsIDs].view(-1,1).float()  #as the outputs are in a 1d vector
            elif outputdatadim==1:
              y_batch = y_train[Curr_obsIDs,:].float()


            #2.3) set the NN gradient to zero
            optimizer.zero_grad()

            #2.4) mini-batch prediction
            output = model(ids=X_batch, mask=Masks_batch)

            #3) compute the attachement term loss
            loss_attach=f_loss_attach(output, y_batch.to(DEVICE))


            #4) prepare and compute the W2 term loss
            lst_loss_regula=[]
            lst_InfoPenaltyTerm=[]
            for VarNbInList in range(len(ID_TreatedVars)):
                #4.1 and 4.2) fill the InfoPenaltyTerm dictionnary after concatenating the histogram information with those of the mini-batch
                lst_InfoPenaltyTerm.append({})
                lst_InfoPenaltyTerm[-1]['mb_S']=lst_S_batch[VarNbInList]
                lst_InfoPenaltyTerm[-1]['o4h_S']=np.concatenate([lst_S_4histo[VarNbInList],lst_S_batch[VarNbInList]],axis=0)
                lst_InfoPenaltyTerm[-1]['o4h_y_pred']=torch.cat([lst_y_pred_4histo[VarNbInList].detach().to('cpu'),output.detach().to('cpu')],dim=0)  # .detach() was added
                lst_InfoPenaltyTerm[-1]['o4h_y_true']=torch.cat([lst_y_4histo[VarNbInList],y_batch],dim=0)
                lst_InfoPenaltyTerm[-1]['DistBetween']=DistBetween    #'Predictions_errors' or 'Predictions'
                lst_InfoPenaltyTerm[-1]['lambdavar']=lambdavar*ID_TreatedVars[VarNbInList][1]
                lst_InfoPenaltyTerm[-1]['ID_TreatedVar']=ID_TreatedVars[VarNbInList][0]

                #4.3) compute the W2 loss
                if DEVICE=='cpu':
                    lst_loss_regula.append(lst_f_loss_regula[VarNbInList](output, y_batch,lst_InfoPenaltyTerm[-1]))  #fair loss - no need to copy in the cpu which would induce memory losses (bug in pytorch when copying from cpu to cpu???)
                else:
                    lst_loss_regula.append(lst_f_loss_regula[VarNbInList](output.to('cpu'), y_batch,lst_InfoPenaltyTerm[-1]))  #fair loss - must be calculated in the CPU but will be detached in the custom regularization function to avoid breaking the NN graph (tested with pytorch 1.3.1)
                #computed in this loop: lst_loss_regula, lst_InfoPenaltyTerm

            #5) compute the whole loss  and perform the gradient descent step
            loss =  loss_attach
            for VarNbInList in range(len(ID_TreatedVars)):
                loss = loss + lst_loss_regula[VarNbInList].to(DEVICE)

            loss.backward()
            optimizer.step()

            #6) update the first observation of the batch
            batch_start+=BATCH_SIZE
            batchNb+=1

            #7) save pertinent information to check the convergence
            locLoss=loss_attach.item()
            locW2=lst_InfoPenaltyTerm[-1]['E_Reg']
            Lists_Results['Loss'].append(locLoss)
            Lists_Results['W2'].append(locW2/lambdavar)
            print("epoch "+str(epoch)+" -- batchNb "+str(batchNb)+": Loss="+str(Lists_Results['Loss'][-1])+' -- W2='+str(locW2/lambdavar)+' --  lambda='+str(lambdavar))


        #save advanced information at the end of the epoch...
        
        """
        if test_data['known']==True:
          
          #... information on the training set
          S0=np.where(S_4histo_merged<0.5)
          S1=np.where(S_4histo_merged>0.5)

          loss_attach=f_loss_attach(y_pred_4histo_merged[S0].to('cpu'), y_4histo_merged[S0])
          Lists_Results['Loss_S0_train'].append(loss_attach.item())
          loss_attach=f_loss_attach(y_pred_4histo_merged[S1].to('cpu'), y_4histo_merged[S1])
          Lists_Results['Loss_S1_train'].append(loss_attach.item())
          
          
          #... inforamtion on the test set
          print("Convergence measured on the test set -> GO")

          #draw the observations
          tst_IDs_S_eq_0=np.where(test_data['S_test']<0.5)[0]
          if len(tst_IDs_S_eq_0)<obs_for_histo:
            obsIDs_4histo_S0=tst_IDs_S_eq_0.copy()
          else:
            np.random.shuffle(tst_IDs_S_eq_0)
            obsIDs_4histo_S0=tst_IDs_S_eq_0[0:obs_for_histo]

          tst_IDs_S_eq_1=np.where(test_data['S_test']>=0.5)[0]
          if len(tst_IDs_S_eq_1)<obs_for_histo:
            obsIDs_4histo_S1=tst_IDs_S_eq_1.copy()
          else:
            np.random.shuffle(tst_IDs_S_eq_1)
            obsIDs_4histo_S1=tst_IDs_S_eq_1[0:obs_for_histo]

          obsIDs_4histo=np.concatenate([obsIDs_4histo_S0,obsIDs_4histo_S1])

          #format the data
          S_4histo=test_data['S_test'][obsIDs_4histo]

          X_4histo = test_data['X_test'][obsIDs_4histo,:].long().to(DEVICE)
          Masks_4histo = test_data['Masks_test'][obsIDs_4histo,:].to(DEVICE)
          
          if outputdatadim==0:
            y_4histo = test_data['y_test'][obsIDs_4histo].view(-1,1).float()  #as the outputs are in a 1d vector
          elif outputdatadim==1:
            y_4histo = test_data['y_test'][obsIDs_4histo,:].float()

          #prediction
          with torch.no_grad():
              y_pred_4histo = model(ids=X_4histo,mask=Masks_4histo)


          #information on the training set
          S0=np.where(S_4histo<0.5)
          S1=np.where(S_4histo>0.5)

          loss_attach=f_loss_attach(y_pred_4histo[S0], y_4histo[S0].to(DEVICE))
          Lists_Results['Loss_S0_test'].append(loss_attach.item())
          loss_attach=f_loss_attach(y_pred_4histo[S1], y_4histo[S1].to(DEVICE))
          Lists_Results['Loss_S1_test'].append(loss_attach.item())

          print("Convergence measured on the test set -> DONE")
        """

        #update the epoch number
        epoch+=1


    model_cpu=model.to('cpu')
    saved_models = { "model": model_cpu }
    pickle.dump( saved_models, open( 'l'+str(lambdavar)+'_saved_model.p', "wb" ) )
    # -> saved_models = pickle.load( open( "saved_model.p", "rb" ) )
    # -> model_cpu=saved_models["model"]
    # -> model=model_cpu.to(DEVICE)

    return Lists_Results



"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
OLD  OLD  OLD  OLD  OLD  OLD  OLD  OLD  OLD  OLD  OLD  OLD  OLD  OLD  OLD  OLD  OLD  OLD
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


def W2R_fit_NLP_old(model,X_train,Masks_train,y_train, S_train_in, lambdavar, f_loss_attach=nn.MSELoss() , EPOCHS = 5, BATCH_SIZE = 32,obs_for_histo=1000,DistBetween='Predictions',DEVICE='cpu',ID_TreatedVar=0,optim_lr=0.0001,reduced_epoch_size=-1,test_data={'known':False}):
    """
    -> X_train: input pytorch tensor are supposed to have a shape structured as [NbObs,:], [NbObs,:,:], or [NbObs,:,:,:].
    -> y_train: true output pytorch tensor. Supposed to be 2D (for one-hot encoding or others), or 1D (eg for binary classification)
    -> Masks_train: input pytorch tensor maksing the useless information. Has the same shape as X_train
    -> S_train : 1D numpy array containing the sensitive variables labels in the mini-batch.  Each label is in {0,1}.
    -> DistBetween can be : 'Predictions_errors' or 'Predictions'
    -> f_loss_attach: Data attachment term in the loss. Can be eg  nn.MSELoss(), nn.BCELoss(), ... . Must be coherent with the model outputs and the true outputs.
    -> ID_TreatedVar: variable in the columns of the y on which the penalty is computed
    -> test_data: dictionnary that eventually contains test data to evaluate the convergence. If test_data['known']==True, then test_data['X_test'], test_data['Masks_test'], test_data['y_test'] and test_data['S_test'] must be filled.
    """

    inputdatadim=len(X_train.shape)-1  #dimension of the input data (-1 is to take into account the fact that the first dimension corresponds to the observations)
    outputdatadim=len(y_train.shape)-1  #dimension of the output data (-1 is to take into account the fact that the first dimension corresponds to the observations)

    optimizer = torch.optim.Adam(model.parameters(),lr=optim_lr) #,lr=0.001, betas=(0.9,0.999))

    f_loss_regula = FairLoss.apply      #fair loss

    model.train()

    if lambdavar==0.:
      lambdavar=0.000000001
      print("lambdavar was set to a negligible value to avoid divisions by zero")

    n=X_train.shape[0]
    if reduced_epoch_size<0:
      reduced_epoch_size=n

    S_train=S_train_in[:]
    obs_not_in_TreatedVar=torch.where(y_train[:,ID_TreatedVar].view(-1)==0)[0].numpy()
    S_train[obs_not_in_TreatedVar]=2                                            #only perform the regularisation on the observations in the class of interest

    IDs_S_eq_0=np.where(S_train==0)[0]
    IDs_S_eq_1=np.where(S_train==1)[0]
    print('Nb obs:',len(S_train))
    print('Nb obs in group 0:',len(IDs_S_eq_0))
    print('Nb obs in group 1:',len(IDs_S_eq_1))  #TO DO: same thing for test when used


    Lists_Results={}
    Lists_Results['Loss']=[]          #loss of the data attachement term
    Lists_Results['W2']=[]            #W2 at the end of each mini-batch
    Lists_Results['Loss_S0_train']=[]
    Lists_Results['Loss_S1_train']=[]
    Lists_Results['Loss_S0_test']=[]
    Lists_Results['Loss_S1_test']=[]


    epoch=0
    while epoch<EPOCHS:
        obsIDs=np.arange(X_train.shape[0])
        np.random.shuffle(obsIDs)

        batch_start=0
        batchNb=0

        while batch_start+BATCH_SIZE < n and batch_start+BATCH_SIZE < reduced_epoch_size:

            #1) additional predictions to those of the mini-batch, in order to properly compute the Wasserstein histograms (may be computed at some iterations only)
            #1.1) get current observation IDs for S_train=0
            if len(IDs_S_eq_0)<obs_for_histo:
                obsIDs_4histo_S0=IDs_S_eq_0.copy()
            else:
                np.random.shuffle(IDs_S_eq_0)
                obsIDs_4histo_S0=IDs_S_eq_0[0:obs_for_histo]

            #1.2) get current observation IDs for S_train=1
            if len(IDs_S_eq_1)<obs_for_histo:
                obsIDs_4histo_S1=IDs_S_eq_1.copy()
            else:
                np.random.shuffle(IDs_S_eq_1)
                obsIDs_4histo_S1=IDs_S_eq_1[0:obs_for_histo]

            #1.3) merge the observation IDs
            obsIDs_4histo=np.concatenate([obsIDs_4histo_S0,obsIDs_4histo_S1])


            #1.4) S_train, X, mask and y_true for the histograms
            S_4histo=S_train[obsIDs_4histo]

            if inputdatadim==1:
              X_4histo = X_train[obsIDs_4histo,:].long().to(DEVICE)
              Masks_4histo = Masks_train[obsIDs_4histo,:].to(DEVICE)
            elif inputdatadim==2:
              X_4histo = X_train[obsIDs_4histo,:,:].long().to(DEVICE)
              Masks_4histo = Masks_train[obsIDs_4histo,:,:].to(DEVICE)
            else:
              X_4histo = X_train[obsIDs_4histo,:,:,:].long().to(DEVICE)
              Masks_4histo = Masks_train[obsIDs_4histo,:,:,:].to(DEVICE)

            if outputdatadim==0:
              y_4histo = y_train[obsIDs_4histo].view(-1,1).float()  #as the outputs are in a 1d vector
            elif outputdatadim==1:
              y_4histo = y_train[obsIDs_4histo,:].float()

            #1.5) prediction
            with torch.no_grad():
                y_pred_4histo = model(ids=X_4histo , mask=Masks_4histo)

            #2) mini-batch predictions

            #2.1) get the observation IDs
            Curr_obsIDs=obsIDs[batch_start:batch_start+BATCH_SIZE]

            #2.2) S, X, Mask, y_true
            S_batch=S_train[Curr_obsIDs]

            if inputdatadim==1:
              X_batch = X_train[Curr_obsIDs,:].long().to(DEVICE)
              Masks_batch = Masks_train[Curr_obsIDs,:].to(DEVICE)
            elif inputdatadim==2:
              X_batch = X_train[Curr_obsIDs,:,:].long().to(DEVICE)
              Masks_batch = Masks_train[Curr_obsIDs,:,:].to(DEVICE)
            else:
              X_batch = X_train[Curr_obsIDs,:,:,:].long().to(DEVICE)
              Masks_batch = Masks_train[Curr_obsIDs,:,:,:].to(DEVICE)

            if outputdatadim==0:
              y_batch = y_train[Curr_obsIDs].view(-1,1).float()  #as the outputs are in a 1d vector
            elif outputdatadim==1:
              y_batch = y_train[Curr_obsIDs,:].float()


            #2.3) set the NN gradient to zero
            optimizer.zero_grad()

            #2.4) mini-batch prediction
            output = model(ids=X_batch, mask=Masks_batch)

            #3) compute the attachement term loss
            loss_attach=f_loss_attach(output, y_batch.to(DEVICE))


            #4) prepare and compute the W2 term loss
            #4.1) concatenate the histogram information with those of the mini-batch
            S_4histo_merged=np.concatenate([S_4histo,S_batch],axis=0)
            y_4histo_merged=torch.cat([y_4histo,y_batch],dim=0)
            y_pred_4histo_merged=torch.cat([y_pred_4histo.detach().to('cpu'),output.detach().to('cpu')],dim=0)  # .detach() was added

            #4.2) fill the InfoPenaltyTerm dictionnary
            InfoPenaltyTerm={}                                         #FOR THE W2 REGULARIZATION
            InfoPenaltyTerm['mb_S']=S_batch
            InfoPenaltyTerm['o4h_S']=S_4histo_merged
            InfoPenaltyTerm['o4h_y_pred']=y_pred_4histo_merged
            InfoPenaltyTerm['o4h_y_true']=y_4histo_merged
            InfoPenaltyTerm['DistBetween']=DistBetween    #'Predictions_errors' or 'Predictions'
            InfoPenaltyTerm['lambdavar']=lambdavar
            InfoPenaltyTerm['ID_TreatedVar']=ID_TreatedVar

            #4.3) compute the W2 loss
            #loss_regula=f_loss_regula(output.detach().to('cpu'), y_batch.view(-1,1),InfoPenaltyTerm)  #if used in the cpu -- memory losses otherwise
            if DEVICE=='cpu':
              loss_regula=f_loss_regula(output, y_batch,InfoPenaltyTerm)  #fair loss - no need to copy in the cpu which would induce memory losses (bug in pytorch when copying from cpu to cpu???)
            else:
              loss_regula=f_loss_regula(output.to('cpu'), y_batch,InfoPenaltyTerm)  #fair loss - must be calculated in the CPU but will be detached in the custom regularization function to avoid breaking the NN graph (tested with pytorch 1.3.1)

            #5) compute the whole loss  and perform the gradient descent step
            loss =  loss_attach+loss_regula.to(DEVICE)
            loss.backward()
            optimizer.step()

            #6) update the first observation of the batch
            batch_start+=BATCH_SIZE
            batchNb+=1

            #7) save pertinent information to check the convergence
            locLoss=loss_attach.item()
            locW2=InfoPenaltyTerm['E_Reg']
            Lists_Results['Loss'].append(locLoss)
            Lists_Results['W2'].append(locW2/lambdavar)
            print("epoch "+str(epoch)+" -- batchNb "+str(batchNb)+": Loss="+str(Lists_Results['Loss'][-1])+' -- W2='+str(locW2/lambdavar)+' --  lambda='+str(lambdavar))


        #save advanced information at the end of the epoch...
        #... inforamtion on the training set
        S0=np.where(S_4histo_merged<0.5)
        S1=np.where(S_4histo_merged>0.5)

        loss_attach=f_loss_attach(y_pred_4histo_merged[S0].to('cpu'), y_4histo_merged[S0])
        Lists_Results['Loss_S0_train'].append(loss_attach.item())
        loss_attach=f_loss_attach(y_pred_4histo_merged[S1].to('cpu'), y_4histo_merged[S1])
        Lists_Results['Loss_S1_train'].append(loss_attach.item())

        if test_data['known']==True:
          print("Convergence measured on the test set -> GO")

          #draw the observations
          tst_IDs_S_eq_0=np.where(test_data['S_test']<0.5)[0]
          if len(tst_IDs_S_eq_0)<obs_for_histo:
            obsIDs_4histo_S0=tst_IDs_S_eq_0.copy()
          else:
            np.random.shuffle(tst_IDs_S_eq_0)
            obsIDs_4histo_S0=tst_IDs_S_eq_0[0:obs_for_histo]

          tst_IDs_S_eq_1=np.where(test_data['S_test']>=0.5)[0]
          if len(tst_IDs_S_eq_1)<obs_for_histo:
            obsIDs_4histo_S1=tst_IDs_S_eq_1.copy()
          else:
            np.random.shuffle(tst_IDs_S_eq_1)
            obsIDs_4histo_S1=tst_IDs_S_eq_1[0:obs_for_histo]

          obsIDs_4histo=np.concatenate([obsIDs_4histo_S0,obsIDs_4histo_S1])

          #format the data
          S_4histo=test_data['S_test'][obsIDs_4histo]

          if inputdatadim==1:
            X_4histo = test_data['X_test'][obsIDs_4histo,:].long().to(DEVICE)
            Masks_4histo = test_data['Masks_test'][obsIDs_4histo,:].to(DEVICE)
          elif inputdatadim==2:
            X_4histo = test_data['X_test'][obsIDs_4histo,:,:].long().to(DEVICE)
            Masks_4histo = test_data['Masks_test'][obsIDs_4histo,:,:].to(DEVICE)
          else:
            X_4histo = test_data['X_test'][obsIDs_4histo,:,:,:].long().to(DEVICE)
            Masks_4histo = test_data['Masks_test'][obsIDs_4histo,:,:,:].to(DEVICE)

          if outputdatadim==0:
            y_4histo = test_data['y_test'][obsIDs_4histo].view(-1,1).float()  #as the outputs are in a 1d vector
          elif outputdatadim==1:
            y_4histo = test_data['y_test'][obsIDs_4histo,:].float()

          #prediction
          with torch.no_grad():
              y_pred_4histo = model(ids=X_4histo,mask=Masks_4histo)


          #information on the training set
          S0=np.where(S_4histo<0.5)
          S1=np.where(S_4histo>0.5)

          loss_attach=f_loss_attach(y_pred_4histo[S0], y_4histo[S0].to(DEVICE))
          Lists_Results['Loss_S0_test'].append(loss_attach.item())
          loss_attach=f_loss_attach(y_pred_4histo[S1], y_4histo[S1].to(DEVICE))
          Lists_Results['Loss_S1_test'].append(loss_attach.item())

          print("Convergence measured on the test set -> DONE")


        #update the epoch number
        epoch+=1

    #memory clean-up
    del Curr_obsIDs, X_batch,  y_batch, output
    del loss_attach
    del loss_regula , loss
    del obsIDs_4histo_S0, obsIDs_4histo_S1, obsIDs_4histo, S_4histo, X_4histo, y_4histo_merged, y_pred_4histo_merged, S_4histo_merged, InfoPenaltyTerm
    del Masks_4histo,Masks_batch

    model_cpu=model.to('cpu')
    saved_models = { "model": model_cpu }
    pickle.dump( saved_models, open( 'l'+str(lambdavar)+'_saved_model.p', "wb" ) )
    # -> saved_models = pickle.load( open( "saved_model.p", "rb" ) )
    # -> model_cpu=saved_models["model"]
    # -> model=model_cpu.to(DEVICE)

    return Lists_Results
