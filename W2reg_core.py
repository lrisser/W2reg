
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





def EstimGrad_W2dist(minibatch_S,minibatch_y_pred,minibatch_y_true,obs4histo_S,obs4histo_y_pred,obs4histo_y_true, NbBins=500,ID_TreatedVar=0,DistBetween='Predictions_errors'):
    """
    Estimate the gradient of the Wasserstein Distance between the histograms of the values of minibatch_y_pred for which minibatch_S=0 and minibatch_S=1
    -> Notations:
        
      --> minibatch_S is the sensitive variable. minibatch_S=0 if minority / minibatch_S=1 if majority
      --> minibatch_y_true is a pytorch tensor representing the (true) selection variable. Y=0 if fail     / Y=1 if success
      --> minibatch_y_pred is a pytorch tensor representing the estimated probability the minibatch_y_true==1

      --> obs4histo_S, obs4histo_y_true, obs4histo_y_pred: same as the minibatch_* variables but to compute the histograms only (should contain more observations but may be updated less often)
       
      --> ID_TreatedVar: If there are multiple outputs, ID_TreatedVar is the index of the output vector on which the regularization will be performed
      
      --> DistBetween: data used to compute the cumulative histograms. Can be:
            * 'All_predictions' -> all data are considered. The regularizer favors similar distribution of predictions, typically low disparate impacts. 
            * 'Predictions_errors' -> all data are used but the distance between the error rates and its gradients are computed
    -> Return:
      --> A list of the gradients for the points defined in minibatch_y_pred and  minibatch_y_true
    """
    
    
    #1) init
    
    #1.1) input conversions (for the minibatch data)
    y_pred=minibatch_y_pred.detach().numpy()
    y_true=minibatch_y_true.detach().numpy()
    
    y_pred=(y_pred[:,ID_TreatedVar]*1.).ravel()
    y_true=(y_true[:,ID_TreatedVar]*1.).ravel()
    
    
    #1.2) input conversions (for the obs4histo data)
    y_pred_4histo=obs4histo_y_pred.detach().numpy()
    y_true_4histo=obs4histo_y_true.detach().numpy()
    
    y_pred_4histo=(y_pred_4histo[:,ID_TreatedVar]*1.).ravel()
    y_true_4histo=(y_true_4histo[:,ID_TreatedVar]*1.).ravel()
    
    
    #2) compute the cumulative distribution functions
    if DistBetween=='Predictions_errors':
        #2.1) pred err -- split the observations w.r.t. the value of obs4histo_S in {0,1}   (for the obs4histo data)
        tmpZip=zip((y_true_4histo-y_pred_4histo)*(y_true_4histo-y_pred_4histo),obs4histo_S)
        zipped_y_true_S_eq_1=list(filter(lambda x: x[1] == 1 , tmpZip))
        err_1_4histo, bidon1 = zip(*zipped_y_true_S_eq_1)
        err_1_4histo=np.array(err_1_4histo)
        n1=err_1_4histo.shape[0]
        
        tmpZip=zip((y_true_4histo-y_pred_4histo)*(y_true_4histo-y_pred_4histo),obs4histo_S)
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
        #2.3) pred -- split the observations w.r.t. the value of obs4histo_S in {0,1}   (for the obs4histo data)
        tmpZip=zip(y_pred_4histo,obs4histo_S)
        zipped_y_pred_S_eq_1=list(filter(lambda x: x[1] == 1, tmpZip))
        y_pred_S_eq_1_4histo, bidon1 = zip(*zipped_y_pred_S_eq_1)
        y_pred_S_eq_1_4histo=np.array(y_pred_S_eq_1_4histo)
        n1=y_pred_S_eq_1_4histo.shape[0]
      
        tmpZip=zip(y_pred_4histo,obs4histo_S)
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
    WGradients=np.zeros([minibatch_y_pred.shape[0],1])
    
    for curObs in range(minibatch_y_pred.shape[0]):
        if DistBetween=='Predictions_errors':
          #curObs_absiss=np.abs(y_pred[curObs]-y_true[curObs])
          curObs_absiss=(y_pred[curObs]-y_true[curObs])*(y_pred[curObs]-y_true[curObs])
        else:
          curObs_absiss=y_pred[curObs]
        
        if minibatch_S[curObs]==1:  #majority
          #... majo 1 ... get the index impacted by curObs_absiss
          loc_index_yp=0
          while cumfreq_absiss[loc_index_yp]<curObs_absiss and loc_index_yp<len(cumfreq_absiss)-1:
            loc_index_yp+=1
          loc_index_yp-=1
          if loc_index_yp<0:
            loc_index_yp=0
        
          #... majo 2 ... linear interpolation to make this estimate finer
          if loc_index_yp==0:
            H_1_loc=cumfreq_S1[loc_index_yp]
          else:
            distPlus=cumfreq_absiss[loc_index_yp+1]-curObs_absiss
            distMinus=curObs_absiss-cumfreq_absiss[loc_index_yp]
            distTot=distMinus+distPlus
            distPlus/=distTot
            distMinus/=distTot
            H_1_loc=(distMinus*cumfreq_S1[loc_index_yp+1])+(distPlus*cumfreq_S1[loc_index_yp])
        
          #... majo 3 ... find the corresponding index in the other cumulative distribution
          loc_index=0
          while cumfreq_S0[loc_index]<H_1_loc and loc_index<len(cumfreq_S0)-1:
            loc_index+=1
          loc_index-=1
          if loc_index<0:
            loc_index=0
        
          #... majo 4 ... linear interpolation to find the correponding value
          if loc_index==0:
            Hinv_0_loc=cumfreq_absiss[loc_index]
          else:
            distPlus=cumfreq_S0[loc_index+1]-H_1_loc
            distMinus=H_1_loc-cumfreq_S0[loc_index]
            distTot=distMinus+distPlus
            if distTot>0.:
                distPlus/=distTot
                distMinus/=distTot
                Hinv_0_loc=(distMinus*cumfreq_absiss[loc_index+1])+(distPlus*cumfreq_absiss[loc_index])
            else:
                Hinv_0_loc=cumfreq_absiss[loc_index]
          
          #... majo 5 ... compute the Wassertein Gradient
          grad_H_1=eps+cumfreq_S1[loc_index_yp+1]-cumfreq_S1[loc_index_yp]
          WGradients[curObs,0]=-2*(Hinv_0_loc-curObs_absiss)/(n1*grad_H_1)
          
          
        else:  #minority
          #... mino 1 ... get the index impacted by curObs_absiss
          loc_index_yp=0
          while cumfreq_absiss[loc_index_yp]<curObs_absiss and loc_index_yp<len(cumfreq_absiss)-1:
            loc_index_yp+=1
          loc_index_yp-=1
          if loc_index_yp<0:
            loc_index_yp=0
        
          #... mino 2 ... linear interpolation to make this estimate finer
          if loc_index_yp==0:
            H_0_loc=cumfreq_S0[loc_index_yp]
          else:
            distPlus=cumfreq_absiss[loc_index_yp+1]-curObs_absiss
            distMinus=curObs_absiss-cumfreq_absiss[loc_index_yp]
            distTot=distMinus+distPlus
            distPlus/=distTot
            distMinus/=distTot
            H_0_loc=(distMinus*cumfreq_S0[loc_index_yp+1])+(distPlus*cumfreq_S0[loc_index_yp])
        
          #... mino 3 ... find the corresponding index in the other cumulative distribution
          loc_index=0
          while cumfreq_S1[loc_index]<H_0_loc and loc_index<len(cumfreq_S1)-1:
            loc_index+=1
          loc_index-=1
          if loc_index<0:
            loc_index=0
        
          #... mino 4 ... linear interpolation to find the correponding value
          if loc_index==0:
            Hinv_1_loc=cumfreq_absiss[loc_index]
          else:
            distPlus=cumfreq_S1[loc_index+1]-H_0_loc
            distMinus=H_0_loc-cumfreq_S1[loc_index]
            distTot=distMinus+distPlus
            if distTot>0.:
                distPlus/=distTot
                distMinus/=distTot
                Hinv_1_loc=(distMinus*cumfreq_absiss[loc_index+1])+(distPlus*cumfreq_absiss[loc_index])
            else:
                Hinv_1_loc=cumfreq_absiss[loc_index]
          
          #... mino 5 ... compute the Wassertein Gradient
          grad_H_0=eps+cumfreq_S0[loc_index_yp+1]-cumfreq_S0[loc_index_yp]
          WGradients[curObs,0]=2*(curObs_absiss-Hinv_1_loc)/(n0*grad_H_0)
          
        if DistBetween=='Predictions_errors':
          WGradients[curObs,0]=2*(y_pred[curObs]-y_true[curObs])*WGradients[curObs,0] 
    
    return [WGradients,W_score]


class FairLoss(torch.autograd.Function):  
    """
    class to manage the loss with a penalty enforcing the fairness. 
      -> The similarity term is a standard Mean Squared Error (MSE)
      -> The regularisation term ensures that the Wassertein distance between the distribution of the 
         predictions for S=0 and S=1 is the same
      -> Designed for 1D neural-network outputs that represent a probability. 
      
    An important structure to compute the Wasserstein distance and its gradient is 'InfoPenaltyTerm', 
    which is given as an input of the 'forward' method. It is a dictionary containing:
      -> InfoPenaltyTerm['lambda']: the weight on the regularization term
      -> InfoPenaltyTerm['mb_S']:  np.array vector containing the sensitive variables labels in the mini-batch.  Each label is in {0,1}.
      -> InfoPenaltyTerm['o4h_S']: np.array vector containing the sensitive variables labels in the observations for the histograms.  Each label is in {0,1}.
      -> InfoPenaltyTerm['o4h_y_pred']: pytorch-tensor vector containing the predicted probabilities that we have label 1 for the histograms. Each probability is in [0,1]. 
      -> InfoPenaltyTerm['o4h_y_true']: pytorch-tensor vector containing the true selection variable for the histograms. Each label is in {0,1}.
      -> InfoPenaltyTerm['DistBetween']: =equal All_predictions' to regularize the predictions or 'Predictions_errors' to regularize the prediction errors
    
    Note that when running 'forward', these values will additionally be saved in 'InfoPenaltyTerm':
      -> InfoPenaltyTerm['E_Reg']: regularization energy
      -> InfoPenaltyTerm['E_Simi']: similarity energy
    """
        
    @staticmethod
    def forward(ctx,  y_pred, y, InfoPenaltyTerm):
        """
        * y_pred:  pytorch-tensor vector containing the predicted probabilities that we have label 1 in the mini-bath. Each probability is in [0,1].
        * y is the true y  -> pytorch-tensor vector containing the true selection variable in the mini-bath. Each label is in {0,1}.
        * InfoPenaltyTerm is the dictionary that contains the pertinent information to compute the regularization term and its gradient 
        """
        
        #0) reshape y if not formated as y_pred, before saving it in the context
        if y.dim()==1:   #check if 1D ref outputs
           y=y.view(-1,1)
        
        if y.size()!=y_pred.size():   #check if transposed ref outputs
          y=y.transpose()
        
        ctx.save_for_backward(y, y_pred)
        
        #1) compute the similarity term
        E_Simi=(y_pred - y).pow(2.).mean()
        
        #2) compute the W2 penalty information and save it for the backward method
        if (InfoPenaltyTerm['lambda']>0.):
           [W_Gradients,W_score]=EstimGrad_W2dist(InfoPenaltyTerm['mb_S'],y_pred,y,InfoPenaltyTerm['o4h_S'],InfoPenaltyTerm['o4h_y_pred'],InfoPenaltyTerm['o4h_y_true'], NbBins=500,ID_TreatedVar=0,DistBetween=InfoPenaltyTerm['DistBetween']) #DistBetween = 'All_predictions' or 'Predictions_errors'
           ctx.W_Gradients=W_Gradients
        else:
            W_score=0.
        
        #3) save the energies
        InfoPenaltyTerm['E_Simi']=E_Simi
        InfoPenaltyTerm['E_Reg']=W_score
        ctx.InfoPenaltyTerm=InfoPenaltyTerm
        
        return E_Simi+ctx.InfoPenaltyTerm['lambda']*W_score 

    @staticmethod
    def backward(ctx, grad_output):
        """
        Requires the information saved in the forward function:
          ctx.saved_tensors -> y and y_pred
          ctx.W_Gradients -> gradient of the wasserstein regularization term
          ctx.InfoPenaltyTerm -> only used for InfoPenaltyTerm['lambda']
        """
        
       
        #0) init
        yy, yy_pred = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        #1) gradient of the similarity term
        grad_input = 2.0*(yy_pred - yy)
        
        #2) update the gradient with the penalty term
        if (ctx.InfoPenaltyTerm['lambda']>0.):
            grad_input += ctx.InfoPenaltyTerm['lambda']*torch.tensor((ctx.W_Gradients).astype(np.float32))
        
        return grad_input, None, None  #second None added because of the context information in forward




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#functions to fit the model to data
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




def W2R_fit(model,X_data,y_data, S, lambdavar, EPOCHS = 5, BATCH_SIZE = 32,obs_for_histo=1000,DistBetween='All_predictions',DEVICE='cpu'):
    """
    -> The input pytorch tensors (X_data) are supposed to have a shape structured as [BatchSize,NbChannels,ImSizeX,ImSizeY]. 
    -> The output pytorch tensors (y_data) are a vector of binary values
    -> S can be a list or a 1D pytorch.tensor or np.array 
    -> DistBetween can be : 'Predictions_errors' or 'All_predictions'
    """

    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001) #,lr=0.001, betas=(0.9,0.999))
    error = FairLoss.apply      #fair loss
    #error = nn.BCELoss()  #alternative standard loss
    #error = nn.MSELoss()  #alternative standard loss

    model.train()
    
    
    n=X_data.shape[0]
    IDs_S_eq_0=np.where(S<0.5)[0]
    IDs_S_eq_1=np.where(S>0.5)[0]

    
    Lists_Results={}
    Lists_Results['Acc']=[]
    Lists_Results['lambda']=[]
    Lists_Results['W2']=[]

    
    epoch=0
    while epoch<EPOCHS:
        obsIDs=np.arange(X_data.shape[0]) 
        np.random.shuffle(obsIDs)
        
        batch_start=0 
        batchNb=0
        
        if epoch==EPOCHS-1:
          BATCH_SIZE*=2
        
        if epoch==EPOCHS-3:
          BATCH_SIZE*=3
        
        while batch_start+BATCH_SIZE < n:
            
            #1) get mini-batch information
            Curr_obsIDs=obsIDs[batch_start:batch_start+BATCH_SIZE]
            
            Curr_obsIDs=obsIDs[batch_start:batch_start+BATCH_SIZE]
            
            var_X_batch = X_data[Curr_obsIDs,:,:,:].float().to(DEVICE)
            var_y_batch = y_data[Curr_obsIDs,0].reshape(-1,1).float()  #added in the 1d case
            
            optimizer.zero_grad()
            
            #2) prediction in the mini-batch
            output = model(var_X_batch)
            
            #3) information to compute the Wasserstein histograms (may be computed at some iterations only)
            #3.1) get current observation IDs for S=0 
            if len(IDs_S_eq_0)<obs_for_histo:
                obsIDs_4histo_S0=IDs_S_eq_0.copy()
            else:
                np.random.shuffle(IDs_S_eq_0)
                obsIDs_4histo_S0=IDs_S_eq_0[0:obs_for_histo]

            #3.2) get current observation IDs for S=1 
            if len(IDs_S_eq_1)<obs_for_histo:
                obsIDs_4histo_S1=IDs_S_eq_1.copy()
            else:
                np.random.shuffle(IDs_S_eq_1)
                obsIDs_4histo_S1=IDs_S_eq_1[0:obs_for_histo]
            
            #3.3) merge the observation IDs
            obsIDs_4histo=np.concatenate([obsIDs_4histo_S0,obsIDs_4histo_S1])
            S_4histo=S[obsIDs_4histo]            
            
            #3.4) predictions for the histograms
            var_X_4histo = X_data[obsIDs_4histo,:,:,:].float().to(DEVICE)
            var_y_4histo = y_data[obsIDs_4histo,0].reshape(-1,1).float()  #added in the 1d case
            with torch.no_grad():
                y_pred_4histo = model(var_X_4histo.float())
                
            
            #3.5) concatenate the histogram information with those of the mini-batch
            #var_X_4histo=torch.cat([var_X_4histo,var_X_batch],dim=0)
            var_y_4histo=torch.cat([var_y_4histo,var_y_batch],dim=0)
            y_pred_4histo=torch.cat([y_pred_4histo,output],dim=0)
                
                
                
            #4) compute the loss and perform the gradient descent step 
            InfoPenaltyTerm={}                                         #FOR THE W2 REGULARIZATION
            InfoPenaltyTerm['lambda']=lambdavar                        #Weight between the data similarity term and the penalty term -> E = E_{simi} + \lambda * E_{pena} 
            InfoPenaltyTerm['mb_S']=S[Curr_obsIDs]
            InfoPenaltyTerm['o4h_S']=S_4histo
            InfoPenaltyTerm['o4h_y_pred']=y_pred_4histo.to('cpu')
            InfoPenaltyTerm['o4h_y_true']=var_y_4histo
            InfoPenaltyTerm['DistBetween']=DistBetween    #'Predictions_errors' or 'All_predictions'
            loss = error(output.to('cpu'), var_y_batch.view(-1,1),InfoPenaltyTerm)  #fair loss - must be calculated in the CPU but backpropagation is OK in the GPU (tested with pytorch 1.3.1)

            loss.backward()
            optimizer.step()
            
            #5) update the first observation of the batch
            batch_start+=BATCH_SIZE
            batchNb+=1
            
            
            
            #9) save pertinent information to check the convergence
            locLoss=InfoPenaltyTerm['E_Simi'].numpy()
            locW2=InfoPenaltyTerm['E_Reg']
              
            Lists_Results['Acc'].append(locLoss)
            Lists_Results['lambda'].append(lambdavar)
            Lists_Results['W2'].append(locW2)
            
            
            print("epoch "+str(epoch)+" -- batchNb "+str(batchNb)+": "+str(Lists_Results['Acc'][-1])+' '+str(locW2)+' --  lambda='+str(Lists_Results['lambda'][-1]))
        
        epoch+=1
    
    torch.save(model.state_dict(), "./modelFinal.pytorchModel")         #save a model

    
    return Lists_Results
