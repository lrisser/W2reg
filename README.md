# W2reg
Train neural networks with Wasserstein-2 regularization to reduce algorithmic bias in future predictions

**Related paper**

This repositroy gives the core functions used in the [Risser et al, JMIV 2022] paper, plus a simple notebook explaining how to use these functions.

[Risser et al, JMIV 2022] Risser L., Gonzalez Sanz A., Vincenot Q., Loubes J.M.: Tackling Algorithmic Bias in Neural-Network Classifiers using Wasserstein-2 Regularization, Journal of Mathematical Image Vision (JMIV), 2022 

**Remarks**

- The code is fully written Python and makes use of PyTorch. 
- The code was succefully tested using pytorch 1.3.1 with or without gpu acceleration.
- The *Wassertein-2 loss* (so-called FairLoss in the code) was only tested on the CPU, as it is based on non-pytroch.autograd functions. It worth emphasising that the backpropagation works even if the neural network model and the mini-batch data are in the GPU memory.
- To give contextual information to the *Wassertein-2 loss* function, the trick was to use the *InfoPenaltyTerm* dictionary. Its items have to be properly filled before computing the loss and its gradient. 
