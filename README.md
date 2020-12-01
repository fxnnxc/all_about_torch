# ALL ABOUT TORCH

Source codes for the specific situations

---
# Classification Models

## 1. Custom Data Using Numpy

* From numpy to Tensor
* From Tensor to TensorDataset
* From TensorDataset to DataLoader

## 2. Plot the loss using visdom

* Visdom Plot with loss
* Visdom Plot with different lines

## 3. Plot the loss using tensorboard

* Tensorboard Plot with loss
* Tensorboard graph structure 

## 4. Fully Connected, CNN, RNN models

* Fully Connected
* Convolution Neural Network
* Recurrent Neural Network

## 5. RNN, LSTM, GRU
 * Recurrent Neural Network
 * Long-Short term memory
 * Gate Recurrent Unit


## 6. Transformer Encoder
* Transformer Encoder (Multihead Self Attention)


---
# Gneration Models

## 1.  Variational AutoEncoder
    
r.f. https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py


conv 1 32
conv 32 64
conv 64 256 
hidden 64


    B x 1  x 32 x 32
    B x 32 x 16 x 16 
    B x 64 x  8 x  8
    B x 256 x 4 x  4
    B x 2056
    B x 64
    B 

