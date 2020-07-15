# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:36:25 2020

@author: SeasonTaiInOTA
"""

import torch 
import numpy  as np 
import matplotlib.pyplot as plt 
import tqdm
import os
from torchvision import transforms, datasets
import torch.nn as nn
from torch.nn import functional as F
from imageio import imsave

class VAE(nn.Module):
    
    def __init__(self,h_dim):
        super(VAE, self).__init__()

        #encoder
        self.f1 = nn.Linear(3*64*64, 4096)
        self.f2 = nn.Linear(4096, 500)
        self.f3 = nn.Linear(500, 150)

        #latent space
        self.w_mean = nn.Linear(150 , h_dim)
        self.w_var = nn.Linear(150, h_dim)

        #decoder 
        self.w1 = nn.Linear(h_dim, 500)
        self.w2 = nn.Linear(500, 4096)
        self.w3 = nn.Linear(4096, 3*64*64)
        
        #activation_function
        self.activation_f = nn.ReLU()

    def encode(self, x):
        self.layer1 = self.activation_f(self.f1(x))
        self.layer2 = self.activation_f(self.f2(self.layer1))
        self.output = self.activation_f(self.f3(self.layer2))
        self.latent_mean = self.w_mean(self.output)
        self.latent_var  = self.w_var(self.output)
        return self.latent_mean, self.latent_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        self.dec_layer1 = self.activation_f(self.w1(z))
        self.dec_layer2 = self.activation_f(self.w2(self.dec_layer1))
        self.pack2 = self.w3(self.dec_layer2)
        self.act_out = torch.sigmoid(self.pack2)
        self.pack3 = self.act_out.view(-1, 3*64*64)
        return self.pack3

    def forward(self,x):
        encoder = self.encode(x)
        self.mu = encoder[0]
        self.var= encoder[1]
        self.z = self.reparameterize(encoder[0], encoder[1])
        self.decoder = self.decode(self.z)
        return self.mu, self.var, self.z, self.decoder

    def loss_f(self,x ):
        recon_x = self.decoder
        loss_first = F.mse_loss(recon_x.view(-1, 3*64*64),x, reduction='sum')
        loss_KL = 0 *-0.5 * torch.sum(1 + self.var - self.mu.pow(2) - self.var.exp())
        return loss_KL + loss_first

    def train_a_epoch(self, train_loader, optimizer, batch_size, epoch):
        self = self.train()
        device = None
        device = torch.device('cpu')
        train_loss = 0
        print(len(train_loader))
        for batch_idx, data in enumerate(train_loader):
            
            x = data[0].to(device)
            x = x.view(-1,3*64*64)
            optimizer.zero_grad()
            mu, var, z, reconst_x  = self.forward(x)
            loss = self.loss_f(x)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % (batch_size-1) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader.dataset)/batch_size,
                    100. * batch_idx / len(train_loader),
                    loss.item()/batch_size))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
        
        return train_loss / len(train_loader.dataset)
    
    def eval_some_sample(self, test_loader, batch_size, epoch, folder):
        self = self.eval()
        device = None
        device = torch.device('cpu')
            
        for batch_idx, data in enumerate(test_loader):
            x = data[0].to(device)
            x_input = x.view(-1,3*64*64)
            mu, var, z, reconst_x  = self.forward(x_input)
            x = x.permute(0,2,3,1).cpu().numpy()
            reconst_x = reconst_x.view(-1,3,64,64).permute(0,2,3,1).detach().cpu().numpy()
            if batch_idx % 50 == 0:
                map_ =np.zeros((64*10, 64*16,3))
                for j in range(0,16,2):
                    for i in range(10):
                        map_[i*64:(i+1)*64,j*64:(j+1)*64,:] = x[j*5+i]
                        map_[i*64:(i+1)*64,(j+1)*64:(j+2)*64,:] = reconst_x[j*5+i]
                if not os.path.exists(folder+"/result_png"):
                    os.makedirs(folder+"/result_png")
                imsave(folder+"/result_png/epoch_{}_batch_{}.png".format(epoch,batch_idx),map_)


if __name__ == "__main__":
    
    log_interval = 5
    lr = 1e-4  #8e-3
    batch_size = 100
    epoch = 100
    latent_z = 50  #20
    folder = "./vae_summary_latent50"
    
    loss = []
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    device = None
#    if torch.cuda.is_available():
#       device = torch.device('cuda:0')
#    else:
    device = torch.device('cpu')
    
    print('-----------data processing------------')
    data_transform = transforms.Compose([transforms.Resize(64),transforms.ToTensor(),])
    train_dataset = datasets.ImageFolder(root='./data',transform= data_transform)
    data = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
    print('-----------data processed------------')

    model = VAE(latent_z).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr) 
    model = model.train()

    print('-----------start training------------')
    for i in tqdm.tqdm(range(epoch)):
        epoch_loss = model.train_a_epoch( data, opt, batch_size, i)
        loss.append(epoch_loss)
        if (i% log_interval == 0):
            model.eval_some_sample(data ,batch_size, i, folder)
    print('-----------end training and generated picture------------')