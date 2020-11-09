import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model_mnist import Generator, Discriminator, init_weights
from utils import save

def train(train_loader, args):
    #Model
    netG = Generator(d=64).to(args.device)
    netD = Discriminator(d=64).to(args.device)
    netG.apply(init_weights)
    netD.apply(init_weights)

    # Optimizer, Loss, SummaryWriter
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    for epoch in range(args.train_epoch):
        # Train
        netG.train()
        netD.train()
        train_start = time.time()
        G_losses = 0
        D_losses = 0
        print("Epoch {0} Train Start".format(epoch)) 

        for i, data in enumerate(train_loader):
            # Load real and fake image
            real_img = data[0].to(args.device)
            b_size = real_img.size(0)
            real_label = torch.full((b_size, ), 1., dtype=torch.float, device=args.device)

            noise = torch.randn(b_size, 100, 1, 1, device=args.device)
            fake_img = netG(noise)
            fake_label = torch.full((b_size, ), 0., dtype=torch.float, device=args.device)

            # Train Discriminator   
            netD.zero_grad()        
            Dreal_output, _ = netD(real_img)
            Dreal_loss = criterion(Dreal_output.view(-1), real_label)        
            Dreal_loss.backward()

            Dfake_output, _ = netD(fake_img.detach())
            Dfake_loss = criterion(Dfake_output.view(-1), fake_label)        
            Dfake_loss.backward()

            D_loss = Dreal_loss + Dfake_loss        
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            Gfake_output, _ = netD(fake_img)
            G_loss = criterion(Gfake_output.view(-1), real_label)        

            G_loss.backward()
            optimizerG.step()

            G_losses += G_loss.item()
            D_losses += D_loss.item()

        G_losses = G_losses/(i+1)
        D_losses = D_losses/(i+1)
        epoch_time = time.time()-train_start


        print("Epoch {0} | G_loss {1:.4f} | D_loss {2:.4f} | Time {3:.2f}".format(epoch, G_losses, D_losses, epoch_time))
        print("-"*100)

        writer.add_scalars('Loss', {'G_loss': G_losses, 'D_loss': D_losses}, epoch)
        save(args.ckpt_dir, netG, netD, epoch)