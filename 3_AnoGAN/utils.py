import os
import torch

def save(ckpt_dir, netG, netD, epoch, best=False):
    os.makedirs(ckpt_dir, exist_ok=True)        
    
    if(best):
        past_best = sorted(os.listdir(ckpt_dir))[0]
        if(past_best.split('_')[0]=='best'):
            os.remove(os.path.join(ckpt_dir, past_best))
        torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict()}, ckpt_dir+"/best_epoch{0}.pth".format(epoch))
        
    else:
        torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict()}, ckpt_dir+"/epoch{0}.pth".format(epoch))
        
def anomaly_score(x, z, netD, netG, lam=0.1):
    netD.eval()
    netG.eval()
    
    G_z = netG(z)
    residual_loss = torch.mean(torch.abs(x - G_z))    
    
    _, x_feature   = netD(x) 
    _, G_z_feature = netD(G_z)
    discrimination_loss = torch.mean(torch.abs(x_feature-G_z_feature))
    
    total_loss = (1-lam)*residual_loss + lam*discrimination_loss
    return total_loss