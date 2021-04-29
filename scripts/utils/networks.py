import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
                
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        
        self.lstm = nn.LSTM(256*4, 256*4, 2, batch_first=True)

        self.conv5_t = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv4_t = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv3_t = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv2_t = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(2,3), stride=(1,2), padding=(1,0), output_padding=(0,1))
        self.conv1_t = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
                   
        self.bn5_t = nn.BatchNorm2d(128)
        self.bn4_t = nn.BatchNorm2d(64)
        self.bn3_t = nn.BatchNorm2d(32)
        self.bn2_t = nn.BatchNorm2d(16)
        self.bn1_t = nn.BatchNorm2d(1)

        self.elu = nn.ELU(inplace=True)
        self.softplus = nn.Softplus()

    def forward(self, x):
        
        out = x.unsqueeze(dim=1)
        e1 = self.elu(self.bn1(self.conv1(out)[:,:,:-1,:].contiguous()))
        e2 = self.elu(self.bn2(self.conv2(e1)[:,:,:-1,:].contiguous()))
        e3 = self.elu(self.bn3(self.conv3(e2)[:,:,:-1,:].contiguous()))
        e4 = self.elu(self.bn4(self.conv4(e3)[:,:,:-1,:].contiguous()))
        e5 = self.elu(self.bn5(self.conv5(e4)[:,:,:-1,:].contiguous()))
        
        out = e5.contiguous().transpose(1, 2)
        q1 = out.size(2)
        q2 = out.size(3)
        out = out.contiguous().view(out.size(0), out.size(1), -1)
        out, _ = self.lstm(out)
        out = out.contiguous().view(out.size(0), out.size(1), q1, q2)
        out = out.contiguous().transpose(1, 2)

        out = torch.cat([out, e5], dim=1)

        d5 = self.elu(torch.cat([self.bn5_t(F.pad(self.conv5_t(out), [0,0,1,0]).contiguous()), e4], dim=1))
        d4 = self.elu(torch.cat([self.bn4_t(F.pad(self.conv4_t(d5), [0,0,1,0]).contiguous()), e3], dim=1))
        d3 = self.elu(torch.cat([self.bn3_t(F.pad(self.conv3_t(d4), [0,0,1,0]).contiguous()), e2], dim=1))
        d2 = self.elu(torch.cat([self.bn2_t(F.pad(self.conv2_t(d3), [0,0,1,0]).contiguous()), e1], dim=1))
        d1 = self.softplus(self.bn1_t(F.pad(self.conv1_t(d2), [0,0,1,0]).contiguous()))
        
        out = torch.squeeze(d1, dim=1)

        return out

        
