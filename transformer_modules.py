import torch
import torch.nn as nn
from torch.nn import functional as F
import math



class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads,
                 ln=False, attention_dropout = 0.1, dim_feedforward = 512):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.dim_V = dim_V
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.num_heads = num_heads

        self.fc_q = nn.Linear(self.dim_Q, self.dim_V) 
        self.fc_k = nn.Linear(self.dim_K, self.dim_V)
        self.fc_v = nn.Linear(self.dim_K, self.dim_V) 

        if ln:
            self.ln0 = nn.LayerNorm(self.dim_Q)
            self.ln1 = nn.LayerNorm(self.dim_V)
            
        self.dropout_attention = nn.Dropout(attention_dropout)
        self.fc_o1 = nn.Linear(self.dim_V, dim_feedforward)
        self.fc_o2 = nn.Linear(dim_feedforward, self.dim_V)
        self.dropout1 = nn.Dropout(attention_dropout)
        self.dropout2 = nn.Dropout(attention_dropout)

    def forward(self, x, y):
        x = x if getattr(self, 'ln0', None) is None else self.ln0(x)
        Q = self.fc_q(x)      
        K, V = self.fc_k(y), self.fc_v(y)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(int(dim_split), 2), 0)
        K_ = torch.cat(K.split(int(dim_split), 2), 0)
        V_ = torch.cat(V.split(int(dim_split), 2), 0)
        A = self.dropout_attention(torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2))
        A =  A.bmm(V_)
        O = torch.cat((Q_ + A).split(Q.size(0), 0), 2)
        O_ = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        O = O + self.dropout2(self.fc_o2(self.dropout1(F.gelu(self.fc_o1(O_))))) 
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=4, ln=False,
                 attention_dropout = 0.1, dim_feedforward = 512):
        super(SAB, self).__init__()
        self.mab = MultiHeadSelfAttentionBlock(dim_in, dim_in, dim_out,
                                               num_heads, ln=ln,
                                               attention_dropout = attention_dropout,
                                               dim_feedforward=dim_feedforward)
    def forward(self, X):
        return self.mab(X, X)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MultiHeadSelfAttentionBlock(dim, dim, dim,
                                               num_heads,
                                               ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        
        self.mab0 = MultiHeadSelfAttentionBlock(dim_out, dim_in,
                                                dim_out, num_heads,
                                                ln=ln)
        self.mab1 = MultiHeadSelfAttentionBlock(dim_in, dim_out,
                                                dim_out, num_heads,
                                                ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)
    
    


class TransformerLayer(nn.Module):
    def __init__(self, dim_input, num_enc_sab = 3,
                 dim_hidden=384, dim_feedforward = 1024,
                 num_heads=8, ln=False, attention_dropout=0.1,
                 eval_mode=False, use_cuda_eval_mode=False, eval_mode_batch_size=200):
        super(TransformerLayer, self).__init__()

        self.use_cuda_eval_mode = use_cuda_eval_mode
        self.eval_mode = eval_mode
        self.dim_hidden = dim_hidden
        self.eval_mode_batch_size = eval_mode_batch_size
        
        
        modules_enc = []
        modules_enc.append(SAB(dim_input, dim_hidden, num_heads,
                               ln=ln, attention_dropout = attention_dropout,
                               dim_feedforward=dim_feedforward))
        for k in range(num_enc_sab):
            modules_enc.append(SAB(dim_hidden, dim_hidden, num_heads,
                                   ln=ln, attention_dropout = attention_dropout,
                                   dim_feedforward=dim_feedforward))
        self.enc = nn.Sequential(*modules_enc)
        

    def change_eval_mode(self, eval_mode, use_cuda_eval_mode=False):
        self.eval_mode = eval_mode
        self.use_cuda_eval_mode = use_cuda_eval_mode
        
    def forward(self, x):
        if self.eval_mode:
            if self.use_cuda_eval_mode:
                self.enc = self.enc.cuda()
            x1 = torch.Tensor()
            index = 0
            batch_size = self.eval_mode_batch_size
            
            while index<x.shape[0]:
                x2 = x[index:(index+batch_size)]
                if self.use_cuda_eval_mode:
                    x2 = x2.cuda()
                        
                x2 = self.enc(x2)
                if self.use_cuda_eval_mode:
                    x2 = x2.cpu()
                        
                x1 = torch.cat((x1, x2), 0)
                    
                index+=batch_size
            x = x1
            if self.use_cuda_eval_mode:
                self.enc = self.enc.cpu()
        else:
            x = self.enc(x)
        return x


    
class TransformerLayer_pooling(nn.Module):
    def __init__(self, dim_input, num_enc_sab = 3,
                 num_outputs = 1, dim_hidden=384, dim_feedforward = 1024,
                 num_heads=8, ln=False, attention_dropout=0.1,
                 eval_mode=False, use_cuda_eval_mode=False,
                 eval_mode_batch_size=200):
        super(TransformerLayer_pooling, self).__init__()

        self.use_cuda_eval_mode = use_cuda_eval_mode
        self.eval_mode = eval_mode
        self.num_outputs = num_outputs
        self.dim_hidden = dim_hidden
        self.eval_mode_batch_size = eval_mode_batch_size

        modules_enc = []
        modules_enc.append(SAB(dim_input, dim_hidden, num_heads,
                               ln=ln, attention_dropout = attention_dropout,
                               dim_feedforward=dim_feedforward))
        for k in range(num_enc_sab):
            modules_enc.append(SAB(dim_hidden, dim_hidden, num_heads,
                                   ln=ln, attention_dropout = attention_dropout,
                                   dim_feedforward=dim_feedforward))
        self.enc = nn.Sequential(*modules_enc)
        modules_dec = []
        modules_dec.append(PMA(dim_hidden, num_heads, num_outputs))
        self.dec = nn.Sequential(*modules_dec)
    
    def change_eval_mode(self, eval_mode, use_cuda_eval_mode=False):
        self.eval_mode = eval_mode
        self.use_cuda_eval_mode = use_cuda_eval_mode
        

    def forward(self, x):
        if self.eval_mode:
            if self.use_cuda_eval_mode:
                self.enc = self.enc.cuda()
                self.dec = self.dec.cuda()
            x1 = torch.Tensor()
            index = 0
            batch_size = self.eval_mode_batch_size
            #pbar = tqdm(total=x.shape[0])
            while index<x.shape[0]:
                x2 = x[index:(index+batch_size)]
                if self.use_cuda_eval_mode:
                    x2 = x2.cuda()
    
                x2 = self.enc(x2)
                x2 = self.dec(x2)
                    
                if self.use_cuda_eval_mode:
                    x2 = x2.cpu()
                        
                x1 = torch.cat((x1, x2), 0)
                index+=batch_size

            x = x1
            if self.use_cuda_eval_mode:
                self.enc = self.enc.cpu()
                self.dec = self.dec.cpu()
        else:
            x = self.enc(x)
            x = self.dec(x)
        feat = x.view(-1, self.num_outputs * self.dim_hidden)
        return feat
    


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3,
                                  stride = stride, padding = 1),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3,
                                  stride = 1, padding = 1))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
    
class downscale_block(nn.Module):
    def __init__(self, c_in, c_out, downscale_factor=1):
        super(downscale_block, self).__init__()
        
        self.conv = nn.Sequential(torch.nn.Conv2d(c_in,
                                                  c_out,
                                                  kernel_size=3,
                                                  stride=1, 
                                                  padding=1),
                                  nn.LeakyReLU(0.1, inplace=False))
        if downscale_factor>1:
            downsample = nn.Conv2d(c_out,
                                   c_out,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1)
        else:
            downsample = None
        self.block = ResidualBlock(in_channels=c_out,
                                   out_channels=c_out,
                                   stride=downscale_factor,
                                   downsample=downsample)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        return x
    

class upscale_block(nn.Module):
    def __init__(self, c_in, c_out, upscale_factor=2):
        super(upscale_block, self).__init__()
        
        self.conv = nn.Sequential(
                            nn.ConvTranspose2d(c_in, c_out, kernel_size=4,
                                               stride=2, padding=1, bias=False),
                            nn.LeakyReLU(0.1, inplace=False)
                            )
        self.block = ResidualBlock(in_channels=c_out,
                                   out_channels=c_out,
                                   stride=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        return x
    
