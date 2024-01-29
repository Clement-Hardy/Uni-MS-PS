import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, trunc_normal_
from transformer_modules import TransformerLayer, TransformerLayer_pooling
from Transformer_8_layer import OverlapPatchEmbed, Block




class Transformer_8(nn.Module):
    def __init__(self,
                 c_in=3, 
                 patch_size=16,
                 dim_hidden=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 linear=False,
                 num_stages = 4,
                 eval_mode=False,
                 use_cuda_eval_mode=False,
                 batch_size_encoder=4,
                 batch_size_transformer=20000):
        
        super(Transformer_8, self).__init__()
        
        self.use_cuda_eval_mode = use_cuda_eval_mode
        self.eval_mode = eval_mode
        self.batch_size_encoder = batch_size_encoder
        self.batch_size_transformer = batch_size_transformer
        
        
        self.c_in = c_in
        self.depths = depths
        self.num_stages = num_stages
        self.use_cuda = False
        
        self.inference_mode = False  
        self.first_device = "cpu"
        self.last_device = "cpu"
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(patch_size=3,
                                            stride=1 if i==0 else 2,
                                            in_chans=c_in if i == 0 else dim_hidden[i - 1],
                                            embed_dim=dim_hidden[i])

            block = nn.ModuleList([Block(dim=dim_hidden[i],
                                         num_heads=num_heads[i],
                                         mlp_ratio=mlp_ratios[i],
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_scale,
                                         drop=drop_rate, 
                                         attn_drop=attn_drop_rate,
                                         drop_path=dpr[cur + j],
                                         norm_layer=norm_layer,
                                         sr_ratio=sr_ratios[i],
                                         linear=linear)
                                            for j in range(depths[i])])
            norm = norm_layer(dim_hidden[i])
            
            if i<num_stages-1:
                light_block = TransformerLayer(dim_input = dim_hidden[i],
                                               num_enc_sab = 2,
                                               dim_hidden = dim_hidden[i],
                                               dim_feedforward = 2*dim_hidden[i],
                                               num_heads = 4,
                                               ln = True,
                                               attention_dropout=0.1,
                                               eval_mode_batch_size=self.batch_size_transformer)
                setattr(self, f"light_block{i + 1}", light_block)
                
            pool_block = TransformerLayer_pooling(dim_input = dim_hidden[i],
                                                  num_enc_sab = 2,
                                                  dim_hidden = dim_hidden[i],
                                                  num_outputs = 1,
                                                  dim_feedforward = 2*dim_hidden[i],
                                                  num_heads=4,
                                                  ln=True,
                                                  attention_dropout=0.1,
                                                  eval_mode_batch_size=self.batch_size_transformer)
            
            
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
            setattr(self, f"pool_block{i + 1}", pool_block)
                
        
        dim_hidden.append(0)
        for i in range(num_stages-1):
            c_in = dim_hidden[::-1][i]
            c_in_up = dim_hidden[::-1][i+1]
            c_out = dim_hidden[::-1][i+1]
            
            c_in+=c_in_up
        
            up_block =  nn.Sequential(nn.ConvTranspose2d(c_in, c_out,
                                                         4, stride=2,
                                                         padding=1),
                               
                                   nn.LeakyReLU(0.1, inplace=False)
                                   )
            setattr(self, f"up_block{i + 1}", up_block)


        self.output_block = nn.Conv2d(c_out+dim_hidden[0],
                                      3,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=True)
        
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():            
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):# or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)


    def forward_stage(self, x, index):

        B, N, C, H, W = x.shape
        patch_embed = getattr(self, f"patch_embed{index + 1}")
        block = getattr(self, f"block{index + 1}")
        norm = getattr(self, f"norm{index + 1}")
        pool_block = getattr(self, f"pool_block{index + 1}")
        
        if index<self.num_stages-1:
            light_block = getattr(self, f"light_block{index + 1}")
        
        x = x.reshape(-1, x.shape[2], x.shape[3],
                      x.shape[4])
        B1 = x.shape[0]
        
        
        if self.eval_mode:
            if self.use_cuda_eval_mode:
                patch_embed = patch_embed.cuda()
            x1 = torch.Tensor()
            
            index_batch = 0
            batch_size = self.batch_size_encoder
            #pbar = tqdm(total=x.shape[0])
            while index_batch<x.shape[0]:
                x2 = x[index_batch:(index_batch+batch_size)]
                if self.use_cuda_eval_mode:
                    x2 = x2.cuda()
                    
                x2, H1, W1 = patch_embed(x2)
                if self.use_cuda_eval_mode:
                    x2 = x2.cpu()
                
                x1 = torch.cat((x1, x2), 0)
                index_batch+=batch_size
                
            x = x1
            if self.use_cuda_eval_mode:
                patch_embed = patch_embed.cpu()
        else:
            x = x.to(next(patch_embed.parameters()).device)
            x, H1, W1 = patch_embed(x)
            
            
        
        if self.use_cuda_eval_mode and self.eval_mode:
            x = x.cpu()
            patch_embed = patch_embed.cpu()
        for blk in block:
            if self.eval_mode:                    
                if self.use_cuda_eval_mode:
                    blk = blk.cuda()
                x1 = torch.Tensor()
                
                index_batch = 0
                batch_size = self.batch_size_encoder
                #pbar = tqdm(total=x.shape[0])
                while index_batch<x.shape[0]:
                    x2 = x[index_batch:(index_batch+batch_size)]
                    if self.use_cuda_eval_mode:
                        x2 = x2.cuda()
                    x2 = blk(x2, H1, W1)
                    if self.use_cuda_eval_mode:
                        x2 = x2.cpu()
                        
                    x1 = torch.cat((x1, x2), 0)
                    index_batch+=batch_size
                    
                x = x1
                if self.use_cuda_eval_mode:
                    blk = blk.cpu()
                    
            else:
                x = x.to(next(blk.parameters()).device)
                x = blk(x, H1, W1)
         
        x = x.to(next(norm.parameters()).device)
        x = norm(x)
        
        x = x.reshape(B1, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x = x.reshape(B,
                      N,
                      x.shape[1],
                      x.shape[2],
                      x.shape[3])
        x1 = x.permute(0, 3, 4, 1, 2)
        shape_x1 = x1.shape
        x1 = x1.reshape(-1,
                        x1.shape[-2],
                        x1.shape[-1])
        
        x1 = x1.to(next(pool_block.parameters()).device)
        x3 = pool_block(x1)
            
            
        x3 = x3.reshape(shape_x1[0],
                        shape_x1[1],
                        shape_x1[2],
                        shape_x1[4])
        x3 = x3.permute(0, -1, 1, 2)
        
        
        if index<self.num_stages-1:
            x1 = x1.to(next(light_block.parameters()).device)
            x2 = light_block(x1)
            
            if self.use_cuda_eval_mode:
                x1 = x1.cpu()
                x2 = x2.cpu()
                
            x2 = x2.reshape(shape_x1)
            x2 = x2.permute(0, 3, 4, 1, 2)
            return x2, x3
        else:
            return x3, None
        
        
    
    def forward_upscale(self, index, x, x_pool):
        up_block = getattr(self, f"up_block{index + 1}")
        if self.eval_mode and self.use_cuda_eval_mode:
            up_block = up_block.cuda()
            x = x.cuda()
        
        x = x.to(next(up_block.parameters()).device)
        x = up_block(x)
        if self.eval_mode and self.use_cuda_eval_mode:
            x = x.cpu()
            up_block = up_block.cpu()
            
        x_pool = x_pool.to(x.device)
        x = torch.cat((x, x_pool), dim=1)
        return x
    
    
    def forward(self, x):
        inputs, mask = x
        x = inputs  
        pools = []
        for i in range(self.num_stages):
            x, x_pool = self.forward_stage(x=x,
                                           index=i)

            pools.append(x_pool)
            
        pools = pools[::-1]
        for i in range(self.num_stages-1):
            x = self.forward_upscale(i,
                                     x=x,
                                     x_pool=pools[i+1])
        x = self.output_block(x)            
        normal = nn.functional.normalize(x, 2, 1)
        
        normal = normal.to(self.last_device)
        mask = mask.to(self.last_device)
        
        pred = {}
        pred['n'] = normal.masked_fill(mask, 0) 
        pred["loss"] = []
        return pred
    
    def change_eval_mode(self, eval_mode,
                         use_cuda_eval_mode=False):
        
        for index in range(self.num_stages-1): 
            getattr(self, f"light_block{index + 1}").change_eval_mode(eval_mode = eval_mode,
                                                                      use_cuda_eval_mode=use_cuda_eval_mode)
        
        for index in range(self.num_stages):
            getattr(self, f"pool_block{index + 1}").change_eval_mode(eval_mode = eval_mode,
                                                                     use_cuda_eval_mode=use_cuda_eval_mode)
        
        self.eval_mode = eval_mode
        self.use_cuda_eval_mode = use_cuda_eval_mode
        
        
    def set_inference_mode(self, use_cuda_eval_mode=False):
        self.inference_mode = True
        self.change_eval_mode(eval_mode=True,
                              use_cuda_eval_mode=use_cuda_eval_mode)            
    
        
    def cuda(self):
        for index in range(self.num_stages):
            getattr(self, f"patch_embed{index + 1}").cuda()
            getattr(self, f"block{index + 1}").cuda()
            getattr(self, f"norm{index + 1}").cuda()
            getattr(self, f"pool_block{index + 1}").cuda()
        
        for index in range(self.num_stages-1): 
            getattr(self, f"up_block{index + 1}").cuda()
            getattr(self, f"light_block{index + 1}").cuda()
            
        self.output_block.cuda()
        
        self.first_device = "cuda"
        self.last_device = "cuda"