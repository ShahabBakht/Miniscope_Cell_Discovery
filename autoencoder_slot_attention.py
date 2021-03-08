import sys

import torch
import torch.nn as nn

from slot_attention import SlotAttention

class conv_backend(nn.Module):
    def __init__(self, out_conv_channels = 64, input_dim = 128, device='cuda'):
        super(conv_backend, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride =1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride =1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride =1, padding = 1)
        self.relu = nn.ReLU()
        self.lnorm = nn.LayerNorm((input_dim**2,out_conv_channels))
        self.linear1 = nn.Linear(in_features=64, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=64)
        self.pos_embed = pos_embedd_layer(out_features = 64, input_dim = (input_dim, input_dim), device=device)
        

    def forward(self, x):
        
        # pass through the conv layers
        x = self.conv1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        
        # permute spaial and channel dim in x
        x = x.permute((0,2,3,1))
        
        # flatten the spatial dimensions
        b, w, h, c = x.shape
        x = x.reshape((b,w*h,c))
        
        # position embedding
        x = self.pos_embed(x)
        
        # layer norm
        x = self.lnorm(x)

        # mlp1
        x = self.linear1(x)
        x = self.relu(x)
        
        # mlp2
        x = self.linear2(x)
        
        return x
         
        
class pos_embedd_layer(nn.Module):
    def __init__(self, out_features = 64, input_dim = (128, 128), device='cuda'):
        super(pos_embedd_layer, self).__init__()
        
        self.linear = nn.Linear(in_features=4, out_features=out_features)
        self.pos_tensor = self._create_position_tensor(input_dim).to(device)
        
        
    def forward(self, x):
        
        w,h,c = self.pos_tensor.shape
        
        #flatten spatial dim
        pos_tensor_flatten = self.pos_tensor.reshape((w*h,c))
        
        # project to out_features dim
        pos_tensor_flatten_prj = self.linear(pos_tensor_flatten)
        
        x = x + pos_tensor_flatten_prj
        
        return x
        
    
    def _create_position_tensor(self,input_dim):
        w, h = input_dim
        P1 = torch.Tensor(torch.arange(start=0, end = 1, step = 1/w).repeat((h,1))).unsqueeze(2)
        P2 = torch.Tensor(torch.arange(start=1, end = 0, step = -1/w).repeat((h,1))).unsqueeze(2)
        P3 = torch.Tensor(torch.arange(start=0, end = 1, step = 1/h).repeat((w,1)).T).unsqueeze(2)
        P4 = torch.Tensor(torch.arange(start=1, end = 0, step = -1/h).repeat((w,1)).T).unsqueeze(2)
        
        P = torch.cat((P1,P2,P3,P4),2)
        
        return P

class deconv_backend(nn.Module):
    def __init__(self, spatial_broadcast_size=(8,8),in_channels = 64, device='cuda'):
        super(deconv_backend, self).__init__()
        
        self.spatial_broadcast_size = spatial_broadcast_size
        self.output_size = (self.spatial_broadcast_size[0]*2**3,self.spatial_broadcast_size[1]*2**3)
        self.pos_embed = pos_embedd_layer(out_features = in_channels, input_dim = spatial_broadcast_size, device=device)
        self.deconv1 = nn.ConvTranspose2d(in_channels = in_channels, out_channels = 64, kernel_size = 5, stride =2, padding = 2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1) #, output_padding=1
        self.deconv5 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        
        self.deconv6 = nn.ConvTranspose2d(in_channels = 64, out_channels = 2, kernel_size = 3, stride = 1, padding = 1) # kernel was 3
        
        self.relu = nn.ReLU()
        
        
        
    
    def forward(self, x):
        
        b,k,d = x.shape # (batch_size,num_slots,d_slot) 
        
        # collapse batch and slots dimensions
        x = x.reshape((b*k,d))
        x = self._spatial_broadcast(x)
        
        # flatten spatial dim
        bk,w,h,d = x.shape
        x = x.reshape((bk,w*h,d))
        
        # embed position
        x = self.pos_embed(x)
        x = x.reshape((bk,w,h,d))
        
        # permute spatial and channel dimensions
        x = x.permute((0,3,1,2))
        x = self.deconv1(x)
        x = self.relu(x)
        
        x = self.deconv2(x)
        x = self.relu(x)
        
        x = self.deconv3(x)
        x = self.relu(x)
        
        x = self.deconv4(x)
        x = self.relu(x)
        
        x = self.deconv5(x)
        x = self.relu(x)
        
        x = self.deconv6(x)
        x = self.relu(x)
        
        
        x = x.reshape((b,k,2,*self.output_size)).permute((0,1,3,4,2))
        
        mask = x[:,:,:,:,1]
        rgb = x[:,:,:,:,0]
        
        mask = nn.functional.softmax(mask,dim=1)
        mask = mask#.unsqueeze(-1).repeat((1,1,1,1,3))
        x_recon = rgb * mask
        x_recon = x_recon.sum(1).unsqueeze(1)#.permute((0,3,1,2))
#         mask = mask.permute((0,1,4,2,3))
        return x_recon, mask
        
        
    
    def _spatial_broadcast(self,x,size=(8,8)):
        
        x = x.repeat((*size,1,1)).permute((2,0,1,3))
        
        return x
        
            
class AE_SlotAttention(nn.Module):
    def __init__(self, out_conv_channels = 64, input_dim = 128, num_slots = 5, slot_iter = 3, slot_hidden_dim = 128, device = 'cuda'):
        super(AE_SlotAttention,self).__init__()
        
        self.conv_backend = conv_backend(out_conv_channels = out_conv_channels, input_dim = input_dim, device='cuda')
        self.slot_attention = SlotAttention(num_slots = num_slots, dim = out_conv_channels, iters = slot_iter, eps = 1e-8, hidden_dim = slot_hidden_dim)
        self.deconv_backend = deconv_backend()
        self._initialize_weights()
        
    def forward(self, x):
        
        if x.ndim == 5:
            multi_frame = True
            B, C, T, W, H = x.shape
            x = x.permute((0,2,1,3,4)).reshape((B * T, C, W, H))
        
        x = self.conv_backend(x)
        
        if multi_frame is True:
            BT, N, D = x.shape
            x = x.reshape((B, T, N, D))
            
        x = self.slot_attention(x)
        
        x_recon, mask = self.deconv_backend(x)
        
        return x_recon, mask
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
if __name__ == '__main__':
    device = 'cuda'
    x = torch.randn((50,1,64,64)).to(device)

    model = AE_SlotAttention(out_conv_channels = 64, 
                             input_dim = 64, 
                             num_slots = 32, 
                             slot_iter = 3, 
                             slot_hidden_dim = 128, 
                             device = 'cuda').to(device)

    x_recon, mask = model(x)
    print(x_recon.shape, x.shape, mask.shape)