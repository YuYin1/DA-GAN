## using the same model as TP-GAN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from model.layers import *
from model.common import elementwise_mult_cast_int
emci = elementwise_mult_cast_int

from model import common
import cv2

def make_model(args):
    net = MFSR(args)

    return net

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = int(in_dim//4) , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = int(in_dim//4) , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature 
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out, attention

class Encoder(nn.Module):
    def __init__(self, args, local_feature_layer_dim = 0 , use_batchnorm = True, scaling_factor = 1.0 , fm_mult = 1.0):
        super(Encoder,self).__init__()
        n_fm_encoder = [64,64,128,256,512]   
        n_fm_encoder = emci(n_fm_encoder , fm_mult)

        #encoder
        #128x128
        self.conv0 = sequential( conv( args.n_colors * 2, n_fm_encoder[0]  , 7 , 1 , 3 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    ResidualBlock( 64 , 64 , 7 , 1 , 3 , "kaiming" , nn.LeakyReLU(1e-2) , scaling_factor = scaling_factor)
                                  )
        #64x64
        self.conv1 = sequential( conv( n_fm_encoder[1]  , n_fm_encoder[1]  , 5 , 2 , 2 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    ResidualBlock( 64 , 64 , 5 , 1 , 2 , "kaiming" , nn.LeakyReLU(1e-2) , scaling_factor = scaling_factor)
                                  )
        #32x32
        self.conv2 = sequential( conv( n_fm_encoder[1]  , n_fm_encoder[2] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    ResidualBlock( 128 , 128 , 3 , 1 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , scaling_factor = scaling_factor)
                                  )
        #16x16
        self.conv3 = sequential( conv( n_fm_encoder[2] , n_fm_encoder[3] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    ResidualBlock( 256 , 256 , 3 , 1 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , is_bottleneck = False , scaling_factor = scaling_factor)
                                  )
        #8x8
        self.conv4 = sequential( conv( n_fm_encoder[3] , n_fm_encoder[4] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm),
                                    *[ ResidualBlock( 512 , 512 , 3 , 1 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , is_bottleneck = False , scaling_factor = scaling_factor) for i in range(4) ]
                                  )
        self.fc1 = nn.Linear( n_fm_encoder[4]*8*8 , 512)
        self.fc2 = nn.MaxPool1d( 2 , 2 , 0)
        torch.nn.functional.max_pool1d

    def forward(self, I128):
        #encoder
        conv0 = self.conv0( I128)#128x128
        conv1 = self.conv1( conv0)#64x64
        conv2 = self.conv2( conv1)#32x32
        conv3 = self.conv3( conv2)#16x16
        conv4 = self.conv4( conv3)#8x8

        fc1 = self.fc1( conv4.view( conv4.size()[0] , -1 ))
        fc1 = self.fc2( fc1.view( fc1.size()[0] , -1 , 2  )).view( fc1.size()[0] , -1 ) 

        return conv0, conv1, conv2, conv3, conv4, fc1

class Decoder(nn.Module):
    def __init__(self, args, zdim , local_feature_layer_dim = 0 , use_batchnorm = True , use_residual_block = True , scaling_factor = 1.0 , fm_mult = 1.0):
        super(Decoder,self).__init__()
        n_fm_encoder = [64,64,128,256,512] 
        n_fm_decoder_initial = [64,32,16,8] 
        n_fm_decoder_reconstruct = [512,256,128,64]
        n_fm_decoder_conv = [64,32]
        n_fm_decoder_initial = emci( n_fm_decoder_initial , fm_mult )
        n_fm_decoder_reconstruct = emci( n_fm_decoder_reconstruct , fm_mult )
        n_fm_decoder_conv = emci( n_fm_decoder_conv , fm_mult )
        self.zdim = zdim
    
        #decoder
        self.initial_8    = deconv( 256 + self.zdim , n_fm_decoder_initial[0] , 8 , 1 , 0 , 0 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.initial_32   = deconv( n_fm_decoder_initial[0] , n_fm_decoder_initial[1] , 3 , 4 , 0 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.initial_64   = deconv( n_fm_decoder_initial[1] , n_fm_decoder_initial[2] , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.initial_128  = deconv( n_fm_decoder_initial[2] , n_fm_decoder_initial[3] , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)

        dim8 = self.initial_8.out_channels + n_fm_encoder[4] #self.conv4.out_channels
        self.before_select_8 = ResidualBlock( dim8 , dim8 , 2 , 1 , padding = [1,0,1,0] , activation = nn.LeakyReLU() )
        self.reconstruct_8 = sequential( *[ResidualBlock( dim8 , dim8 , 2 , 1 , padding = [1,0,1,0] , activation = nn.LeakyReLU() ) for i in range(2)] )

        
        self.reconstruct_deconv_16 = deconv( self.reconstruct_8.out_channels , n_fm_decoder_reconstruct[0] , 3 , 2 , 1 , 1, 'kaiming' , nn.ReLU() , use_batchnorm )
        dim16 = n_fm_encoder[3] # self.conv3.out_channels
        self.before_select_16 = ResidualBlock( dim16 , activation =nn.LeakyReLU() )
        self.reconstruct_16 = sequential( *[ResidualBlock( self.reconstruct_deconv_16.out_channels + self.before_select_16.out_channels , activation = nn.LeakyReLU() )for i in range(2)])

        self.reconstruct_deconv_32 = deconv( self.reconstruct_16.out_channels , n_fm_decoder_reconstruct[1] , 3 , 2 , 1 , 1, 'kaiming' , nn.ReLU() , use_batchnorm )
        dim32 = n_fm_encoder[2] + self.initial_32.out_channels + args.n_colors * 2 # self.conv2.out_channels
        self.before_select_32 = ResidualBlock( dim32 , activation = nn.LeakyReLU() )
        self.reconstruct_32 = sequential( *[ResidualBlock( dim32 + n_fm_decoder_reconstruct[1]   , activation = nn.LeakyReLU()) for i in range(2) ]  )
        self.decoded_img32 = conv( self.reconstruct_32.out_channels , args.n_colors , 3 , 1 , 1 , None ,  None )

        self.reconstruct_deconv_64 = deconv( self.reconstruct_32.out_channels , n_fm_decoder_reconstruct[2] , 3 , 2 , 1 , 1 , 'kaiming' , nn.ReLU() , use_batchnorm )
        dim64 = n_fm_encoder[1] + self.initial_64.out_channels + args.n_colors * 2 # self.conv1.out_channels
        self.before_select_64 = ResidualBlock(  dim64 , kernel_size =  5 , activation = nn.LeakyReLU()   ) 
        self.reconstruct_64 = sequential( *[ResidualBlock( dim64 + n_fm_decoder_reconstruct[2] + args.n_colors , activation = nn.LeakyReLU()) for i in range(2)])
        self.decoded_img64 = conv( self.reconstruct_64.out_channels , args.n_colors , 3 , 1 , 1 , None ,  None )

        self.reconstruct_deconv_128 = deconv( self.reconstruct_64.out_channels , n_fm_decoder_reconstruct[3] , 3 , 2 , 1 , 1 , 'kaiming' , nn.ReLU() , use_batchnorm )
        dim128 = n_fm_encoder[0] + self.initial_128.out_channels + args.n_colors * 2 # self.conv0.out_channels
        self.before_select_128 = ResidualBlock( dim128  , kernel_size = 7 , activation = nn.LeakyReLU()  )
        self.reconstruct_128 = sequential( *[ResidualBlock( dim128 + n_fm_decoder_reconstruct[3] + args.n_colors, kernel_size = 5 , activation = nn.LeakyReLU())] )
        self.conv5 = sequential( conv( self.reconstruct_128.out_channels , n_fm_decoder_conv[0] , 5 , 1 , 2 , 'kaiming' , nn.LeakyReLU() , use_batchnorm  ) , \
                ResidualBlock(n_fm_decoder_conv[0] , kernel_size = 3 , activation = nn.LeakyReLU() ))
        self.conv6 = conv( n_fm_decoder_conv[0] , n_fm_decoder_conv[1] , 3 , 1 , 1 , 'kaiming' , nn.LeakyReLU() , use_batchnorm )
        self.decoded_img128 = conv( n_fm_decoder_conv[1] , args.n_colors , 3 , 1 , 1 , None , activation = None )
        
        self.attn1 = Self_Attn(self.reconstruct_32.out_channels)
        self.attn2 = Self_Attn(self.reconstruct_64.out_channels)

    def forward(self, conv0, conv1, conv2, conv3, conv4, fc2, I128 , I64 , I32 , z ):
        #decoder
        initial_8   = self.initial_8( torch.cat([fc2,z] , 1).view( fc2.size()[0] , -1 , 1 , 1 )  )
        initial_32  = self.initial_32( initial_8)
        initial_64  = self.initial_64( initial_32)
        initial_128 = self.initial_128( initial_64)

        conv4 = self.before_select_8( torch.cat( [initial_8,conv4] , 1 ) )
        conv4 = self.reconstruct_8( conv4 )
        assert conv4.shape[2] == 8

        conv4 = self.reconstruct_deconv_16( conv4 )
        conv3 = self.before_select_16( conv3 )
        conv3 = self.reconstruct_16( torch.cat( [conv4 , conv3] , 1 ) )
        assert conv3.shape[2] == 16

        conv3 = self.reconstruct_deconv_32( conv3 )
        conv2 = self.before_select_32( torch.cat( [initial_32 , conv2 , I32] ,  1 ) )
        conv2 = self.reconstruct_32( torch.cat( [conv3,conv2] , 1 ) )
        conv2, p1 = self.attn1(conv2)
        decoded_img32 = self.decoded_img32( conv2 )
        assert decoded_img32.shape[2] == 32

        conv2 = self.reconstruct_deconv_64( conv2 )
        conv1 = self.before_select_64( torch.cat( [initial_64 , conv1 , I64] , 1 ) )
        conv1 = self.reconstruct_64( torch.cat( [conv2 , conv1 , torch.nn.functional.upsample( decoded_img32.data, (64,64) , mode = 'bilinear') ] , 1))
        conv1, p2 = self.attn2(conv1)
        decoded_img64 = self.decoded_img64( conv1 )
        assert decoded_img64.shape[2] == 64

        conv1 = self.reconstruct_deconv_128( conv1 )
        conv0 = self.before_select_128( torch.cat( [initial_128 , conv0 , I128 ] , 1 ) )
        conv0 = self.reconstruct_128( torch.cat( [conv1 , conv0 , torch.nn.functional.upsample(decoded_img64 , (128,128) , mode = 'bilinear' ) ] , 1 ) )
        conv0 = self.conv5( conv0 )
        conv0 = self.conv6( conv0 )
        decoded_img128 = self.decoded_img128( conv0 )
        return decoded_img128 , decoded_img64 , decoded_img32 , fc2

class MFSR(nn.Module):
    def __init__(self, args, use_batchnorm = True , use_residual_block = True):
        super(MFSR,self).__init__()
        print("Model: self attention")
        zdim = 64
        num_classes = 1000
        self.encoder = Encoder(args, use_batchnorm = use_batchnorm)
        self.decoder = Decoder(args, zdim , use_batchnorm = use_batchnorm)
        

    def forward( self, Img , use_dropout = True  ):
        # Img: hr_p, lr_p_x2, lr_p_x4
        I128 = self.mc(Img[0])
        I64  = self.mc(Img[1])
        I32  = self.mc(Img[2])

        z = Variable(torch.FloatTensor( np.random.uniform(-1,1,(Img[0].size()[0], 64)) ).cuda())
        
        conv0, conv1, conv2, conv3, conv4, fc2 = self.encoder(I128)
        I128_fake, I64_fake, I32_fake, encoder_feature = self.decoder( 
                                                            conv0, conv1, conv2, conv3, conv4, fc2, 
                                                            I128, I64, I32, z)

        return I128_fake , I64_fake , I32_fake


    def mc(self, tensor): 
        # mirror and then concat
        inv_idx = torch.arange(tensor.size(0)-1, -1, -1).long().cuda()
        inv_tensor = tensor.index_select(0, inv_idx)
        inv_tensor = tensor[inv_idx]
        return torch.cat([tensor, inv_tensor], 1)

