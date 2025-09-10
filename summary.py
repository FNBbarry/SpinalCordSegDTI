import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.unet import Unet

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 21
    backbone        = 'vgg'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(num_classes = num_classes, backbone = backbone).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#  
    # FLOPs*2 is due to the fact that the profiling tool does not treat convolution operations as two separate components.  
    # Some studies account for both multiplication and addition operations in convolution, in which case the FLOPs are doubled.  
    # Other studies consider only multiplication operations and neglect additions, in which case no doubling is applied.  
    # This implementation adopts the doubling strategy, following the methodology used in YOLOX.  
    #--------------------------------------------------------#  
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
