"""Improved U-Net Implementation for MRI Reconstruction with Gradient Flow Fixes"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder path
        self.input_block = InputBlock(in_channels, 64)
        self.enc1 = EncoderBlock(64, 128)
        self.enc2 = EncoderBlock(128, 256)
        self.enc3 = EncoderBlock(256, 512)
        self.enc4 = EncoderBlock(512, 1024)
        
        # Bottleneck
        self.bottleneck = BottleneckBlock(1024, 1024)
        
        # Decoder path 
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        
        # Final output layer
        self.out = OutputBlock(64, self.out_channels)
        
        # Initialize weights properly
        self.apply(self._init_weights)
        self.to(device)
    
    def _init_weights(self, module):
        """Proper weight initialization for better gradient flow"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            if module.weight is not None: 
                nn.init.constant_(module.weight, 1)
            if module.bias is not None: 
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Encoder path 
        x1 = self.input_block(x)         
        x2 = self.enc1(x1) 
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        
        # Bottleneck
        bottleneck = self.bottleneck(x5)
        
        # Decoder path 
        dec4 = self.dec4(bottleneck, x4)
        dec3 = self.dec3(dec4, x3)
        dec2 = self.dec2(dec3, x2)
        dec1 = self.dec1(dec2, x1)
    
        # Output layer        
        out = self.out(dec1)
        return out
    
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

class DoubleConv(nn.Module):
    """Double convolution block with residual connection"""
    def __init__(self, in_channels, out_channels, use_residual=True):
        super(DoubleConv, self).__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),  # Better than InstanceNorm for gradient flow
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
        )
        
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # 1x1 conv for residual connection if channel dimensions don't match
        if in_channels != out_channels and self.use_residual:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_conv = None
    
    def forward(self, x):
        identity = x
        out = self.conv(x)
        
        # Add residual connection
        if self.use_residual:
            if self.residual_conv is not None:
                identity = self.residual_conv(identity)
            out = out + identity
            
        return self.activation(out)

class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, use_residual=False)
        
    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(EncoderBlock, self).__init__() 
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x): 
        x = self.pool(x) 
        x = self.conv(x) 
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                         kernel_size=2, stride=2, bias=False)
        
        self.up_norm = nn.InstanceNorm2d(in_channels // 2)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True),

        
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up_conv(x)
        x = self.up_norm(x)
        x = self.act(x)
                
        # Ensure skip connection has same spatial dimensions
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        x = torch.cat([skip, x], dim=1)       
        x = self.conv(x)              
        return x

class BottleneckBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(BottleneckBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.2),  # Add dropout for regularization
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # Residual connection
        self.residual = in_channels == out_channels
        
    def forward(self, x):
        out = self.conv(x)
        if self.residual:
            out = out + x
        return out

class OutputBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(OutputBlock, self).__init__()
        
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
    
    def forward(self, x): 
        return self.out(x)

# Training utilities for better gradient flow
class GradientClipping:
    """Utility class for gradient clipping"""
    @staticmethod
    def clip_gradients(model, max_norm=1.0):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def get_optimizer(model, lr=1e-4, weight_decay=1e-4):
    """Get optimizer with proper settings for UNET training"""
    return torch.optim.AdamW(model.parameters(), lr=lr, 
                             weight_decay=weight_decay, betas=(0.9, 0.999), 
                             eps=1e-8)

def get_scheduler(optimizer, num_epochs):
    """Get learning rate scheduler"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    model = UNet(in_channels=1, out_channels=2, device=device) 
    x = torch.randn(2, 1, 320, 320).to(device)  # Smaller batch for testing
    print("Input shape:", x.shape)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        print("Output shape:", output.shape)
    
    # Check parameter count
    total, trainable = model.parameter_count()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Example training setup
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, num_epochs=100)
    
    print("Model initialized with gradient-friendly architecture!")