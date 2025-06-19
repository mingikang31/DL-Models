import torch 
import torch.nn as nn
from torchsummary import summary


from layers2d import (
    Conv2d_NN,
    Conv2d_NN_Spatial,
    Conv2d_NN_Attn,
    Conv2d_NN_Attn_Spatial,
    Attention2d,
    Conv2d_ConvNN_Branching,
    Conv2d_ConvNN_Spatial_Branching,
    Conv2d_ConvNN_Attn_Branching,
    Conv2d_ConvNN_Attn_Spatial_Branching,
    Attention_ConvNN_Branching,
    Attention_ConvNN_Spatial_Branching,
    Attention_ConvNN_Attn_Branching,
    Attention_ConvNN_Attn_Spatial_Branching,
    Attention_Conv2d_Branching,    
    
)

class AllConvNet(nn.Module): 
    def __init__(self, args): 
        super(AllConvNet, self).__init__()
        self.args = args
        self.model = "All Convolutional Network"
        self.layer = args.layer
        
        # Model Parameters
        self.K = int(args.K)
        self.kernel_size = int(args.kernel_size)
        self.sampling = args.sampling
        self.shuffle_pattern = args.shuffle_pattern
        self.shuffle_scale = int(args.shuffle_scale)
        self.magnitude_type = args.magnitude_type
        self.location_channels = args.location_channels
        
        self.num_heads = int(args.num_heads)
        
        self.num_samples = int(args.num_samples) if args.num_samples != 0 else "all"
        self.num_classes = int(args.num_classes)
        self.device = args.device
        
        # In Channels, Middle Channels, and Number of Layers
        self.img_size = args.img_size
        self.in_ch = int(self.img_size[0]) # Number of Channels
        self.num_layers = int(args.num_layers)
        self.channels = args.channels
        
        assert self.num_layers >= 2, "Number of layers must be at least 2"

        layers = [] 
        
        for i in range(self.num_layers):
            self.mid_ch = int(self.channels[i])
            if args.layer == "Conv2d":
                layers.append(
                    nn.Conv2d(
                        in_channels=self.in_ch, 
                        out_channels=self.mid_ch, 
                        kernel_size=self.kernel_size, 
                        stride=1, 
                        padding=(self.kernel_size - 1) // 2 if self.kernel_size % 2 == 1 else self.kernel_size // 2
                    )
                )
            elif args.layer == "ConvNN": 
                if self.sampling == "All" or "Random": 
                    layers.append(
                        Conv2d_NN(
                            in_channels=self.in_ch, 
                            out_channels=self.mid_ch, 
                            K=self.K,
                            stride=self.K, 
                            padding=0, 
                            shuffle_pattern=self.shuffle_pattern, 
                            shuffle_scale=self.shuffle_scale,
                            samples=self.num_samples,
                            magnitude_type=self.magnitude_type,
                            location_channels=self.location_channels
                        )
                    )
                elif self.sampling == "Spatial":
                    layers.append(
                        Conv2d_NN_Spatial(
                            in_channels=self.in_ch, 
                            out_channels=self.mid_ch, 
                            K=self.K,
                            stride=self.K, 
                            padding=0, 
                            shuffle_pattern=self.shuffle_pattern, 
                            shuffle_scale=self.shuffle_scale,
                            samples=self.num_samples,
                            sample_padding=0,
                            magnitude_type=self.magnitude_type,
                            location_channels=self.location_channels
                        )
                    )
            elif args.layer == "ConvNN_Attn":
                if self.sampling == "All" or "Random": 
                    layers.append(
                        Conv2d_NN_Attn(
                            in_channels=self.in_ch, 
                            out_channels=self.mid_ch, 
                            K=self.K,
                            stride=self.K, 
                            padding=0, 
                            shuffle_pattern=self.shuffle_pattern, 
                            shuffle_scale=self.shuffle_scale,
                            samples=self.num_samples,
                            magnitude_type=self.magnitude_type,
                            location_channels=self.location_channels, 
                            image_size=self.img_size[1:]
                        )
                    )   
                elif self.sampling == "Spatial":
                    layers.append(
                        Conv2d_NN_Attn_Spatial(
                            in_channels=self.in_ch, 
                            out_channels=self.mid_ch, 
                            K=self.K,
                            stride=self.K, 
                            padding=0, 
                            shuffle_pattern=self.shuffle_pattern, 
                            shuffle_scale=self.shuffle_scale,
                            samples=self.num_samples,
                            magnitude_type=self.magnitude_type,
                            location_channels=self.location_channels, 
                            image_size=self.img_size[1:],
                        )   
                    )
            elif args.layer == "Attention":
                layers.append(
                    Attention2d(
                        in_channels=self.in_ch, 
                        out_channels=self.mid_ch,
                        shuffle_pattern=self.shuffle_pattern,
                        shuffle_scale=self.shuffle_scale,
                        num_heads=self.num_heads,
                        location_channels=self.location_channels,
                    )
                )
            elif args.layer == "Conv2d/ConvNN":
                if self.sampling == "All" or "Random": 
                    layers.append(
                        Conv2d_ConvNN_Branching(
                            in_channels=self.in_ch, 
                            out_channels = self.mid_ch,
                            channel_ratio=(self.mid_ch, self.mid_ch),
                            K = self.K,
                            kernel_size=self.kernel_size,
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            samples=self.num_samples,
                            magnitude_type=self.magnitude_type, 
                            location_channels=self.location_channels,
                        )
                    )
                elif self.sampling == "Spatial":
                    layers.append(
                        Conv2d_ConvNN_Spatial_Branching(
                            in_channels=self.in_ch, 
                            out_channels = self.mid_ch,
                            channel_ratio=(self.mid_ch, self.mid_ch),
                            K = self.K,
                            kernel_size=self.kernel_size,
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            samples=self.num_samples,
                            magnitude_type=self.magnitude_type, 
                            location_channels=self.location_channels,
                        )
                    )
            elif args.layer == "Conv2d/ConvNN_Attn": 
                if self.sampling == "All" or "Random": 
                    layers.append(
                        Conv2d_ConvNN_Attn_Branching(
                            in_channels=self.in_ch, 
                            out_channels = self.mid_ch,
                            channel_ratio=(self.mid_ch, self.mid_ch),
                            K = self.K,
                            kernel_size=self.kernel_size,
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            samples=self.num_samples,
                            magnitude_type=self.magnitude_type, 
                            location_channels=self.location_channels,
                            image_size=self.img_size[1:]
                        )
                    )
                elif self.sampling == "Spatial":
                    layers.append(
                        Conv2d_ConvNN_Attn_Spatial_Branching(
                            in_channels=self.in_ch, 
                            out_channels = self.mid_ch,
                            channel_ratio=(self.mid_ch, self.mid_ch),
                            K = self.K,
                            kernel_size=self.kernel_size,
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            samples=self.num_samples,
                            magnitude_type=self.magnitude_type, 
                            location_channels=self.location_channels,
                            image_size=self.img_size[1:],
                        )
                    )
            elif args.layer == "Attention/ConvNN":
                if self.sampling == "All" or "Random": 
                    layers.append(
                        Attention_ConvNN_Branching(
                            in_channels=self.in_ch, 
                            out_channels = self.mid_ch,
                            channel_ratio=(self.mid_ch, self.mid_ch),
                            K = self.K,
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            samples=self.num_samples,
                            magnitude_type=self.magnitude_type, 
                            location_channels=self.location_channels,
                        )
                    )
                elif self.sampling == "Spatial":
                    layers.append(
                        Attention_ConvNN_Spatial_Branching(
                            in_channels=self.in_ch, 
                            out_channels = self.mid_ch,
                            channel_ratio=(self.mid_ch, self.mid_ch),
                            K = self.K,
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            samples=self.num_samples,
                            magnitude_type=self.magnitude_type, 
                            location_channels=self.location_channels,
                        )
                    )
            elif args.layer == "Attention/ConvNN_Attn":
                if self.sampling == "All" or "Random": 
                    layers.append(
                        Attention_ConvNN_Attn_Branching( 
                            in_channels=self.in_ch, 
                            out_channels = self.mid_ch,
                            channel_ratio=(self.mid_ch, self.mid_ch),
                            K = self.K,
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            samples=self.num_samples,
                            magnitude_type=self.magnitude_type, 
                            location_channels=self.location_channels,
                            image_size=self.img_size[1:]
                        )
                    )
                elif self.sampling == "Spatial":
                    layers.append(
                        Attention_ConvNN_Attn_Spatial_Branching( 
                            in_channels=self.in_ch, 
                            out_channels = self.mid_ch,
                            channel_ratio=(self.mid_ch, self.mid_ch),
                            K = self.K,
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            samples=self.num_samples,
                            magnitude_type=self.magnitude_type, 
                            location_channels=self.location_channels,
                            image_size=self.img_size[1:],
                        )
                    )
            elif args.layer == "Conv2d/Attention":
                    layers.append(
                        Attention_Conv2d_Branching(
                            in_channels=self.in_ch, 
                            out_channels = self.mid_ch,
                            channel_ratio=(self.mid_ch, self.mid_ch),
                            kernel_size=self.kernel_size,
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels,
                        )
                    )
         
            layers.append(nn.BatchNorm2d(self.mid_ch))        
            layers.append(nn.ReLU(inplace=True))
            
            self.in_ch = self.mid_ch
            
        
        self.features = nn.Sequential(*layers)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
            
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.mid_ch, self.num_classes)
        )
        
        self.to(self.device)
        self.name = f"{self.model} {self.layer}"

        
    def forward(self, x): 
        x = self.features(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def summary(self): 
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but img_size doesn't include it
            summary(self, input_size=self.img_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)
        
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    import torch
    from types import SimpleNamespace
    
    # Set up arguments
    # This is a placeholder. You can replace it with your actual argument parsing logic.
    args = SimpleNamespace(
        layer = "ConvNN", 
        num_layers = 4,
        channels = [16, 32, 64, 128],
        kernel_size = 3,
        K = 9, 
        sampling = "All",
        num_samples = 0,
        num_heads=4, 
        shuffle_pattern = "NA", 
        shuffle_scale = 4, 
        magnitude_type = "similarity",
        location_channels = False,
        num_classes = 100,
        img_size = (3, 32, 32),  # Example image size
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else "cpu"),
    )
    
    x = torch.randn(32, 3, 32, 32).to(args.device)  

    # Create the model
    model = AllConvNet(args)
    
    print("Conv2d_NN All")
    # Print parameter count
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # print(model.summary())
    # print()
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.numel():,} parameters")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}\n")
    
    #######
    print("Conv2d/Attention")
    args.num_samples =64 
    args.sampling = "Random"
    model = AllConvNet(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # print(model.summary())
    # print()
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.numel():,} parameters")        
    output = model(x)
    print(f"Output shape: {output.shape}\n")
    
    print("Attention Random")
    args.layer = "Attention/ConvNN"
    args.num_samples =8 
    args.sampling = "Spatial"
    model = AllConvNet(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # print(model.summary())
    # print()
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.numel():,} parameters")        
    output = model(x)
    print(f"Output shape: {output.shape}\n")
    
    #######
    print("Conv2d_NN Spatial")
    num_samples = 8 
    args.sampling = "Spatial"
    model = AllConvNet(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # print(model.summary())
    # print()
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.numel():,} parameters")        
    output = model(x)
    print(f"Output shape: {output.shape}\n")
    
    
    
    
    
    
    print("Conv2d")
    args.layer = "Conv2d"
    model = AllConvNet(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # print(model2.summary())
    # print()
    # for name, param in model2.named_parameters():
    #     print(f"{name}: {param.numel():,} parameters")
    
    output2 = model(x)
    
    print(f"Output shape: {output2.shape}")
