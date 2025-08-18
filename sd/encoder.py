import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBLock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            #(Batch_Size,Channel ,Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size = 3, padding = 1),

            #(Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            #(Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            #(Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            #(Batch_Size, 128, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),

            #(Batch_Size, 256, Height/2, Width/) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),

            #(Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            #(Batch_Size, 256, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),

            #(Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),
            
            #(Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),
            
            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_AttentionBLock(512),

            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512),

            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8) !!!!ACTIVATION FUNCTION!!!!
            nn.SiLU(),

            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8) !!Bottle Neck!!
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            #(Batch_Size, 8, Height/8, Width/8) -> (Batch_Size, 8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x (batch_size, 3, height, width    !!input image!!!
        # noise *Batch_Size, Output_Channels, Height, Width) !!noise to be added to the output!!

        for module in self:
            #padding works (left, right, top, bottom) so we add padding to the right and bottom
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)

            # (batch_size, 8, height/8, width/8) - two tensors of shape (batch_size, 4, height/8, width/8) are concatenated
            mean, logvar = torch.chunk(x, 2, dim=1)

            # (batch_size, 4, height/8, width/8) -> (batch_size, 4, height/8, width/8)
            logvar = torch.clamp(logvar, -30.0, 20.0)
            # (batch_size, 4, height/8, width/8) -> (batch_size, 4, height/8, width/8)
            variance= torch.exp(logvar)

            # (batch_size, 4, height/8, width/8) -> (batch_size, 4, height/8, width/8)
            std = torch.sqrt(variance)

            #Z =N(0, 1) -> N(mean, variance) = mean + std * N(0, 1)
            x = mean + std * noise
            
            #scale constant for stability, taken from paper 

            x = 0.18215 * x


