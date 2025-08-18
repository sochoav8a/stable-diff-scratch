import torch 
from torch import nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    
    def __init__(self, num_heads: int, dim_embed: int, in_proj_bias =True, out_proj_bias=True):
        super().__init__()
        
        self.in_proj = nn.Linear(dim_embed, dim_embed * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(dim_embed, dim_embed, bias=out_proj_bias)
        self.num_heads = num_heads
        self.dim_head = dim_embed // num_heads
        
    def forward(self, x: torch.Tensor, causal_mask = True) -> torch.Tensor:
        
        # x : (Batch_Size, seq_len, dim_embed)
        
        input_shape = x.shape
        batch_size, sequence_length, dim_embed = input_shape

        intermin_shape = (batch_size, sequence_length, self.num_heads, self.dim_head)
        
        # (batch_size, seq_len, dim) --- batch_size, seq_len, 3 * dim)
        #!!3 tensor of shape batch_size, seq_len, dim!!
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        
        q = q.view(intermin_shape).transpose(1, 2)  # (batch_size, H, seq_len, dim/H)
        v = v.view(intermin_shape).transpose(1, 2)  # (batch_size, H, seq_len, dim/H)
        k = k.view(intermin_shape).transpose(1, 2)  # (batch_size, H, seq_len, dim/H)

        # (batch_size, H, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)  
        
        if causal_mask:
            mask = torch.ones_like(weight, dtype = torch.bool).triu(diagonal=1)
            weight = weight.masked_fill(mask, -torch.inf)
            
        weight/= torch.sqrt(torch.tensor(self.dim_head, dtype=torch.float32))
        weight = F.softmax(weight, dim=-1)
        
        
        # batch_size, H, seq_len, seq_len) -> (batch_size, H, seq_len, dim/H)
        output = weight @ v  
        
        #(batch_size, H, seq_len, dim/H) -> (batch_size, seq_len, H, dim/H)
        output = output.transpose(1, 2)
        
        output = output.reshape(input_shape)
        
        output = self.out_proj(output)
           
        return output
    
    
        