import torch
from torch import nn
import torch.nn.functional as F
from attention import SelfAttention



class CLIPEmbedding(nn.Module):
    
    def __init__(self, vocab_size: int, n_embedding: int, n_tokens: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embedding))
        
        
    def forward(self, tokens):
        #(batch_size, seq_len) -> (batch_size, seq_len, dim)
        
        x = self.token_embedding(tokens)
        
        x += self.position_embedding
        
        return x
    
class CLIPLayer(nn.Module):
    
    def __init__(self, n_heads: int, n_embedding: int):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(n_embedding)
        self.attention = SelfAttention(n_heads, n_embedding)
        self.layer_norm2 = nn.LayerNorm(n_embedding)
        self.linear_1 = nn.Linear(n_embedding, n_embedding * 4)
        self_linear_2 = nn.Linear(n_embedding * 4, n_embedding)
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        
        # (batch_size, seq_len, dim) !!!dim is 768!!
        residue = x
        
        #SELF ATTENTION
        
        x = self.layer_norm1(x)
        
        x = self.attention(x, causal_mask = True)
        
        x = x + residue
        
        ##FEED FORWARD LAYER
        
        residue = x
        
        self = self.layer_norm2(x)
        
        x = self.linear_1(x)
        
        #QUICK GELU ACTIVATION
        x = x * torch.sigmoid(1.702 * x)
        
        x = self.linear_2(x)
        
        x = x + residue
        
        return x
    
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        
        #x: (batch_size, seq_len, dim)
        
        x = x + self.attention(x)
        x = self.layerNorm1(x)
        
        x = x + self.mlp(x)
        x = self.layerNorm2(x)
        
        return x
        

class CLIP(nn.Module):
    
    def __init__(self):
        self.embbeding = CLIPEmbedding(49408, 768, 77)
        
        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])
        
        self.layerNorm = nn.LayerNorm(768)
        
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        
        tokens = tokens.type(torch.long)
        
        #batch_size, seq_len --- batch_size, seq_len, dim
        state = self.embbeding(tokens)
        
        for layer in self.layers:
            state = layer(state)
        
        #batch_size, seq_len, dim
        output = self.layerNorm