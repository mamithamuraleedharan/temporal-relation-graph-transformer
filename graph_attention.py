import torch.nn as nn


class GraphMultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim = 256,num_heads = 4,batch_first = True)

    def forward(self,x,attention_mask):
        attn_output,attn_output_weights = self.attention(x,x,x,attn_mask= attention_mask)
        return attn_output,attn_output_weights
        
        

