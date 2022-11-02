import torch
import torch.nn as nn

class Attention_LM(nn.Module):
    def __init__(self,embedding_size,n_class):
        super(Attention_LM,self).__init__()
        #n_class coresponse to |V|

        self.C = nn.Embedding(n_class,embedding_size)
        self.attn1 = nn.MultiheadAttention(embed_dim=embedding_size,num_heads=1,batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=embedding_size,num_heads=1,batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.W = nn.Linear(embedding_size,n_class,bias=True)

    def forward(self, input):
        embeddings = self.C(input)  #should be [B,n_step,embedding_size]
        hidden, _ = self.attn1(embeddings,embeddings,embeddings)
        hidden = self.layer_norm1(hidden + embeddings)
        
        outputs, _ = self.attn2(hidden,hidden,hidden)
        outputs = self.layer_norm2(outputs+hidden)
        output = self.W(outputs[:,-1,:])
        return output