import torch
import torch.nn as nn

class NNLM(nn.Module):
    def __init__(self,embedding_size,hidden_size,n_step,n_class):
        super(NNLM,self).__init__()
        #n_class coresponse to |V|
        self.n_step = n_step
        self.embedding_size = embedding_size
        self.C = nn.Embedding(n_class,embedding_size)
        self.H = nn.Linear(n_step*embedding_size,hidden_size,bias=True)
        self.U = nn.Linear(hidden_size,n_class,bias=False)
        self.W = nn.Linear(n_step*embedding_size,n_class,bias=True)

    def forward(self, input):
        embeddings = self.C(input)
        embeddings = embeddings.view(-1,self.n_step*self.embedding_size)
        hidden = torch.tanh(self.H(embeddings))
        output = self.W(embeddings) + self.U(hidden)
        return output