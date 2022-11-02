import torch
import torch.nn as nn

class RNNtext(nn.Module):
    def __init__(self,hidden_size,embedding_size,n_class):
        super(RNNtext,self).__init__()
        #n_class coresponse to |V|
        self.hidden_size = hidden_size
        self.C = nn.Embedding(n_class,embedding_size)
        self.rnn = nn.RNN(embedding_size,hidden_size)
        self.W = nn.Linear(hidden_size,n_class,bias=True)

    def forward(self, input):
        B,_= input.shape
        hidden = torch.zeros(1,B,self.hidden_size)
        input = self.C(input).permute(1,0,2)
        outputs, hidden = self.rnn(input,hidden)
        output = outputs[-1]
        model_output = self.W(output)    
        return model_output