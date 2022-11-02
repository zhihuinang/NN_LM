import torch
import numpy as np
from torch.utils.data import Dataset

class NNLM_dataset(Dataset):
    def __init__(self,list_sample,n_step,vocab):
        super().__init__()
        self.list_sample = list_sample
        self.n_step = n_step
        self.vocab = vocab

    def __len__(self):
        return int((len(self.list_sample)-self.n_step)/self.n_step)

    def __getitem__(self, index):
        start_idx = index * self.n_step
        data_list = self.vocab(self.list_sample[start_idx:start_idx+self.n_step-1])
        target_list = [self.vocab([self.list_sample[start_idx+self.n_step-1]])]
        
        data_list = torch.tensor(data_list)
        target_list = torch.tensor(target_list)
        return {'data':data_list,'target':target_list}


class RNN_dataset(Dataset):
    def __init__(self,list_sample,n_step,vocab):
        super(RNN_dataset,self).__init__()
        self.list_sample = list_sample
        self.n_step = n_step
        self.vocab = vocab

    def __len__(self):
        return int((len(self.list_sample)-self.n_step)/self.n_step)

    def __getitem__(self, index):
        start_idx = index * self.n_step
        data_list = self.vocab(self.list_sample[start_idx:start_idx+self.n_step-1])
        target_list = [self.vocab([self.list_sample[start_idx+self.n_step-1]])]
        data_list = torch.tensor(data_list)
        target_list = torch.tensor(target_list)
        return {'data':data_list,'target':target_list}


class Attn_dataset(Dataset):
    def __init__(self,list_sample,n_step,vocab):
        super(Attn_dataset,self).__init__()
        self.list_sample = list_sample
        self.n_step = n_step
        self.vocab = vocab

    def __len__(self):
        return int((len(self.list_sample)-self.n_step)/self.n_step)

    def __getitem__(self, index):
        start_idx = index * self.n_step
        data_list = self.vocab(self.list_sample[start_idx:start_idx+self.n_step-1])
        target_list = [self.vocab([self.list_sample[start_idx+self.n_step-1]])]
        
        data_list = torch.tensor(data_list)
        target_list = torch.tensor(target_list)
        return {'data':data_list,'target':target_list}