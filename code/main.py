import os
import torch
import random
import yaml
import numpy as np
import argparse
import logging
from tqdm import tqdm
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam

from model.NNLM import *
from model.RNNtext import *
from model.Attention import *
from dataset.data_preprocess import *
from dataset.dataloader import *
from utils import *

logger = logging.getLogger(__name__)


def evaluate(model,test_loader,config):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    t_total = len(test_loader)
    total_loss = 0
    epoch_iterator = tqdm(test_loader,
                              desc="Testing (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
    for data in epoch_iterator:
        input_data = data['data']
        target = data['target'].squeeze()
        output = model(input_data)
        loss = criterion(output,target)
        epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, loss.item()))
        
        total_loss += loss.item()
        global_step += 1
    ppl = np.exp(total_loss/len(test_loader))
    logger.info('finish test, ppl = {} for label'.format(ppl))
        


def train(model,optimizer,train_loader,config):
    epoch = config['epoch']
    criterion = nn.CrossEntropyLoss()

    best_ppl = 999999
    global_step = 0
    t_total = epoch * len(train_loader)
    

    for i in range(epoch):
        total_loss = 0
        model.zero_grad()
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for data in epoch_iterator:
            input_data = data['data']
            target = data['target'].squeeze()
            output = model(input_data)
            loss = criterion(output,target)
            epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, loss.item()))
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        ppl = np.exp(total_loss/len(train_loader))
        logger.info('finish training {} epoch, ppl = {} for label'.format(i+1, ppl))
        
        if ppl < best_ppl:
            save_checkpoint(model,config)



    return 
    


def get_dataloader(config,train_set,test_set,vocab):
    model_type = config['model_type']

    if model_type == 'NNLM':
        n_step = config['n_step']
        train_dataset = NNLM_dataset(train_set,n_step,vocab)
        test_dataset = NNLM_dataset(test_set,n_step,vocab)
    elif model_type == 'RNN':
        n_step = config['n_step']
        train_dataset = RNN_dataset(train_set,n_step,vocab)
        test_dataset = RNN_dataset(test_set,n_step,vocab)
    elif model_type == 'Attention':
        n_step = config['n_step']
        train_dataset = Attn_dataset(train_set,n_step,vocab)
        test_dataset = Attn_dataset(test_set,n_step,vocab)
    
    train_loader = DataLoader(train_dataset,
                            batch_size = 2048,
                            shuffle=True,
                            drop_last=True,
                            num_workers = 4)
    test_loader = DataLoader(test_dataset,
                            batch_size = 2048,
                            shuffle=True,
                            drop_last=True,
                            num_workers = 4)
    return train_loader, test_loader

def get_model(vocab,config):
    model_type = config['model_type']
    
    if model_type == 'NNLM':
        embedding_size = config['embedding_size']
        hidden_size = config['hidden_size']
        n_step = config['n_step']
        n_class = len(vocab)
        
        model = NNLM(embedding_size=embedding_size,hidden_size=hidden_size,n_step=n_step-1,n_class=n_class)    
        return model    
    elif model_type == 'RNN':
        embedding_size = config['embedding_size']
        hidden_size = config['hidden_size']
        n_class = len(vocab)
        
        model = RNNtext(hidden_size=hidden_size,embedding_size = embedding_size,n_class=n_class)
        return model
    elif model_type == 'Attention':
        embedding_size = config['embedding_size']
        n_class = len(vocab)
        model = Attention_LM(embedding_size=embedding_size,n_class=n_class)
        return model

def main(config):

    ckpt = config['ckpt']
    train_set, test_set,vocab = data_preprocess()
    
    train_loader, test_loader = get_dataloader(config,train_set,test_set,vocab)


    model = get_model(vocab,config)
    
    if os.path.isfile('{}/model/model.pth'.format(ckpt)):
        logger.info("------resuming last training------")
        checkpoint = torch.load('{}/model/model.pth'.format(ckpt),map_location='cpu')
        model.load_state_dict(checkpoint['net'])
    
    
    
    optimizer = Adam(model.parameters(),
                    lr = config['lr'])

    
    train(model,optimizer,train_loader,config)

    evaluate(model, test_loader,config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    
    
    config_path = args.config

    if os.path.isfile(config_path):
        f = open(config_path)
        config = yaml.load(f,Loader=yaml.FullLoader)
        print("***********************************")
        print(yaml.dump(config, default_flow_style=False, default_style=''))
        print("***********************************")
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

    random.seed(1234)
    torch.manual_seed(1234)

    ckpt = config['ckpt']
    os.makedirs(ckpt,exist_ok=True)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s',datefmt='%m/%d/%Y %H:%M:%S')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(ckpt+'/result.log',encoding='utf8',mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    main(config)











