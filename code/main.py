import argparse
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
import random
import numpy as np
import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch_geometric.nn import GCNConv,GATConv,GraphConv,GATv2Conv,RGATConv,RGCNConv
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import glob
import gc

from collections import defaultdict
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import AutoTokenizer, AutoModel, AutoConfig,get_cosine_schedule_with_warmup,DebertaV2TokenizerFast

import re
import os


from utils import get_logger,seed_everything,get_optimizer_params,get_optimizer_params_diff,get_scheduler,train_fn,validate_fn,get_tokenizer
from processor import MuTualProcessor,_truncate_seq_pair,convert_examples_to_features
from dataset import MuTualDataset
from model import MuTualModel
from criterion import criterion1,criterion2
from metricmonitor import MetricMonitor




RANDOM_SEED = 42



def parse_args():
    parser = argparse.ArgumentParser('Training args...')
    parser.add_argument('--data_dir', default="data/mutual", help='input data path')
    parser.add_argument('--max_seq_length', default = 512, help='max_len')
    parser.add_argument('--max_utterance_num', default=30, help='max_utterance_num')
    parser.add_argument('--model_name', default="microsoft/deberta-v2-xxlarge", help='model name')
    parser.add_argument('--epochs', default= 10, help='epochs')
    parser.add_argument('--lr', default= 2e-6, help='learning_rate')
    parser.add_argument('--epoch_number', default = 200, help='Epoch number.')
    parser.add_argument('--learning_rate', default = 2e-5, help='Learning rate.')
    parser.add_argument('--output_dir', default="output/", help='output path')
    parser.add_argument('--batch_size', default=2, help='batch size(all)')
    parser.add_argument('--h_dim', default = 1536, help="the model'hidden size ")
    parser.add_argument('--max_grad_norm', default = 100, help="grad_norm")
    parser.add_argument('--eps', default = 1e-6, help="optimizer'eps ")
    parser.add_argument('--num_options', default = 4, help="the number of options ")
    parser.add_argument('--num_train_data', default = 7088, help="the number of traning data ")
    parser.add_argument('--betas', default = (0.9, 0.999), help="betas")
    parser.add_argument('--scheduler_name', default = 'OneCycleLR', help="scheduler")
    parser.add_argument('--max_lr', default = 2e-6, help="paras of scheduler ")
    parser.add_argument('--pct_start', default = 0.1, help="paras of scheduler ")
    parser.add_argument('--anneal_strategy', default = 'cos', help="paras of scheduler ")
    parser.add_argument('--div_factor', default = 1e2, help="paras of scheduler ")
    parser.add_argument('--final_div_factor', default = 1e2, help="paras of scheduler ")
    

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    args.params = {
    'scheduler_name': args.scheduler_name,
    'max_lr': args.max_lr,                 
    'pct_start': args.pct_start,               
    'anneal_strategy': args.anneal_strategy,       
    'div_factor': args.div_factor,              
    'final_div_factor': args.final_div_factor,        
}
    
    logger = get_logger(args.output_dir+'train')
    seed_everything(RANDOM_SEED)
    
    # Device Optimization
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("GPU found (", torch.cuda.get_device_name(torch.cuda.current_device()), ")")
        print("num device avail: ", torch.cuda.device_count())
    else:
        device = torch.device('cpu')
    
    print(f'Using device: {device}')
    
    
    tokenizer = get_tokenizer(args)
                             
    processor = MuTualProcessor(logger)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    train_examples = processor.get_train_examples(args.data_dir)
    val_examples = processor.get_dev_examples(args.data_dir)
    
    train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, args.max_utterance_num, tokenizer,logger)
    
    val_features = convert_examples_to_features(
                val_examples, label_list, args.max_seq_length, args.max_utterance_num, tokenizer,logger)
    
    
    gc.collect()
    train_dataset = MuTualDataset(train_features)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    
    val_dataset = MuTualDataset(val_features)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    model = MuTualModel(args.model_name,args)
    model = model.to(device)
    
    
    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=args.lr, 
                                                decoder_lr=args.lr,
                                                weight_decay=0.01)
    optimizer = AdamW(optimizer_parameters)
    
    scheduler = get_scheduler(optimizer,args.params,args)
    best=0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f'******************** Training Epoch: {epoch} ********************')
        train_fn(train_dataloader, model, criterion1, optimizer,args,device,epoch,scheduler)
        r4_1,r4_2,mrr = validate_fn(val_dataloader, model, criterion2,args,device,epoch)
        logger.info(f"Epoch: {epoch:02}. Valid. R4_1:{r4_1} R4_2:{r4_2} MRR:{mrr}")
        if r4_1 >= best:
            best = r4_1
            logger.info(f'{r4_1} model saved')
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_model.bin"))
