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
from utils import batch_graphify



class Co_attention_head(nn.Module):
    def __init__(self,cfg):
        super(Co_attention_head, self).__init__()
        self.q1 = nn.Linear(cfg.h_dim,cfg.h_dim //4)
        self.q2 = nn.Linear(cfg.h_dim,cfg.h_dim //4)
        self.q3 = nn.Linear(cfg.h_dim,cfg.h_dim //4)
        self.q4 = nn.Linear(cfg.h_dim,cfg.h_dim //4)
    
        self.k1 = nn.Linear(cfg.h_dim,cfg.h_dim //4)
        self.k2 = nn.Linear(cfg.h_dim,cfg.h_dim //4)
        self.k3 = nn.Linear(cfg.h_dim,cfg.h_dim //4)
        self.k4 = nn.Linear(cfg.h_dim,cfg.h_dim //4)
        
        self.v1 = nn.Linear(cfg.h_dim,cfg.h_dim //4)
        self.v2 = nn.Linear(cfg.h_dim,cfg.h_dim //4)
        self.v3 = nn.Linear(cfg.h_dim,cfg.h_dim //4)
        self.v4 = nn.Linear(cfg.h_dim,cfg.h_dim //4)
        
        self.layer_norm = nn.LayerNorm(cfg.h_dim)
    def forward(self, node1,node2,cfg):
        q1 = self.q1(node1)
        q2 = self.q2(node1)
        q3 = self.q3(node1)
        q4 = self.q4(node1)
        
        k1 = self.k1(node2)
        k2 = self.k2(node2)
        k3 = self.k3(node2)
        k4 = self.k4(node2)
        
        v1 = self.v1(node2)
        v2 = self.v2(node2)
        v3 = self.v3(node2)
        v4 = self.v4(node2)
        
        r1 = (torch.matmul(q1,k1.permute(1,0))/((cfg.h_dim/4)**0.5))*v1
        r2 = (torch.matmul(q2,k2.permute(1,0))/((cfg.h_dim/4)**0.5))*v2
        r3 = (torch.matmul(q3,k3.permute(1,0))/((cfg.h_dim/4)**0.5))*v3
        r4 = (torch.matmul(q4,k4.permute(1,0))/((cfg.h_dim/4)**0.5))*v4
        
        return self.layer_norm(torch.cat((r1,r2,r3,r4),-1)) 

class MuTualModel(nn.Module):
    def __init__(self,model_name,cfg):
        super(MuTualModel, self).__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name,config=self.config)
        
        self.conv3 = GraphConv(768,64)
    
        self.conv1 = RGCNConv(cfg.h_dim,768,8)
    
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=cfg.h_dim, nhead=2,batch_first=True)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=64, nhead=2,batch_first=True)
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.coattention =Co_attention_head(cfg)
        
        self.linear4 = nn.Linear(cfg.h_dim,cfg.h_dim)
        self.linear5 = nn.Linear(cfg.h_dim,64)
        self.linear6 = nn.Linear(64,8)

        self.relu = nn.ReLU()
        self.softmax1 = nn.Softmax(dim=0)


  
    def forward(self, input_ids, input_mask,segment_ids, cls_pos,cfg,device):
        outputs = self.model(input_ids,attention_mask=input_mask,token_type_ids=segment_ids)
        sequence_output = outputs[0]  
    

    
        nodes_feature_batch,edge_index_batch,options_cls_batch,options_edge_index_batch = batch_graphify(sequence_output,cls_pos,cfg,device)
        batch_size = len(sequence_output)
        options_batch_raw = torch.zeros((batch_size*cfg.num_options,cfg.h_dim),dtype = torch.float32).to(device)
        for index,i in enumerate(options_cls_batch):
            options_batch_raw[index] = nodes_feature_batch[i]
        options_batch_raw = options_batch_raw.view(batch_size,cfg.num_options,cfg.h_dim)
    
        options_batch_mutual = self.encoder_layer1(options_batch_raw)
   
        options_batch_mutual = options_batch_mutual.view(batch_size*cfg.num_options,cfg.h_dim)

        for index,i in enumerate(options_cls_batch):
            nodes_feature_batch[i] = options_batch_mutual[index]

        
        
        edge_type = torch.zeros(edge_index_batch.size(1),dtype=torch.long).to(device)

        for i in range(edge_index_batch.size(1)):
            start = edge_index_batch[0,i]
            end = edge_index_batch[1,i]
            temp = self.relu(self.coattention(nodes_feature_batch[start].unsqueeze(0),nodes_feature_batch[end].unsqueeze(0),cfg))
            #temp = torch.cat((word_nodes_feature_batch_a[start],word_nodes_feature_batch_a[end]),-1)
            #temp = self.tanh(self.linear3(temp))
            temp = self.relu(self.linear4(temp))
            temp = self.relu(self.linear5(temp))
            result = self.softmax1(self.linear6(temp))
            edge_type[i]=torch.argmax(result,dim=1)
    
            # edge_index_batch = edge_index_batch[:,edge_type != 8]
            # edge_type = edge_type[edge_type != 8]

        output = self.conv1(nodes_feature_batch,edge_index_batch,edge_type)
        output = self.conv3(output,edge_index_batch)

        options_batch = torch.zeros((batch_size*cfg.num_options,64),dtype = torch.float32).to(device)
        cnt = 0
        for i in options_cls_batch:
            options_batch[cnt] = output[i]
            cnt+=1
    
        options_batch = options_batch.view(batch_size,cfg.num_options,64)
    
        options_batch = self.encoder_layer2(options_batch)
        options_batch = self.tanh(self.linear1(options_batch))
    
        logits = self.linear2(options_batch)
    

        return logits