import os
import numpy as np
import random
import torch
from tqdm import tqdm

from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from transformers import AutoTokenizer, AutoModel, AutoConfig,get_cosine_schedule_with_warmup,DebertaV2TokenizerFast
from metricmonitor import MetricMonitor
from torch.optim.lr_scheduler import OneCycleLR


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_logger(filename):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def get_tokenizer(cfg):
    if "deberta-v2" in cfg.model_name:
        tokenizer = DebertaV2TokenizerFast.from_pretrained(cfg.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    
    return tokenizer


def get_edge_index(length,cfg):
    all_edges = set()
    all_options_edges = set()
    num_edges = (length-cfg.num_options)*(length-cfg.num_options-1) + (length-cfg.num_options)*cfg.num_options
    #edge_index=numpy.zeros((2,num_labels,dtype=int)

    for i in range(length-cfg.num_options,length):
        for j in range(length-cfg.num_options,length):
            all_options_edges.add((i,j))
  
    #cnt=0
    for i in range(length-cfg.num_options):
        for j in range(length-cfg.num_options):
            if i != j:
                all_edges.add((i,j))
    for i in range(length-cfg.num_options):
        for j in range(length-cfg.num_options,length):
            all_edges.add((i,j))
    
    assert num_edges == len(all_edges)

    return list(all_edges),len(all_edges),list(all_options_edges)




def batch_graphify(batch_output,batch_cls_pos,cfg,device):
    #edge_length_sum = 0
    batch_size = len(batch_output)
    

    node_length_sum = 0
    edge_index_batch = []
    options_edge_index_batch = []
    nodes_feature_batch_first = batch_output[:,0,:] 
    
    nodes_feature_list = []
    options_cls_batch = []
    for i in range(batch_size):
        edges,edges_length,options_edges = get_edge_index(batch_cls_pos[i,-1],cfg)
        
        edges_s = [(item[0]+node_length_sum, item[1]+node_length_sum) for item in edges]
        
        for item in edges_s:
            edge_index_batch.append(torch.tensor([item[0], item[1]]))
        options_edges_s = [(item[0]+node_length_sum, item[1]+node_length_sum) for item in options_edges]
        #
        for item in options_edges_s:
            options_edge_index_batch.append(torch.tensor([item[0], item[1]]))


    #edge_length_sum+=edges_length
        node_length_sum+=batch_cls_pos[i,-1]
    
        nodes_feature = nodes_feature_batch_first[i].unsqueeze(0) 
    
        for j in range(len(batch_cls_pos[i])-1):
            if batch_cls_pos[i,j] != 0:
                nodes_feature = torch.cat((nodes_feature,batch_output[i,batch_cls_pos[i,j]].unsqueeze(0)),0)
        
        nodes_feature_list.append(nodes_feature)
        for k in range(-cfg.num_options,0):
            options_cls_batch.append(node_length_sum+k)
    

  
    nodes_feature_batch = nodes_feature_list[0]
    for i in range(len(nodes_feature_list)):
        if i !=0:
            nodes_feature_batch = torch.cat((nodes_feature_batch,nodes_feature_list[i]),0)

    nodes_feature_batch = nodes_feature_batch.to(device)
    edge_index_batch = torch.stack(edge_index_batch).transpose(0, 1).to(device)
    options_edge_index_batch = torch.stack(options_edge_index_batch).transpose(0, 1).to(device)

    options_cls_batch = torch.tensor(options_cls_batch).to(device)

    return nodes_feature_batch,edge_index_batch,options_cls_batch,options_edge_index_batch



def train_fn(train_loader, model, criterion1, optimizer ,cfg,device,epoch,scheduler=None):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    for step,batch in enumerate(stream):
        
        input_ids = batch["input_ids"].to(device)
        input_mask = batch["input_mask"].to(device)
        segment_ids = batch["segment_ids"].to(device)
        cls_pos = batch["cls_pos"].to(device)
        label = batch["label_id"].to(device)
        #edge_index = batch["edge_index"].to(device)
        with torch.cuda.amp.autocast(enabled=True):
            preds = model(input_ids,input_mask,segment_ids,cls_pos,cfg,device)

        
        loss = criterion1(preds, label)
        
        metric_monitor.update('Loss', loss.item())

        
        scaler.scale(loss).backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        optimizer.zero_grad()
        stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")

        
def validate_fn(val_loader, model, criterion2,cfg,device,epoch):
    metric_monitor1 = MetricMonitor()
    metric_monitor2 = MetricMonitor()
    metric_monitor3 = MetricMonitor()
    #results =[]
    model.eval()
    stream = tqdm(val_loader)
    all_r4_1 = []
    all_r4_2=[]
    all_mrr=[]
    with torch.no_grad():
        for i, batch in enumerate(stream ):
            input_ids = batch["input_ids"].to(device)
            input_mask = batch["input_mask"].to(device)
            segment_ids = batch["segment_ids"].to(device)
            cls_pos = batch["cls_pos"].to(device)
            label = batch["label_id"].to(device)
          

            preds = model(input_ids,input_mask,segment_ids,cls_pos,cfg,device)
            #for i in range(len(preds)):
            #  results.append(np.argmax(preds.cpu().numpy(),axis=1)

            r4_1,r4_2,mrr = criterion2(preds,label)

          
            all_r4_1.append(r4_1)
            all_r4_2.append(r4_2)
            all_mrr.append(mrr)
            metric_monitor1.update('R4_1', r4_1)
            metric_monitor2.update('R4_2', r4_2)
            metric_monitor3.update('MRR', mrr)
            stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor1} {metric_monitor2} {metric_monitor3}")
          
    return np.mean(all_r4_1),np.mean(all_r4_2),np.mean(all_mrr) 



def get_scheduler(optimizer, scheduler_params,cfg):
    if scheduler_params['scheduler_name'] == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_params['T_0'],
            eta_min=scheduler_params['min_lr'],
            last_epoch=-1
        )
    elif scheduler_params['scheduler_name'] == 'OneCycleLR':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=scheduler_params['max_lr'],
            steps_per_epoch=int(cfg.num_train_data / cfg.batch_size) + 1,
            epochs=cfg.epochs,
            pct_start=scheduler_params['pct_start'],
            anneal_strategy=scheduler_params['anneal_strategy'],
            div_factor=scheduler_params['div_factor'],
            final_div_factor=scheduler_params['final_div_factor'],
        )
    return scheduler



def get_optimizer_params_diff(model, encoder_lr, decoder_lr, weight_decay=0.0):
    named_parameters = list(model.named_parameters())    
    parameters = []

    # increase lr every second layer
    increase_lr_every_k_layer = 1
    lrs = np.linspace(1, 2, 48 // increase_lr_every_k_layer)
    num = 0
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    for layer_num, (name, params) in enumerate(named_parameters):
        weight_decay = 0.0 if any(nd in name for nd in no_decay) else 0.01
        splitted_name = name.split('.')
        lr = encoder_lr
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[3]):
            layer_num = int(splitted_name[3])
            lr = lrs[layer_num // increase_lr_every_k_layer] * encoder_lr
            # num+=1
            print(name,lr)
        if 'model' not in splitted_name:
            lr = lrs[-1]*encoder_lr
            print(name,lr)
#         if splitted_name[0] in ['fc']:
#             lr = 10*encoder_lr
#             print(name,lr)

#         if splitted_name[0] in ['head']:
#             lr = 10*encoder_lr
#             print(name,lr)
        # print(num)
        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})
    return parameters



def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay, 'initial_lr':encoder_lr},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0, 'initial_lr':encoder_lr},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0, 'initial_lr':decoder_lr}
    ]
    return optimizer_parameters