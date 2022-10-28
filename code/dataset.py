import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

NUM_OPTIONS = 4
class MuTualDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.length = len(features)

    def __len__(self): 
        return self.length
  
    def __getitem__(self, idx):
        input_ids = self.features[idx].choices_features[0]['input_ids']
        input_mask = self.features[idx].choices_features[0]['input_mask']
        segment_ids = self.features[idx].choices_features[0]['segment_ids']
        cls_pos = self.features[idx].choices_features[0]['cls_pos']
        label = self.features[idx].label
    

        label_id = np.zeros((NUM_OPTIONS,1),dtype=int)

        label_id[label,0] = 1
        #length = len(cls_pos)
        
        #edge_index = _get_edge_index(length, edge_index)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids =torch.tensor(segment_ids, dtype=torch.long)
        cls_pos = torch.tensor(cls_pos, dtype=torch.long)
        label_id = torch.tensor(label_id, dtype=torch.float)

        return {"input_ids":input_ids,"input_mask": input_mask,"segment_ids": segment_ids,"cls_pos":cls_pos,"label_id":label_id}