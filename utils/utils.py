import torch
import torch.nn as nn
import torch.nn.functional as F
import sqlite3
from torch.utils.data import Dataset
import json
import csv
import random




# scores shape = (batch_size, target_size)
# output shape = (batch_size, 1)
class CBoW_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CBoW_loss, self).__init__()
        self.reduction = reduction
    
    def forward(self, input_data):
        input_data[:, 1:] *= -1
        weights = F.logsigmoid(input_data)
        #  print(weights)
        loss = -torch.sum(weights, dim=-1)
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss

    #  def __call__(self, input_data):
    #      return self.forward(input_data)
        




class SQLiteDataset(Dataset):
    def __init__(self, db_path, query, freq_path, negative_sample_size):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.cursor.execute(query)
        self.data = self.cursor.fetchall()
        self.negative_sample_size = negative_sample_size
        self.encoding_list = list()
        self.freq_list = list()
        

        with open(freq_path, 'r') as f:
            csv_reader = csv.reader(f)
            
            for row in csv_reader:
                self.encoding_list.append(int(row[0]))
                self.freq_list.append(int(row[1]))

            f.close()


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        serialized_context, target = self.data[idx]
        context = json.loads(serialized_context)

        negative_targets = random.choices(self.encoding_list, weights=self.freq_list, k=self.negative_sample_size)
        
        #  print(len(context))
        #  print(target.dtype)

        

        concated_targets = [target] + negative_targets
        context_tensor = torch.tensor(context, dtype=torch.int32)
        targets_tensor = torch.tensor(concated_targets, dtype=torch.int32)
        return context_tensor, targets_tensor


