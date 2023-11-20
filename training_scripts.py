import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, dataloader
from model.cbow import CBoW
from utils.utils import CBoW_loss
from utils.utils import SQLiteDataset
import utils.utils

#  from 




def train_CBoW(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    '''
    model
    '''
    cbow_model = CBoW(
            vocab_size=training_config['vocab_size'],
            embedding_dim=training_config['embedding_dim']
            ).to(device)

    #  load
    if training_config.get('model_path_src') is not None:
        model_dict = torch.load(training_config['model_path_src'])
        cbow_model.load_state_dict(model_dict)


    '''
    optimizer
    '''
    optimizer = optim.Adam(cbow_model.parameters(), lr=0.1)


    '''
    loss 
    '''
    criterion = CBoW_loss()



    '''
    dataloader
    '''
    dataset = SQLiteDataset(db_path='data/BBCnews_processed.db', query='SELECT context, target FROM pairs', freq_path='data/word_occurrence.csv', negative_sample_size=training_config['negative_sample_size'])
    dataloader = DataLoader(dataset, batch_size=training_config['batch_size'], shuffle=True)



    for epoch in range(training_config['num_of_epochs']):
        loss_sum = 0
        for i, inputs in enumerate(dataloader):
            optimizer.zero_grad()
            scores = cbow_model(inputs)
            loss = criterion(scores)
            loss_sum += torch.sum(loss)
            loss.backward()
            optimizer.step()
        print(loss_sum)

    torch.save(cbow_model.state_dict(), training_config['model_path_dst'])




if __name__ == "__main__":
    training_config = dict()
    training_config['vocab_size']           = 8836
    training_config['embedding_dim']        = 16
    training_config['negative_sample_size'] = 4
    training_config['num_of_epochs']        = 40
    training_config['batch_size']           = 20
    training_config['model_path_dst']       = 'saved_embedding.pth'
    training_config['learning_rate']        = 0.1
    #  training_config['model_path_src']    = 'saved_embedding.pth'
    train_CBoW(training_config)
