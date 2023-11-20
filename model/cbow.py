import torch
import torch.nn as nn



''' 
input: (context, target)
return: (scores)
where v_c is the average of the context vectors
'''
class CBoW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBoW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #  self.linear shape = nn.Linear(embedding_dim, vocab_size)

    
    '''
    skip init because pytorch has already done it for 
    us, using Kaiming He initialization
    '''
    #  def init_params(self, default_initialization = False):


    '''
    return the scores of the targets
    '''
    def forward(self, inputs):
        #  context shape = (batch_size, context_size)
        #  targets shape = (batch_size, target_size)
        context, targets = inputs
        

        # embedded_context shape = (batch_size, context_size, embedding_dim)
        #  v_c shape = (batch_size, embedding_dim)
        embedded_context = self.embedding(context)
        v_c = torch.sum(embedded_context, dim = 1) / embedded_context.shape[1]


        # embedded_targets shape = (batch_size, target_size, embedding_dim)
        # scores shape = (batch_size, target_size)
        embedded_targets = self.embedding(targets)
        scores = torch.matmul(v_c.unsqueeze(1), embedded_targets.transpose(-2,-1)).squeeze(dim=1)
        return scores
        #  scores[1:] *= -1