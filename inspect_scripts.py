import torch
import torch.nn as nn
from model.cbow import CBoW
import csv



'''
return the cross score of src to every word
value of src should be of type int 
return tensor shape (vocab_size)
'''
class CrossScoreCalculator():
    def __init__(self, vocab_size, embedding_dim, model_path = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.vocab_size = vocab_size


        '''
        load cbow model
        '''
        self.cbow_model = CBoW(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                ).to(self.device)

        if model_path is not None:
            model_dict = torch.load(model_path)
            self.cbow_model.load_state_dict(model_dict)
        
    def __call__(self, src_idx):
        #  src_tensor shape: (1)
        #  targets_tensor shape: (vocab_size)
        src_tensor = torch.tensor(src_idx, dtype=torch.int32).view(1,-1).to(self.device)
        trgs_tensor = torch.arange(self.vocab_size).view(1,1,-1).to(self.device)

        
        scores = self.cbow_model(src_tensor, trgs_tensor).view(-1).to('cpu')
        return scores



'''
return the n words of max scores to the src word
input: a word, in string format
output: a list of words
'''
class NeighboursExtractor():
    def __init__(self, vocab_size, embedding_dim, model_path, encoding_path):
        self.cross_score_calculator = CrossScoreCalculator(vocab_size, embedding_dim, model_path)
        self.idx2word = dict()
        self.word2idx = dict()
        self.idx2word[0] = 'UNK'
        
        #  read the coding dict of every encoded token
        with open(encoding_path, 'r') as f:
            csv_reader = csv.reader(f)
            
            for row in csv_reader:
                idx = int(row[0])
                word = row[1]
                self.idx2word[idx] = word
                self.word2idx[word] = idx

            f.close()

    
    def __call__(self, src, n, limit = 2000):
        if self.word2idx.get(src) is None:
            return None
        else:
            scores = self.cross_score_calculator(self.word2idx[src])
            scores = scores.tolist()
            scores = scores[:limit]
            indexed_scores = list(enumerate(scores))
            sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
            top_n_words = [self.idx2word[index] for index, _ in sorted_scores[:n]]
            return top_n_words






if __name__ == "__main__":
    vocab_size = 6957
    embedding_dim = 32
    model_path = 'saved_embedding.pth'
    encoding_path = 'data/encoding_table.csv'
    

    neighbours_extractor = NeighboursExtractor(vocab_size, embedding_dim, model_path, encoding_path)

    while True:
        src = input("enter the src word:\n")
        neighbours = neighbours_extractor(src, 5)
        if neighbours is None:
            print("This word is not in the vocab!")
        else:
            print(neighbours_extractor(src, 5))
