# models.py

from sentiment_data import *
from typing import List
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
np.random.seed(0)
torch.manual_seed(0)


class FFNN(nn.Module):
    def __init__(self, inp, hid1, hid2, out):
        super(FFNN, self).__init__()
        '''
        two FC layers followed by an activation and a final classifier FC layer
        '''
        self.V1 = nn.Linear(inp, hid1)
        self.V2 = nn.Linear(hid1, hid2)
        self.g = nn.ReLU()
        self.W = nn.Linear(hid2, out)
        nn.init.xavier_uniform(self.V1.weight)
        nn.init.xavier_uniform(self.V2.weight)
        nn.init.xavier_uniform(self.W.weight)

    def forward(self, x):
        return self.W(self.g(self.V2(self.V1(x))))

class FancyModel(nn.Module):
    def __init__(self, inp, hid1, out):
        '''
        one lstm layer followed by a FC layer
        '''
        super(FancyModel, self).__init__()
        self.lstm = nn.LSTM(inp, hid1, batch_first = True, bidirectional = True)
        self.W = nn.Linear(2*hid1, out) #because its bidirectional
        
    def forward(self, x, seq_lens):

        output, hidden = self.lstm(x, self.prev_hidden)
        required_output = []
        for i, seq_len in enumerate(seq_lens): #selecting the last output according to the seq length
            required_output.append(output[i, seq_len-1, :])
        required_output = torch.stack(required_output)
        return self.W(required_output)

class FancyModel2(nn.Module):
    def __init__(self, inp, hid1, hid2, out):
        '''
        one lstm layer followed by a FC layer
        '''
        super(FancyModel, self).__init__()
        self.lstm = nn.LSTM(inp, hid1, batch_first = True, bidirectional = True)
        self.W = nn.Linear(2*hid1, hid2) #because its bidirectional
        self.g = nn.ReLU()
        self.W2 = nn.Linear(hid2, out)

    def forward(self, x, seq_lens):

        output, hidden = self.lstm(x, self.prev_hidden)
        required_output = []
        for i, seq_len in enumerate(seq_lens): #selecting the last output according to the seq length
            required_output.append(output[i, seq_len-1, :])
        required_output = torch.stack(required_output)
        return self.W2(self.g(self.W(required_output)))

def pad_to_length(np_arr, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    result = np.zeros(length) - 1
    result[0:np_arr.shape[0]] = np_arr
    return result


def get_embeddings(train_mat, word_vectors):

    ans = []
    for ex in train_mat:
        sub_ans = [word_vectors.get_embedding_from_index(int(idx)) for idx in ex]
        ans.append(sub_ans)
    
    return np.array(ans)

def shuffle_together(a, b):

    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def shuffle_together_3(a, b, c):

    assert len(a) == len(b)
    assert len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

def get_onehot(labels, num_classes):

    ans = []
    for idx in labels:
        ans.append(np.arange(num_classes) ==idx)
    
    ans = np.array(ans, dtype=np.int32)
    return ans


def get_average_embeddings_and_labels(exs: List[SentimentExample], word_vectors: WordEmbeddings, seq_max_len):

    mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in exs])
    seq_lens = np.expand_dims(np.array([len(ex.indexed_words) for ex in exs]), -1)
    labels_arr = np.array([ex.label for ex in exs])
    embeddings = get_embeddings(mat, word_vectors)
    average_embeddings = np.sum(embeddings, axis=1)/ seq_lens
    return average_embeddings, labels_arr

embeddings_mean = None
embeddings_var = None

def get_embeddings_and_labels(exs: List[SentimentExample], word_vectors: WordEmbeddings, seq_max_len):

    mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in exs])
    seq_lens = np.array([len(ex.indexed_words) for ex in exs])
    labels_arr = np.array([ex.label for ex in exs])
    embeddings = get_embeddings(mat, word_vectors)
    embeddings = normalize(embeddings)
    return embeddings, labels_arr, seq_lens

def normalize(embeddings):

    global embeddings_mean, embeddings_var
    orig_shape = embeddings.shape
    embeddings = np.reshape(embeddings, (-1, embeddings.shape[2]))
    if embeddings_mean is None: #will be calcuulated only for train
        embeddings_mean = np.expand_dims(np.mean(embeddings, axis=0), axis=0) #the feature axis
        embeddings_var = np.expand_dims(np.std(embeddings, axis=0) , axis=0)
    embeddings = (embeddings-embeddings_mean)/embeddings_var
    embeddings = np.reshape(embeddings, orig_shape)
    return embeddings

def train_evaluate_ffnn(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    """
    Train a feedforward neural network on the given training examples, using dev_exs for development, and returns
    predictions on the *blind* test examples passed in. Returned predictions should be SentimentExample objects with
    predicted labels and the same sentences as input (but these won't be read by the external code). The code is set
    up to go all the way to test predictions so you have more freedom to handle example processing as you see fit.
    :param train_exs:
    :param dev_exs:
    :param test_exs:
    :param word_vectors:
    :return:
    """
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    num_classes = 2


    train_embeddings, train_labels_arr = get_average_embeddings_and_labels(train_exs, word_vectors, seq_max_len)
    dev_embeddings, dev_labels_arr = get_average_embeddings_and_labels(dev_exs, word_vectors, seq_max_len)
    test_embeddings, _ = get_average_embeddings_and_labels(test_exs, word_vectors, seq_max_len)

    dev_embeddings = torch.from_numpy(dev_embeddings).float()
    test_embeddings = torch.from_numpy(test_embeddings).float()

    #Hyper-parameters
    num_epochs = 15
    # hidden_1 = 32
    # hidden_2 = 4
    batch_size = 8
    lr = 0.01

    
    #grid search
    hidden_sizes_1 = [4, 8, 16, 32, 64]
    hidden_sizes_2 = [2, 4, 8, 16, 32]

    all_best_dev_acc = 0
    for hidden_1 in hidden_sizes_1:
        for hidden_2 in hidden_sizes_2:
            if hidden_1 > hidden_2:
                feat_vec_size = train_embeddings.shape[1]
                ffnn = FFNN(inp = feat_vec_size, hid1=hidden_1, hid2=hidden_2, out=num_classes)
                
                optimizer = optim.Adam(ffnn.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [7, 12], 0.1) #epcohs when lr will be reduced
                criterion = nn.CrossEntropyLoss()
                best_dev_acc = 0
                for epoch in range(-1, num_epochs):
                    train_embeddings, train_labels_arr = shuffle_together(train_embeddings, train_labels_arr)
                    total_loss = 0.0
                    total_count = 0
                    cursor = 0


                    while(cursor < len(train_embeddings)):
                        batch_x = torch.from_numpy(train_embeddings[cursor:cursor+batch_size]).float()
                        batch_y = torch.from_numpy(train_labels_arr[cursor:cursor+batch_size]).long()
                        cursor += batch_size
                        ffnn.zero_grad()
                        pred_y = ffnn(batch_x)
                        loss = criterion(pred_y, batch_y)
                        total_loss += loss.data
                        total_count += 1
                        loss.backward()
                        if(epoch != -1):
                            optimizer.step()

                    scheduler.step()
                    
                    # print("loss at epoch {}: {}".format(epoch, total_loss/total_count))
                    dev_pred = np.argmax(ffnn(dev_embeddings).data.numpy(), axis=1)
                    dev_acc = np.sum(dev_pred == dev_labels_arr)/len(dev_labels_arr)

                    # print("dev acc: {}".format(dev_acc))
                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                    
                    if dev_acc > all_best_dev_acc:
                        all_best_dev_acc = dev_acc
                        torch.save(ffnn, "best_ff.model")
                        # print("best model saved")
                
                print("{},{} : {}".format(hidden_1, hidden_2, best_dev_acc))
            
    print("loading best model")
    ffnn = torch.load("best_ff.model")

    test_pred = np.argmax(ffnn(test_embeddings).data.numpy(), axis=1)

    for i in range(len(test_exs)):
        test_exs[i].label = test_pred[i]

    return test_exs

# Analogous to train_ffnn, but trains your fancier model.
def train_evaluate_fancy(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    seq_max_len = 60
    num_classes = 2


    train_embeddings, train_labels_arr, train_seq_len = get_embeddings_and_labels(train_exs, word_vectors, seq_max_len)
    dev_embeddings, dev_labels_arr, dev_seq_len  = get_embeddings_and_labels(dev_exs, word_vectors, seq_max_len)
    test_embeddings, _, test_seq_len  = get_embeddings_and_labels(test_exs, word_vectors, seq_max_len)

    dev_embeddings = torch.from_numpy(dev_embeddings).float()
    test_embeddings = torch.from_numpy(test_embeddings).float()



    #Hyper-parameters
    num_epochs = 10
    lstm_hiiden_dim = 64
    batch_size = 32
    lr = 0.01

    
    feat_vec_size = train_embeddings.shape[2]
    model = FancyModel(inp = feat_vec_size, hid1=lstm_hiiden_dim, out=num_classes)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3], 0.1) #epcohs when lr will be reduced
    criterion = nn.CrossEntropyLoss()
    best_dev_acc = 0
    for epoch in range(0, num_epochs):
        train_embeddings, train_labels_arr, train_seq_len = shuffle_together_3(train_embeddings, train_labels_arr, train_seq_len)
        total_loss = 0.0
        total_count = 0
        cursor = 0
        model.prev_hidden = None


        while(cursor + batch_size < len(train_embeddings)):
            batch_x = torch.from_numpy(train_embeddings[cursor:cursor+batch_size]).float()
            batch_y = torch.from_numpy(train_labels_arr[cursor:cursor+batch_size]).long()
            bathc_seq_len = train_seq_len[cursor:cursor+batch_size]
            cursor += batch_size
            model.zero_grad()
            pred_y = model(batch_x, bathc_seq_len)
            loss = criterion(pred_y, batch_y)
            total_loss += loss.data
            total_count += 1
            loss.backward()
            optimizer.step()

        scheduler.step()
        
        # model.prev_hidden = None
        print("loss at epoch {}: {}".format(epoch, total_loss/total_count))
        dev_pred = np.argmax(model(dev_embeddings, dev_seq_len).data.numpy(), axis=1)
        dev_acc = np.sum(dev_pred == dev_labels_arr)/len(dev_labels_arr)
        print("dev acc: {}".format(dev_acc))

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model, "best.model")
            print("best model saved")
    
    print("loading best model")
    model = torch.load("best.model")
    # model.prev_hidden = None
    dev_pred = np.argmax(model(dev_embeddings, dev_seq_len).data.numpy(), axis=1)
    dev_acc = np.sum(dev_pred == dev_labels_arr)/len(dev_labels_arr)
    print("dev acc: {}".format(dev_acc))
    test_pred = np.argmax(model(test_embeddings, test_seq_len).data.numpy(), axis=1)

    for i in range(len(test_exs)):
        test_exs[i].label = test_pred[i]

    return test_exs