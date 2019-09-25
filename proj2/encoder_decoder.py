import pickle
import torch
from torch import nn
import numpy as np
import torch.optim as optim


class EncoderDecoder(nn.Module):

    def __init__(self, num_input_features, num_output_features):

        super().__init__()
        self.w1 = nn.Parameter(torch.empty(num_input_features, num_output_features))
        nn.init.uniform_(self.w1, -0.1, 0.1)
        self.w2 = nn.Parameter(torch.empty(num_output_features, num_input_features))
        nn.init.uniform_(self.w2, -0.1, 0.1)
    
    def encode(self, x):
        return torch.mm(x, self.w1)

    def decode(self, x):
        return torch.mm(x, self.w2)

    def forward(self, x):

        encode_x = self.encode(x)
        decode_x = self.decode(encode_x)
        return decode_x



if __name__ == "__main__":


    feature_indexer_file = "feature_indexer.pkl"
    feature_cache_file = "features.pkl"
    print("Loading features")
    feature_indexer = pickle.load(open(feature_indexer_file, "rb"))
    feature_cache = pickle.load(open(feature_cache_file, "rb"))

    num_features = 300

    model = EncoderDecoder(len(feature_indexer), num_features)

    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr)
    loss_fn = torch.nn.SmoothL1Loss()

    

    total_loss = 0.0
    total_count = 0.0

    
    # for i in range(len(feature_indexer)):

    #     all_indices = np.array(feature_cache[i])

        
    #     for indices in all_indices:
    #         x = np.zeros((indices.shape[0], len(feature_indexer)))
    #         for ind1 in range(indices.shape[0]):
    #             for ind2 in range(indices.shape[1]):
    #                 x[ind1, indices[ind1][ind2]]  = 1
            
    #         all_x.append(x)

    #         if(len(all_x) >= 100):
    #             all_x = np.stack(all_x)
    #             all_x = np.reshape(all_x, (-1, len(feature_indexer)))
    #             print(all_x.shape)

    #             model.zero_grad()
    #             x_tensor = torch.from_numpy(all_x).float()
    #             recon_x = model(x_tensor)
    #             loss = loss_fn(recon_x, x_tensor)
    #             print(loss.data)
    #             total_loss += loss.data
    #             total_count += 1
    #             loss.backward()
    #             optimizer.step()
    #             all_x = []
    #             print(loss.data)
    #             torch.save(model, "simple.embedder")
        
    #     if(i%10==0):
    #         print("{} done".format(i))

    embedder = torch.load("simple.embedder")
    all_embedded = []
    for i in range(len(feature_cache)):

        all_indices = np.array(feature_cache[i])

        all_x = []
        for indices in all_indices:
            x = np.zeros((indices.shape[0], len(feature_indexer)))
            for ind1 in range(indices.shape[0]):
                for ind2 in range(indices.shape[1]):
                    x[ind1, indices[ind1][ind2]]  = 1
            all_x.append(x)
        all_x = np.stack(all_x)
        all_x = np.reshape(all_x, (-1, len(feature_indexer)))
        x_tensor = torch.from_numpy(all_x).float()
        embedded_x = embedder.encode(x_tensor).data
        embedded_x = np.reshape(embedded_x, (-1, 9, 300))
        print(embedded_x.shape)
        all_embedded.append(embedded_x)
    
        if(i%100 == 0):
            print("{} done".format(i))
            pickle.dump(all_embedded, open("embedded.cache", "wb"))
        
    
    # print("epoch : {}, loss {}".format(epoch, total_loss/total_count))

        
            


