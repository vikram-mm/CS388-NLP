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


    feature_indexer_file = "german_feature_indexer.pkl"
    feature_cache_file = "german_features.pkl"
    german_feature_indexer = pickle.load(open(feature_indexer_file, "rb"))
    german_feature_cache = pickle.load(open(feature_cache_file, "rb"))


    num_features = 300

    model = EncoderDecoder(len(german_feature_indexer), num_features)

    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr)
    loss_fn = torch.nn.SmoothL1Loss()

    

    total_loss = 0.0
    total_count = 0.0


    i = 0
    j = 0

    while i<1500:
        all_x = []
 
        

       

        
        if i<=j:
            all_indices = np.array(feature_cache[i])
            for indices in all_indices:
                x = np.zeros((indices.shape[0], len(german_feature_indexer) ))
                for ind1 in range(indices.shape[0]):
                    for ind2 in range(indices.shape[1]):
                        x[ind1, indices[ind1][ind2]]  = 1
                
                all_x.append(x)

                if len(all_x) >= 2:
                    break
                

            
            i += 1

        else: 
            all_indices = np.array(german_feature_cache[j])       
            for indices in all_indices:
                x = np.zeros((indices.shape[0], len(german_feature_indexer) ))
                for ind1 in range(indices.shape[0]):
                    for ind2 in range(indices.shape[1]):
                        x[ind1, indices[ind1][ind2]]  = 1
                
                all_x.append(x)

                if len(all_x) >= 2:
                    break     
                
            j += 1
            
        all_x = np.stack(all_x)
        all_x = np.reshape(all_x, (-1, len(german_feature_indexer)))


        if all_x.shape[0] > 10:
            all_x = all_x[0:10, :]

        print(all_x.shape)

        model.zero_grad()
        x_tensor = torch.from_numpy(all_x).float()
        recon_x = model(x_tensor)
        loss = loss_fn(recon_x, x_tensor)
        print(loss.data)
        total_loss += loss.data
        total_count += 1
        loss.backward()
        optimizer.step()
        torch.save(model, "multi_lingual.embedder")
            
        if(i%10==0):
            print("{} done".format(i))

    # embedder = torch.load("multi_lingual.embedder")
    # all_embedded = []
    # for i in range(len(feature_cache)):

    #     all_indices = np.array(feature_cache[i])

    #     all_x = []
    #     for indices in all_indices:
    #         x = np.zeros((indices.shape[0], len(feature_indexer)))
    #         for ind1 in range(indices.shape[0]):
    #             for ind2 in range(indices.shape[1]):
    #                 x[ind1, indices[ind1][ind2]]  = 1
    #         all_x.append(x)
    #     all_x = np.stack(all_x)
    #     all_x = np.reshape(all_x, (-1, len(feature_indexer)))
    #     x_tensor = torch.from_numpy(all_x).float()
    #     embedded_x = embedder.encode(x_tensor).data
    #     embedded_x = np.reshape(embedded_x, (-1, 9, 300))
    #     print(embedded_x.shape)
    #     all_embedded.append(embedded_x)
    
    # pickle.dump(all_embedded, open("embedded.cache", "wb"))
        
    
    # print("epoch : {}, loss {}".format(epoch, total_loss/total_count))

        
            


