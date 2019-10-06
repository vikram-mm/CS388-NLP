# feedforward_example_pytorch.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random


# DEFINING THE COMPUTATION GRAPH
# Define the core neural network: one hidden layer, tanh nonlinearity
# Returns probabilities; in general your network can be set up to return probabilities, log probabilities,
# or (log) probabilities + loss
class FFNN(nn.Module):
    def __init__(self, inp, hid1, hid2, out):
        super(FFNN, self).__init__()
        self.V1 = nn.Linear(inp, hid1)
        self.V2 = nn.Linear(hid1, hid2)
        self.g = nn.Tanh()
        self.W = nn.Linear(hid, out)
        self.softmax = nn.Softmax(dim=0)
        # Initialize weights according to the Xavier Glorot formula
        nn.init.xavier_uniform(self.V.weight)
        nn.init.xavier_uniform(self.W.weight)

    # Forward computation. Backward computation is done implicitly (nn.Module already has an implementation of
    # it that you shouldn't need to override)
    def forward(self, x):
        return self.softmax(self.W(self.g(self.V2(self.V1(x)))))


# Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
# of data, does some computation, handles batching, etc.
def form_input(x):
    return torch.from_numpy(x).float()


# Example of training a feedforward network with one hidden layer to solve XOR.
if __name__=="__main__":
    # MAKE THE DATA
    # Synthetic data for XOR: y = x0 XOR x1
    train_xs = np.array([[0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]], dtype=np.float32)
    train_ys = np.array([0, 1, 1, 1, 1, 0], dtype=np.float32)
    # Define some constants
    # Inputs are of size 2
    feat_vec_size = 2
    # Let's use 10 hidden units
    embedding_size = 10
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # RUN TRAINING AND TEST
    num_epochs = 100
    ffnn = FFNN(feat_vec_size, embedding_size, num_classes)
    initial_learning_rate = 0.1
    optimizer = optim.Adam(ffnn.parameters(), lr=0.1)
    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_ys))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x = form_input(train_xs[idx])
            y = train_ys[idx]
            # Build one-hot representation of y
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            ffnn.zero_grad()
            probs = ffnn.forward(x)
            # Can also use built-in NLLLoss as a shortcut here (takes log probabilities) but we're being explicit here
            loss = torch.neg(torch.log(probs)).dot(y_onehot)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Loss on epoch %i: %f" % (epoch, total_loss))
    # Evaluate on the train set
    train_correct = 0
    for idx in range(0, len(train_xs)):
        # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
        # quantities from the running of the computation graph, namely the probabilities, prediction, and z
        x = form_input(train_xs[idx])
        y = train_ys[idx]
        probs = ffnn.forward(x)
        prediction = torch.argmax(probs)
        if y == prediction:
            train_correct += 1
        print("Example " + repr(train_xs[idx]) + "; gold = " + repr(train_ys[idx]) + "; pred = " +\
              repr(prediction) + " with probs " + repr(probs))
    print(repr(train_correct) + "/" + repr(len(train_ys)) + " correct after training")
