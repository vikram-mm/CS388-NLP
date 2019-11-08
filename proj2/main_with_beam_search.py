import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from lf_evaluator import *
from models import *
from data import *
from utils import *
from typing import List

random.seed(11)
np.random.seed(11)
torch.manual_seed(11)


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='main.py')
    
    # General system running and configuration options
    parser.add_argument('--do_nearest_neighbor', dest='do_nearest_neighbor', default=False, action='store_true', help='run the nearest neighbor model')

    parser.add_argument('--train_path', type=str, default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/geo_test.tsv', help='path to blind test data')
    parser.add_argument('--test_output_path', type=str, default='geo_test_output.tsv', help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo', help='domain (geo for geoquery)')
    
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes
    args = parser.parse_args()
    return args


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # N.B. a list!
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])

        # print(test_derivs)
        # exit(0)
        return test_derivs

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0) 

class Seq2SeqSemanticParser(object):
    def __init__(self, input_embedding_layer, encoder, decoder, output_indexer,\
     beam_size=3):
        # raise Exception("implement me!")
        # Add any args you need here
        self.input_embedding_layer = input_embedding_layer
        self.encoder = encoder
        self.decoder = decoder
        self.output_indexer = output_indexer
        self.beam_size = beam_size

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:

        # raise Exception("implement me!")
        ans  = []
        
        for ii, ex in enumerate(test_data):
            pred_tokens = []
            input_batch = np.array([ex.x_indexed])
            inp_lens = np.sum(input_batch!=0, axis=1)
            x_tensor = torch.from_numpy(input_batch).long()
            inp_lens_tensor = torch.from_numpy(inp_lens).long()
            all_enc_output, _, enc_output = encode_input_for_decoder(x_tensor,\
             inp_lens_tensor, self.input_embedding_layer,\
            self.encoder)
            
            hidden, cell = enc_output
            input = torch.ones((1)).long()
            context = torch.zeros(1, 1, self.decoder.hid_dim)
            token = "<SOS>"
            count  = 0

            final_beam = Beam(self.beam_size)

            all_beams = [Beam(self.beam_size) for x in range(70)]
            all_beams[0].add(elt = (pred_tokens, input, context,"<SOS>", hidden, cell), score = 0.0)


            while count<69:
                
                for beam_element, score in all_beams[count].get_elts_and_scores():

                    pred_tokens, input, context, prev_token, hidden, cell = beam_element

                    if(prev_token == "<EOS>"):
                        final_beam.add(beam_element, score)
                        continue

                    output, hidden, cell, context = self.decoder(input, \
                    hidden, cell, context, all_enc_output)
                    output = output.squeeze().data.numpy()
                    output = softmax(output)
                    top_indices = output.argsort()
                    top_indices = top_indices[::-1]

                    for index in range(self.beam_size):
                        # top1 = output.argmax(1) 
                        i = top_indices[index]
                        input = torch.tensor([i]).long()
                        token = output_indexer.get_object(i)
                        prob = output[i]
                        
                        new_pred_list = pred_tokens.copy()
                        if token != "<EOS>":
                            new_pred_list.append(token)

                        all_beams[count+1].add(elt= (new_pred_list,\
                         input, context, token, hidden, cell), score = (score*(count)+prob)/(count+1.0))
                
                # for beam_elt, score in all_beams[count+1].get_elts_and_scores():
                #     print(beam_elt[0])
                
                # print('-----------------------------------------------')


                count += 1
            sub_ans = []
            # exit(0)
            for beam_elt, score in final_beam.get_elts_and_scores():
                sub_ans.append(Derivation(ex, 1.0, beam_elt[0]))
                # print(beam_elt[0])
                # break
            ans.append(sub_ans)

            # print("{}/{} done".format(ii+1, len(test_data)))
        
        return ans
            

            


    


def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb: 
    EmbeddingLayer, model_enc: RNNEncoder):
    """
    Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
    inp_lens_tensor lengths.
    YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
    as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
    :param x_tensor: [batch size, sent len] tensor of input token indices
    :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
    :param model_input_emb: EmbeddingLayer
    :param model_enc: RNNEncoder
    :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
    are real and which ones are pad tokens), and the encoder final states (h and c tuple)
    E.g., calling this with x_tensor (0 is pad token):
    [[12, 25, 0, 0],
    [1, 2, 3, 0],
    [2, 0, 0, 0]]
    inp_lens = [2, 3, 1]
    will return outputs with the following shape:
    enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
    enc_final_states = 3 x dim
    """
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_context_mask, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


def train_model_encdec(train_data: List[Example], test_data: List[Example], input_indexer, output_indexer, load_epoch, args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param test_data:
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, reverse_input=False)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    # print("Train length: %i" % input_max_len)
    # print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    # print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # First create a model. Then loop over epochs, loop over examples, and given some indexed words, call
    # the encoder, call your decoder, accumulate losses, update parameters
    

    input_vocab_size = len(input_indexer)
    output_vocab_size = len(output_indexer)

    do_training = False
    # load_epoch = 15
    input_embedding_layer = torch.load("models_fresh/{}.embed".format(load_epoch))
    encoder = torch.load("models_fresh/{}.encoder".format(load_epoch))
    decoder = torch.load("models_fresh/{}.decoder".format(load_epoch))
    
    return Seq2SeqSemanticParser(input_embedding_layer, encoder, decoder, output_indexer)

    # raise Exception("Implement the rest of me to train your encoder-decoder model")


def evaluate(test_data: List[Example], decoder, example_freq=50, print_output=True, outfile=None):
    """
    Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
    every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
    executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
    example with a valid denotation (if you've provided more than one).
    :param test_data:
    :param decoder:
    :param example_freq: How often to print output
    :param print_output:
    :param outfile:
    :return:
    """
    e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    java_crashes = False

    # for true, pred in zip([ex.y for ex in test_data], pred_derivations):
    #     print(true)
    #     print(pred)
    #     print('-----')

    # exit(0)

    if java_crashes:
        selected_derivs = [derivs[0] for derivs in pred_derivations]
        denotation_correct = [False for derivs in pred_derivations]
    else:
        selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations, quiet=True)
    t, r = print_evaluation_results(test_data, selected_derivs, denotation_correct, example_freq, print_output=False)
    
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()

    return t,r


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev, test, args.decoder_len_limit)
    print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s" % input_indexer)
    print("Output indexer: %s" % output_indexer)
    print("Here are some examples post tokenization and indexing:")
    for i in range(0, min(len(train_data_indexed), 10)):
        print(train_data_indexed[i])
    if args.do_nearest_neighbor:
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        evaluate(dev_data_indexed, decoder)
    else:
        for load_epoch in range(4, 40):
            decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, load_epoch, args)
            t, r = evaluate(dev_data_indexed, decoder)
            print(load_epoch, t, r)
    exit(0)
    print("=======FINAL EVALUATION ON BLIND TEST=======")
    evaluate(test_data_indexed, decoder, print_output=False, outfile="geo_test_output.tsv")


