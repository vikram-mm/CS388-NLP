import argparse
import sys
import time
from nerdata import *
from utils import *
from collections import Counter
from optimizers import *
from typing import List
# from classifier_main import PersonExample, transform_for_classification
import re


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD, CLASSIFIER)')
    parser.add_argument('--train_path', type=str, default='data/eng.train', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/eng.testa', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/eng.testb.blind', help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args

class PersonExample(object):
    """
    Data wrapper for a single sentence for person classification, which consists of many individual tokens to classify.

    Attributes:
        tokens: the sentence to classify
        labels: 0 if non-person name, 1 if person name for each token in the sentence
    """
    def __init__(self, tokens: List[str], labels: List[int]):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)


def transform_for_classification(ner_exs: List[LabeledSentence]):
    """
    :param ner_exs: List of chunk-style NER examples
    :return: A list of PersonExamples extracted from the NER data
    """
    for labeled_sent in ner_exs:
        tags = bio_tags_from_chunks(labeled_sent.chunks, len(labeled_sent))
        labels = [1 if tag.endswith("PER") else 0 for tag in tags]
        yield PersonExample([tok.word for tok in labeled_sent.tokens], labels)


def generate_vocabDict(ner_exs: List[PersonExample]):

    vocab_dict = {}
    index = 0
    for ex in ner_exs:
        for token in ex.tokens:
            if re.match('^[a-zA-Z][\w-]+$', token.lower()):   #only words and not numbers, dates etc.
                if token.lower() not in vocab_dict:
                    vocab_dict[token.lower()] = index
                    index += 1
    return vocab_dict

def get_frequencyDict(ner_exs: List[PersonExample]):

    frequency_dict = {}
    inverse_document_dict = {}
    for ex in ner_exs:
        for token in ex.tokens:
            if re.match('^[a-zA-Z][\w-]+$', token.lower()):   #only words and not numbers, dates etc.
                if token.lower() not in frequency_dict:
                    frequency_dict[token.lower()] = 1
                else:
                    frequency_dict[token.lower()] += 1
        
        for token in list(set(ex.tokens)):

             if re.match('^[a-zA-Z][\w-]+$', token.lower()):   #only words and not numbers, dates etc.
                if token.lower() not in inverse_document_dict:
                    inverse_document_dict[token.lower()] = 1
                else:
                    inverse_document_dict[token.lower()] += 1
        
    # for key, value in sorted(frequency_dict.items(), key=lambda item: item[1]):
    #     print("%s: %s" % (key, value))
    #threshold is heuristically selected to be 40

    return frequency_dict, inverse_document_dict

def get_frequentNameDict(ner_exs: List[PersonExample]):

    frequentNameDict = {}
    for ex in ner_exs:
        for i, token in enumerate(ex.tokens):
            if re.match('^[a-zA-Z][\w-]+$', token.lower()):   #only words and not numbers, dates etc.

                if ex.labels[i] == 1:
                    if token.lower() not in frequentNameDict:
                        frequentNameDict[token.lower()] = 1
                    else:
                        frequentNameDict[token.lower()] += 1

    # for key, value in sorted(frequentNameDict.items(), key=lambda item: item[1]):
    #     print("%s: %s" % (key, value))
    #threshold is heuristically selected to be 15

    return frequentNameDict
    

def get_oneBeforeNameDict(ner_exs: List[PersonExample]):

    oneBeforeNameDict = {}
    for ex in ner_exs:
        for i, token in enumerate(ex.tokens):
            if re.match('^[a-zA-Z][\w-]+$', token.lower()):   #only words and not numbers, dates etc.

                if i<len(ex)-1 and ex.labels[i+1] == 1 and ex.labels[i]!=1:
                    if token.lower() not in oneBeforeNameDict:
                        oneBeforeNameDict[token.lower()] = 1
                    else:
                        oneBeforeNameDict[token.lower()] += 1

    # for key, value in sorted(oneBeforeNameDict.items(), key=lambda item: item[1]):
    #     print("%s: %s" % (key, value))
    #threshold is heuristically selected to be 18

    return oneBeforeNameDict

def get_twoBeforeNameDict(ner_exs: List[PersonExample]):

    twoBeforeNameDict = {}
    for ex in ner_exs:
        for i, token in enumerate(ex.tokens):
            if re.match('^[a-zA-Z][\w-]+$', token.lower()):   #only words and not numbers, dates etc.

                if i<len(ex)-2 and ex.labels[i+2] == 1 and ex.labels[i]!=1:
                    if token.lower() not in twoBeforeNameDict:
                        twoBeforeNameDict[token.lower()] = 1
                    else:
                        twoBeforeNameDict[token.lower()] += 1

    # for key, value in sorted(twoBeforeNameDict.items(), key=lambda item: item[1]):
    #     print("%s: %s" % (key, value))
    # threshold is heuristically selected to be 16

    return twoBeforeNameDict

def get_oneAfterNameDict(ner_exs: List[PersonExample]):

    oneAfterNameDict = {}
    for ex in ner_exs:
        for i, token in enumerate(ex.tokens):
            if re.match('^[a-zA-Z][\w-]+$', token.lower()):   #only words and not numbers, dates etc.

                if i<len(ex)-1 and ex.labels[i-1] == 1 and ex.labels[i]!=1:
                    if token.lower() not in oneAfterNameDict:
                        oneAfterNameDict[token.lower()] = 1
                    else:
                        oneAfterNameDict[token.lower()] += 1

    # for key, value in sorted(oneAfterNameDict.items(), key=lambda item: item[1]):
    #     print("%s: %s" % (key, value))
    #threshold is heuristically selected to be 2

    return oneAfterNameDict

def get_posDict(ner_exs: List[PersonExample]):
    pos_dict = {}
    index = 0
    for ex in ner_exs:
        for i, pos in enumerate(ex.pos_tags):
           if re.match('^[a-zA-Z][\w-]+$', pos.lower()):   #only words and not numbers, dates etc.

                if pos.lower() not in pos_dict:
                    pos_dict[pos.lower()] = index
                    index += 1

    total_posTags = index
                
    return pos_dict, index




class Featurizer():


    def __init__(self, train_class_exs: List[PersonExample]):
        '''
        used to initiialize the featurizer and calculate various priors of the train set
        '''
        self.vocab_dict = generate_vocabDict(train_class_exs)
        self.frequencyDict, self.inverse_document_dict =  get_frequencyDict(train_class_exs)
        self.frequentNameDict = get_frequentNameDict(train_class_exs)
        self.oneBeforeNameDict = get_oneBeforeNameDict(train_class_exs)
        self.twoBeforeNameDict = get_twoBeforeNameDict(train_class_exs)
        self.oneAfterNameDict = get_oneAfterNameDict(train_class_exs)
        self.pos_dict, self.total_posTags = get_posDict(train_class_exs)

        self.vocab_size = len(self.vocab_dict)  
        self.num_dimensions = 15 + 1 + 3*(self.vocab_size + 1) + 3*self.total_posTags #plus one for bias, plus 3 for none of the words
        self.num_direct_dimensions = 15 + 1 # 1 is bias

        #for tf-idf
        self.total_frequencyDict = 0.0
        for key, value in self.frequencyDict.items():
            self.total_frequencyDict += value       
        self.total_docs = len(train_class_exs)

        #for frequency features
        self.max_frequencyDict = max(self.frequencyDict.values())
        self.max_frequentNameDict = max(self.frequentNameDict.values())
        self.max_oneBeforeNameDict = max(self.oneBeforeNameDict.values())
        self.max_twoBeforeNameDict = max(self.oneBeforeNameDict.values())
        self.max_oneAfterNameDict = max(self.oneAfterNameDict.values())

        print('vocabulary size: {}'.format(self.vocab_size))
        print('number of pos tags: {}'.format(self.total_posTags))
        print('number of dimensions: {}'.format(self.num_dimensions))
        
       



    def get_tf_idf(self, token):

        if token not in self.frequencyDict:
            return 0
        
        tf = self.frequencyDict[token] /self.total_frequencyDict
        idf = np.log(self.total_docs / self.inverse_document_dict[token])

        return tf*idf

    
    def featurize(self, tokens: List[str], pos_tags: List[str]):
        '''
        featurization over a list of tokens
        '''

        assert len(tokens) == len(pos_tags)

        vocab_size = self.vocab_size
        features_list = []
        for i, token in enumerate(tokens):

            pos_tag = pos_tags[i]
            feature = self.featurize_oneInstance(tokens, pos_tags, i)
            features_list.append(feature)
        
        return features_list

    def featurize_oneInstance(self, tokens: List[str], pos_tags: List[str], idx):

        '''
        featurization over of a token at a given index
        '''

        #Feature dimensions:
        # 0 -> is it the first word of the sentence? (0 or 1)
        # 1 -> Does it start with a capital letter? (0 or 1)
        # 2 -> is it a simple number with no hyphens? (0 or 1)  
        # 3 -> is it a number with hyphens? (o or 1) (such as dates)
        # 4 -> is it a word? (0 or 1)
        # 5 -> is it a word with hyphens (0 or 1)
        # 6 -> length of word (integer)
        # 7 -> prior probability of token being a name.
        # 8 -> normalized frequency of the token
        # 9 -> prior probability of the previous token occuring before a name
        # 10 -> prior probability of the two token before occuring before a name
        # 11 -> prior probability of the next token occuring after a name
        # 12 -> tf-idf of current word
        # 13 -> tf-idf of previous word
        # 14 -> tf-idf of next word
        # 15 -> bias

        #the above features are referred to as the "direct features"
        # and the below ones are the compressed features
        
        # 16 -> index of current term (indicator)
        # 17 -> index of previous term (indicator)
        # 18 -> index of next term (indicator)
        # 19 -> index of current term (indicator)
        # 20 -> index of previous term (indicator)
        # 21 -> index of next term (indicator)

        #Therefore, dimensions of the feature is 11 + vocab_size*3 + 1
        # last one is bias

        #the variou

        vocab_size = self.vocab_size
        i = idx
        token = tokens[idx]     
        pos_tag = pos_tags[idx]

        
        feature = np.zeros(15 + 1 + 3 + 3)

        #word pattern features
        feature[0] = int(i==0)
        feature[1] = int(re.match('[A-Z][\w-]+$', token) is not None)
        feature[2] = int(re.match('^[\d]+$', token) is not None)
        feature[3] = int(re.match('^[\d][\d-]+[-]+[\d-]+$', token) is not None)
        feature[4] = int(re.match('^[a-zA-Z][\w-]+$', token) is not None)
        feature[5] = int(re.match('^[a-zA-Z][\w-]+[-]+[\w-]+$', token) is not None)
        feature[6] = int(re.match('^[a-zA-Z][\w-]+$', token) is not None) * len(token) / 10.0

        #frequency features
        if token.lower() in self.frequentNameDict:
            feature[7] = self.frequentNameDict[token.lower()]/ self.max_frequentNameDict
        
        if token.lower() in self.frequencyDict:
            feature[8] = self.frequencyDict[token.lower()]/ self.max_frequencyDict

        if i>=1 and tokens[i-1].lower() in self.oneBeforeNameDict:
            feature[9] = self.oneBeforeNameDict[tokens[i-1].lower()]/ self.max_oneBeforeNameDict

        if i>=2 and tokens[i-2].lower() in self.twoBeforeNameDict:
            feature[10] = self.twoBeforeNameDict[tokens[i-2].lower()]/ self.max_twoBeforeNameDict
        
        if i+1 < len(tokens) and tokens[i+1].lower() in self.oneAfterNameDict:
            feature[11] = self.oneAfterNameDict[tokens[i+1].lower()]/ self.max_oneAfterNameDict


        #tf - idf features
        feature[12] = self.get_tf_idf(token.lower())
        if i>= 1:
            feature[13] = self.get_tf_idf(tokens[i-1].lower())
        if i+1 < len(tokens):
            feature[14] = self.get_tf_idf(tokens[i+1].lower())
    
        #bias
        feature[15] = 1

        #bag of word indices
        # feature[16:] = 16 + 3*self.vocab_size - 1#words not in vocab

        if token.lower() in self.vocab_dict:
            feature[16] = 16 + self.vocab_dict[token.lower()]
        else:
            feature[16] = 16 + self.vocab_size

        if i >= 1 and tokens[i-1].lower() in self.vocab_dict:
            feature[17] = 16 + self.vocab_dict[tokens[i-1].lower()] + self.vocab_size + 1
        else:
            feature[17] = 16 + 2*self.vocab_size + 1

            
        if i+1 < len(tokens) and tokens[i+1].lower() in self.vocab_dict:
            feature[18] = 16 + self.vocab_dict[tokens[i+1].lower()] + 2*self.vocab_size + 2
        else:
            feature[18] = 16 + 3*self.vocab_size + 2
        
        if pos_tag.lower() in self.pos_dict:
            feature[19] = 16 + 3*self.vocab_size + 3 + self.pos_dict[pos_tag.lower()]
        
        if i>=1 and pos_tags[i-1].lower() in self.pos_dict:
            feature[20] = 16 + 3*self.vocab_size + 3 + self.pos_dict[pos_tags[i-1].lower()] + self.total_posTags
        
        if i+1 < len(tokens) and pos_tags[i+1].lower() in self.pos_dict:
            feature[21] = 16 + 3*self.vocab_size + 3 + self.pos_dict[pos_tags[i+1].lower()] + 2*self.total_posTags

        
        return feature