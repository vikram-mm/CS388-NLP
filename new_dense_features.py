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
    for ex in ner_exs:
        for token in ex.tokens:
            if re.match('^[a-zA-Z][\w-]+$', token.lower()):   #only words and not numbers, dates etc.
                if token.lower() not in frequency_dict:
                    frequency_dict[token.lower()] = 0
                else:
                    frequency_dict[token.lower()] += 1
    
    # for key, value in sorted(frequency_dict.items(), key=lambda item: item[1]):
    #     print("%s: %s" % (key, value))
    #threshold is heuristically selected to be 40

    return frequency_dict

def get_frequentNameDict(ner_exs: List[PersonExample]):

    frequentNameDict = {}
    for ex in ner_exs:
        for i, token in enumerate(ex.tokens):
            if re.match('^[a-zA-Z][\w-]+$', token.lower()):   #only words and not numbers, dates etc.

                if ex.labels[i] == 1:
                    if token.lower() not in frequentNameDict:
                        frequentNameDict[token.lower()] = 0
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
                        oneBeforeNameDict[token.lower()] = 0
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
                        twoBeforeNameDict[token.lower()] = 0
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
                        oneAfterNameDict[token.lower()] = 0
                    else:
                        oneAfterNameDict[token.lower()] += 1

    # for key, value in sorted(oneAfterNameDict.items(), key=lambda item: item[1]):
    #     print("%s: %s" % (key, value))
    #threshold is heuristically selected to be 2

    return oneAfterNameDict



class Featurizer():

    def __init__(self, train_class_exs: List[PersonExample]):
        '''
        computes the token level statistics of the train data, which will be used to computer
        the features of the tokens
        '''
        self.vocab_dict = generate_vocabDict(train_class_exs)
        self.frequencyDict =  get_frequencyDict(train_class_exs)
        self.frequentNameDict = get_frequentNameDict(train_class_exs)
        self.oneBeforeNameDict = get_oneBeforeNameDict(train_class_exs)
        self.twoBeforeNameDict = get_twoBeforeNameDict(train_class_exs)
        self.oneAfterNameDict = get_oneAfterNameDict(train_class_exs)
        self.vocab_size = len(self.vocab_dict) + 1 # plus one for none of the words
        self.num_dimensions = 11 + 1 + self.vocab_size #plus one for bias
        self.num_direct_dimensions = 11 + 1 # 1 is bias

    
    def featurize(self, tokens: List[str], frequency_threshold=40, frequentName_threshold = 15, oneBefore_threshold=18,\
     twoBefore_threshold = 16, oneAfter_threshold=2):
        
        '''
        the four dictionaries passed in the argument are generated only using the training data.
        the thresholds were heuristically deterimined 
        '''

        #Feature dimensions:
        # 1 -> is it the first word of the sentence? (0 or 1)
        # 2 -> Does it start with a capital letter? (0 or 1)
        # 3 -> is it a simple number with no hyphens? (0 or 1)  
        # 4 -> is it a number with hyphens? (o or 1) (such as dates)
        # 5 -> is it a word? (0 or 1)
        # 6 -> is it a word with hyphens (0 or 1)
        # 7 -> length of word (integer)
        # 8 -> is it a frequent word? (frequency in frequencyDict atleast frequency_threshold) (0 or 1)
        # 9 -> is the previous word frequent before a name? (frequency of previous word in oneBeforeNameDict
        #atleast oneBefore_threshold ) 
        # 10 -> same as 9 but for two words before (0 or 1)
        # 11 -> same as 9 but for one word after (0 or 1)
        # 12 - 12 + vocab_size - the index at which the current token is present
        # (if present) in the vocabDict is set to 1
        # 12 + vocab_size - 12 + 2*vocab_size - the index at which the previous token is present
        # (if present) in the vocabDictis set to 1\
        # 12 + 2*vocab_size - 12 + 3*vocab_size - the index at which the previous token is present
        # (if present) in the vocabDictis set to 1

        #Therefore, dimensions of the feature is 11 + vocab_size*3 + 1
        # last one is bias

        vocab_size = self.vocab_size
        features_list = []
        gradient_key_list = []
        for i, token in enumerate(tokens):
            
            feature = np.zeros(11 + 1 + 3)
            gradient_keys = [] #only weights corresponding to these dimension numbers need to be updated
            feature[0] = int(i==0)
            feature[1] = int(re.match('[A-Z][\w-]+$', token) is not None)
            feature[2] = int(re.match('^[\d]+$', token) is not None)
            feature[3] = int(re.match('^[\d][\d-]+[-]+[\d-]+$', token) is not None)
            feature[4] = int(re.match('^[a-zA-Z][\w-]+$', token) is not None)
            feature[5] = int(re.match('^[a-zA-Z][\w-]+[-]+[\w-]+$', token) is not None)
            feature[6] = int(token in self.frequentNameDict and self.frequentNameDict[token] > frequentName_threshold)
            feature[7] = int(token in self.frequencyDict and self.frequencyDict[token] > frequency_threshold)
            feature[8] = int(i>=1 and tokens[i-1] in self.oneBeforeNameDict and self.oneBeforeNameDict[tokens[i-1]]\
            > oneBefore_threshold)
            feature[9] = int(i>=2 and tokens[i-2] in self.twoBeforeNameDict and self.twoBeforeNameDict[tokens[i-2]]\
            > twoBefore_threshold)
            feature[10] = int(i+1 < len(tokens) and tokens[i+1] in self.oneAfterNameDict and self.oneAfterNameDict[tokens[i+1]]\
            > oneAfter_threshold)

            for j in range(10):
                if(feature[j] == 1):
                    gradient_keys.append(j)

            feature[11] = 1 #for bias
            
            feature[12:] = 12 + self.vocab_size - 1#words not in vocab

            if token in self.vocab_dict:
                feature[12] = 12 + self.vocab_dict[token]
                gradient_keys.append(12 + self.vocab_dict[token])
            else:
                gradient_keys.append(12 + self.vocab_size -1)

            if i >= 1 and tokens[i-1] in self.vocab_dict:
                feature[13] = 12 + self.vocab_dict[tokens[i-1]]
                gradient_keys.append(12 + self.vocab_dict[tokens[i-1]])
            else:
                gradient_keys.append(12 + self.vocab_size -1)
                
            if i+1 < len(tokens) and tokens[i+1] in self.vocab_dict:
                feature[14] = 12 + self.vocab_dict[tokens[i+1]]
                gradient_keys.append(12 + self.vocab_dict[tokens[i+1]])
            else:
                gradient_keys.append(12 + self.vocab_size -1)

            features_list.append(feature)
            gradient_key_list.append(gradient_keys)
        
        return features_list, gradient_key_list

    def featurize_oneInstance(self, tokens: List[str], idx, frequency_threshold=40, frequentName_threshold = 15, oneBefore_threshold=18,\
     twoBefore_threshold = 16, oneAfter_threshold=2):

        '''
        same as the featurize but only computes the feature for the token 
        at the given index
        '''

        vocab_size = self.vocab_size
        i = idx
        token = tokens[idx]     

        
        feature = np.zeros(11 + 1 + 3)
        gradient_keys = [] #only weights corresponding to these dimension numbers need to be updated
        feature[0] = int(i==0)
        feature[1] = int(re.match('[A-Z][\w-]+$', token) is not None)
        feature[2] = int(re.match('^[\d]+$', token) is not None)
        feature[3] = int(re.match('^[\d][\d-]+[-]+[\d-]+$', token) is not None)
        feature[4] = int(re.match('^[a-zA-Z][\w-]+$', token) is not None)
        feature[5] = int(re.match('^[a-zA-Z][\w-]+[-]+[\w-]+$', token) is not None)
        feature[6] = int(token in self.frequentNameDict and self.frequentNameDict[token] > frequentName_threshold)
        feature[7] = int(token in self.frequencyDict and self.frequencyDict[token] > frequency_threshold)
        feature[8] = int(i>=1 and tokens[i-1] in self.oneBeforeNameDict and self.oneBeforeNameDict[tokens[i-1]]\
        > oneBefore_threshold)
        feature[9] = int(i>=2 and tokens[i-2] in self.twoBeforeNameDict and self.twoBeforeNameDict[tokens[i-2]]\
        > twoBefore_threshold)
        feature[10] = int(i+1 < len(tokens) and tokens[i+1] in self.oneAfterNameDict and self.oneAfterNameDict[tokens[i+1]]\
        > oneAfter_threshold)

        for j in range(10):
            if(feature[j] == 1):
                gradient_keys.append(j)

        feature[11] = 1 #for bias
            
        feature[12:] = 12 + self.vocab_size - 1#words not in vocab

        if token in self.vocab_dict:
            feature[12] = 12 + self.vocab_dict[token]
            gradient_keys.append(12 + self.vocab_dict[token])
        else:
            gradient_keys.append(12 + self.vocab_size -1)

        if i >= 1 and tokens[i-1] in self.vocab_dict:
            feature[13] = 12 + self.vocab_dict[tokens[i-1]]
            gradient_keys.append(12 + self.vocab_dict[tokens[i-1]])
        else:
            gradient_keys.append(12 + self.vocab_size -1)
            
        if i+1 < len(tokens) and tokens[i+1] in self.vocab_dict:
            feature[14] = 12 + self.vocab_dict[tokens[i+1]]
            gradient_keys.append(12 + self.vocab_dict[tokens[i+1]])
        else:
            gradient_keys.append(12 + self.vocab_size -1)

        return feature












if __name__ == '__main__':                                                                                                                                                                                                                                                                                                      

    args = _parse_args()
    train_class_exs = list(transform_for_classification(read_data(args.train_path)))
    dev_class_exs = list(transform_for_classification(read_data(args.dev_path)))

    featurizer = Featurizer(train_class_exs)

    from sklearn.linear_model import LogisticRegression as model
    from sklearn.metrics import f1_score
    import pickle

    X = []
    Y = []
    for ex in train_class_exs:
        features = featurizer.featurize(ex.tokens)
        X += features
        Y += ex.labels

    np_y = np.array(Y)
    pos_class = np.sum(np_y==1)
    neg_class = np.sum(np_y==0)

    # print(len(np_y)/(2 * pos_class))
    # print(len(np_y)/(2* neg_class))
    # exit(0)
        
    print(len(X))
    print(len(Y))

    model_instance = model(class_weight= 'balanced', max_iter=1000)
    model_instance.fit(X, Y)
    print(model_instance.score(X, Y))

    pickle.dump(model_instance, open('LR_model.pkl', 'wb'))
    predY = model_instance.predict(X)
    print(np.sum(predY))
    print(f1_score(Y, predY))
    exit(0)

    model_instance = pickle.load(open('LR_model.pkl', 'rb'))

    devX = []
    devY = []

    for ex in dev_class_exs:
        features = featurize(ex.tokens, vocab_dict, frequencyDict, oneBeforeNameDict, twoBeforeNameDict, oneAfterNameDict)
        devX += features
        devY += ex.labels
    predY = model_instance.predict(devX)
    print(np.sum(predY))
    print(f1_score(devY, predY))
    print(model_instance.score(devX, devY))
