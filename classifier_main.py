# classifier_main.py

import argparse
import sys
import time
from nerdata import *
from utils import *
from collections import Counter
from optimizers import *
from typing import List
from featurizers import Featurizer
import random
random.seed(11)
np.random.seed(11)

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
    parser.add_argument('--test_output_path', type=str, default='eng.testb_new.out', help='output path for test predictions')
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
    def __init__(self, tokens: List[str], pos_tags: List[str], labels: List[int]):
        self.tokens = tokens
        self.pos_tags = pos_tags
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
        yield PersonExample([tok.word for tok in labeled_sent.tokens], [tok.pos for tok in labeled_sent.tokens], labels)


class CountBasedPersonClassifier(object):
    """
    Person classifier that takes counts of how often a word was observed to be the positive and negative class
    in training, and classifies as positive any tokens which are observed to be positive more than negative.
    Unknown tokens or ties default to negative.
    Attributes:
        pos_counts: how often each token occurred with the label 1 in training
        neg_counts: how often each token occurred with the label 0 in training
    """
    def __init__(self, pos_counts: Counter, neg_counts: Counter):
        self.pos_counts = pos_counts
        self.neg_counts = neg_counts

    def predict(self, tokens: List[str], idx: int):
        if self.pos_counts[tokens[idx]] > self.neg_counts[tokens[idx]]:
            return 1
        else:
            return 0


def train_count_based_binary_classifier(ner_exs: List[PersonExample]):
    """
    :param ner_exs: training examples to build the count-based classifier from
    :return: A CountBasedPersonClassifier using counts collected from the given examples
    """
    pos_counts = Counter()
    neg_counts = Counter()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            if ex.labels[idx] == 1:
                pos_counts[ex.tokens[idx]] += 1.0
            else:
                neg_counts[ex.tokens[idx]] += 1.0
    print(repr(pos_counts))
    print(repr(pos_counts["Peter"]))
    print(repr(pos_counts["aslkdjtalk;sdjtakl"]))

    return CountBasedPersonClassifier(pos_counts, neg_counts)


def sigmoid(Z):
    """
    calculates the sigmoid function 
    """
    return 1/(1+ np.e**(-Z))

def logistic_loss(y, y_pred, scale_toBalance=1):
    """
    the scale_toBalance indicates the additional weight given to one class
    """
    return -np.sum(scale_toBalance *y * np.log(y_pred+10e-5) + (1-y) * np.log(1 - y_pred+10e-5))

class PersonClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self, weights: np.ndarray, featurizer: Featurizer):
        self.W = weights
        self.featurizer = featurizer

    def predict(self, tokens: List[str], pos_tags: List[str], idx: int):
        """
        Makes a prediction for token at position idx in the given PersonExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
        feature = self.featurizer.featurize_oneInstance(tokens, pos_tags, idx)
        X = expand_features(np.array(feature), self.featurizer.num_dimensions,\
             self.featurizer.num_direct_dimensions)               
        z = np.matmul(X, self.W)
        y_pred = sigmoid(z)

        if(y_pred > 0.55):
            return 1
        
        return 0
    # def predict(self, tokens, idx):
        # raise Exception("Implement me!")

def shuffle_together(a, b):
    """
    used to shuffle to np arrays together,
    such as the features and their corresponding labels
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def expand_features(X, num_dimensions, num_direct_dimensions):
    """
    the higher dimensional features (indicator and pos features) are uncompressed
    to their one hot form in this function
    """

    should_squeeze = False
    if(len(X.shape) == 1):
        X = np.expand_dims(X, axis=0)
        should_squeeze = True
    new_X = np.zeros((len(X), num_dimensions))
    new_X[:, 0:num_direct_dimensions] = X[:, 0:num_direct_dimensions]

    for i in range(len(new_X)): # 3 pos tags and 3 indicataor features
        new_X[i, X[i,-6].astype(int)] = 1
        new_X[i, X[i,-5].astype(int)] = 1
        new_X[i, X[i,-4].astype(int)] = 1
        new_X[i, X[i,-3].astype(int)] = 1
        new_X[i, X[i,-2].astype(int)] = 1
        new_X[i, X[i,-1].astype(int)] = 1

    if(should_squeeze):
        new_X = np.squeeze(new_X)
    return new_X

def get_gradient_keys(X, num_direct_dimensions):
    """
    return the list of  non zero indices of X, so that the 
    corresponding gradients can be updated
    """

    gradient_key_list = []
    for x in X:
        gradient_keys = []
        for i in range(num_direct_dimensions):
            if(x[i] != 0.0):
                gradient_keys.append(i)

        for i in range(num_direct_dimensions, X.shape[1]):
            gradient_keys.append(int(x[i]))
        # gradient_keys.append(int(x[-2]))
        # gradient_keys.append(int(x[-1]))

        gradient_key_list.append(gradient_keys)
    
    return gradient_key_list


def train_classifier(ner_exs: List[PersonExample]):
    """
    the training functions which return a predictor instance
    initialized with the trained weights
    """

    featurizer = Featurizer(ner_exs)
    W = np.random.rand(featurizer.num_dimensions)
    optimizer1 = UnregularizedAdagradTrainer(W)

    num_epochs = 13
    swap_epoch = 100
    scale_toBalance = 4.0
    batch_size = 1
    regularization = False
    all_X = []
    all_Y = []

    #gathering all train examples
    for ex in train_class_exs:
        features = featurizer.featurize(ex.tokens, ex.pos_tags)
        X = np.array(features)
        Y = np.array(ex.labels)
        all_X.append(X)
        all_Y.append(Y)
    
    all_X = np.concatenate(all_X)
    all_Y = np.concatenate(all_Y)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_count = 0.0
            

        all_X, all_Y = shuffle_together(all_X, all_Y)
        cursor = 0

        while cursor + batch_size < len(all_X):
            
            X = expand_features(all_X[cursor: cursor+batch_size, :], featurizer.num_dimensions,\
            featurizer.num_direct_dimensions)
            gradient_key_list = get_gradient_keys(all_X[cursor: cursor+batch_size, :], featurizer.num_direct_dimensions)
            y = all_Y[cursor: cursor+batch_size]
            z = np.matmul(X, W)
            y_pred = sigmoid(z)
            loss = logistic_loss(y, y_pred, scale_toBalance)
            gradient = y - y_pred
            
            gradient_counter = Counter()
            for i in range(len(gradient_key_list)):        
                if(y[i]==0):
                    scale_factor = 1.0
                else:
                    scale_factor = scale_toBalance       
                for key in gradient_key_list[i]:
                    if not regularization or key > featurizer.num_direct_dimensions:
                        gradient_counter[key] += gradient[i] * X[i, key] *scale_factor *10

            optimizer1.apply_gradient_update(gradient_counter, batch_size)

            cursor += batch_size
            total_loss += np.sum(loss)
            total_count += batch_size
        
        # np.save('dense_SGD_weights_final_18_{}.npy'.format(epoch), W)
        print('Loss at epoch {} : {}'.format(epoch, total_loss/total_count))
    
    W[15:] *= 1.2 # slightly increasing the weight given to indicator and pos features, a simple trick to avoid overfitting

    return PersonClassifier(W, featurizer)

def evaluate_classifier(exs: List[PersonExample], classifier: PersonClassifier):
    """
    Prints evaluation of the classifier on the given examples
    :param exs: PersonExample instances to run on
    :param classifier: classifier to evaluate
    """
    predictions = []
    golds = []
    for ex in exs:
        for idx in range(0, len(ex)):
            golds.append(ex.labels[idx])
            predictions.append(classifier.predict(ex.tokens, ex.pos_tags, idx))
    print_evaluation(golds, predictions)


def print_evaluation(golds: List[int], predictions: List[int]):
    """
    Prints statistics about accuracy, precision, recall, and F1
    :param golds: list of {0, 1}-valued ground-truth labels for each token in the test set
    :param predictions: list of {0, 1}-valued predictions for each token
    :return:
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision: %i / %i = %f" % (num_pos_correct, num_pred, prec))
    print("Recall: %i / %i = %f" % (num_pos_correct, num_gold, rec))
    print("F1: %f" % f1)


def predict_write_output_to_file(exs: List[PersonExample], classifier: PersonClassifier, outfile: str):
    """
    Runs prediction on exs and writes the outputs to outfile, one token per line
    :param exs:
    :param classifier:
    :param outfile:
    :return:
    """
    f = open(outfile, 'w')
    for ex in exs:
        for idx in range(0, len(ex)):
            prediction = classifier.predict(ex.tokens, ex.pos_tags,  idx)
            f.write(ex.tokens[idx] + " " + repr(int(prediction)) + "\n")
        f.write("\n")
    f.close()

if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)
    # Load the training and test data

    train_class_exs = list(transform_for_classification(read_data(args.train_path)))
    dev_class_exs = list(transform_for_classification(read_data(args.dev_path)))

    
    # Train the model
    if args.model == "BAD":
        classifier = train_count_based_binary_classifier(train_class_exs)
    else:
        classifier = train_classifier(train_class_exs)

    print("Data reading and training took %f seconds" % (time.time() - start_time))
    # Evaluate on training, development, and test data
    print("===Train accuracy===")
    evaluate_classifier(train_class_exs, classifier)
    print("===Dev accuracy===")
    evaluate_classifier(dev_class_exs, classifier)
    if args.run_on_test:
        print("Running on test")
        test_exs = list(transform_for_classification(read_data(args.blind_test_path)))
        predict_write_output_to_file(test_exs, classifier, args.test_output_path)
        print("Wrote predictions on %i labeled sentences to %s" % (len(test_exs), args.test_output_path))



