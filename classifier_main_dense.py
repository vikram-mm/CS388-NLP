# classifier_main.py

import argparse
import sys
import time
from nerdata import *
from utils import *
from collections import Counter
from optimizers import *
from typing import List
from dense_features_3 import Featurizer
import random
random.seed(11)

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
    return 1/(1+ np.e**(-Z))

def logistic_loss(y, y_pred, scale_toBalance=1):
    return -np.sum(scale_toBalance *y * np.log(y_pred+10e-5) + (1-y) * np.log(1 - y_pred+10e-5))

class PersonClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self, weights: np.ndarray, featurizer: Featurizer):
        self.W = weights
        self.featurizer = featurizer

    def predict(self, tokens: List[str], idx: int):
        """
        Makes a prediction for token at position idx in the given PersonExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
        feature = self.featurizer.featurize_oneInstance(tokens, idx)
        X = expand_features(np.array(feature), self.featurizer.num_dimensions,\
             self.featurizer.num_direct_dimensions)

        z = np.matmul(X, self.W)
        y_pred = sigmoid(z)

        if(y_pred > 0.5):
            return 1
        
        return 0
    # def predict(self, tokens, idx):
        # raise Exception("Implement me!")

def shuffle_together(a, b):

    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def expand_features(X, num_dimensions, num_direct_dimensions):

    should_squeeze = False
    if(len(X.shape) == 1):
        X = np.expand_dims(X, axis=0)
        should_squeeze = True

    new_X = np.zeros((len(X), num_dimensions))
    new_X[:, 0:num_direct_dimensions] = X[:, 0:num_direct_dimensions]

    for i in range(len(new_X)):
        new_X[i, X[i,-3].astype(int)] = 1
        new_X[i, X[i,-2].astype(int)] = 1
        new_X[i, X[i,-1].astype(int)] = 1

    if(should_squeeze):
        new_X = np.squeeze(new_X)
    
   
    return new_X

def get_gradient_keys(X, num_direct_dimensions):

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


def train_classifier(ner_exs: List[PersonExample], should_train = True):

    featurizer = Featurizer(ner_exs)
    W = np.zeros(featurizer.num_dimensions)
    # W = np.load('dense_SGD_weights_4.npy')


    if should_train:
        # optimizer = BatchSGDOptimizer(W, alpha=0.1)
        optimizer = UnregularizedAdagradTrainer(W)
        # optimizer = L1RegularizedAdagradTrainer(W)

        num_epochs = 16
        scale_toBalance = 1.0
        
        all_X = []
        all_Y = []

        #gathering all train examples
        for ex in train_class_exs:
            features = featurizer.featurize(ex.tokens)

            for feature, label in zip(features, ex.labels):
                if(label == 1): #increasing positive examples
                    for j in range(10): 
                        all_X.append(np.expand_dims(feature, axis=0))
                        all_Y.append(np.expand_dims(label, axis=0))

            X = np.array(features)
            Y = np.array(ex.labels)
            all_X.append(X)
            all_Y.append(Y)
        
        all_X = np.concatenate(all_X)
        all_Y = np.concatenate(all_Y)

        batch_size = 16
        regularization = False
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_count = 0.0

            # if((epoch+1) % 5 == 0):
            #     print('stepping down')
            #     optimizer.alpha /= 10.0

            if epoch > 8:
                regularization = True
                prob = 0.8

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
                    for key in gradient_key_list[i]:
                        if not regularization or np.random.random_sample() > 1 - prob:
                            gradient_counter[key] += gradient[i] * X[i, key]
                
                optimizer.apply_gradient_update(gradient_counter, batch_size)

                cursor += batch_size
                total_loss += np.sum(loss)
                total_count += batch_size
            
            np.save('dense_SGD_weights_4.npy', W)

            print('Loss at epoch {} : {}'.format(epoch, total_loss/total_count))

    else:
        W = np.load('dense_SGD_weights_3.npy')

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
            predictions.append(classifier.predict(ex.tokens, idx))
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
            prediction = classifier.predict(ex.tokens, idx)
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
        classifier = train_classifier(train_class_exs, True)

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



