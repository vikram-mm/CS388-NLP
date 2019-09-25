# models.py

from optimizers import *
from nerdata import *
from utils import *

from collections import Counter
from typing import List
from crf import CRF
import numpy as np
import pickle
import os
import torch
import torch.optim as optim


class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs


    def decode(self, sentence_tokens: List[Token]):
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """

        N = len(self.init_log_probs)
        T = len(sentence_tokens)
        dp = np.zeros((N, T)) - np.inf #initialized to a low value as we have to find maximum
        prev = np.zeros((N, T)) #backpointers

        #initialization 
        for state_idx in range(N):
            word_idx = self.word_indexer.index_of(sentence_tokens[0].word)
            if(word_idx == -1):
               word_idx = self.word_indexer.index_of("UNK")
            dp[state_idx,0] = self.init_log_probs[state_idx] + self.emission_log_probs[state_idx, word_idx]
        
        #forward
        for t in range(1, T):
            token = sentence_tokens[t]
            word_idx = self.word_indexer.index_of(token.word)
            if(word_idx == -1):
               word_idx = self.word_indexer.index_of("UNK")
               
            for cur_state_idx in range(N):
                tmp = dp[:, t-1] + self.transition_log_probs[:, cur_state_idx]
                best_prev = tmp.argmax()
                prev[cur_state_idx, t] = best_prev
                dp[cur_state_idx, t] = tmp[best_prev] + self.emission_log_probs[cur_state_idx, word_idx]
         
        #backtracing
        pred_tag_indexes = [] #will be stored in reverse order
        temp_status = np.argmax(dp[:, -1])
        pred_tag_indexes.append(temp_status)
        
        for t in range(T-1, 0, -1):
           temp_status = prev[int(temp_status), t]
           pred_tag_indexes.append(temp_status)
        
        pred_tag_indexes = pred_tag_indexes[::-1]
        pred_tags = []

        for tag_index in pred_tag_indexes:
            pred_tags.append(self.tag_indexer.get_object(tag_index))


        assert len(pred_tags) == T
        

        # raise Exception("IMPLEMENT ME")
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))



def train_hmm_model(sentences: List[LabeledSentence]) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.zeros((len(tag_indexer)), dtype=float) + 0.0001
    transition_counts = np.zeros((len(tag_indexer),len(tag_indexer)), dtype=float)  + 0.000000001
    emission_counts = np.zeros((len(tag_indexer),len(word_indexer)), dtype=float)   + 0.0001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print("Tag indexer: %s" % tag_indexer)
    print("Initial state log probabilities: %s" % init_counts)
    print("Transition log probabilities: %s" % transition_counts)
    print("Emission log probs too big to print...")
    print("Emission log probs for India: %s" % emission_counts[:,word_indexer.add_and_get_index("India")])
    print("Emission log probs for Phil: %s" % emission_counts[:,word_indexer.add_and_get_index("Phil")])
    print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, crf_model):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.model = crf_model

    def decode(self, sentence_tokens):
        tag_indexer = self.tag_indexer
        feature_indexer = self.feature_indexer

        all_features = []
        for word_idx in range(0, len(sentence_tokens)):
            features = []
            for tag_idx in range(0, len(tag_indexer)):
                features.append(extract_emission_features(sentence_tokens,\
                    word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=False))

            all_features.append(features)
        
        # print(all_features)
        all_features = np.array(all_features)
        # print(all_features.shape)
        best_tags = self.model(all_features)

        
        
        pred_tags = []

        for tag in best_tags:
            if tag_indexer.get_object(tag) is None:
                print(tag)
                print(best_tags)
                exit(0)
            pred_tags.append(tag_indexer.get_object(tag))
        
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))
            


# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences):
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    

    feature_indexer_file = "german_feature_indexer.pkl"
    feature_cache_file = "german_features.pkl"

    if not os.path.isfile(feature_indexer_file) or not os.path.isfile(feature_cache_file):
        print("Extracting german features")
        feature_indexer = Indexer()
        feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
        for sentence_idx in range(0, len(sentences)):
            if sentence_idx % 100 == 0:
                print("Ex %i/%i" % (sentence_idx, len(sentences)))
            for word_idx in range(0, len(sentences[sentence_idx])):
                for tag_idx in range(0, len(tag_indexer)):
                    feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
    
        pickle.dump(feature_indexer, open(feature_indexer_file, "wb"))
        pickle.dump(feature_cache, open(feature_cache_file, "wb"))

    else:
        print("Loading features")
        feature_indexer = pickle.load(open(feature_indexer_file, "rb"))
        feature_cache = pickle.load(open(feature_cache_file, "rb"))

    lr = 0.01
  

    num_epochs = 3
    train = True
    if train:
        crf_model = CRF(num_features = len(feature_indexer), nb_labels = len(tag_indexer), )
        transmission_optimizer = optim.SGD([crf_model.transitions], lr=lr)
        emmision_optimizer = optim.Adam([crf_model.emmision_weights], lr=lr)
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_count = 0.0
            for sentence_idx in range(len(feature_cache)):
                sentence = sentences[sentence_idx]
                true_tags = []
                for tag in sentence.get_bio_tags():
                    true_tags.append(tag_indexer.index_of(tag))
                x = np.array(feature_cache[sentence_idx])

                true_tags  = np.expand_dims(np.array(true_tags), axis=0)

                crf_model.zero_grad()
                loss = crf_model.loss(x, true_tags)
                total_loss += loss.item()
                total_count += 1
                loss.backward()
                emmision_grads = crf_model
                # transmission_optimizer.step()
                emmision_optimizer.step()

                if(sentence_idx%100 == 0):
                    print("epoch {} {}/{} done loss {}".format(epoch, sentence_idx, len(feature_cache), total_loss/total_count))

                # if(sentence_idx == 2000):
                #     break
            print("epoch {}, loss {}".format(epoch, total_loss/total_count))
            save_path = "model_crf_confirm.crf"
            torch.save(crf_model, save_path)
            print("model saved to {}".format(save_path))
    else:
        # print(tag_indexer.__repr__)
        crf_model = torch.load("model.crf")

    return CrfNerModel(tag_indexer, feature_indexer, crf_model)


    # raise Exception("IMPLEMENT THE REST OF ME")


def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    
    while len(feats) < 14:
        feats.append(0)
    return np.asarray(feats, dtype=int)

