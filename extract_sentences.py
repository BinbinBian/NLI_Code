__author__ = 'david_torrejon'

"""
This module extracts the sentences from the snli corpus:
 Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015.
 A large annotated corpus for learning natural language inference.
 In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)
 http://nlp.stanford.edu/projects/snli/
"""
import sys
import pandas as pd
import numpy as np
import re
import string
from keras.preprocessing.text import text_to_word_sequence, base_filter
from keras.preprocessing.sequence import pad_sequences
from random import seed, uniform

tokenizing_errors = 0

def build_glove_dictionary():
    """
        builds a dictionary based on the glove model.
        http://nlp.stanford.edu/projects/glove/
        dictionary will have the form of key = token, value = numpy array with the pretrained values

        REALLY IMPORTANT the glove dataset. with the big one finds nearly everything....
        smallest one...quite baaaaaad...
    """
    print ('building glove dictionary...')
    glove_file = '../TBIR/glove.840B.300d.txt'
    glove_dict = {}
    with open(glove_file) as fd_glove:
        for i, input in enumerate(fd_glove):
            if i%5000 == 0:
                print i, 'entries on the dictionary'
            input_split = input.split(" ")
            #print input_split
            key = input_split[0] #get key
            del input_split[0]  # remove key
            values = []
            for value in input_split:
                values.append(float(value))
            np_values = np.asarray(values)
            glove_dict[key] = np_values

    print 'dictionary build with length', len(glove_dict)

    return glove_dict




def return_sparse_vector(sentence, vocab_size):
    """
        @params:
        sentence: array with the encoded sentence [1 534 232 ... 3 ...0]
        returns len(sentence) np.arrays with 1 hot encoded vectors
    """
    sparse_vector = []
    for item in sentence:
        one_hot_vector = np.zeros(vocab_size)
        one_hot_vector[item] = 1
        sparse_vector.append(one_hot_vector)

    return np.asarray(sparse_vector)


def read_json_file():

    file_to_read = 'snli_1.0/snli_1.0_dev.jsonl'

    try:
        #read whole file into python array
        print 'Opening File ' + file_to_read
        with open(file_to_read, 'rb') as f:
            data = f.readlines()
        '''
        Each element of 'data' is an individual JSON object.
        I want to convert it into an *array* of JSON objects
        which, in and of itself, is one large JSON object
        basically... add square brackets to the beginning
        and end, and have all the individual business JSON objects
        separated by a comma
        '''
        data_json_str = "[" + ','.join(data) + "]"
        # now, load it into pandas
        data_df = pd.read_json(data_json_str)

        # Only need sentence 1 and sentence 2 + id + annotator_labels
        """
        sample
        {"annotator_labels": ["neutral"], "captionID": "3416050480.jpg#4", "gold_label": "neutral", "pairID": "3416050480.jpg#4r1n", "sentence1": "A person on a horse jumps over a broken down airplane.",
        "sentence1_binary_parse": "( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )",
        "sentence1_parse": "(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))",
        "sentence2": "A person is training his horse for a competition.", "sentence2_binary_parse": "( ( A person ) ( ( is ( ( training ( his horse ) ) ( for ( a competition ) ) ) ) . ) )",
        "sentence2_parse": "(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (VP (VBG training) (NP (PRP$ his) (NN horse)) (PP (IN for) (NP (DT a) (NN competition))))) (. .)))"}
        """
        # ,['sentence1_binary_parse'],['sentence1_parse'],['sentence2_binary_parse'],['sentence2_parse']
        del data_df['captionID']
        del data_df['sentence1_binary_parse']
        del data_df['sentence1_parse']
        del data_df['sentence2_binary_parse']
        del data_df['sentence2_parse']
        print data_df.head(3)
        print data_df.tail(3)

        return data_df
    except IOError as e:
        print e

def make_unicode(source_text):
    not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
    no_unicode = ''
    for char in source_text:
        if char in not_letters_or_digits:
            char = " "
        no_unicode+=char
    return no_unicode

def label_output_data(label):
    labels = {'neutral':np.array([0,1,0]),
                'entailment':np.array([1,0,0]),
                'contradiction':np.array([0,0,1]),
                ' ': np.array([0,0,0])
                }
    return labels[label]

"""
Generates a encoded vector, each cell 1 numbered refering to a word in the word2idx dictionary
could use keras text to word, but gives some trouble with nonunicode tokens...have to convert to unicode all the time...see glove.py
"""

def create_vectorized_sentence(sentence, word2idx):

    vectorized_sentence = []

    regex = re.compile('[%s]' % re.escape(string.punctuation))
    sentence = regex.sub('', sentence).lower()
    tokenized_sentence = sentence.split(" ")

    for token in tokenized_sentence:
        if word2idx.get(token):
            vectorized_sentence.append(word2idx[token])
        else:
            vectorized_sentence.append(0) # dunno how to deal with mistakes, ask!

    return np.asarray(vectorized_sentence)


def pad_sentence(sentence, max_len=35, pad_with=0):
        '''
            @pads a single sentence
            @if sentence is below max_len, returns a sentence(vectorized) of max_len with 0s on the right
            @sentence is a list of tokens (right now, a numpy array)
            @if sentence length larger than max_len, truncates the sentence to the max_len first values
        '''
        padded_sentence=np.zeros(max_len, dtype=int)

        if len(sentence) < max_len:
            #pads
            for i,value in enumerate(sentence):
                padded_sentence[i] = value
        else:
            #truncates
            for i in range (len(padded_sentence)):
                padded_sentence[i] = sentence[i]

        return padded_sentence


def create_sentence_ds(sentences_df, word2idx, cut_ds,  maxlen=35,):
    # create pair [[s1,s2], label]
    seed = 1337 #great seed

    data_set = []
    print('Generating dataset')
    list_premises = sentences_df['sentence1'].tolist()
    list_hypothesis = sentences_df['sentence2'].tolist()
    list_label = sentences_df['gold_label'].tolist()

    for premise, hypothesis, label_text in zip(list_premises, list_hypothesis, list_label):
            num = uniform(1.0, 0.0)
            if num < cut_ds:
                label_no_unicode = make_unicode(label_text)
                numpy_label = label_output_data(label_no_unicode)
                premise_encoded = create_vectorized_sentence(premise, word2idx)
                hypothesis_encoded = create_vectorized_sentence(hypothesis, word2idx)
                padded_premise = pad_sentence(premise_encoded, max_len=maxlen)
                padded_hypothesis = pad_sentence(hypothesis_encoded, max_len=maxlen)
                #print numpy_label
                #([pre-hypo],[encoded pre-hypo],[100]output) first pair of values is unnecesary right now 27/02, just for debug purpouses
                data_set.append([[premise, hypothesis], [padded_premise, padded_hypothesis], numpy_label])

    return data_set

def give_vocabulary(sentences_df):
    '''
    @parameter: the dataframe from the json file with the 5 columns we need
    @returns: the vocabulary in a set.
    '''
    vocabulary = []
    list_of_sentences1 = sentences_df['sentence1'].tolist()
    list_of_sentences2 = sentences_df['sentence2'].tolist()
    list_sentence_words = []
    '''
    # Do same with keras
    for sentence in list_of_sentences1:
        sentence.lower()
        #tokenize or split by " "
        tokens1 = sentence.split(" ")
        for token1 in tokens1:
            if token1 not in vocabulary:
                vocabulary.append(token1)
    '''
    list_sentence_word_tmp = []
    for s1, s2 in zip (list_of_sentences1, list_of_sentences2):
        sentence_unicode1 = make_unicode(s1)
        sentence_unicode2 = make_unicode(s2)
        #print sentence_no_unicode
        list_sentence_word_tmp += text_to_word_sequence(sentence_unicode1.encode('ascii'), filters=base_filter(), lower=True, split=" ")
        list_sentence_word_tmp += text_to_word_sequence(sentence_unicode2.encode('ascii'), filters=base_filter(), lower=True, split=" ")

    set_words = set(list_sentence_word_tmp)
    word2idx = {}
    for i, word in enumerate(set_words):
        word2idx[word] = int(i)

    #print word2idx
    print "length of vocabulary: %d"%len(set_words)
    return set_words, len(set_words), word2idx

# simple test of extracting a json file and showing the len of the vocabulary
def test():
    df_data = read_json_file()
    vocab, len_vocab = give_vocabulary(df_data)
def test_labeling():
    labels = ['neutral','entailment','contradiction']
    for label in labels:
        print label_output_data(label)
