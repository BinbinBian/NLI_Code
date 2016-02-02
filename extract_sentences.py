__author__ = 'david_torrejon'

"""
This module extracts the sentences from the snli corpus:
 Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015.
 A large annotated corpus for learning natural language inference.
 In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)

 http://nlp.stanford.edu/projects/snli/
"""
import sys
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np

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



def embed_words(sentences_df):
    '''
    @parameter: the dataframe from the json file with the 5 columns we need
    @returns: the vocabulary in a set.
    '''
    vocabulary = []
    list_of_sentences1 = sentences_df['sentence1'].tolist()
    list_of_sentences2 = sentences_df['sentence2'].tolist()
    # print list_of_sentences1[:3]
    # print list_of_sentences2[:3]
    # idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    # idx2word  = dict((k,v) for v,k in dic['words2idx'].iteritems())

    for sentence in list_of_sentences1:
        sentence.lower()
        #tokenize or split by " "
        tokens1 = sentence.split(" ")
        for token1 in tokens1:
            if token1 not in vocabulary:
                vocabulary.append(token1)
    print "length of vocabulary: %d"%len(vocabulary)







df_data = read_json_file()
vocab = embed_words(df_data)
