__author__ = 'david_torrejon'

from random import seed, uniform
from extract_sentences import make_unicode, label_output_data
import numpy as np
import re
import string

def create_embedding_sentence(sentence, glove_dict, maxlen=50):

    vectorized_sentence = []

    regex = re.compile('[%s]' % re.escape(string.punctuation))
    sentence = regex.sub('', sentence).lower()
    tokenized_sentence = sentence.split(" ")
    for token in tokenized_sentence:
        try:
            if token in glove_dict:
                vectorized_sentence.append(glove_dict[token])
            else:
                vectorized_sentence.append(np.zeros(300)) # dunno how to deal with mistakes, ask!
        except:
            print ('token: ', token, 'went wrong...')

    while len(vectorized_sentence) < maxlen:
        vectorized_sentence.append(np.zeros(300))


    return np.asarray(vectorized_sentence)


def create_embeddings(df_data, glove_dict, cut_ds):

    embedded_sentences = []
    seed = 1337 #great seed

    data_set = []
    print('Generating dataset')
    list_premises = df_data['sentence1'].tolist()
    list_hypothesis = df_data['sentence2'].tolist()
    list_label = df_data['gold_label'].tolist()

    for premise, hypothesis, label_text in zip(list_premises, list_hypothesis, list_label):
            num = uniform(1.0, 0.0)
            if num < cut_ds:
                label_no_unicode = make_unicode(label_text)
                numpy_label = label_output_data(label_no_unicode)
                premise_encoded=create_embedding_sentence(premise, glove_dict)
                hypothesis_encoded=create_embedding_sentence(hypothesis, glove_dict)
                embedded_sentences.append([[premise_encoded, hypothesis_encoded], numpy_label])

    return embedded_sentences
