__author__ = 'david_torrejon'

from random import seed, uniform
from extract_sentences import make_unicode, label_output_data
import numpy as np
import re
import string
from sys import stdout

def create_embedding_sentence(sentence, glove_dict, maxlen=45):

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

    if len(vectorized_sentence) > maxlen:
        return np.asarray(vectorized_sentence[:maxlen])
    else:
        return np.asarray(vectorized_sentence)


def create_embeddings(df_data, glove_dict, batch_i, batch_size, test_tbir=False):

    embedded_sentences = []

    print('Generating embeddings')
    if test_tbir is False:
        list_premises = df_data['sentence1'].tolist()
        list_hypothesis = df_data['sentence2'].tolist()
        list_label = df_data['gold_label'].tolist()
        list_annotator = df_data['annotator_labels'].tolist()

        if batch_i+batch_size > len(df_data):
            limit = len(df_data)-1
        else:
            limit = batch_i+batch_size

        for i in range(batch_i, limit): #zip(list_premises, list_hypothesis, list_label, list_annotator):
            stdout.write("\rloading embeddings: %d" % i)
            stdout.flush()

            label_no_unicode = make_unicode(list_label[i])
            numpy_label = label_output_data(label_no_unicode)
            premise_encoded=create_embedding_sentence(list_premises[i], glove_dict)
            hypothesis_encoded=create_embedding_sentence(list_hypothesis[i], glove_dict) # do the cut here whether it has same labels or not
            embedded_sentences.append([[premise_encoded, hypothesis_encoded], numpy_label, len(set(list_annotator[i]))])
    else:
        for i,data_point in enumerate(df_data):
            stdout.write("\rloading embeddings tbir: %d" % i)
            #embeddings creation here
            #print data_point
            #raise SystemExit(0)
            premise_encoded = create_embedding_sentence(data_point[0], glove_dict)
            hypothesis_encoded=create_embedding_sentence(data_point[1], glove_dict)
            #numpy_label = label_output_data(data_point[3])
            embedded_sentences.append([[premise_encoded, hypothesis_encoded], data_point[2], data_point[3], data_point[4]])
    #print embedded_sentences[0] #make sure that works! remove at the end
    print ""
    return embedded_sentences
