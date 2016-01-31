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

file_to_read = 'snli_1.0/snli_1.0_train.jsonl'

try:
    #read whole file into python array
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
    print data_df[0]
except IOError as e:
    print e
