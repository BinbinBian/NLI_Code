__author__ = 'david_torrejon'


from extract_sentences import give_vocabulary, read_json_file, create_sentence_ds
from model_simple_rnn import paper_model
from shuffle_data import get_test_train_sets

import numpy as np


#split parameters for dev set
percentage_split_ds = 0.80
shitty_pc = True #to run it in shitty pcs turn it on
#it will shrink the ds by the fixed amount in the dev set consisting of 10k sent 0.4 will
#create a ds with only 4k pairs or both train and test...
# no need shitty_pc just fixate cut_ds to 1
cut_ds = 0.6
LOAD_W = True # LOAD weights

df_data = read_json_file()


vocabulary, size_vocabulary, word2idx = give_vocabulary(df_data)
dataset = create_sentence_ds(df_data, word2idx, shitty_pc, cut_ds)
print dataset[1][0][0] # ([[premise, hypothesis],numpy_label]) premise[n][0][0], hypothesis[n][0][1], label[n][1]



train_set, test_set = get_test_train_sets(dataset, percentage_split_ds)
# feed model with premise, hypothesis

#create class model
snli_model = paper_model(3, size_vocabulary)
#create MODEL ADD TYPE OF MODEL RNN OR LSTM build(model='type') word2idx might not be needed in the model anymore
snli_model.build_model(size_vocabulary, train_set, test_set, word2idx, LOAD_W)
#train model

#model test
#snli_model.model_evaluate(test_set)
