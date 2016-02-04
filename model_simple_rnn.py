__author__ = 'david_torrejon'


"""
This module implements a simple RNN model for the snli paper:
http://nlp.stanford.edu/pubs/snli_paper.pdf
"""


from keras.layers import recurrent
from keras.models import Sequential, slice_X
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Merge, Dense, Dropout
from keras.regularizers import l2, activity_l2
from keras.layers.embeddings import Embedding
import numpy as np

#parameters
RNN = recurrent.SimpleRNN
vocab_size = 5000  # extract it from dataset! (extract_sentences already does that (set_words))

#NN paramters!
stacked_layers = 3

"""
simply a stack of three 200d
tanh layers, with the bottom layer taking the concatenated
sentence representations as input and the
top layer feeding a softmax classifier, all trained
jointly with the sentence embedding model itself.
"""

"""
models are randomly
initialized using standard techniques and trained using AdaDelta (Zeiler, 2012) minibatch SGD until
performance on the development set stops improving.
We applied L2 regularization to all models, manually tuning the strength coefficient (lambda) for
each, and additionally applied dropout (Srivastava et al., 2014) to the inputs
and outputs of the sentence embedding models (though not to its internal
connections) with a fixed dropout rate. All models were implemented in a common framework for this paper.
"""

#DROPOUT TO INPUT AND OUTPUTS OF THE SENTENCE EMBEDDINGS!!

print('Build embeddings model...')
premise_model = Sequential()
hypothesis_model = Sequential()
# 2 embedding layers 1 per premise 1 per hypothesis
premise_model.add(Embedding(vocab_size, 100, init='normal',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01), input_length=16))
premise_model.add(Dropout(0.2))
hypothesis_model.add(Embedding(vocab_size, 100, init='normal', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01), input_length=16))
hypothesis_model.add(Dropout(0.2))
print('Concat premise + hypothesis...')

nli_model = Sequential()
nli_model.add(Merge([premise_model, hypothesis_model], mode='concat', concat_axis=-1))


for i in range(1, stacked_layers):
    print ('stacking %d layer')%i
    nli_model.add(RNN(input_dim=200, output_dim=200, init='normal', inner_init='orthogonal', activation='tanh', return_sequences=True))

print ('stacking last recurrent layer')
nli_model.add(RNN(input_dim=200, output_dim=3, init='normal', inner_init='orthogonal', activation='tanh', return_sequences=False))


# 3 way softmax (entail, neutral, contradiction)
nli_model.add(Dense(3, init='uniform'))
nli_model.add(Activation('softmax')) # care! 3way softmax!

print('Compiling model...')
nli_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print('Model Compiled')

# model.fit()
