__author__ = 'david_torrejon'


"""
This module implements a simple RNN model for the snli paper:
http://nlp.stanford.edu/pubs/snli_paper.pdf
"""


from __future__ import print_function
from keras.layers import recurrent
from keras.models import Sequential, slice_X
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector
from keras.regularizers import l2, activity_l2
import numpy as np

#parameters
RNN = recurrent.SimpleRNN
#more parameters!

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
We applied L2 regularization to all models, manually tuning the strength coefficient λ for
each, and additionally applied dropout (Srivastava et al., 2014) to the inputs
and outputs of the sentence embedding models (though not to its internal
connections) with a fixed dropout rate. All models were implemented in a common framework for this paper.
"""

#DROPOUT TO INPUT AND OUTPUTS OF THE SENTENCE EMBEDDINGS!!


print('Build model...')
model = Sequential()

model.add(RNN(200, init='he_uniform', inner_init='orthogonal', activation='tanh'))
model.add(RNN(200, init='he_uniform', inner_init='orthogonal', activation='tanh'))
model.add(RNN(200, init='he_uniform', inner_init='orthogonal', activation='tanh'))

model.add(Activation('softmax')) # care! 3way softmax!

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
