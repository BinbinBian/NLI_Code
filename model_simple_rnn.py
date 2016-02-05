__author__ = 'david_torrejon'


"""
This module implements a simple RNN model for the snli paper:
http://nlp.stanford.edu/pubs/snli_paper.pdf
"""


from keras.layers import recurrent
from keras.models import Sequential, slice_X
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Merge, Dense, Dropout
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
import numpy as np

class paper_model():
    #parameters
    def __init__(self, number_stacked_layers, vocabulary_size):
        self.RNN = recurrent.SimpleRNN
        self.stacked_layers = number_stacked_layers
        self.vocab_size = vocabulary_size

    #NN paramters!
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
    def build_model(self):
            #DROPOUT TO INPUT AND OUTPUTS OF THE SENTENCE EMBEDDINGS!!
        print('Build embeddings model...')
        premise_model = Sequential()
        hypothesis_model = Sequential()
        # 2 embedding layers 1 per premise 1 per hypothesis
        premise_model.add(self.RNN(input_dim=self.vocab_size, output_dim=100, init='normal', activation='tanh'))
        premise_model.add(Dropout(0.2))
        hypothesis_model.add(self.RNN(input_dim=self.vocab_size, output_dim=100, init='normal', activation='tanh'))
        hypothesis_model.add(Dropout(0.2))

        print('Concat premise + hypothesis...')
        nli_model = Sequential()
        nli_model.add(Merge([premise_model, hypothesis_model], mode='concat', concat_axis=-1))

        for i in range(1, self.stacked_layers):
            print ('stacking %d layer')%i
            nli_model.add(Dense(input_dim=200, output_dim=200, init='normal', activation='tanh'))

        print ('stacking last recurrent layer')
        nli_model.add(Dense(input_dim=200, output_dim=3, init='normal', activation='tanh'))
        print ('Softmax layer...')
        # 3 way softmax (entail, neutral, contradiction)
        nli_model.add(Dense(3, init='uniform'))
        nli_model.add(Activation('softmax')) # care! 3way softmax!

        print('Compiling model...')
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        nli_model.compile(loss='rmse', optimizer=sgd)
        print('Model Compiled')

    # model.fit()

    def model_train(self):
        model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)

    def model_evaluate(self):
        score = model.evaluate(X_test, Y_test, batch_size=16)
