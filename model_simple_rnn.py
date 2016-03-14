__author__ = 'david_torrejon'


"""
This module implements a simple RNN model for the snli paper:
http://nlp.stanford.edu/pubs/snli_paper.pdf
"""


from keras.layers import recurrent
from keras.models import Sequential, slice_X
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Merge, Dense, Dropout, Flatten
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer, one_hot, base_filter

import numpy as np
import glove
from extract_sentences import return_sparse_vector

class paper_model():
    #parameters
    def __init__(self, number_stacked_layers, vocabulary_size):
        self.RNN = recurrent.SimpleRNN
        self.stacked_layers = number_stacked_layers
        self.vocab_size = vocabulary_size
        self.weights_path = "./weights.hdf5"

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

    def data_preparation_nn(self, sentences):
        premises_encoded = []
        hypothesis_encoded = []
        expected_output = []
        for data in sentences:
            premises_encoded.append( return_sparse_vector(data[1][0], self.vocab_size ))
            hypothesis_encoded.append( return_sparse_vector(data[1][1], self.vocab_size ))
            expected_output.append(data[2])

        return np.asarray(premises_encoded), np.asarray(hypothesis_encoded), np.asarray(expected_output)

    def build_model(self, max_features, data_train, data_test, word2idx, LOAD_W=True):
            #DROPOUT TO INPUT AND OUTPUTS OF THE SENTENCE EMBEDDINGS!!
        print('Build embeddings model...')
        #check this maxlen
        maxlen = 35

        premise_model = Sequential()
        hypothesis_model = Sequential()
        # 2 embedding layers 1 per premise 1 per hypothesis
        #premise_model.add(Flatten(i))
        premise_model.add(self.RNN(100, init='normal', activation='tanh', input_shape=(maxlen,self.vocab_size)))
        #premise_model.add(Dropout(0.1))

        #hypothesis_model.add(Embedding(input_dim=self.vocab_size, output_dim=self.vocab_size, input_length=maxlen))
        hypothesis_model.add(self.RNN(100, init='normal', activation='tanh', input_shape=(maxlen,self.vocab_size)))
        #hypothesis_model.add(Dropout(0.1))

        print('Concat premise + hypothesis...')
        nli_model = Sequential()
        nli_model.add(Merge([premise_model, hypothesis_model], mode='concat', concat_axis=1))

        for i in range(1, self.stacked_layers):
            print ('stacking %d layer')%i
            nli_model.add(Dense(input_dim=100, output_dim=200, init='normal', activation='tanh'))

        print ('stacking last layer')
        nli_model.add(Dense(input_dim=200, output_dim=3, init='normal', activation='tanh'))
        print ('Softmax layer...')
        # 3 way softmax (entail, neutral, contradiction)
        nli_model.add(Dense(3, init='uniform'))
        nli_model.add(Activation('softmax')) # care! 3way softmax!

        print('Compiling model...')
        #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        nli_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        print('Model Compiled')
        print('generating sparse vectors(1hot encoding) from sentences...')


        #split data
        #data preparation
        premises_encoded, hypothesis_encoded, expected_output = self.data_preparation_nn(data_train)

        """
        nb_samples, timesteps, input_dim) means:
        - nb_samples samples (examples)
        - for each sample, a number of time steps (the same for all samples in the batch)
        - for each time step of each sample, input_dim features.
        NEEEEED TO CONVER DATA TO SPARSE VECTORSSASDASD
        """

        '''
        TODO PREPARE DATA OUTSIDE HERE TO GENERATE BOTH TEST AND TRAIN
        '''

        print('premsises shape and sample....')
        print premises_encoded.shape
        print('hypothesis shape and sample....')
        print hypothesis_encoded.shape
        print('output shape and sample....')
        print expected_output.shape
        print expected_output[0]

        #(nb_samples, timesteps, input_dim). -> (expected_output[0], [1], len_vocab)

        X = [premises_encoded, hypothesis_encoded]


        if LOAD_W:
            print('loading weights...')
            nli_model.load_weights(self.weights_path)
        # I dont want the conversion here, make the conversion somewhere else, for esthetic purpouses
        print('training....')
        nli_model.fit(X, expected_output, batch_size=64, nb_epoch=10, verbose=1, sample_weight=None, show_accuracy=True)

        print ('testing....')
        premises_encoded_t, hypothesis_encoded_t, expected_output_t = self.data_preparation_nn(data_test)
        print('saving weights')
        nli_model.save_weights(self.weights_path, overwrite=True)
        X_t = [premises_encoded_t, hypothesis_encoded_t]
        score = nli_model.evaluate(X_t, expected_output_t, batch_size=64, show_accuracy=True, verbose=1)
        predictions = nli_model.predict_classes(X_t, batch_size=64, verbose=1)

        """
        store results?
        """

        f = open('predictions.txt', 'w')
        for pred, e_out in zip(predictions, expected_output_t):
            sup = str(pred) + str(e_out)
            f.write(sup)
            f.write('\n')
        f.close()



    # model.fit()
"""
    def model_train(self, data_train):


    def model_evaluate(self):
        score = model.evaluate(X_test, Y_test, batch_size=16)
"""
