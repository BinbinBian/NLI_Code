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
from keras.preprocessing.text import Tokenizer, one_hot, base_filter
import numpy as np
import glove

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
    def build_model(self, max_features, data_train, word2idx):
            #DROPOUT TO INPUT AND OUTPUTS OF THE SENTENCE EMBEDDINGS!!
        print('Build embeddings model...')
        #check this maxlen
        maxlen = 30

        premise_model = Sequential()
        hypothesis_model = Sequential()
        # 2 embedding layers 1 per premise 1 per hypothesis
        #premise_model.add(Embedding(input_dim=self.vocab_size, output_dim=self.vocab_size, input_length=maxlen))
        premise_model.add(self.RNN(input_dim=self.vocab_size, output_dim=100, init='normal', activation='tanh', input_length=maxlen))
        premise_model.add(Dropout(0.2))

        #hypothesis_model.add(Embedding(input_dim=self.vocab_size, output_dim=self.vocab_size, input_length=maxlen))
        hypothesis_model.add(self.RNN(input_dim=self.vocab_size, output_dim=100, init='normal', activation='tanh', input_length=maxlen))
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
        nli_model.compile(loss='mse', optimizer=sgd, class_mode='categorical'  )
        print('Model Compiled')
        print('training model...')
        premises = []
        hypothesis = []
        premises_encoded = []
        hypothesis_encoded = []
        expected_output = []
        #split data
        for data in data_train:
            premises.append(data[0][0])
            hypothesis.append(data[0][1])
            premises_encoded.append(data[1][0])
            hypothesis_encoded.append(data[1][1])
            expected_output.append(data[2])


        print(premises_encoded[0], hypothesis_encoded[0], expected_output[0])
        #train model

        #print('writing shitty dataset log')
        print(len(premises_encoded), len(hypothesis_encoded), len(expected_output))
        """
        f = open('dataset.txt', 'w')

        f.write('premises...\n')
        f.write(str(premises_encoded))
        f.write('hypothesis...\n')
        f.write(str(hypothesis_encoded))
        f.write('output...\n')
        f.write(str(expected_output))

        f.write('dictionary...\n')
        f.write(str(word2idx))
        """
        print('training....')
        #debug errors in dataset?????


        premises_encoded = np.asarray([premises_encoded])
        hypothesis_encoded = np.asarray([hypothesis_encoded])
        expected_output = np.asarray(expected_output)


        X = [premises_encoded, hypothesis_encoded]
        y = expected_output


        if len(set([len(a) for a in X] + [len(y)])) != 1:
            for a in X:
                print len(a)
            print('im retard, ', len(set([len(a) for a in X] + [len(y)])))



        # I dont want the conversion here, make the conversion somewhere else, for esthetic purpouses
        nli_model.fit([premises_encoded, hypothesis_encoded], expected_output, batch_size=128, nb_epoch=2, verbose=2)

    # model.fit()
"""
    def model_train(self, data_train):


    def model_evaluate(self):
        score = model.evaluate(X_test, Y_test, batch_size=16)
"""
