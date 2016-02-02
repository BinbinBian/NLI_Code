__author__ = 'david_torrejon'

import word2vec

f = open('/Users/David/Downloads/text8/text8','r')
print f
# word2vec.word2phrase('/Users/David/Downloads/text8/text8', '/Users/David/Downloads/text8/text8phrases', verbose=True)

word2vec.word2vec('/Users/David/Downloads/text8/text8', '/Users/David/Downloads/text8/text8phrases', size=100, verbose=True)
