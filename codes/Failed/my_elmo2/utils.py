from keras.preprocessing import sequence
import numpy as np
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

class Vocab() :
  def __init__(self, x) :
    words = np.unique(tokenizer.tokenize(' '.join(x)))
    self.idx2word = {i+3:w for i,w in enumerate(words)}
    self.idx2word[0] = "<PAD>"
    self.idx2word[1] = "<START>"
    self.idx2word[2] = "<UNK>"
    self.word2idx = {w:i for i,w in self.idx2word.items()}

  def size(self) :
    return len(self.idx2word)

  def words(self) :
    return self.idx2word.values()

  def indexes(self) :
    return self.idx2word.keys()

  def sent2idx(self, s) :
    return list(map(self.word2idx.__getitem__, tokenizer.tokenize(s)))

class DataKeeper() :

  def __init__(self, xtrain, ytrain, xtest, ytest) :
    self.vocab = Vocab(xtrain+xtest)
    self.xtrain = np.array(xtrain)
    self.xtest = np.array(xtest)
    self.ytrain = np.array(ytrain)
    self.ytest = np.array(ytest)
    self.xtrainidx = np.array(list(map(lambda x: [1]+self.vocab.sent2idx(x), self.xtrain)))
    self.xtestidx = np.array(list(map(lambda x: [1]+self.vocab.sent2idx(x), self.xtest)))
    self.trainsize = len(ytrain)
    self.testsize = len(ytest)

  def get_train(self) :
    return self.xtrain, self.xtrainidx, self.ytrain

  def get_test(self) :
    return self.xtest, self.xtestidx, self.ytest

  def load_data(self, max_sequence_length):    
    _, x_train, y_train = self.get_train()
    _, x_test,  y_test  = self.get_test()
    x_train = sequence.pad_sequences(x_train, maxlen=max_sequence_length, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_sequence_length, padding='post', truncating='post')
    return (x_train, y_train), (x_test, y_test)

  def get_idx2word(self) :
    return self.vocab.idx2word
