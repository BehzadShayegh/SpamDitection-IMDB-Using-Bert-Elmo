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
  def shuffle(self, alpha=0.2) :
    indexes = np.zeros(self.size).astype(bool)
    indexes[np.random.choice(self.size, int(alpha*self.size), replace=False)] = True
    return indexes

  def __init__(self, x, y) :
    self.vocab = Vocab(x)
    self.x = np.array(x)
    self.y = np.array(y)
    self.xidx = np.array(list(map(lambda x: [1]+self.vocab.sent2idx(x), self.x)))
    self.size = len(y)
    self.splitter = self.shuffle()

  def get_train(self) :
    return self.x[~self.splitter], self.xidx[~self.splitter], self.y[~self.splitter]

  def get_test(self) :
    return self.x[self.splitter], self.xidx[self.splitter], self.y[self.splitter]

  def load_data(self, max_sequence_length):    
    _, x_train, y_train = self.get_train()
    _, x_test,  y_test  = self.get_test()
    x_train = sequence.pad_sequences(x_train, maxlen=max_sequence_length, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_sequence_length, padding='post', truncating='post')
    return (x_train, y_train), (x_test, y_test)

  def get_idx2word(self) :
    return self.vocab.idx2word