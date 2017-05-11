#! /usr/bin/python
import nltk
import sys
import codecs

def create_vocab(corpus_file,vocab_file, vocab_size):
  """Creates a txt vocab file from a corpus"""
  vs = int(vocab_size)
  with open(corpus_file, 'r') as f:
    data = f.read().replace('\n',' ')
    allwords = nltk.tokenize.word_tokenize(data.decode('utf-8'))
    dist = nltk.FreqDist(w.lower() for w in allwords)
    most_common = dist.most_common(vs)
    handle = codecs.open(vocab_file, 'w', encoding='utf8')
    for i in range(vs):
      handle.write(most_common[i][0])
      handle.write('\n')

if __name__ == "__main__":
  if len(sys.argv) == 4:
    cf = sys.argv[1]
    vf = sys.argv[2]
    sz = sys.argv[3]
    create_vocab(cf, vf, sz)
  else:
    print "Incorrect number of args"
