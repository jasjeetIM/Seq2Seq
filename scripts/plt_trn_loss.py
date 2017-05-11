#! /usr/bin/python
import sys
import matplotlib.pyplot as plt
import numpy as np

def get_training_loss(train_log):
  """Get the training loss from a caffe logfile into an array"""
  with open(train_log) as f:
    lines = f.readlines()
  train_loss = []
  for line in lines:
    if "cross_entropy_loss = " not in line:
      continue
    else:
      train_loss.append(float(line[line.index("=")+2:line.index("(")-1]))

  #Plot Results
  iters = np.arange(0, len(train_loss)*100, 100)
  plt.plot(iters, train_loss)

  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  plt.title('Training Loss')
  plt.grid(True)
  plt.savefig("test.png")
  plt.show()


if __name__ == "__main__":
  if len(sys.argv) == 2:
    log_file = sys.argv[1]
    get_training_loss(log_file)
  else:
    print "Incorrect number of args\nUsage: print_training_loss training_file.log"
