'''
This file is to build the vacabulary
'''
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from collections import Counter
from matplotlib import pyplot
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.utils.data as utils

def create_vocab(filename, vocab):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	tokens = text.split()
	vocab.update(tokens)
	return vocab

def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

vocab = Counter()
vocab = create_vocab('train_val.txt', vocab)
vocan = create_vocab('test_val.txt', vocab)
print (len(vocab))
tokens = [k for k, c in vocab.items() if c >= 1]
save_list(tokens, 'vocab.txt')