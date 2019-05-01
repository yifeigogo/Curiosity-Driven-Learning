'''
The deep learning model using pre-trained word-embedding
'''
from string import punctuation
from os import listdir
from collections import Counter
from collections import Counter
from matplotlib import pyplot
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.utils.data as utils


class Word_Embed(nn.Module):
	def __init__(self, vocab_size, embedding_dim,out_feature, pretrained_weight, n_layers, bidirectional, dropout, hidden_dim = 1):
		super(Word_Embed, self).__init__()
		
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight))
		self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional=bidirectional, dropout = dropout)
		
		self.linear = nn.Linear(hidden_dim*2, out_feature)
		self.sig = nn.Sigmoid()
		self.dropout = nn.Dropout(dropout)

	def forward(self, input):
		
		input = torch.transpose(input, 0, 1)
		x = self.embeddings(input)

		output_1, (hidden, cell) = self.rnn(x)
		
		hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
		#assert torch.equal(output_1[-1,:,:], hidden.squeeze(0))
		output_2 = self.linear(hidden.squeeze(0))
		
		output = self.sig(output_2)
		
		return output




def create_vocab(filename, vocab):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	tokens = text.split()
	vocab.update(tokens)
	return vocab

def load_clean(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	tokens = text.split()
	return tokens

def doc_to_line(filename, vocab):
	tokens = load_clean(filename)
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)

def process_docs(filename, vocab):
	lines = list()
	line = doc_to_line(filename, vocab)
	lines.append(line)
	return lines

def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

def generate_vect(filename, vocab, x_length):
	text = load_clean(filename)
	y_length = len(vocab)
	bag_matrix = np.zeros((x_length,y_length))
	j = 0
	with open(filename) as f:	
		for line in f:
			for w in line:
				if(w in vocab):
					bag_matrix[j][vocab[w]] += 1
			j += 1

	return bag_matrix

def get_max(filename, x_length):
	max_len = 0
	with open(filename) as f:
		for line in f:
			line = line.split()
			max_len = max(max_len, len(line))
			
	return max_len

def padding(filename, vocab, x_length):
	max_len = get_max(filename, x_length)
	word_embed = np.zeros((x_length, max_len))
	actual_len = np.zeros(x_length)
	j = 0
	with open(filename) as f:
		for line in f:
			line = line.split()
			actual_len[j] = len(line)
			#word_embed[j][max_len] = len(line)
			#print(word_embed[j][len(line)])
			for i in range(len(line)):
				if(line[i] in vocab):
					word_embed[j][i] = vocab[line[i]]
			j += 1
	#print(word_embed[:,word_embed.shape[1]-1])
	return word_embed, actual_len


		
def get_label(y_pred):
	for i in range(y_pred.shape[0]):
		if(y_pred[i] >= 0.5):
			y_pred[i] = 1
		else:
			y_pred[i] = 0
	return y_pred

def get_glove(filename):
	words = []
	idx = 0
	word_to_idx = {}
	vectors = []
	with open(filename) as f:
		for line in f:
			line = line.split()
			word = line[0]
			words.append(word)
			word_to_idx[word] = idx
			idx += 1
			vec = np.array(line[1:]).astype(np.float)
			vectors.append(vec)
	return word_to_idx, vectors

def generate_dic(filename, target_vocab, emb_dim):
	word_to_idx, vectors = get_glove(filename)
	matrix_len = len(target_vocab)
	weights_matrix = np.zeros((matrix_len, emb_dim))
	words_found = 0

	for i, word in enumerate(target_vocab):
		if word in word_to_idx:
			weights_matrix[i] = vectors[word_to_idx[word]]
			words_found += 1
		else:
			weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
	return weights_matrix

def main():
	#glove = torchtext.vocab.GloVe(name='6B', dim=100)
	vocab_temp = load_clean('vocab.txt')
	vocab_temp = sorted(list(set(vocab_temp)))
	vocab_size = len(vocab_temp)
	vocab = {}

	for i, word in enumerate(vocab_temp):
		vocab[word] = i
	
	#get training data
	Y_train = load_clean('train_label_b.txt')

	x_length = len(Y_train)
	#generate_embed('train_val.txt', vocab, x_length)
	Y_t = np.zeros(x_length)
	for i in range(x_length):
		if (Y_train[i] >= '0.5'):
			Y_t[i] = 1
		#Y_t[i] = int(Y_train[i])
	
	X_train, train_len = padding('train_val.txt', vocab, x_length)
	X_train, Y_t = torch.from_numpy(X_train), torch.from_numpy(Y_t)
	
	#get testing data
	
	Y_test = load_clean('test_label_b.txt')
	x_length = len(Y_test)
	Y_tt = np.zeros(x_length)
	for i in range(x_length):
		if (Y_test[i] >= '0.5'):
			Y_tt[i] = 1
		#Y_tt[i] = int(Y_test[i])
	X_test, test_len = padding('test_val.txt', vocab, x_length)
	X_test, Y_tt = torch.from_numpy(X_test), torch.from_numpy(Y_tt)

	X_train, Y_t, X_test, Y_tt = X_train.type(torch.LongTensor), Y_t.type(torch.FloatTensor), X_test.type(torch.LongTensor), Y_tt.type(torch.IntTensor)
	



	train_dataset = utils.TensorDataset(X_train, Y_t)
	test_dataset = utils.TensorDataset(X_test, Y_tt)
	
	train_loader = utils.DataLoader(train_dataset, batch_size=10, shuffle=True)
	test_loader = utils.DataLoader(test_dataset, batch_size=10, shuffle=True)
	
	# training
	embedding_dim = 100
	pretrained_weight = generate_dic('glove.6B.100d.txt', vocab_temp,embedding_dim)

	hidden_dim = 128*2
	n_layers = 1
	bidirectional = True
	dropout = 0.5
	model = Word_Embed(vocab_size, embedding_dim, 1, pretrained_weight, n_layers, bidirectional, dropout, hidden_dim)
	criterion = torch.nn.BCELoss()
	#criterion = nn.NLLLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
	running_loss = 0
	for epoch in range(30):
		running_loss = 0
		total = 0
		correct = 0
		for i, data in enumerate(train_loader, 0):
			inputs, labels = data
			#print(inputs.type)
			#print(train_len.type)
			optimizer.zero_grad()

			y_pred = model(inputs)
			#print(y_pred)

			#print("..................................")
			#print(y_pred.shape)
			#print(labels.shape)
			predicted = (y_pred > 0.5).type(torch.int).view(-1,)
			total += labels.shape[0]
			opt_labels = (labels > 0.5).type(torch.int).view(-1,)
			correct += (predicted == opt_labels).sum().item()

		

			loss = criterion(y_pred, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		print('epoch: ', epoch, ' loss: ', running_loss)
		print('Accuracy of training the network: %d %%' % (100 * correct / total))
	#test
	correct = 0
	total = 0
	with torch.no_grad():
		for i, data in enumerate(test_loader, 0):
			inputs, labels = data
			outputs = model(inputs)
		
			predicted = (outputs > 0.5).type(torch.int).view(-1,)
			#print (inputs)
			print (predicted)
			total += labels.shape[0]
			opt_labels = (labels > 0.5).type(torch.int).view(-1,)
			correct += (predicted == opt_labels).sum().item()

	print('Accuracy of the network: %d %%' % (100 * correct / total))
	print(correct)
	print(total)

	
	
if __name__== "__main__":
    main()
