'''
When adding new data to the cureent dataset, please run preprocessing first to generate the
sentence file and label file.
'''
from string import punctuation
from os import listdir
from collections import Counter


def create_dataset(filename):
	file1 = open(filename + '_val.txt', 'w')
	file2 = open(filename + '_label.txt', 'w')
	filename_t = filename + '.txt'
	with open(filename_t) as f:
		for line in f:
			line.split()
			file1.write(line[4:])
			file2.write(line[0:3])
			file2.write('\n')
	file1.close()
	file2.close()
#create_dataset('train')
create_dataset('train')
create_dataset('test')

