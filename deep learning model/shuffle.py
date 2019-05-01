'''
This file is to shuffle all the data to make the test dataset and training dataset.
'''
import random

def shuffle_data(filename):
	file = open("data.txt", 'w')
	with open(filename) as f:
		lines = f.readlines()
		random.shuffle(lines)
		for line in lines:
			file.write(line)
			#file.write('\n')
def check():
	lines = []
	with open("data.txt") as f:
		lines = f.readlines()
	with open("t_results_1.txt") as f:
		for line in f:
			if(line in lines):
				print("!")
			else:
				print(line)

#check()

shuffle_data("t_results_1.txt")

