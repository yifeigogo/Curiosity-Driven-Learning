'''
The data we get has label as a number from 0 to 1. However, we need to make
the label to be 0 or 1 to make it as a binary classifier. This file is to transfer the labels.
'''
def create_label(filename):
	file1 = open(filename + '_b.txt', 'w')
	filename_t = filename + '.txt'
	with open(filename_t) as f:
		for line in f:
			#line.split(' ')
			k = line[0:3]
			
			print(k)
			if(k < '0.5'):
				file1.write('0')
			else:
				file1.write('1')
			file1.write('\n')
	file1.close()
#create_dataset('train')
create_label('train_label')
create_label('test_label')

