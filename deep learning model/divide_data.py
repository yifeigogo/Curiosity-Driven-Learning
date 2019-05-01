def divide_data():
	with open("data.txt") as f:
		lines = f.readlines()
	num = len(lines)
	file1 = open("train.txt", 'w')
	file2 = open("test.txt", 'w')
	for i in range (num):
		if i < 0.80 * num:
			file1.write(lines[i])
		else:
			file2.write(lines[i])
divide_data()