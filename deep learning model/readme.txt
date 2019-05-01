readme
If you want to train the model, you only need to run model.py.
If you have more data, you need to rebuild the dataset and train the model:
First, run preprocessing.py to seperate data into value and label
Second, run label_trans.py to make the label file to have the form we want.
Thrid, run shuffle.py, to shuffle all the data and run build_vocab.py to build vacabulary.
Fourth, run divide_data.py to divide the dataset into training and testing.
Finally, run model.py to train the model