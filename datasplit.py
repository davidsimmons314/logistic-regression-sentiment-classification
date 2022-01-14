# Function to randomly split data into 80% train set and 20% test set
# saves to folders '/review_polarity/txt_sentoken/train/pos', '/review_polarity/txt_sentoken/train/neg'
# and  '/review_polarity/txt_sentoken/test/pos', '/review_polarity/txt_sentoken/test/neg'

import os
import numpy as np

cwd = os.getcwd()
dir = cwd + '/review_polarity/txt_sentoken'

# array of strings for filenames and respective sentiment of total dataset
total_dataset_filenames = np.empty(0, dtype = str)
total_dataset_sentiment = np.empty(0, dtype = str)
# add all positive filenames
pos_dir = dir + '/pos'
for filename in os.listdir(pos_dir):
    if filename.endswith(".txt"):
        total_dataset_filenames = np.append(total_dataset_filenames, filename)
        total_dataset_sentiment = np.append(total_dataset_sentiment, 'pos')

# add all negative filenames
neg_dir = dir + '/neg'
for filename in os.listdir(neg_dir):
    if filename.endswith(".txt"):
        total_dataset_filenames = np.append(total_dataset_filenames,filename)
        total_dataset_sentiment = np.append(total_dataset_sentiment, 'neg')

# compute sizes
dataset_size = len(total_dataset_filenames)
train_size = int(0.80 * dataset_size)
test_size = dataset_size - train_size

# generate random indices and split among train_indices and test_indices
indices = np.arange(0, dataset_size)
random_indices = np.random.permutation(indices)
train_indices = random_indices[np.arange(0, train_size)]
test_indices = random_indices[np.arange(train_size, dataset_size)]

dir = cwd + '/review_polarity/txt_sentoken'

# first 80% of shuffled files are saved as train data
os.mkdir(dir + '/train')
os.mkdir(dir + '/train/pos')
os.mkdir(dir + '/train/neg')
for i in range(train_size):
    filename = total_dataset_filenames[random_indices[i]]
    sentiment = total_dataset_sentiment[random_indices[i]]

    src_path = dir + '/' + sentiment + '/' + filename
    dst_path = dir + '/train/' + sentiment + '/' + filename

    src_file = open(src_path, 'r')
    dst_file = open(dst_path, 'w')

    dst_file.write(src_file.read())

    src_file.close()
    dst_file.close()

# last 20% of shuffled files are saved as test data
os.mkdir(dir + '/test')
os.mkdir(dir + '/test/pos')
os.mkdir(dir + '/test/neg')
for i in range(train_size, dataset_size):
    filename = total_dataset_filenames[random_indices[i]]
    sentiment = total_dataset_sentiment[random_indices[i]]

    src_path = dir + '/' + sentiment + '/' + filename
    dst_path = dir + '/test/' + sentiment + '/' + filename

    src_file = open(src_path, 'r')
    dst_file = open(dst_path, 'w')

    dst_file.write(src_file.read())

    src_file.close()
    dst_file.close()
