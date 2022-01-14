# reads in 16 randomly selected train files and finds the 1000 most common words. writes to '/review_polarity/txt_sentoken/dictionary.txt'

import os
import numpy as np

def tokenize(file_path):
    file = open(file_path, 'r', encoding="ISO-8859-1")
    text = file.read()
    file.close()
    parsed_text = text.split()

    punctuation = [',', '.', '?', '!', '(', ')', ':', '"', '*','-',';']
    for word in parsed_text:
        if word in punctuation:
            parsed_text.remove(word)

    return parsed_text


# set up train data
cwd = os.getcwd()
train_dir = cwd + '/review_polarity/txt_sentoken/train'

X_train_path = np.empty(0, dtype = str)
y_train = np.empty(0, dtype = int)

pos_dir = train_dir + '/pos'
for filename in os.listdir(pos_dir):
    if filename.endswith(".txt"):
        file_path = pos_dir + '/' + filename
        X_train_path = np.append(X_train_path, file_path)
        y_train = np.append(y_train, 1)

neg_dir = train_dir + '/neg'
for filename in os.listdir(neg_dir):
    if filename.endswith(".txt"):
        file_path = neg_dir + '/' + filename
        X_train_path = np.append(X_train_path, file_path)
        y_train = np.append(y_train, -1)

train_size = len(X_train_path)

words_list = []
size = 16
X_paths = X_train_path[np.random.permutation(np.arange(train_size))[0:size]]
size = len(X_paths)
for i in range(size):
    parsed_text = tokenize(X_paths[i])
    words_list += parsed_text

print(words_list)
print(len(words_list))

dict = cwd + '/review_polarity/txt_sentoken/dictionary.txt'

file = open(dict, 'w')

ordered_list = []
ordered_list_count = []

for j in range(1000):
    count = np.zeros(len(words_list), int)
    for i in range(len(words_list)):
        count[i] = words_list.count(words_list[i])

    max_index = np.argmax(count)
    max_word = words_list[max_index]
    max_word_count = count[max_index]

    ordered_list.append(words_list[max_index])
    ordered_list_count.append(count[max_index])


    for i in range(max_word_count):
        words_list.remove(max_word)

    file.writelines(max_word + ',' + str(max_word_count) + '\n')

file.close()
