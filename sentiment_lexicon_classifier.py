# Sentiment lexicon-based classifier for CSE517-A1

import os
import numpy as np

class Sentiment_Lexicon_Classifier:
    def __init__(self):
        self.pos_words_list = []
        self.neg_words_list = []

        cwd = os.getcwd()

        # read in positive words into list from file
        pos_words_path = cwd + '/opinion_lexicon_English/positive-words.txt'
        pos_words_file = open(pos_words_path, 'r', encoding="ISO-8859-1")
        for line in pos_words_file:
            stripped_line = line.strip()
            if len(stripped_line) != 0:
                if stripped_line[0] != ';':
                    self.pos_words_list.append(stripped_line)

        pos_words_file.close()

        # read in negative words into list from file
        neg_words_path = cwd + '/opinion_lexicon_English/negative-words.txt'
        neg_words_file = open(neg_words_path, 'r', encoding="ISO-8859-1")

        for line in neg_words_file:
            stripped_line = line.strip()
            if len(stripped_line) != 0:
                if stripped_line[0] != ';':
                    self.neg_words_list.append(stripped_line)

        neg_words_file.close()

    def tokenize(self, file_path):
        file = open(file_path, 'r', encoding="ISO-8859-1")
        text = file.read()
        file.close()
        parsed_text = text.split()

        punctuation = [',', '.', '?', '!', '(', ')', ':']
        for word in parsed_text:
            if word in punctuation:
                parsed_text.remove(word)

        return parsed_text

    def predict(self, x):
        # x - list of words

        pos_count = 0
        neg_count = 0

        for word in x:
            if word in self.pos_words_list:
                pos_count += 1
            if word in self.neg_words_list:
                neg_count += 1

        if pos_count >= neg_count:
            return 1
        else:
            return -1

if __name__ == "__main__":
    # set up test data
    cwd = os.getcwd()
    test_dir = cwd + '/review_polarity/txt_sentoken/test'

    X_test_path = np.empty(0, dtype = str)
    y_test = np.empty(0, dtype = int)

    pos_dir = test_dir + '/pos'
    for filename in os.listdir(pos_dir):
        if filename.endswith(".txt"):
            file_path = pos_dir + '/' + filename
            X_test_path = np.append(X_test_path, file_path)
            y_test = np.append(y_test, 1)

    neg_dir = test_dir + '/neg'
    for filename in os.listdir(neg_dir):
        if filename.endswith(".txt"):
            file_path = neg_dir + '/' + filename
            X_test_path = np.append(X_test_path, file_path)
            y_test = np.append(y_test, -1)

    test_size = len(X_test_path)

    # initiate Sentiment Lexicon Classifier
    classifier = Sentiment_Lexicon_Classifier()

    true_positive_count = 0
    false_positive_count = 0
    true_negative_count = 0
    false_negative_count = 0

    for i in range(test_size):
        x_parsed = classifier.tokenize(X_test_path[i])
        y_label = y_test[i]

        output_label = classifier.predict(x_parsed)

        if y_label == 1 and output_label == 1:
            true_positive_count += 1
        elif y_label == -1 and output_label == 1:
            false_positive_count += 1
        elif y_label == -1 and output_label == -1:
            true_negative_count += 1
        elif y_label == 1 and output_label == -1:
            false_negative_count += 1

    test_accuracy = (true_positive_count + true_negative_count) / (true_positive_count + false_positive_count + true_negative_count + false_negative_count)

    precision = true_positive_count / (true_positive_count + false_positive_count)
    recall = true_positive_count / (true_positive_count + false_negative_count)

    F_1 = 2 * (precision * recall) / (precision + recall)

    print('test accuracy = {val:.3f}'.format(val = test_accuracy))
    print('F_1 = {val:.3f}'.format(val = F_1))
