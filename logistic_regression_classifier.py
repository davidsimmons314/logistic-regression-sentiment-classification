# Logistic Regressin Classifier for CSE517-A1

import os
import numpy as np

class Logistic_Regression_Classifier:
    def __init__(self):

        self.words_list = []

        cwd = os.getcwd()
        dict = cwd + '/review_polarity/txt_sentoken/dictionary.txt'

        words_file = open(dict, 'r', encoding="ISO-8859-1")

        for line in words_file:
            split_line = line.split(',')
            self.words_list.append(split_line[0])

        words_file.close()

        self.words_list = np.array(self.words_list)

        # features = frequency of each word, bias
        self.d = 1 + len(self.words_list)
        self.theta = np.zeros(self.d)

        #self.theta = np.random.normal(0, 1, self.n)

        self.mu= None
        self.sigma = None

    def compute_standardizers(self, phi_X_train):
        n, d = np.shape(phi_X_train)

        self.mu = np.zeros(d)
        self.sigma = np.ones(d)

        # compute standarizers
        for i in range(0, d):
            self.mu[i] = np.mean(phi_X_train[:, i])
            self.sigma[i] = np.std(phi_X_train[:, i])
            # corner case if all entries are the same
            if self.sigma[i] == 0:
                self.sigma[i] = 1

        for i in range(n):
            phi_X_train[i,:] = self.standardize(phi_X_train[i,:])

    def standardize(self, phi_x):
        for i in range(self.d):
            phi_x[i] = (phi_x[i] - self.mu[i]) / self.sigma[i]

        return phi_x

    def tokenize(self, file_path):
        file = open(file_path, 'r', encoding="ISO-8859-1")
        text = file.read()
        file.close()
        parsed_text = text.split()

        punctuation = [',', '.', '?', '!', '(', ')', ':', '"', '*']
        for word in parsed_text:
            if word in punctuation:
                parsed_text.remove(word)

        return np.array(parsed_text)

    def to_feature_vector(self, parsed_text):
        phi_x = np.ones(self.d)
        for i in range(len(self.words_list)):
            phi_x[i] = np.count_nonzero(parsed_text == self.words_list[i])
        return phi_x

    def compute_grad(self, phi_x, y):
        return - y * phi_x / (1 + np.exp(y * self.theta.T * phi_x))

    def SGD(self, phi_X_train, y_train, epochs, learning_rate):
        self.d = 1 + len(self.words_list)
        self.theta = np.zeros(self.d)

        n, d = np.shape(phi_X_train)

        for t in range(epochs):
            theta = np.copy(self.theta)
            random_indices = np.random.permutation(np.arange(0,n))
            for i in random_indices:
                self.theta += - learning_rate * self.compute_grad(phi_X_train[i,:], y_train[i])
            #print('epoch = ', str(t), np.linalg.norm(theta-self.theta))

    def predict(self, phi_x):
        return np.sign(self.theta.dot(phi_x))

if __name__ == "__main__":
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

    classifier = Logistic_Regression_Classifier()

    n = len(X_train_path)
    d = classifier.d
    phi_X_train = np.zeros([n, d])

    for i in range(n):
        #print(i, '/' , n)
        parsed_text = classifier.tokenize(X_train_path[i])
        phi_x = classifier.to_feature_vector(parsed_text)
        phi_X_train[i,:] = phi_x

    #print(phi_X_train)

    # normalize phi_X_train
    classifier.compute_standardizers(phi_X_train)

    '''
    print(phi_X_train)
    print(classifier.mu)
    print(classifier.sigma)
    '''

    theta = np.copy(classifier.theta)
    classifier.SGD(phi_X_train, y_train, 100, 0.01)

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

    true_positive_count = 0
    false_positive_count = 0
    true_negative_count = 0
    false_negative_count = 0

    for i in range(test_size):
        x_parsed = classifier.tokenize(X_test_path[i])
        y_label = y_test[i]

        phi_x = classifier.to_feature_vector(x_parsed)
        phi_x = classifier.standardize(phi_x)

        output_label = classifier.predict(phi_x)

        #print(y_label, output_label)
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
