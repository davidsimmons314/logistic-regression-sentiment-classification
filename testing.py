# CSE517-A1

import os
import numpy as np
from logistic_regression_classifier import Logistic_Regression_Classifier
from sentiment_lexicon_classifier import Sentiment_Lexicon_Classifier

if __name__ == "__main__":
    # initiate Sentiment Lexicon Classifier
    lexicon_classifier = Sentiment_Lexicon_Classifier()

    # set up train data
    cwd = os.getcwd()
    train_dir = cwd + '/review_polarity/txt_sentoken/train'

    X_train_path = np.empty(0, dtype=str)
    y_train = np.empty(0, dtype=int)

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

    # initiate logistic classifier
    regression_classifier = Logistic_Regression_Classifier()

    n = len(X_train_path)
    d = regression_classifier.d
    phi_X_train = np.zeros([n, d])

    for i in range(n):
        # print(i, '/' , n)
        parsed_text = regression_classifier.tokenize(X_train_path[i])
        phi_x = regression_classifier.to_feature_vector(parsed_text)
        phi_X_train[i, :] = phi_x

    # print(phi_X_train)

    # normalize phi_X_train
    regression_classifier.compute_standardizers(phi_X_train)
    regression_classifier.SGD(phi_X_train, y_train, 100, 0.01)

    stop = False
    while not stop:
        print()
        print(" *** Please enter one of the following commands *** ")
        print("> filename")
        print("> quit")
        print()
        response = ""
        print("> Enter: ", end='')

        try:
            response = str(input())
        except ValueError:
            print("Type in a valid argument")
            break

        response = response.lower()
        tokens = response.split(" ")
        if len(tokens) == 0:
            ValueError("Try Again")
            continue
        operation = tokens[0]
        if operation == "quit":
            print("Goodbye!")
            stop = True
        elif operation.endswith(".txt"):
            lexicon_parsed_x = lexicon_classifier.tokenize(operation)
            lexicon_output = lexicon_classifier.predict(lexicon_parsed_x)

            regression_parsed_x = regression_classifier.tokenize(operation)
            phi_x = regression_classifier.to_feature_vector(regression_parsed_x)
            non = np.copy(phi_x)
            phi_x = regression_classifier.standardize(phi_x)

            regression_output = regression_classifier.predict(phi_x)

            file = open(operation, 'r',encoding="ISO-8859-1")
            print(file.read())
            file.close()
            print("Sentiment Lexicon Classification: ", lexicon_output)
            print("Logistic Regression Classification: ", regression_output)

            # /Users/davidsimmons/Documents/GitHub/CSE517-A1/review_polarity/txt_sentoken/breakinggood.txt
        else:
            print("Invalid Argument")