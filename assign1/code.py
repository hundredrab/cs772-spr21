"""
About the task:

You are provided with a codeflow- which consists of functions to be implemented(MANDATORY).

You need to implement each of the functions mentioned below, you may add your own function parameters if needed(not to main).
Execute your code using the provided auto.py script(NO EDITS PERMITTED) as your code will be evaluated using an auto-grader.
"""

import math
import string
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def vocab_dict(text):
    uniq = set()
    text.apply(uniq.update)
    int_to_word = dict(enumerate(uniq))
    int_to_word[len(int_to_word)] = '<UNK>'
    word_to_int = {v: k for k, v in int_to_word.items()}
    return int_to_word, word_to_int


def encode_data(text, vdic):
    """Encode text to integers.

    This function will be used to encode the reviews using a dictionary
    (created using corpus vocabulary)

    Example of encoding :
        "The food was fabulous but pricey"
        {'The':1,'food':2,'was':3 'fabulous':4 'but':5 'pricey':6}
    """

    def encode_sent(sent: list):
        return [vdic.get(x, -1) for x in sent]

    return text.apply(encode_sent)


def convert_to_lower(text: pd.Series):
    """Return the reviews after convering them to lowercase."""
    return text.str.lower()


def remove_punctuation(text):
    """Return the reviews after removing punctuations."""
    return text.str.replace("[{}]".format(string.punctuation), "")


def perform_tokenization(text):
    """Return the reviews after performing tokenization."""
    return text.apply(word_tokenize)


def remove_stopwords(text):
    """Return the reviews after removing the stopwords."""
    stop = stopwords.words("english")
    return text.apply(lambda x: [item for item in x if item not in stop])


def perform_padding(data, n=30):
    """Return the reviews after padding the reviews to maximum length."""

    def pad_sent(sent, n=n):
        """Pad or trim a given sentence to make it of length n."""
        return sent[:n] + [-1] * (n - len(sent))

    return data.apply(pad_sent)


def preprocess_data(data, vdic=None):
    """Preprocesses data using the above defined functions."""
    data = convert_to_lower(data)
    data = remove_punctuation(data)

    data = perform_tokenization(data)

    data = remove_stopwords(data)

    _, w2i = vocab_dict(data)
    # data = encode_data(data, w2i)

    # data = perform_padding(data)


    return data, w2i


"""
        review = data["reviews"]
        review = convert_to_lower(review)
        review = remove_punctuation(review)
        review = remove_stopwords(review)
        review = perform_tokenization(review)
        review = encode_data(review)
        review = perform_padding(review)
"""
# return processed data


def softmax_activation(x):
    """
    Write your own implementation from scratch and return softmax values

    (using predefined softmax is prohibited)
    """
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x-maxes)
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    return x_exp/x_exp_sum

    # num = torch.exp(x)
    # return num / torch.sum(num)
    # return np.exp(x) / np.sum(np.exp(x), axis=0)


class NeuralNet(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # self.reviews = reviews.float()
        # self.ratings = ratings
        self.build_nn(num_features)

    def build_nn(self, num_features):
        print(num_features, 98765)
        self.fc1 = nn.Linear(num_features, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = softmax_activation(x)
        return x

    def train(self, train_df):#, batch_size, epochs):
        from torch.optim import Adam

        for i in range(2000):
            batch = train_df.groupby('ratings').apply(lambda x: x.sample(16))
            # import pdb; pdb.set_trace();
            optm = Adam(self.parameters(), lr=0.001)
            self.zero_grad()
            x = torch.tensor(np.matrix(batch.reviews.tolist())).float()
            y = torch.tensor(batch.ratings.tolist())
            output = self(x)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output.float(), y)
            loss.backward()
            optm.step()
            print(loss)


    def predict(self, reviews):
        print(type(reviews))
        reviews = torch.tensor(np.matrix(reviews.tolist()))
        with torch.no_grad():
            out = self.forward(reviews.float())
            print(out)
            return np.argmax(out, axis=1)
        # return a list containing all the ratings predicted by the trained model


# DO NOT MODIFY MAIN FUNCTION'S PARAMETERS
def main(train_file, test_file):
    from sklearn.model_selection import train_test_split
    from keras.utils import to_categorical
    from sklearn.feature_extraction.text import CountVectorizer


    batch_size, epochs = 1, 2
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    train_reviews = train["reviews"]
    train_ratings = train["ratings"] - 1
    test_reviews = test["reviews"]

    # train_reviews, vocab_dict = preprocess_data(train_reviews)
    train_df = pd.concat([train_reviews, train_ratings], axis=1)
    train_df, validation_df = train_test_split(train_df, random_state=42)
    vectorizer = CountVectorizer(min_df=2, stop_words=stopwords.words('english'))
    train_df.reviews = vectorizer.fit_transform(train_df.reviews).toarray().tolist()
    validation_df.reviews = vectorizer.transform(validation_df.reviews).toarray().tolist()
    X_test = pd.Series(vectorizer.transform(test_reviews).toarray().tolist())
    # y = train_df.ratings
    # y_val = validation_df.ratings
    # import pdb; pdb.set_trace();


    # h1_enc = OneHotEncoder(handle_unknown='ignore')
    # import pdb; pdb.set_trace();
    # h1_enc.fit(train_reviews.tolist())
    # import pdb; pdb.set_trace();



    # h1_enc.transform(train_reviews)
    # train_reviews = train_reviews.apply(lambda x: to_categorical(x))
    # import pdb; pdb.set_trace();
    # test_reviews, _ = preprocess_data(test_reviews, vocab_dict)


    model = NeuralNet(len(vectorizer.get_feature_names()))
    # model.build_nn()
    # model.train_nn(batch_size, epochs)
    model.train(train_df)
    # print(torch.tensor(np.matrix(test_reviews.to_list())))
    print('hii')
    # import pdb; pdb.set_trace()
    predictions = model.predict(validation_df.reviews).tolist()#(model.predict(torch.tensor(np.matrix(test_reviews.to_list())))).tolist()
    correct = (sum(np.array(predictions) == np.array(validation_df.ratings)))
    print(len(predictions))
    print(f"Accuracy: {correct/len(predictions)}")
    from collections import Counter
    # import pdb; pdb.set_trace()
    print(Counter(predictions))

    return
