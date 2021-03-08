"""
About the task:

You are provided with a codeflow- which consists of functions to be implemented(MANDATORY).

You need to implement each of the functions mentioned below, you may add your own function parameters if needed(not to main).
Execute your code using the provided auto.py script(NO EDITS PERMITTED) as your code will be evaluated using an auto-grader.
"""

import math
import os
import pickle
import string
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertModel, BertTokenizer

##
train_file='train.csv'
test_file='test.csv'
glove = defaultdict(lambda x: 0)
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove[word] = coefs
f.close()

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
train_reviews = train["reviews"]
train_ratings = train["ratings"] - 1
test_reviews = test["reviews"]
# train_reviews = "[CLS] " + train_reviews + " [SEP]"
##
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_reviews = train_reviews.apply(tokenizer.tokenize)
train_reviews = train_reviews.apply(tokenizer.convert_tokens_to_ids)
##
train_reviews = perform_padding(train_reviews)
##
train_reviews = torch.tensor(np.matrix(train_reviews.tolist()))
##
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
with torch.no_grad():
    outputs = model(train_reviews)
##
def get_embed(row: list):
    zero = torch.tensor([row])
    one = torch.tensor([[0]*len(row)])
    with torch.no_grad():
        return model(zero, one)
##
train_test_split
##
##


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


##
def perform_padding(data, n=30):
    """Return the reviews after padding the reviews to maximum length."""

    def pad_sent(sent, n=n):
        """Pad or trim a given sentence to make it of length n."""
        return sent[:n] + [-1] * (n - len(sent))

    return data.apply(pad_sent)
##


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
        self.fc1 = nn.Linear(num_features, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = softmax_activation(x)
        return x

    def train(self, train_df, validation_df):#, batch_size, epochs):
        from torch.optim import Adam

        EPOCHS = 2000
        LR = 0.001
        for i in range(EPOCHS):
            batch = train_df.groupby('ratings').apply(lambda x: x.sample(64))
            # import pdb; pdb.set_trace();
            optm = Adam(self.parameters(), lr=LR)
            self.zero_grad()
            x = torch.tensor(np.matrix(batch.reviews.tolist())).float()
            y = torch.tensor(batch.ratings.tolist())
            output = self(x)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output.float(), y)
            loss.backward()
            optm.step()
            if i%100 == 0:
                print(100*i//EPOCHS, loss)
                predictions = self.predict(validation_df.reviews).tolist()
                #print(accuracy_score(np.array(validation_df.ratings), np.array(predictions)))
                #print(accuracy_score(np.array(train_df.ratings), np.array(self.predict(train_df.reviews).tolist())))


    def predict(self, reviews):
        reviews = torch.tensor(np.matrix(reviews.tolist()))
        with torch.no_grad():
            out = self.forward(reviews.float())
            print(out)
            return np.argmax(out, axis=1)
        # return a list containing all the ratings predicted by the trained model


# DO NOT MODIFY MAIN FUNCTION'S PARAMETERS
def main(train_file, test_file):
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer #TfidfVectorizer as CountVectorizer
    from sklearn.metrics import f1_score
    from nltk.stem.porter import PorterStemmer

    stemmer = PorterStemmer()
    analyzer = CountVectorizer().build_analyzer()

    def stemmed_words(doc):
        return (stemmer.stem(w) for w in analyzer(doc))

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    train_reviews = train["reviews"]
    train_ratings = train["ratings"] - 1
    test_reviews = test["reviews"]


    ##### Check whether pretrained model exists
    MODEL_PATH = "model.pt"
    if os.path.isfile(MODEL_PATH):
        print("Loading model from disk; press ENTER to continue")
        input()
        vectorizer = pickle.load(open("vector.pickel", "rb"))
        model = NeuralNet(len(vectorizer.get_feature_names()))
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("Model does not exist; training")
        train_df = pd.concat([train_reviews, train_ratings], axis=1)
        train_df, validation_df = train_test_split(train_df, random_state=42)
        vectorizer = CountVectorizer(min_df=1, stop_words=stopwords.words('english'))#, analyzer=stemmed_words)
        train_df.reviews = vectorizer.fit_transform(train_df.reviews).toarray().tolist()
        validation_df.reviews = vectorizer.transform(validation_df.reviews).toarray().tolist()
        model = NeuralNet(len(vectorizer.get_feature_names()))
        model.train(train_df, validation_df)
        torch.save(model.state_dict(), MODEL_PATH)
        pickle.dump(vectorizer, open("vector.pickel", "wb"))

    X_test = pd.Series(vectorizer.transform(test_reviews).toarray().tolist())


    print(len(vectorizer.get_feature_names()), "uniq vocab")
    test_pred = model.predict(X_test).tolist()
    gold = (pd.read_csv('gold_test.csv').ratings - 1).tolist()
    print(f"min_df=1, with stemming {accuracy_score(gold, test_pred)}, {f1_score(gold, test_pred, average='macro')}, {f1_score(gold, test_pred, average='weighted')}")


    return test_pred

if __name__ == '__main__':
    MODEL_PATH = "model.pt"
    if os.path.isfile(MODEL_PATH):
        print("Loading model from disk")
        vectorizer = pickle.load(open("vector.pickel", "rb"))
        model = NeuralNet(len(vectorizer.get_feature_names()))
        model.load_state_dict(torch.load(MODEL_PATH))
        X_test = pd.Series(vectorizer.transform(pd.Series(input("Enter review: "))).toarray().tolist())
        predictions = model.predict(X_test)
        print(predictions + 1)
    else:
        print("Model does not exist; exiting")

