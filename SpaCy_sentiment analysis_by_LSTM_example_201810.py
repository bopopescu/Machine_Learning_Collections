#!/usr/bin/env python
# coding: utf-8

"""
Using LSTM Deep Learning for Sentiment Analysis for Movie Reviews
using SpaCy and Keras
"""

import plac
import random
import pathlib
import cytoolz
import numpy
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import thinc.extra.datasets
from spacy.compat import pickle
import spacy




class SentimentAnalyser(object):
    """
    This class is for loading the NLP model that
    is saved in the model directory
    """
    def __init__(self, model, max_length=100):
        self._model = model
        self.max_length = max_length
    def __call__(self, doc):
        X = get_features([doc], self.max_length)
        y = self._model.predict(X)
        self.set_sentiment(doc, y)
    @classmethod
    def load(cls, path, nlp, max_length=100):
        with (path / 'config.json').open() as file_:
            model = model_from_json(file_.read())
        with (path / 'model').open('rb') as file_:
            lstm_weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, max_length=max_length)
    def set_sentiment(self, doc, y):
        doc.sentiment = float(y[0])
        # Sentiment has a native slot for a single float.
        # For arbitrary data storage, there's:
        # doc.user_data['my_data'] = y
    def pipe(self, docs, batch_size=1000, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            sentences = []
            for doc in minibatch:
                sentences.extend(doc.sents)
            Xs = get_features(sentences, self.max_length)
            ys = self._model.predict(Xs)
            for sent, label in zip(sentences, ys):
                sent.doc.sentiment += label - 0.5
            for doc in minibatch:
                yield doc


def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, numpy.asarray(labels, dtype='int32')


def get_features(docs, max_length):

    """Get the matrix of doc into the feature matrix of length
    """
    docs = list(docs)
    Xs = numpy.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                Xs[i, j] = vector_id
            else:
                Xs[i, j] = 0
            j += 1
            if j >= max_length:
                break
    return Xs


def get_embeddings(vocab):
    return vocab.vectors.data


def compile_lstm(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    model.add(TimeDistributed(Dense(shape['nr_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['nr_hidden'],
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(Dense(shape['nr_class'], activation='sigmoid'))
    model.compile(optimizer=Adam(lr=settings['lr']), loss='binary_crossentropy',
  metrics=['accuracy'])
    return model


def train(train_texts, train_labels, dev_texts, dev_labels,
           lstm_shape, lstm_settings, lstm_optimizer, batch_size=100,
           nb_epoch=5, by_sentence=True):
    print("Loading spaCy")
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    embeddings = get_embeddings(nlp.vocab)
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)
    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts))  # convert list of texts to spacy.tokens.doc.Doc class
    dev_docs = list(nlp.pipe(dev_texts))
    if by_sentence:
        train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
                    ## split each doc into sentences and labels for each sentence
        dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)
    train_X = get_features(train_docs, lstm_shape['max_length'])
    dev_X = get_features(dev_docs, lstm_shape['max_length'])
    model.fit(train_X, train_labels, validation_data=(dev_X, dev_labels),
              epochs=nb_epoch, batch_size=batch_size)
    return model


def evaluate(model_dir, texts, labels, max_length=100):
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.add_pipe(SentimentAnalyser.load(model_dir, nlp, max_length=max_length))

    correct = 0
    i = 0
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):
        correct += bool(doc.sentiment >= 0.5) == bool(labels[i])
        i += 1
    return float(correct) / i


def read_data(data_dir, limit=0):
    # iterate through the folder
    examples = []
    for subdir, label in (('pos', 1), ('neg', 0)):
        for filename in (data_dir / subdir).iterdir(): # this is based on pathlib
            with filename.open() as file_:
                text = file_.read()
            examples.append((text, label))
    random.shuffle(examples)
    if limit >= 1: examples = examples[:limit] # set limit of texts
    return zip(*examples) # Unzips into two lists




##############################   MAIN PROCEDURE FUNCTION   ##############################
# if __name__ == "__main__":

# Example is from https://github.com/explosion/spaCy/tree/master/examples
# Command line to install spacy models
# python -m spacy download en
# python -m spacy download en_vectors_web_lg


model_dir = "/Users/mpeng/Desktop/spacymodels"
train_dir = "/Users/mpeng/Desktop/spacymodels"
dev_dir = "/Users/mpeng/Desktop/spacymodels"
is_runtime = True
# Neural network parameters
nr_hidden=64; max_length=100 # Shape
dropout=0.5; learn_rate=0.001 # General NN config
nb_epoch=5; batch_size=256; nr_examples=-1

if model_dir is not None:
    model_dir = pathlib.Path(model_dir)
#if train_dir is None or dev_dir is None:
imdb_data = thinc.extra.datasets.imdb()  #load in the imdb movie database
# IMDB data is movie user revies. It's tuple of two tuples
    # first tuple is the list of tuples of (review, 1/0 label), used for training
    # the second tuple is the list of tuples of (review, 1/0 label) for validation

# Validation data dev_texts and labels
dev_texts, dev_labels = zip(*imdb_data[1])
# Train texts and train labels
train_texts, train_labels = zip(*imdb_data[0])
# train_texts, train_labels = read_data(train_dir, limit=nr_examples) for loading the data from folder, not online
dev_texts, dev_labels = zip(*imdb_data[1])
train_labels = numpy.asarray(train_labels, dtype='int32')
dev_labels = numpy.asarray(dev_labels, dtype='int32')
lstm = train(train_texts, train_labels, dev_texts, dev_labels,
                lstm_shape={'nr_hidden': nr_hidden, 'max_length': max_length, 'nr_class': 1},
                lstm_settings={'dropout': dropout, 'lr': learn_rate},
                lstm_optimizer={},
                nb_epoch=nb_epoch, batch_size=batch_size)
weights = lstm.get_weights()
if model_dir is not None:
    with (model_dir / 'model').open('wb') as file_:
        pickle.dump(weights[1:], file_)
    with (model_dir / 'config.json').open('w') as file_:
        file_.write(lstm.to_json())

# This is to load in NLP model and evaluate the model performance on the validation
# texts dev_texts
acc = evaluate(model_dir, dev_texts, dev_labels, max_length=max_length)
print(acc)

