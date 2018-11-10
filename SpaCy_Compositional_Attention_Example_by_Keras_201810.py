#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Natural language inference using spaCy and Keras

Introduction
This notebook details an implementation of the natural language inference model "Cecompositional Attention"
    presented in (Parikh et al, 2016).
The model is notable for the small number of paramaters and hyperparameters it specifices,
    while still yielding good performance.
"""


import spacy
import numpy as np
import ujson as json
from keras.utils import to_categorical
from keras import layers, Model, models
from keras import backend as K

from keras.callbacks import TensorBoard
from time import time


def read_snli(path):
    texts1 = []
    texts2 = []
    labels = []
    with open(path, 'r') as file_:
        for line in file_:
            eg = json.loads(line)
            label = eg['gold_label']
            if label == '-':  # per Parikh, ignore - SNLI entries
                continue
            texts1.append(eg['sentence1'])
            texts2.append(eg['sentence2'])
            labels.append(LABELS[label])
    return texts1, texts2, to_categorical(np.asarray(labels, dtype='int32'))


def create_dataset(nlp, texts, hypotheses, num_oov, max_length, norm_vectors = True):
    """
        norm_vectors: specifies whether normalize the token vector representations
        max_length: max length of words to take in the doc (or a text or a hypotheses)
        num_oov: OOV terms (tokens for which no semantic vector is available) are
                assigned to one of a set of randomly-generated OOV vectors
                nlp.vocab.vectors_length is 300, length of vector representation
    We use spaCy to tokenize the sentences and return, when available, a semantic vector for each token.
    Note that we will clip sentences to 50 words maximum when max_length=50.
    """
    # put texts and hypothteses together
    sents = texts + hypotheses
    # the extra +1 is for a zero vector represting NULL for padding
    num_vectors = max(lex.rank for lex in nlp.vocab) + 2  # num_vectors is 1070972
    # create random vectors for OOV tokens
        # OOV terms (tokens for which no semantic vector is available) are
        # assigned to one of a set of randomly-generated OOV vectors
        # nlp.vocab.vectors_length is 300, the max dimension
        # for exmaple, if num_oov=100, the oov is of size (100, 300)
    oov = np.random.normal(size=(num_oov, nlp.vocab.vectors_length))
    oov = oov / oov.sum(axis=1, keepdims=True)
    vectors = np.zeros((num_vectors + num_oov, nlp.vocab.vectors_length), dtype='float32')
    vectors[num_vectors:, ] = oov
    # Token.has_vector (lex.has_vector): boolean value indicating whether
        # a word vector is associated with the token.
        # Check out the spacy token api (https://spacy.io/api/token)
    # Token.vector_norm (lex.vector_norm): The L2 norm of the vector representation.
    # Token.vector (lex.vector): (300,) np.array. A 1D numpy array representing the token's semantics.
    # Token.rank (lex.rank): Sequential ID of the token's lexical type, used to index into tables, e.g. for word vectors.
    for lex in nlp.vocab:
        if lex.has_vector and lex.vector_norm > 0:
            vectors[lex.rank + 1] = lex.vector / lex.vector_norm if norm_vectors == True else lex.vector
    # Go through all texts (and hypotheses)
    sents_as_ids = []
    for sent in sents:
        doc = nlp(sent) # tokenize the doc
        word_ids = []
        for i, token in enumerate(doc):  # go through each word (token)
            # skip odd spaces from tokenizer
            if token.has_vector and token.vector_norm == 0: continue
            if i > max_length: break  # take only max_length of words in the doc
            if token.has_vector:
                word_ids.append(token.rank + 1)  # take the word id (sequential id of token's lexical type) for that word using it's token.rank
            else:
                # if we don't have a vector, pick an OOV entry
                word_ids.append(token.rank % num_oov + num_vectors)
        # there must be a simpler way of generating padded arrays from lists...
            # padding word_ids to to max_length, where ending is 0. if there are
            # less than max_length words in the text
        word_id_vec = np.zeros((max_length), dtype='int')
        clipped_len = min(max_length, len(word_ids))
        word_id_vec[:clipped_len] = word_ids[:clipped_len]
        sents_as_ids.append(word_id_vec)
    # sents_as_ids[:len(texts)], np.arrays of the texts
    # sents_as_ids[len(texts):], np.arrays of the hypotheses
    return vectors, np.array(sents_as_ids[:len(texts)]), np.array(sents_as_ids[len(texts):])


########  Building the model by Keras  ########
# The embedding layer copies the 300-dimensional GloVe vectors into GPU memory.
# Per (Parikh et al, 2016), the vectors, which are not adapted during training,
#    are projected down to lower-dimensional vectors using a trained projection matrix.
def create_embedding(vectors, max_length, projected_dim):
    return models.Sequential([
        layers.Embedding(
            vectors.shape[0],
            vectors.shape[1],
            input_length=max_length,
            weights=[vectors],
            trainable=False),
        layers.TimeDistributed(
            layers.Dense(projected_dim,
                         activation=None,
                         use_bias=False))
    ])

# The Parikh model makes use of three feedforward blocks that
    # construct nonlinear combinations of their input.
    # Each block contains two ReLU layers and two dropout layers.
def create_feedforward(num_units=200, activation='relu', dropout_rate=0.2):
    return models.Sequential([
        layers.Dense(num_units, activation=activation),
        layers.Dropout(dropout_rate),
        layers.Dense(num_units, activation=activation),
        layers.Dropout(dropout_rate)
    ])

# We need a couple of little functions for Lambda layers to normalize and aggregate weights:
def normalizer(axis):
    def _normalize(att_weights):
        exp_weights = K.exp(att_weights)
        sum_weights = K.sum(exp_weights, axis=axis, keepdims=True)
        return exp_weights/sum_weights
    return _normalize
def sum_word(x):
    return K.sum(x, axis=1)  # sum of columns for each row


"""
The basic idea of the (Parikh et al, 2016) model is to:

Align: Construct an alignment of subphrases in the text and hypothesis using an
    attention-like mechanism, called "decompositional" because the layer is
    applied to each of the two sentences individually rather than to their product.
    The dot product of the nonlinear transformations of the inputs is then
    normalized vertically and horizontally to yield a pair of "soft" alignment
    structures, from text->hypothesis and hypothesis->text. Concretely,
    for each word in one sentence, a multinomial distribution is computed over
    the words of the other sentence, by learning a multinomial logistic with softmax target.
Compare: Each word is now compared to its aligned phrase using a function modeled as
    a two-layer feedforward ReLU network. The output is a high-dimensional
    representation of the strength of association between word and aligned phrase.
Aggregate: The comparison vectors are summed, separately, for the text and
    the hypothesis. The result is two vectors: one that describes the degree
    of association of the text to the hypothesis, and the second,
    of the hypothesis to the text.
Finally, these two vectors are processed by a dense layer followed by a softmax
    classifier, as usual.
Note that because in entailment the truth conditions of the consequent must
    be a subset of those of the antecedent, it is not obvious that we need
    both vectors in step (3). Entailment is not symmetric. It may be enough
    to just use the hypothesis->text vector. We will explore this possibility later.
"""

def build_model(vectors, max_length, num_hidden, num_classes, projected_dim, entail_dir='both'):
    """
    sem_vectors: the semantic vectors created from function create_dataset, the first element in its return
    num_hidden: num of hidden neuros
    num_classes: 3 classes in this case, {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    projected_dim: the dimension where the embedding projected input dim (max_length=50, the
        max length of words to take in each doc). Here, we use 200 in the model
    max_length, num_hidden, num_classes, projected_dim, entail_dir = 50, 200, 3, 200, 'both'
    """
    # two independent inputs, texts, and hypotheses
    input1 = layers.Input(shape=(max_length,), dtype='int32', name='words1')
    input2 = layers.Input(shape=(max_length,), dtype='int32', name='words2')
    # embeddings (projected)
    embed = create_embedding(vectors, max_length, projected_dim)
    a = embed(input1)  # a.shape: TensorShape([Dimension(None), Dimension(50), Dimension(200)])
    b = embed(input2)
    # step 1: attend
        # F(a).shape or F(b).shape: TensorShape([Dimension(None), Dimension(50), Dimension(200)])
            # Dimension(None) is place holder of whatever # of rows to put in
            # Dimension(50), Dimension(50) are place holders of the embedding dimensions for a,b
                # i.e., for input1, input2
    F = create_feedforward(num_hidden)
    G = create_feedforward(num_hidden)
    # att_weights.shape: TensorShape([Dimension(None), Dimension(50), Dimension(50)])
    att_weights = layers.dot([F(a), F(b)], axes=-1)
    if entail_dir == 'both':
        # normalize for a or input1, norm_weights_a has same dimension as att_weights
        norm_weights_a = layers.Lambda(normalizer(1))(att_weights)
        norm_weights_b = layers.Lambda(normalizer(2))(att_weights)
        # alpha.shape, beta.shape: TensorShape([Dimension(None), Dimension(50), Dimension(200)])
            # alpha and beta are the soft alightment of sentence and hypotheses
        alpha = layers.dot([norm_weights_a, a], axes=1)
        beta  = layers.dot([norm_weights_b, b], axes=1)
        # step 2: compare
            # comp1.shape: TensorShape([Dimension(None), Dimension(50), Dimension(400)])
            # layers.concatenate([a, beta]) each word from input1 is now compared to its aligned phrase
            #    from input2
        comp1 = layers.concatenate([a, beta])
        comp2 = layers.concatenate([b, alpha])
        # layers.TimeDistributed:
            # This wrapper applies a layer to every temporal slice of an input.
            # for example: Consider a batch of 32 samples, where each sample is
                # a sequence of 10 vectors of 16 dimensions. The batch input
                # shape of the layer is then (32, 10, 16), and the input_shape,
                # not including the samples dimension, is (10, 16).
            # $ model = Sequential()
            # $ model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
            # output has dimension (32, 10, 8)
        # v1.shape: TensorShape([Dimension(None), Dimension(50), Dimension(200)])
            # where, Dimension(200) comes from the num_hidden of G (above)
        v1 = layers.TimeDistributed(G)(comp1)
        v2 = layers.TimeDistributed(G)(comp2)
        # step 3: aggregate
            # v1_sum: <tf.Tensor 'lambda_7/Sum:0' shape=(?, 200) dtype=float32>
            #     sum up all 50 (max_length) words' weights, hence
            #     v1 dim is changed from (?, 50, 200) to v1_sum dim (?, 200)
        v1_sum = layers.Lambda(sum_word)(v1)
        v2_sum = layers.Lambda(sum_word)(v2)
        concat = layers.concatenate([v1_sum, v2_sum])  # concat.shape: (?, 400)
    elif entail_dir == 'left':
        norm_weights_a = layers.Lambda(normalizer(1))(att_weights)
        alpha = layers.dot([norm_weights_a, a], axes=1)
        comp2 = layers.concatenate([b, alpha])
        v2 = layers.TimeDistributed(G)(comp2)
        v2_sum = layers.Lambda(sum_word)(v2)
        concat = v2_sum
    else:
        norm_weights_b = layers.Lambda(normalizer(2))(att_weights)
        beta  = layers.dot([norm_weights_b, b], axes=1)
        comp1 = layers.concatenate([a, beta])
        v1 = layers.TimeDistributed(G)(comp1)
        v1_sum = layers.Lambda(sum_word)(v1)
        concat = v1_sum
    H = create_feedforward(num_hidden)
    out = H(concat)  # feed-forward the concatenation 'concat'
    out = layers.Dense(num_classes, activation='softmax')(out)
    model = Model([input1, input2], out)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model




#####################  MAIN PROCEDURE  #####################

nlp = spacy.load('en_vectors_web_lg')  # specify which NLP model to load
LABELS = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
texts,hypotheses,labels = read_snli('./snli_1.0_train.jsonl')
texts_test,hypotheses_test,labels_test = read_snli('./snli_1.0_test.jsonl')

"""
$ texts[0], hypotheses[0], labels[0]
    Here, Labels are {'entailment': 0, 'contradiction': 1, 'neutral': 2}
[1] ('A person on a horse jumps over a broken down airplane.',
    'A person is training his horse for a competition.',
    array([0., 0., 1.], dtype=float32))
"""


sem_vectors, text_vectors, hypothesis_vectors = create_dataset(nlp, texts, hypotheses, 100, 50, True)
_, text_vectors_test, hypothesis_vectors_test = create_dataset(nlp, texts_test, hypotheses_test, 100, 50, True)

# Train Model
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# check Tensorboard by typing in command line the following
# > tensorboard --logdir=logs/
# then check browser http://localhost:6006

K.clear_session()
m = build_model(sem_vectors, 50, 200, 3, 200)
m.summary()
m.fit([text_vectors, hypothesis_vectors], labels, batch_size=1024, epochs=50,
    validation_data=([text_vectors_test, hypothesis_vectors_test], labels_test),
    callbacks=[tensorboard])

