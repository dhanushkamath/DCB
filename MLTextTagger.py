import SimpleTextTagger
import random
from xgboost import XGBClassifier
from collections import Counter
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pandas as pd
import re as regex
import numpy as np
stemmer = LancasterStemmer()
import gensim

# DOWNLOAD AND STORE THE word2vec MODEL TRAINED BY GOOGLE in the directory 'data/' from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True,
                                                        limit=30000)

from glob import glob
import pickle
model_path = 'data/XGBOOST_trained1.pickle.dat'


def get_w2v(word):
    return model.word_vec(str(word))

def get_w2v_sent(sentence,show=False):
    sentence=sentence.lower()
    words = sentence.split()
    w2v_sent_list=[]
    for word in nltk.word_tokenize(sentence):
        if word in model.vocab:
            if show:
                print(word)
            w2v_sent_list.append(get_w2v(word.lower()))
    #print(np.array(w2v_sent_list).shape)
    return list(np.array(w2v_sent_list).mean(axis=0))

def generate_train_data(path):
    train_data=[]
    cor_words, clas_words = SimpleTextTagger.pre_process_text(path)
    vocabulary = sorted(cor_words.keys())
    target_i = sorted(clas_words.keys())
    #print(target_i)
    train_data_row = np.zeros((len(vocabulary)))
    targets = []
    tags, sents = SimpleTextTagger.open_chats(path)
    for i, sen in enumerate(sents):
        train_data_row = np.zeros((len(vocabulary)))
        for word in nltk.word_tokenize(sen.lower()):
            if word not in ["?", "'s"]:
                # stem and lowercase each word
                stemmed_word = stemmer.stem(word.lower())
                # have we not seen this word already?
                if stemmed_word in vocabulary:
                    # increase the count in train_data_row
                    train_data_row[vocabulary.index(str(stemmed_word))]+=1
        w2v = get_w2v_sent(sen)
        train_data_row = list(train_data_row) + list(w2v)
        train_data.append(train_data_row)
        targets.append(target_i.index(tags[i]))
    return train_data,targets



#
# train_data, targets = generate_train_data(path = SimpleTextTagger.path)
# train_data=np.array(train_data)

# print(len(targets))
# print(train_data)
# w2v = get_w2v_sent("Hi my name is Dhanush")
# print(np.array(get_w2v_sent("Hi my name is dhanush")))

def preprocess_sentence_test(sentence):
    cor_words, clas_words = SimpleTextTagger.pre_process_text(path=SimpleTextTagger.path)
    target_i = sorted(clas_words.keys())
    vocabulary = sorted(cor_words.keys())
    train_data_row = np.zeros((len(vocabulary)))
    for word in nltk.word_tokenize(sentence.lower()):
        if word not in ["?", "'s"]:
            # stem and lowercase each word
            stemmed_word = stemmer.stem(word.lower())
            # have we not seen this word already?
            if stemmed_word in vocabulary:
                # increase the count in train_data_row
                train_data_row[vocabulary.index(str(stemmed_word))] += 1
        w2v = get_w2v_sent(sentence)
    train_data_row = list(train_data_row) + list(w2v)
    return train_data_row,target_i






def train_model():
    train_data, targets = generate_train_data(path=SimpleTextTagger.path)
    seed =1234
    # Shuffling the train_data and targets together
    combined = list(zip(train_data, targets))
    random.shuffle(combined)
    train_data[:], targets[:] = zip(*combined)

    # fit model on training data
    model = XGBClassifier(seed=1234)
    model.fit(np.array(train_data), np.array(targets))
    pickle.dump(model,open(str(model_path), "wb"))
    return model

def test_model(sentence):
    if glob(model_path):
        model = pickle.load(open(str(model_path),"rb"))
    else:
        model = train_model()
    test_data,target_i = preprocess_sentence_test(sentence)
    prediction = model.predict(test_data)
    return target_i[prediction[0]]

print(test_model("send some money to my mother in australia"))
print(test_model("Hi how was your day?"))
print(test_model("pay my friend 500rs"))
print(test_model("how much money do i have in my account?"))
print(test_model("Have a wonderful day")) # Misclassified
