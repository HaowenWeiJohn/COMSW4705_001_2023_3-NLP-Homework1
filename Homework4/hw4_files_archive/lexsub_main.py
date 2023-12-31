#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers

from typing import List


def tokenize(s):
    """
    a naive tokenizer that splits on punctuation and whitespaces.
    """
    string = s.replace("-", " ")
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()


def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    candidates = set()
    for l in wn.lemmas(lemma, pos):
        s = l.synset()
        lexemes_in_synset = s.lemmas()
        for lexeme in lexemes_in_synset:
            candidates.add(lexeme.name().replace("_", " "))

    candidates.remove(lemma)

    return candidates


def smurf_predictor(context: Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'


def wn_frequency_predictor(context: Context) -> str:
    target_synsets = wn.synsets(context.lemma, context.pos)


    frequency_counter = {}
    # get the frequency counter for each synset
    for synset in target_synsets:
        for lemma in synset.lemmas():
            lemma = lemma.name().replace("_", " " ) # replace _ with space
            if lemma != context.lemma:
                if lemma not in frequency_counter:
                    frequency_counter[lemma] = 1
                else:
                    frequency_counter[lemma] += 1

    # find the most frequent lemma
    most_frequent_lemma = max(frequency_counter, key=frequency_counter.get)
    return most_frequent_lemma




    # Part 2



    return None  # replace for part 2


def wn_simple_lesk_predictor(context: Context) -> str:
    return None  # replace for part 3


class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self, context: Context) -> str:
        return None  # replace for part 4


class BertPredictor(object):

    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context: Context) -> str:
        return None  # replace for part 5


if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    for context in read_lexsub_xml('lexsub_trial.xml'):
    # for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = wn_frequency_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

# if __name__ == '__main__':
#     # Part 1 - get_candidates
#     test_candidates = get_candidates('slow', 'a')
#     print(test_candidates)
#
#     # Part 2 - wn_frequency_predictor
