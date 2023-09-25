import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2023 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """

    sequence_pad = ["START"] + sequence + ["STOP"]
    if n > 2:
        sequence_pad = ["START"]*(n-2) + sequence_pad

    ngrams = []
    for i in range(len(sequence_pad)-n+1):
        ngrams.append(tuple(sequence_pad[i:i+n]))



    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """

        self.unigramcounts = {}  # might want to use defaultdict or Counter instead
        self.bigramcounts = {}
        self.trigramcounts = {}

        ##Your code here
        self.total_number_of_sentences = 0
        for sentence in corpus:
            self.total_number_of_sentences += 1
            for ngram in get_ngrams(sentence, 1):
                if ngram in self.unigramcounts:
                    self.unigramcounts[ngram] += 1
                else:
                    self.unigramcounts[ngram] = 1

            for ngram in get_ngrams(sentence, 2):
                if ngram in self.bigramcounts:
                    self.bigramcounts[ngram] += 1
                else:
                    self.bigramcounts[ngram] = 1

            for ngram in get_ngrams(sentence, 3):
                if ngram in self.trigramcounts:
                    self.trigramcounts[ngram] += 1
                else:
                    self.trigramcounts[ngram] = 1

        # remove "START" from unigramcounts
        if ("START",) in self.unigramcounts:
            del self.unigramcounts[("START",)]

        self.total_number_of_tokens = sum(self.unigramcounts.values())

        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        c_uvw = 0
        c_uv = 0
        if trigram in self.trigramcounts:
            c_uvw = self.trigramcounts[trigram]

        # checkif the trigram[0:2] is ('START', 'START')
        if trigram[0:2] == ('START', 'START'):
            c_uv = self.total_number_of_sentences

        if trigram[0:2] in self.bigramcounts:
            c_uv = self.bigramcounts[trigram[0:2]]

        if c_uv == 0:
            return self.raw_unigram_probability((trigram[-1],)) # return unigram probability of the last word in trigram if trigram not in trigramcounts
        else:
            return c_uvw/c_uv




    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """

        c_uw = 0
        c_u = 0

        if bigram in self.bigramcounts:
            c_uw = self.bigramcounts[bigram]
        if bigram[0] in self.unigramcounts:
            c_u = self.unigramcounts[bigram[0]]

        if c_u == 0:
            return self.raw_unigram_probability((bigram[-1],))
        else:
            return c_uw/c_u





    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.

        if unigram in self.unigramcounts:
            return self.unigramcounts[unigram]/self.total_number_of_tokens
        else:
            return 0.0

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return 0.0
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        log_prob = 0.0

        # sentence is a list of tokens
        for i in range(2, len(sentence) - 1):

            trigram = (sentence[i - 2], sentence[i - 1], sentence[i])
            trigram_prob = self.smoothed_trigram_probability(trigram)
            log_prob += math.log2(trigram_prob)

        return log_prob # return log probability of sentence


    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        return float("inf") 


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # .. 
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # .. 
        
        return 0.0

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)

