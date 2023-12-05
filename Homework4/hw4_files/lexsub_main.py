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
import string # for tokenizer


def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """

    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    # Part 1
    candidates = set()
    for l in wn.lemmas(lemma, pos):
        s = l.synset()
        lexemes_in_synset = s.lemmas()
        for lexeme in lexemes_in_synset:
            candidates.add(lexeme.name().replace("_", " "))

    candidates.remove(lemma)

    return candidates

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
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
    return most_frequent_lemma # replace for part 2

def wn_simple_lesk_predictor(context : Context) -> str:

    stop_words = stopwords.words('english')

    stop_words = set(stop_words)

    target_context = context.left_context + context.right_context # left and right context
    target_context = set(target_context)

    # remove stop words in target context
    # target_context = [w for w in target_context if not w in stop_words]

    target_context = target_context - stop_words # use set is faster

    target_synsets = wn.synsets(context.lemma, context.pos)

    table = {}
    for synset in target_synsets:
        word_set = set()
        # according to professor's instruction: "I recommend just using a set. The overlap score would then be 1."
        definitions = set(tokenize(synset.definition())) - stop_words # remove stop words in definition
        word_set |= definitions
        for example in synset.examples():
            word_set |= set(tokenize(example)) - stop_words

        for hypernym in synset.hypernyms():
            word_set |= set(tokenize(hypernym.definition())) - stop_words # hypernyms definition
            for example in hypernym.examples():
                word_set |= set(tokenize(example)) - stop_words # hypernyms examples

        overlap = len(word_set & target_context) # intersection, overlap score

        count = [word.count() for word in synset.lemmas() if word.name().replace('_', ' ') == context.lemma] # word.count() is the frequency of the word in the corpus

        # My friend helped me with this part
        for word in synset.lemmas():
            if word.name().replace('_',' ') != context.lemma:
                # overlap, count, word count
                normalized_frequency = 1000*overlap + 100*(count[0] if count else 0) + word.count()

                table[(synset, word)] = normalized_frequency # this synset and this word
    similar_pair = max(table, key=table.get) # get the pair with the highest score
    return similar_pair[1].name().replace('_', ' ') # replace for part 3



class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self,context : Context) -> str:
        # Part 4
        # use part 1 to get candidates
        candidates = get_candidates(context.lemma, context.pos)
        # remove all candidates that are not in the model's vocabulary, key_to_index
        candidates = list(candidates)
        candidates = [candidate for candidate in candidates if candidate in self.model.key_to_index]
        # remove
        table = {}
        if context.lemma in self.model.key_to_index:
            for candidate in candidates:
                table[candidate] = self.model.similarity(context.lemma, candidate)
        else:
            return None # if the lemma is not in the model's vocabulary, return None, not sure if this will happen

        return max(table, key=table.get) # replace for part 4


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # Part 5

        # Obtain a set of candidate synonyms (for example, by calling get_candidates).
        # Convert the information in context into a suitable masked input representation for the DistilBERT model (see example above). Make sure you store the index of the masked target word (the position of the [MASK] token).
        # Run the DistilBERT model on the input representation.
        # Select, from the set of wordnet derived candidate synonyms, the highest-scoring word in the target position (i.e. the position of the masked word). Return this word.

        wordnet_candidates = get_candidates(context.lemma, context.pos) # convert to list
        wordnet_candidates = list(wordnet_candidates)

        context_string = context.left_context + ['[MASK]'] + context.right_context # left and right context with [MASK]
        input_toks = self.tokenizer.encode(' '.join(context_string)) # encode

        mask_index = input_toks.index(self.tokenizer.mask_token_id) # get the index of [MASK]

        # input mat
        input_mat = np.array(input_toks).reshape((1,-1))

        # predict
        outputs = self.model.predict(input_mat, verbose=0) # don't need the verbose

        # get the predictions
        predictions = outputs[0][0]
        target_predictions = predictions[mask_index] # get the predictions of the target word

        wordnet_candidates_id = [self.tokenizer.convert_tokens_to_ids(word) for word in wordnet_candidates]

        # get the candidates' scores
        wordnet_candidates_score = [target_predictions[candidate_id] for candidate_id in wordnet_candidates_id]
        max_score_id = np.argmax(wordnet_candidates_score) # get the index of the highest score
        return wordnet_candidates[max_score_id] # replace for part 5


class CustomPredictor:
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self,context : Context) -> str:

        # here is the approch, we first get the candidates,
        synset_vector_dict = {}


        for synset in wn.synsets(context.lemma, context.pos):
            vector_sum = np.zeros(300)
            synset_vector_dict[synset] = vector_sum
            for lemmas in synset.lemmas():
                if lemmas.name() != context.lemma:
                    if lemmas.name() in self.model.key_to_index:
                        synset_vector_dict[synset] += self.model.get_vector(lemmas.name())

        # get the lemma's vector
        target_word_vector = self.model.get_vector(context.lemma)

        synset_score_dict = {}
        for synset in synset_vector_dict:
            synset_score_dict[synset] = np.dot(target_word_vector, synset_vector_dict[synset])

        # get the highest score synset
        highest_score_synset = max(synset_score_dict, key=synset_score_dict.get)

        # lemma height score
        lemma_score_dict = {}
        for lemma in highest_score_synset.lemmas():
            if lemma.name() != context.lemma:
                if lemma.name() in self.model.key_to_index:
                    lemma_score_dict[lemma] = self.model.similarity(context.lemma, lemma.name())

        # get the highest score lemma
        highest_score_lemma = max(lemma_score_dict, key=lemma_score_dict.get)
        return highest_score_lemma.name()








            # for lemma in synset.lemmas():
            #     if lemma.name() == context.lemma:
            #         return lemma.name()





# if __name__=="__main__":
#
#     # At submission time, this program should run your best predictor (part 6).
#
#     W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin'
#     predictor = Word2VecSubst(W2VMODEL_FILENAME)
#
#     for context in read_lexsub_xml(sys.argv[1]):
#         #print(context)  # useful for debugging
#         prediction = smurf_predictor(context)
#         print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))



# Part 1

# if __name__=="__main__":
#     import io
#
#     f = io.open('smurf.predict', 'w', newline='\n')
#     for context in read_lexsub_xml("lexsub_trial.xml"):
#         #print(context)  # useful for debugging
#         prediction = smurf_predictor(context)
#         a = "{}.{} {} :: {}\n".format(context.lemma, context.pos, context.cid, prediction)
#         f.write(a)
#
#         print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))






# # Part 2
#
# if __name__=="__main__":
#     import io
#
#     f = io.open('part2.predict', 'w', newline='\n')
#     for context in read_lexsub_xml("lexsub_trial.xml"):
#         #print(context)  # useful for debugging
#         prediction = wn_frequency_predictor(context)
#         a = "{}.{} {} :: {}\n".format(context.lemma, context.pos, context.cid, prediction)
#         f.write(a)
#
#         print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
#
#     # Total = 298, attempted = 298
#     # precision = 0.074, recall = 0.074
#     # Total with mode 206 attempted 206
#     # precision = 0.107, recall = 0.107





# # Part 3
#
#
# if __name__=="__main__":
#     import io
#
#     f = io.open('part3.predict', 'w', newline='\n')
#     for context in read_lexsub_xml("lexsub_trial.xml"):
#         #print(context)  # useful for debugging
#         prediction = wn_simple_lesk_predictor(context)
#         a = "{}.{} {} :: {}\n".format(context.lemma, context.pos, context.cid, prediction)
#         f.write(a)
#
#         print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
#
# # Total = 298, attempted = 298
# # precision = 0.112, recall = 0.112
# # Total with mode 206 attempted 206
# # precision = 0.155, recall = 0.155




# # # Part 4
#
#
# if __name__=="__main__":
#     import io
#
#     f = io.open('part4.predict', 'w', newline='\n')
#     word2vec = Word2VecSubst('GoogleNews-vectors-negative300.bin')
#     for context in read_lexsub_xml("lexsub_trial.xml"):
#         #print(context)  # useful for debugging
#         prediction = word2vec.predict_nearest(context)
#         a = "{}.{} {} :: {}\n".format(context.lemma, context.pos, context.cid, prediction)
#         f.write(a)
#
#         print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
#
# # Total = 298, attempted = 298
# # precision = 0.115, recall = 0.115
# # Total with mode 206 attempted 206
# # precision = 0.170, recall = 0.170



# #
# # # Part 5
# #
# if __name__=="__main__":
#     import io
#
#     f = io.open('part5.predict', 'w', newline='\n')
#     bert_predictor = BertPredictor()
#     for context in read_lexsub_xml("lexsub_trial.xml"):
#         #print(context)  # useful for debugging
#         prediction = bert_predictor.predict(context)
#         a = "{}.{} {} :: {}\n".format(context.lemma, context.pos, context.cid, prediction)
#         f.write(a)
#
#         print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
#
#
# # Total = 298, attempted = 298
# # precision = 0.115, recall = 0.115
# # Total with mode 206 attempted 206
# # precision = 0.170, recall = 0.170
#








# # Part 6

if __name__=="__main__":
    import io

    f = io.open('part6.predict', 'w', newline='\n')
    custom_predictor = CustomPredictor('GoogleNews-vectors-negative300.bin')
    for context in read_lexsub_xml("lexsub_trial.xml"):
        #print(context)  # useful for debugging
        prediction = custom_predictor.predict_nearest(context)
        a = "{}.{} {} :: {}\n".format(context.lemma, context.pos, context.cid, prediction)
        f.write(a)

        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))



