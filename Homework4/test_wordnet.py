from nltk.corpus import wordnet as wn

wn.lemmas('break', pos='n') # Retrieve all lexemes for the noun 'break'


l1 = wn.lemmas('break', pos='n')[0]

s1 = l1.synset() # get the synset for the first lexeme