import gensim
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
v1 = model.get_vector('computer')
def cos(v1,v2):
    return np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))


cos_sim = cos(model.get_vector('computer'),model.get_vector('calculator'))