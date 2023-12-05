# import gensim
# model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)


import transformers
import tensorflow as tf
import numpy as np


model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')


tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

test_tokenizer = tokenizer.tokenize("If your money is tight, don't cut corners.")

input_toks = tokenizer.encode("If your money is tight, don't cut corners")

result = tokenizer.convert_ids_to_tokens(input_toks)

input_mat = np.array(input_toks).reshape((1,-1))