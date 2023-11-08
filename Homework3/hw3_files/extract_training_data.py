#
# Part 2 - Extracting Input/Output matrices for training (35 pts)
# To train the neural network we first need to obtain a set of input/output training pairs. More specifically, each training example should be a pair (x,y), where x is a parser state and y is the transition the parser should make in that state.
#
# Take a look at the file extract_training_data.py
# States: The input will be an instance of the class State, which represents a parser state. The attributes of this class consist of a stack, buffer, and partially built dependency structure deps. stack and buffer are lists of word ids (integers).
# The top of the stack is the last word in the list stack[-1]. The next word on the buffer is also the last word in the list, buffer[-1].
# Deps is a list of (parent, child, relation) triples, where parent and child are integer ids and relation is a string (the dependency label).
#
# Transitions: The output is a pair (transition, label), where the transition can be one of "shift", "left_arc", or "right_arc" and the label is a dependency label. If the transition is "shift", the dependency label is None. Since there are 45 dependency relations (see list deps_relations), there are 45*2+1 possible outputs.
#
# Obtaining oracle transitions and a sequence of input/output examples.
# As discussed in class, we cannot observe the transitions directly from the treebank. We only see the resulting dependency structures. We therefore need to convert the trees into a sequence of (state, transition) pairs that we use for training. That part has already been implemented in the function get_training_instances(dep_structure). Given a DependencyStructure instance, this method returns a list of (State, Transition) pairs in the format described above.
#
# TODO: Extracting Input Representations
# Your task will be to convert the input/output pairs into a representation suitable for the neural network. You will complete the method get_input_representation(self, words, pos, state) in the class FeatureExtractor. The constructor of the class FeatureExtractor takes the two vocabulary files as inputs (file objects). It then stores a word-to-index dictionary in the attribute word_vocab and POS-to-index dictionary in the attribute pos_vocab.
#
# get_input_representation(self, words, pos, state) takes as parameters a list of words in the input sentence, a list of POS tags in the input sentence and an instance of class State. It should return an encoding of the input to the neural network, i.e. a single vector.
#
# To represent a state, we will use the top-three words on the buffer and the next-three word on the stack, i.e. stack[-1], stack[-2], stack[-3] and buffer[-1], buffer[-2], buffer[-3]. We could use embedded representations for each word, but we would like the network to learn these representations itself. Therefore, the neural network will contain an embedding layer and the words will be represented as a one-hot representation. The actual input will be the concatenation of the one-hot vectors for each word.
#
# This would typically require a 6x|V| vector, but fortunately the keras embedding layer will accept integer indices as input and internally convert them. We therefore just need to return a vector (a 1-dimensional numpy array) of length 6.
#
# So for example, if the next words on the buffer are "dog eats a" and "the" and <ROOT> are on the stack , the return value should be a numpy array numpy.array([4047, 3, 4, 8346, 8995, 14774]). Here 4 is the index for the <NULL> symbol, 3 is the index for the <ROOT> symbol, 4047 is the index for "the" and 8346, 8995, 14774 are the indices for "dog", "eats" and "a". (Your indices may differ from this example because the get_vocab script outputs a different index mapping each time it is run.)
#
# Note that you need to account for the special symbols (<CD>,<NNP>,<UNK>,<ROOT>,<NULL>) in creating the input representation. Make sure you take into account states in which there are less than 3 words on the stack or buffer.
#
# This representation is a subset of the features in the Chen & Manning (2014) paper. Feel free to experiment with the complete feature set once you got the basic version running.
#
# TODO: Generating Input and Output matrices
#
# Write the method get_output_representation(self, output_pair), which should take a (transition, label) pair as its parameter and return a one-hot representation of these actions. Because there are 45*2+1 = 91 possible outputs, the output should be represented as a one-hot vector of length 91.
#
# Saving training matrices
# The neural network will take two matrices as its input, a matrix of training data (in the basic case a N x 6 matrix, where N is the number of training instances) and an output matrix (an Nx91 matrix).
#
# The function get_training_matrices(extractor, in_file) will take a FeatureExtractor instance and a file object (a CoNLL formatted file) as its input. It will then extract state-transition sequences and call your input and output representation methods on each to obtain input and output vectors. Finally it will assemble the matrices and return them.
#
# The main program in extrac_training_data.py calls get_training_matrices to obtain the matrices and then writes them to two binary files (encoded in the numpy array binary format). You can call it like this:
#
# python extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy
#
# You can also obtain matrices for the development set, which is useful to tune network parameters.
#
# python extract_training_data.py data/dev.conll data/input_dev.npy data/target_dev.npy


from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
import keras
import numpy as np

class State(object):
    def __init__(self, sentence = []):
        self.stack = []
        self.buffer = []
        if sentence: 
            self.buffer = list(reversed(sentence))
        self.deps = set() 
    
    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        self.deps.add( (self.buffer[-1], self.stack.pop(),label) )

    def right_arc(self, label):
        parent = self.stack.pop()
        self.deps.add( (parent, self.buffer.pop(), label) )
        self.buffer.append(parent)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)

   

def apply_sequence(seq, sentence):
    state = State(sentence)
    for rel, label in seq:
        if rel == "shift":
            state.shift()
        elif rel == "left_arc":
            state.left_arc(label) 
        elif rel == "right_arc":
            state.right_arc(label) 
         
    return state.deps
   
class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None    
    def __repr__(self):
        return "<ROOT>"

     
def get_training_instances(dep_structure):

    deprels = dep_structure.deprels
    
    sorted_nodes = [k for k,v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident,node in deprels.items():
        childcount[node.head] += 1
 
    seq = []
    while state.buffer: 
        if not state.stack:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy() 
        else:
            stackword = deprels[state.stack[-1]]
        bufferword = deprels[state.buffer[-1]]
        if stackword.head == bufferword.id:
            childcount[bufferword.id]-=1
            seq.append((copy.deepcopy(state),("left_arc",stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id]-=1
            seq.append((copy.deepcopy(state),("right_arc",bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else: 
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
    return seq   


dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']


class FeatureExtractor(object):
       
    def __init__(self, word_vocab_file, pos_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)        
        self.pos_vocab = self.read_vocab(pos_vocab_file)        
        self.output_labels = self.make_output_labels()

    def make_output_labels(self):
        labels = []
        labels.append(('shift',None))
    
        for rel in dep_relations:
            labels.append(("left_arc",rel))
            labels.append(("right_arc",rel))
        return dict((label, index) for (index,label) in enumerate(labels))

    def read_vocab(self,vocab_file):
        vocab = {}
        for line in vocab_file: 
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab     

    def get_input_representation(self, words, pos, state):
        # TODO: Write this method for Part 2
        # print(words)
        # print(pos)
        # print(state)
        pos[0] = 'ROOT'
        # special symbols (<CD>,<NNP>,<UNK>,<ROOT>,<NULL>)
        special_pos = ['CD', 'NNP', 'UNK', 'ROOT', 'NULL']
        # convert all words to lower case
        for i in range(1, len(words)):
            words[i] = words[i].lower()

        # To represent a state, we will use the top-three words on the buffer and the next-three word on the stack, i.e. stack[-1], stack[-2], stack[-3] and buffer[-1], buffer[-2],
        # buffer[-3]. We could use embedded representations for each word, but we would like the network to learn these representations itself. Therefore, the neural network will
        # contain an embedding layer and the words will be represented as a one-hot representation. The actual input will be the concatenation of the one-hot vectors for each word.
        stack_encoding = np.zeros(3)
        # stack:
        if len(state.stack) >= 3:
            for i in range(0,3):
                index = -(i+1)
                word = words[state.stack[index]]
                tag = pos[state.stack[index]]
                if tag in special_pos:
                    stack_encoding[i] = self.word_vocab['<'+tag+'>']
                else:
                    if word not in self.word_vocab:
                        stack_encoding[i] = self.word_vocab['<UNK>']
                    else:
                        stack_encoding[i] = self.word_vocab[word]
        else:

            for i in range(0, len(state.stack)):
                index = -(i+1)
                word = words[state.stack[index]]
                tag = pos[state.stack[index]]
                if tag in special_pos:
                    stack_encoding[i] = self.word_vocab['<'+tag+'>']
                else:
                    if word not in self.word_vocab:
                        stack_encoding[i] = self.word_vocab['<UNK>']
                    else:
                        stack_encoding[i] = self.word_vocab[word]

            for i in range(len(state.stack), 3):
                stack_encoding[i] = self.word_vocab['<NULL>']

        buffer_encoding = np.zeros(3)
        # buffer:
        if len(state.buffer) >= 3:
            for i in range(0,3):
                index = -(i+1)
                word = words[state.buffer[index]]
                tag = pos[state.buffer[index]]
                if tag in special_pos:
                    buffer_encoding[i] = self.word_vocab['<'+tag+'>']
                else:
                    if word not in self.word_vocab:
                        buffer_encoding[i] = self.word_vocab['<UNK>']
                    else:
                        buffer_encoding[i] = self.word_vocab[word]
        else:
                for i in range(0, len(state.buffer)):
                    index = -(i+1)
                    word = words[state.buffer[index]]
                    tag = pos[state.buffer[index]]
                    if tag in special_pos:
                        buffer_encoding[i] = self.word_vocab['<'+tag+'>']
                    else:
                        if word not in self.word_vocab:
                            buffer_encoding[i] = self.word_vocab['<UNK>']
                        else:
                            buffer_encoding[i] = self.word_vocab[word]

                for i in range(len(state.buffer), 3):
                    buffer_encoding[i] = self.word_vocab['<NULL>']

        encoding = np.concatenate((stack_encoding, buffer_encoding)).astype(int)
        # print(encoding)

        return encoding

    def get_output_representation(self, output_pair):  
        # TODO: Write this method for Part 2
        encoding = np.zeros(91)
        encoding[self.output_labels[output_pair]] = 1
        return encoding

     
    
def get_training_matrices(extractor, in_file):
    inputs = []
    outputs = []
    count = 0 
    for dtree in conll_reader(in_file):
        words = dtree.words()
        pos = dtree.pos()
        for state, output_pair in get_training_instances(dtree):
            inputs.append(extractor.get_input_representation(words, pos, state))
            outputs.append(extractor.get_output_representation(output_pair))
        if count%100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        count += 1
    sys.stdout.write("\n")
    return np.vstack(inputs),np.vstack(outputs)
       


# if __name__ == "__main__":
#
#     WORD_VOCAB_FILE = 'data/words.vocab'
#     POS_VOCAB_FILE = 'data/pos.vocab'
#
#     try:
#         word_vocab_f = open(WORD_VOCAB_FILE,'r')
#         pos_vocab_f = open(POS_VOCAB_FILE,'r')
#     except FileNotFoundError:
#         print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
#         sys.exit(1)
#
#
#     with open(sys.argv[1],'r') as in_file:
#
#         extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
#         print("Starting feature extraction... (each . represents 100 sentences)")
#         inputs, outputs = get_training_matrices(extractor,in_file)
#         print("Writing output...")
#         np.save(sys.argv[2], inputs)
#         np.save(sys.argv[3], outputs)


if __name__ == "__main__":
    import os

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    train_conll = 'D:/HaowenWei\PycharmProjects\COMSW4705_001_2023_3-NLP-Homework1\Homework3\hw3_files\data/dev.conll'
    input_train = 'data/input_dev.npy'
    output_train = 'data/output_dev.npy'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 


    with open(train_conll,'r') as in_file:

        extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor,in_file)
        print("Writing output...")
        np.save(input_train, inputs)
        np.save(output_train, outputs)


