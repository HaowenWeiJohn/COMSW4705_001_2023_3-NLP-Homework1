from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys
import tensorflow as tf
import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)


        while state.buffer:
            # TODO: Write the body of this loop for part 4
            model_input = self.extractor.get_input_representation(words, pos, state)
            model_input = np.array([model_input])
            output = self.model.predict(model_input)
            output = output[0]

            transitions = [transition for transition in self.output_labels.values()]

            # sort transitions based on the output
            transitions = [x for _,x in sorted(zip(output,transitions), reverse=True)]


            for transition in transitions:
                # if stack is empty, left-arc and right-arc are not possible
                if len(state.stack) == 0:
                    if transition[0] == 'left_arc' or transition[0] == 'right_arc':
                        # skip this transition
                        continue

                # shifting the only word out of the buffer is also illegal, unless the stack is empty
                if len(state.buffer) == 1 and len(state.stack) > 0:
                    if transition[0] == 'shift':
                        # skip this transition
                        continue


                # the root node must never be the target of a left-arc
                if len(state.stack) > 0 and state.stack[-1] == 0:
                    if transition[0] == 'left_arc':
                        # skip this transition
                        continue




                # found valid transition
                if transition[0] == 'shift':
                    state.shift()
                if transition[0] == 'left_arc':
                    state.left_arc(transition[1])
                if transition[0] == 'right_arc':
                    state.right_arc(transition[1])

                # print(state.stack, state.buffer, state.deps)

                break







        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

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
#     extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
#     parser = Parser(extractor, sys.argv[1])
#
#     with open(sys.argv[2],'r') as in_file:
#         for dtree in conll_reader(in_file):
#             words = dtree.words()
#             pos = dtree.pos()
#             deps = parser.parse_sentence(words, pos)
#             print(deps.print_conll())
#             print()
#


if __name__ == "__main__":

    tf.compat.v1.disable_eager_execution()

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    model_path = 'data\model.h5'
    dev_conll = 'data\dev.conll'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, model_path)

    with open(dev_conll,'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()

