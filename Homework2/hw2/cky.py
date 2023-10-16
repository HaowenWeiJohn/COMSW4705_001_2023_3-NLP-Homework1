"""
COMS W4705 - Natural Language Processing - Fall 2023
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2

        table, probs = self.parse_with_backpointers(tokens)
        start_symbol = self.grammar.startsymbol
        if start_symbol in table[(0,len(tokens))]:
            return True
        else:
            return False
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table= None
        probs = None

        table = {}
        probs = {}

        # initialization
        for i in range(len(tokens)):
            table[(i,i+1)] = {}
            probs[(i,i+1)] = {}
            token = tokens[i]
            rules = self.grammar.rhs_to_rules[(token,)]
            for rule in rules:
                table[(i,i+1)][rule[0]] = rule[1][0]
                probs[(i,i+1)][rule[0]] = math.log(rule[2])


            # table[(i,i+1)] = {}
            # token = tokens[i]
            # rules = self.grammar.rhs_to_rules[(token,)]
            # for rule in rules:
            #     table[(i,i+1)][rule[0]] = (rule[1][0],i,i+1)

        # cky algorithm
        for length in range(2, len(tokens)+1):
            for i in range(0, len(tokens)-length+1):
                j = i + length
                table[(i,j)] = {}
                probs[(i,j)] = {}
                for k in range(i+1, j):
                    left = (i,k)
                    right = (k,j)
                    # record the possible non-terminal
                    left_nonterminal = table[left].keys()
                    right_nonterminal = table[right].keys()
                    for left_nt in left_nonterminal:
                        for right_nt in right_nonterminal:
                            rules = self.grammar.rhs_to_rules[(left_nt, right_nt)]
                            for rule in rules:
                                if rule[0] not in table[(i,j)]:
                                    table[(i,j)][rule[0]] = ((left_nt, i, k), (right_nt, k, j))
                                    probs[(i,j)][rule[0]] = math.log(rule[2]) + probs[left][left_nt] + probs[right][right_nt]
                                else:
                                    if probs[(i,j)][rule[0]] < math.log(rule[2]) + probs[left][left_nt] + probs[right][right_nt]:
                                        table[(i,j)][rule[0]] = ((left_nt, i, k), (right_nt, k, j))
                                        probs[(i,j)][rule[0]] = math.log(rule[2]) + probs[left][left_nt] + probs[right][right_nt]


        return table, probs


def get_tree(chart, i,j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4

    # find the most possible non-terminal

    rule = chart[(i,j)][nt]

    if isinstance(rule, str):
        return (nt, rule)

    elif isinstance(rule, tuple):
        left_rule = rule[0]
        right_rule = rule[1]
        return (nt, get_tree(chart, left_rule[1], left_rule[2], left_rule[0]), get_tree(chart, right_rule[1], right_rule[2], right_rule[0]))


       
if __name__ == "__main__":

    print("Testing parser:")

    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.']

        print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)

        tree = get_tree(table, 0, len(toks), grammar.startsymbol)

        # This is copied from the homework pdf
        tree_correct = tree == ('TOP', ('NP', ('NP', 'flights'), ('NPBAR', ('PP', ('FROM', 'from'), ('NP', 'miami')), ('PP', ('TO', 'to'), ('NP', 'cleveland')))), ('PUN', '.'))

        print("Tree is correct: " + str(tree_correct))