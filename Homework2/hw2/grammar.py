"""
COMS W4705 - Natural Language Processing - Fall 2023
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum
import math

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1

        for key in self.lhs_to_rules:
            rule = self.lhs_to_rules[key]

            print(key)
            print(rule)

            sum = 0
            for r in rule:
                sum += r[2]
                # check if the key and rule follows Chomsky Normal Form
                if not r[0].isupper():
                    return False

                # if len(r[1]) == 1:
                #     if not r[1][0].islower() or : Note: some are digits in the rule
                #         return False

            # use close function to check if the sum of all rules for a key is 1
            if not math.isclose(sum, 1.0):
                return False

        # check sum for all lhs_to_rules


        return True


if __name__ == "__main__":
    # with open(sys.argv[1],'r') as grammar_file:
    #     grammar = Pcfg(grammar_file)
    with open('atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file)

    grammar_is_valid = grammar.verify_grammar()
    print("Grammar is valid: {}".format(grammar_is_valid))
