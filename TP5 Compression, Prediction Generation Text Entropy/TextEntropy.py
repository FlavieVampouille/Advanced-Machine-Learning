"""Compute the entropy of different models for text
# coding: utf-8      
      
Usage: compress [-m <model>] [-f <file>] [-o <order>]

Options:
-h --help      Show the description of the program
-f <file> --filename <file>  filename of the text to compress [default: Dostoevsky.txt]
-o <order> --order <order>  order of the model [default: 4]
-m <model> --model <model>  model for compression [default: MarkovModel]
"""

# -*- coding: utf-8 -*-

from __future__ import division
import re
import numpy as np
#import argparse
#import math
from collections import defaultdict, deque, Counter
from docopt import docopt
import random


class IIDModel:
    """An interface for the text model"""
    def __init__(self, order):                 # CHANGE ORDER IN OPTIONS ABOVE
        print("Creation of the model")
        self.order = order
        
    def process(self,text):
        self.length = len(text)
        self.frequency = {}
        # remplir le dictionnaire en creant un compteur pour chaque caractere
        for symb in xrange(self.length - self.order):
            self.frequency[text[symb:symb+self.order]] = 0
        for symb in xrange(self.length - self.order):
            self.frequency[text[symb:symb+self.order]] += 1
        return self.frequency

    def getEntropy(self, text):
        self.process(text)
        self.entropy = 0
        self.sum_val = np.sum(self.frequency.values())
        # calcul la frequence de chaque symbol p(x) et l'entropie sum_x(p(x).log2 p(x))
        for key, value in self.frequency.iteritems():
            self.frequency[key] = value / self.sum_val
            self.entropy -= self.frequency[key] * np.log2(self.frequency[key])
        # sauvegarde du dictionnaire
        self.frequencyScr = self.frequency
        return self.entropy

    def getCrossEntropy(self, text):
        self.process(text)
        self.cross_entropy = 0
        self.n_Scr = len(self.frequencyScr)
        self.n = len(self.frequency)
        self.sum_val = np.sum(self.frequency.values())
        # calcul la frequence de chaque symbol q(x)
        for key, value in self.frequency.iteritems():
            self.frequency[key] = value / self.sum_val
        # calcul la cross entropie des deux textes
        for key in set(self.frequencyScr) | set(self.frequency):
            if key in set(self.frequencyScr) & set(self.frequency):
                self.cross_entropy -= self.frequencyScr[key] * np.log2(self.frequency[key])
            else:
                if key in set(self.frequency):
                    self.cross_entropy -= np.log2(self.frequency[key]) / self.n_Scr
                else: 
                    self.cross_entropy -= self.frequencyScr[key] * np.log2(1/self.n)
        return self.cross_entropy  

    def getSentences(self,text):
        self.process(text)
        self.token = ''
        for token_id in range(0,1000):  # CHOOSE HERE THE DESIRED SENTENCE LENGTH
            self.count = 0
            for key, value in self.frequency.iteritems():
                self.count += value
                if random.randint(0, self.count-1) < value:
                    self.sample = key
            self.token = ''.join([self.token,self.sample])
        print(self.token)
        

class MarkovModel:
    """An interface for the text model"""
    def __init__(self, order):
        print("Creation of the model")
        self.order = order

    def process(self, text):
        self.frequency_char, self.frequency_symb = defaultdict(Counter), Counter()
        circular_buffer = deque(maxlen = self.order)   
        for token in text:
            prefix = tuple(circular_buffer)
            circular_buffer.append(token)
            if len(prefix) == self.order:
                self.frequency_symb[prefix] += 1 
                self.frequency_char[prefix][token] += 1
        return self.frequency_char , self.frequency_symb

    def getEntropy(self,text):
        self.process(text)
        self.frequency_char_scr = self.frequency_char
        self.frequency_symb_scr = self.frequency_symb
        def entropy(stats, normalization_factor):
            return -sum((freq/normalization_factor) * np.log2(freq/normalization_factor) 
                        for freq in stats.values()) 
        def entropy_markov(characters , symbols):
            return sum(symbols[key] * entropy(characters[key],symbols[key]) 
                       for key in symbols) / sum(symbols.values())
        return entropy_markov(self.frequency_char,self.frequency_symb)
    
    
    def getCrossEntropy(self, text):
        self.process(text)
        self.n_scr = len(self.frequency_symb_scr)
        self.n = len(self.frequency_symb)
        self.cross_entropy = 0
        self.sum_val_scr = sum(self.frequency_symb_scr.values())
        self.sum_val = sum(self.frequency_symb.values())
        
        for key in set(self.frequency_symb_scr) | set(self.frequency_symb):
            if key in set(self.frequency_symb_scr) & set(self.frequency_symb):
                for stat_scr,stat_target in zip(self.frequency_char_scr[key].values(),self.frequency_char[key].values()):
                    self.cross_entropy -= self.frequency_symb_scr[key] * (stat_scr/self.frequency_symb_scr[key]) * np.log2(stat_target/self.frequency_symb[key]) / self.sum_val_scr     
            else:
                if key in set(self.frequency_symb):
                    for stat_target in self.frequency_char[key].values():
                        self.cross_entropy -= (1/self.n_scr)**2 * np.log2(stat_target/self.frequency_symb[key])
                else:
                    for stat_scr in self.frequency_char_scr[key].values():
                        self.cross_entropy -= self.frequency_symb_scr[key] * (stat_scr/self.frequency_symb_scr[key]) * np.log2(1/self.n) / self.sum_val_scr
        return self.cross_entropy 
        
    def getSentences(self,text):
        self.process(text)
        def pick(counter):
            sample, accumulator = None, 0
            for key, count in counter.items():
                accumulator += count
                if random.randint(0, accumulator - 1) < count:
                    sample = key
            return sample
        def generate(model, state, length):
            for token_id in range(0, length):
                yield state[0]
                state = state[1:] + (pick(model[state]), )
        prefix = pick(self.frequency_symb)
        sentences = "".join(generate(self.frequency_char, prefix, 1000))
        print(sentences)


def preprocess(text):
    text = re.sub("\s\s+", " ", text)
    text = re.sub("\n", " ", text)
    return text

# Experiencing encoding issues due to UTF8 (on possibly other texts)? Consider:
#  f.read().decode('utf8')
#  blabla.join(u'dgfg')
#              ^


if __name__ == '__main__':

    # Retrieve the arguments from the command-line
    args = docopt(__doc__)
    print(args)

    # Read and preprocess the text
    src_text = preprocess(open(args["--filename"]).read().decode('utf8'))
    target_text = preprocess(open("Hamlet.txt").read().decode('utf8'))

    # Create the model
    if(args["--model"]=="IIDModel"):
        model = IIDModel(int(args["--order"]))
    elif(args["--model"]=="MarkovModel"):
        model = MarkovModel(int(args["--order"]))
    
    print(model.getEntropy(src_text))
    print(model.getCrossEntropy(target_text))
    #model.getSentences(src_text)
    #print("KL divergence : {0}".format(-model.getEntropy(src_text)+model.getCrossEntropy(target_text)))