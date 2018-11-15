import pandas as pd
from sys import stdout

from collections import Counter

class WordProcess(object):
    def __init__(self, data): 
        self.data = data
             
    def dict_count(self, 
                   verbose = 2,
                   num = 0):
        count = Counter() # like a dict, but missing element count is 0 instead of KeyError
        for text in self.data:
            num += 1
            count_list = Counter(text)
            if verbose:
                print('\ndocument #{}:\n {}\n'.format(num, count_list))
                verbose -= 1
            for token_ in count_list:
                count[token_] += count_list[token_]
        return count             
                
    def count_words(self, num=0):
        for text in self.data:
            phrase = text.split()
            num += len(phrase)
        return num       
    
       