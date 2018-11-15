import pandas as pd

import os 
import re
import unicodedata
import string

from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer


class DataLoader(object):
    def __init__(self, filename, #string, filename with extension
                       data_folder, # string, folder where data is
                       colname, # list of strings, column(s) to remove weird chars
                       rm_NAs,  # bool, remove NA/NaNs?
                       removed_language=[], # list containing languages to remove
                       normalize_unicode=True):
        
        self.filename = filename
        self.data_folder = data_folder
        self.colname = colname        
        self.rm_NAs = rm_NAs
        self.removed_language = removed_language
        self.normalize_unicode = normalize_unicode

    # read csv data file 
    def read_data(self):
        cur_path = os.getcwd() + self.data_folder + self.filename
        data = pd.read_csv(cur_path, engine='python', error_bad_lines=False)
        return data

    def _clean_data(self, data):
        if self.rm_NAs:
            for col in self.colname:  
                data = data[~data[col].duplicated(keep='first')]
#                 data = data[~data[col].isnull()]
#                 data = data[~data[col].isna()]          
                data = data[pd.notnull(data[col])]                
                print('Removing NAs and duplicates from column {}...\n'.format(col))                  
        if len(self.removed_language) > 0:
            for lang in self.removed_language:
                print('Removing {} words...\n'.format(lang))
                data = data[~data['Languages'].str.contains(lang)]       
        return data      
    
    def _remove_weird_chars(self, data):
        for col in self.colname:
            words = []
            for row in data[col]:
           
                phrase = re.sub( r'Ã©', 'e', row)
                phrase = re.sub( r'[âÂ]', '', phrase)
                phrase = re.sub( r'Ã§', 'c', phrase)
                phrase = ''.join(x for x in unicodedata.normalize('NFKD', phrase) if x in string.printable).lower()
                words.append(phrase)
        return words
    
    def data_loader(self):
        data = self.read_data()
        data = self._clean_data(data)
        if self.normalize_unicode:
            data = self._remove_weird_chars(data)
        return data



class CleanText(object):
    def __init__(self, text, 
                       col,
                       stop=[],
                       stemmer='Porter',
                       remove_urls=False):
        self.text = text
        self.col = col
        self.stop = stop
        self.stemmer = stemmer
        self.remove_urls = remove_urls
        
    def lemmatize_stemming(self, word):
        if self.stemmer == 'Porter':
            stemming = PorterStemmer()
        elif self.stemmer == 'Lancaster':
            stemming = LancasterStemmer()
        else:
            print('No word stemming, pick \'Porter\' or \'Lancaster\',\n')
        lemmatizer = WordNetLemmatizer()
        return stemming.stem(lemmatizer.lemmatize(word, pos='v'))    

    def preprocess(self,
                   result=[],
                   num_urls = 0,
                   num_change = 0,
                   word_min_len = 3):
        regex = r'[a-zA-Z+]{'+ str(word_min_len) + ',}'
        for line in self.text:
            phrase = line
            if self.remove_urls:
                (phrase, num_repl) = re.subn(
                           r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                           '', phrase)
                num_urls += num_repl    
            phrase = re.findall(regex, phrase)
            if phrase:
                num_change += 1
            phrase = [self.lemmatize_stemming(w) for w in phrase
                                                     if w not in self.stop]
            result.append(phrase)
        if num_change:
            print('Only plain letters kept and words of {} letters or more.'.format(word_min_len))
        
        print('{} url(s) removed.'.format(num_urls))
        return result   


    
# # NOTE: MAKE LABELS LATER WITHIN THIS CLASS   
#     def _to_bool(self, s, string):
#         return 1 if s == string else 0
    
#     def _pre_process_labels(data, string=''):
#         list_y = []
#         len_data = len(data)
#         for row in data:
#             bins = self.to_bool(row, string)
#             list_y.append(bins)
#         arr_y = np.array(list_y)
#         return arr_y
        
     
        


