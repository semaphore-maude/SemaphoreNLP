import pandas as pd
import numpy as np

import os 
import re
import unicodedata
import string
from collections import Counter

from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer, SnowballStemmer

# plot libraries
from plotly.offline import iplot
import plotly.graph_objs as go
# Static image generation with plotly.io requires orca commandline util and psutil library
import plotly.io as pio
from matplotlib import pyplot as plt
from wordcloud import WordCloud


from helper.helper_funs import time_filename, save_folder_file


class DataLoader:
    '''
    A class for loading and cleaning the dataset. 
    This was written with the ASC dataset in mind, but it could be modified 
    to fit other datasets more generally. The things specific to this data are
    removed_language in the _clean_data() method, and the characters in re.sub 
    in the _remove_weird_chars method.
    '''
    def __init__(self, filename, 
                       data_folder, 
                       colname, 
                       rm_NAs,  
                       removed_language=[], # keep empty list for no language to remove
                       normalize_unicode=True):
        '''
          `filename`:          str, full filename including file extension (.csv)
          `data_folder`:       str, folder containing dataset 
          `colname`:           list of str, column(s) to remove weird chars (ascii artefacts)
          `rm_NAs`:            bool, remove NA/NaNs (True) or replace with empty str (False)
          `removed_language`:  list of str, containing languages to remove (lower case)
          `normalize_unicode`: bool, remove all accents and replace with plain letter (lower case) 

        '''
        self.filename = filename
        self.data_folder = data_folder
        self.colname = colname        
        self.rm_NAs = rm_NAs
        self.removed_language = removed_language
        self.normalize_unicode = normalize_unicode

    def read_data(self):
        '''
        Read csv data file 
        Returns pandas DataFrame
        '''
        cur_path = os.path.join(os.getcwd(), self.data_folder, self.filename)
        data = pd.read_csv(cur_path, engine='python', error_bad_lines=False)
        return data

    def _clean_data(self, data):
        '''
        Private function to remove NaNs for selected columns, duplicates 
        In both NaN removal and duplicate removal, only the first column is considered.
        for example: 
             [['a', NaN, 'b', 'c', 'b']^T, ['d', 'e', 'f', NaN, 'g']^T] would result in:
             [['a', 'b', 'c']^T, ['d', 'f', '']^T] then concatenated as a single list:
             ['a d', 'b f', 'c'] 
        Also removes records with unwanted languages

        Returns pandas DataFrame
        '''
        rmNAs = self.rm_NAs
        print('Removing duplicates in column(s): {}...\n'.format(self.colname))

        for num, col in enumerate(self.colname):  # loop over text cols
            if num == 0: # using first col specified
                data.drop_duplicates(keep='first', subset=col, inplace=True)

            if rmNAs and len(self.colname) > 1:  
                data = data[pd.notnull(data[col])] 
                print('Can only remove NaNs in first column, '\
                      'subsequent columns will make NaNs into empty strings\n')
                rmNAs = False

            elif rmNAs: # remove NaNs for only one column
                data = data[pd.notnull(data[col])]

            else:       # one or many cols where NaNs are replaced by empty strings
                data[col].replace( np.nan, '', regex = True, inplace = True)
            data[col].astype(str).str.lower()                               

        if len(self.removed_language):
            for lang in self.removed_language:
                print('Removing {} words...\n'.format(lang))
                low_lang = data['Languages'].astype(str).str.lower()
                data = data[~low_lang.str.contains(lang)]   
  
        return data      
    
    def _remove_weird_chars(self, data): 
        '''
        Deals with weird ascii artefacts in dataset.
        Turns into lists (1 list per col), appends each statement, 
        no accents, just plain letters
        Then concatenates each columns together, empty str if no words.
        If all cols are empty in a records, it is skipped

        Returns:
         A list of strings
         Output example: ['(from 1st record) words from col 1, ..., words from col n',
                          ' . . .',
                          '(from mth record) words from col 1, ..., words from col n']
            for m records and n columns                           
        '''
        all_words = {}
        for col in self.colname:   
            words = []
            for row in data[col]:
                phrase = re.sub( r'Ã©', 'e', row)
                phrase = re.sub( r'[âÂ]', '', phrase)
                phrase = re.sub( r'Ã§', 'c', phrase)
                phrase = ''.join(x for x in unicodedata.normalize(
                                 'NFKD', phrase) if x in string.printable).lower()
                words.append(phrase)
            all_words[col] = words   

        words_list_tup = list(zip(*[val for val in all_words.values()]))
        joined_words = [' '.join(
                       [word for word in words if word != '']) for words in words_list_tup]
        joined_words = list(filter(None, joined_words)) # remove records with no text after cleaning

        return joined_words


    def data_loader(self):
        '''
        Method used to import and pre-clean data
        Returns:
         A list of strings
         Output example: ['(from 1st record) words from col 1, ..., words from col n',
                          ' . . .',
                          '(from mth record) words from col 1, ..., words from col n']
            for m records and n columns 
        '''
        print('Reading and cleaning data file...\n')
        data = self.read_data()
        data = self._clean_data(data)
        if self.normalize_unicode:
            data = self._remove_weird_chars(data)
        print('Done!\n')

        return data      


class CleanText:
    '''
    Class for prepping and cleaning for LDA, LSI or HDP
    '''
    def __init__(self, text, 
                       stop=['english'],
                       stemmer='Porter',
                       remove_urls=False):
        """
         `text`:    A list of strings, with each strings representing a document
         `stop`:    A list of strings containing the stopwords *language*
                    ex.: ['english', 'german'] 
         `stemmer`: str, type of stemming process.
                         Choices: 'Porter': often default stemmer
                                  'Lancaster': more conservative (smaller stems) than Porter
                                  'Snowball': language-specific (only implemented in English) 
          `remove_urls`: bool, do we cut urls out of the text?                               
        """
        self.text = text
        self.stop = stop
        self.stemmer = stemmer
        self.remove_urls = remove_urls
        
    def _lemmatize_stemming(self, word, is_notebook):
        '''
        `word`:      str, single word from a document
        `is_notebook`: bool, is this a jupyter notebook?
        '''
        if self.stemmer == 'Porter':
            stemming = PorterStemmer()
        elif self.stemmer == 'Lancaster':
            stemming = LancasterStemmer()
        elif self.stemmer == 'Snowball':
            stemming = SnowballStemmer("english")    
        else:
            if is_notebook:
                print('No word stemming, pick \'Porter\', \'Lancaster\' or \'Snowball\'')
            else:
                print('No word stemming, pick \'Porter\', \'Lancaster\' or \'Snowball\''\
                                                                 'with CL flag -stem\n') 

        lemmatizer = WordNetLemmatizer()
        stemlemma = stemming.stem(lemmatizer.lemmatize(word, pos='v')) 
        # lemmatize groups semantics, here pos='v' will take care of verbs
        # for example, 'are', 'were' and 'is' should all be categorized under 'be'

        return stemlemma   

    def preprocess(self,
                   word_min_len = 3,
                   is_notebook = True,
                   numeric = False):
        '''
        Full preprocess to prepare for LDA
         `word_min_len`: int, the smallest number of letters required to keep a word
                              (this should be done in the )
         `is_notebook`:  bool, is this a jupyter notebook?
         `numeric`:      bool, keep alphanumerics (True) or only letters (False)? 
        '''
        count_urls = 0
        num_change = 0
        result=[]

        if numeric:
            regex = r'[a-zA-Z0-9+]{'+ str(word_min_len) + ',}'
        else:
            regex = r'[a-zA-Z+]{'+ str(word_min_len) + ',}'

        for line in self.text:
            phrase = line   # `phrase` is redefined if remove_urls is True (without the urls)
            if self.remove_urls:
                phrase, num_repl = re.subn(
                           r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                           '', phrase)

                count_urls += num_repl  # count the number of urls found   
            phrase = re.findall(regex, phrase)
            if phrase:
                num_change += 1  # number of records with new changes
            phrase = [self._lemmatize_stemming(w, is_notebook) for w in phrase
                                                     if w not in self.stop]
            result.append(phrase)
        if num_change:
            print('\nOnly plain letters kept and words of {} letters or more.'.format(word_min_len))
        
        print('\n{} url(s) removed.'.format(count_urls))
        return result   



class WordCount:
    """
    Class to deal with word frequency related output.
    Frequency plots, wordplots and counting dictionaries for printing summaries
    """
    def __init__(self, data): 
        '''
         `data`: list of prepared (cleaned, lemmatized, stemmed) strings (1 element is 1 doc)
        '''
        self.data = data
             
    def dict_count(self, 
                   verbose = 2):
        '''
         Returns a {'word': overall frequency} dictionary
         `verbose`: int, for printing output. 
                         Number of documents printed in output
                         Each document is setup with key/value as word/frequency
         * NOTE * the dict returned does not separate docs, only the `verbose` output                      
        '''

        count = Counter() # like a dict, but missing element count is 0 instead of KeyError
        if verbose:
            print('Showing stemmed/lemmatized words and counts of {} first documents'.format(verbose))
        else:    
            for num, text in enumerate(self.data):
                count_list = Counter(text)
                if verbose:
                    print('\ndocument #{}:\n {}\n'.format(num, count_list))
                    verbose -= 1
                for token_ in count_list:
                    count[token_] += count_list[token_]
                    # print(count[token_])
        return count             
                
    def count_words(self):
        '''
        Small function to count the total number of words
        '''
        num=0

        for text in self.data:
            phrase = text.split()
            num += len(phrase)
        return num       


class PlotWords:
    '''
    Class to build interactive plots on the local server and word cloudplots
    '''
    def __init__(self, count_dict):
        '''
         `count_dict` is an instance of the WordCount class
        '''
        self.count_dict = count_dict
        
    def order_count(self):   

        '''
         Returns a sort of ordered dictionary, 
         where values are ordered from largest (i.e. most frequent) to smallest
        '''
        count_dict = self.count_dict
        freqs = count_dict.items()
        sorted_count = sorted(freqs, key=lambda t: t[1], reverse=True)

        return sorted_count
    

    def freq_plot(self, top_n=50,
                        width=1.0,
                        c_scale='Portland',
                        title = 'Top word frequencies (after cleanup and lemmatization)',
                        plotname='word_count_bar',
                        image_format ='png',
                        save_plot=True,
                        save_dir='visualization',
                        filename='',
                        is_notebook = True):
        """
        Interactive bar frequency plot

        `top_n`:        int, to plot a number top_n of most frequent words
        `width`:        float, bar width
        `c_scale`:      str, colour scheme (see matplotlib colour schemes)
        `title`:        str, title to display on image
        `plotname`:     str, for notebook display
        `image_format`: str, image extension, of the for 'png', 'pdf', etc - NO dot
        `save_plot`:    bool, is the plot saved
        `save_dir`:     str, folder to save plot (child of the working directory)
                             folder will be created if it doesn't exists
                             NOTE: orca must be installed to save a still image of the plot
        `filename`:     str, filename for the still image to save
        `is_notebook`:  bool, is this displayed on a notebook?
        """

        ordered_count = self.order_count()
        sorted_word = [count[0] for count in ordered_count[:top_n]]
        sorted_freq = [count[1] for count in ordered_count[:top_n]]

        data_word = [go.Bar(x = sorted_word,
                            y = sorted_freq,
                            marker = dict(colorscale = c_scale,
                                          color = sorted_freq,
                                          line = dict(color = 'rgb(0,0,0)', 
                                                    width = width)),
                    text = 'Word count')]
        
        layout = go.Layout(title=title)        
        fig = go.Figure(data=data_word, layout=layout)

        if is_notebook:
            iplot(fig, filename=plotname, image=image_format);

        if save_plot:   
            if len(filename) == 0:
                filename = 'word_frequency_barplot_top{}_words_'.format(top_n)     

            full_path = save_folder_file(save_dir, filename, ext= '.' + image_format)
       
            print('Pyplot word frequency bar chart saved to `{}`.\n'.format(full_path))

            pio.write_image(fig, full_path)

    def cloud_plot(self, size=(9,6),
                         background_color="black", 
                         max_words=1000, 
                         max_font_size= 60,
                         min_font_size=5,
                         collocations = False,
                         colormap="coolwarm",
                         plot_title="Most common words",
                         plot_fontsize=30,
                         interpolation='lanczos',
                         save_plot='True',
                         save_dir='visualization',
                         filename='',
                         image_format ='.png',
                         is_notebook=True):
        '''
         `size`:             tuple of ints, image size
         `background_color`: str, colour name
         `max_words`:        int, maximum number of words to plot 
         `max_font_size`:    int, maximum font size
         `min_font_size`:    int, minimum font size
         `collocations`:     bool, * set to False * to avoid duplicates
         `colormap`:         str, colour scheme for letters (see matplotlib colours)
         `plot_title`:       str, title 
         `plot_fontsize`:    int, average fontsize
         `interpolation`:    str, smoother, example of possible choices: 
                                  'nearest', 'bilinear', 'hamming', 'quadric', 'lanczos'
        `save_plot`:    bool, is the plot saved
        `save_dir`:     str, folder to save plot (child of the working directory)
                             folder will be created if it doesn't exists
        `filename`:     str, filename for the still image to save
        `image_format`: str, extension, of the form '.png', '.pdf', etc
        `is_notebook`:  bool, is this displayed on a notebook?
        '''

        self.text_cloud = " ".join(word for word in self.count_dict.elements())

        plt.figure(figsize=size);
        wc = WordCloud(background_color=background_color, 
                       max_words=max_words, 
                       max_font_size=max_font_size,
                       min_font_size=min_font_size,
                       collocations = collocations,
                       colormap=colormap);
        
        wc.generate(self.text_cloud);
        plt.title(plot_title, fontsize=plot_fontsize);
        plt.margins(x=0.25, y=0.25);
        plt.axis('off');

        plt.imshow(wc, interpolation=interpolation);
        if is_notebook:
            plt.show();

        if save_plot:   
            if len(filename) == 0:
                filename = 'wordcloud_plot_'    

            full_path = save_folder_file(save_dir, filename, ext= image_format)
       
            print('Wordcould plot saved to `{}`.\n'.format(full_path))
            # store to file
            wc.to_file(full_path);
            plt.savefig(full_path);



    
     


