import pandas as pd
import numpy as np

import os 
import json
import re
import unicodedata
from html.parser import HTMLParser
import string
from collections import Counter

import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)


from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer, SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool

import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import to_unicode
from gensim.parsing.preprocessing import strip_multiple_whitespaces

import nltk
from nltk.corpus import wordnet, stopwords

# plotting libraries
from plotly.offline import iplot
import plotly.graph_objs as go
# Static image generation with plotly.io requires orca commandline util and psutil library
import plotly.io as pio
from matplotlib import pyplot as plt
from wordcloud import WordCloud

# local modules
from helper.helper_funs import time_filename, save_folder_file



class MLStripper(HTMLParser):
    '''
    Remove html escape sequences in text
    Use by calling the strip_tags() function
    '''
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
        self.strict = False
        self.convert_charrefs= True

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


class DataLoader:
    '''
    A class for loading and cleaning the dataset. 
    This was written with the ASC dataset in mind (since there were specific kinds of characters), 
    but it could be modified slightly to fit other datasets. The things specific to this data are
    removed_language in the _clean_data() method, and the characters in re.sub 
    in the _remove_weird_chars method.
    '''
    def __init__(self, filename, 
                       data_folder, 
                       colname, 
                       rm_NAs=False,  
                       removed_language=[], # keep empty list for no language to remove
                       normalize_unicode=True,
                       label_colname='Organization Name',
                       no_nums=True,
                       greedy_url_regex=True):
        '''
          `filename`:          str, full filename including file extension (.csv)
          `data_folder`:       str, folder containing dataset 
          `colname`:           list of str, column(s) to remove weird chars (ascii artefacts)
          `rm_NAs`:            bool, remove NA/NaNs (True) or replace with empty str (False)
          `removed_language`:  list of str, containing languages to remove (lower case)
          `normalize_unicode`: bool, remove all accents and replace with plain letter (lower case) 
          `label_colname`:     str, name of column that contains the labels (any kind of ID column)
          `no_nums`:           bool, remove numerics that are not part of a word
          `greedy_url_regex`:  bool, whether to use regex that matches as many urls as possible,
                                     possibly at the cost of matching a few non-url patterns

        '''
        self.filename = filename
        self.data_folder = data_folder
        self.colname = colname        
        self.rm_NAs = rm_NAs
        self.removed_language = removed_language
        self.normalize_unicode = normalize_unicode
        self.label_colname = label_colname
        self.no_nums = no_nums
        self.greedy_url_regex = greedy_url_regex

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

        for num, col in enumerate(self.colname):  # loop over text cols
            if num == 0: # using first col specified

                data.drop_duplicates(keep='first', subset=col, inplace=True)
                print('Removing duplicates based on column: {}...\n'.format(col))

                if rmNAs and len(self.colname) > 1:  
                    data = data[pd.notnull(data[col])] 
                    print('Can only remove NaNs in {}, subsequent columns will change NaNs into '\
                              'empty strings and later concatenate to first column\n'.format(col))
                    rmNAs = False
    
                elif rmNAs: # remove NaNs for only one column
                    data = data[pd.notnull(data[col])]

                else:
                    print('Look into data, potential conflict in arguments related to NaNs.\n')    

            else:       # one or many cols where NaNs are replaced by empty strings
                data[col].replace( np.nan, '', regex = True, inplace = True)

            data[col].astype(str).str.lower()                               

        if len(self.removed_language):
            for lang in self.removed_language:
                print('Removing {} texts...\n'.format(lang))
                low_lang = data['Languages'].astype(str).str.lower()
                data = data[~low_lang.str.contains(lang)]   

        return data     

    def _load_regexes(self):
        '''
        `regex_path`, `typo_key`, `accent_key`... defined in self.data_loader() call
        '''
        with open(self.regex_path) as json_file:  
            clean_regex = json.load(json_file)
            clean_words = clean_regex[self.typo_key] 
            accents = clean_regex[self.accent_key]

            return clean_words, accents    
    
    def _remove_weird_chars_line(self, row):

        # remove remnants of html
        phrase = strip_tags(row)        

        # weird characters, remove before normalizing below 
        for key, value in self.accents.items():
            phrase = re.sub(r"{}".format(value), r"{}".format(key), phrase)   

        # line below will remove accents from all words (i.e. 'é' -> 'e')
        phrase = ''.join(x for x in unicodedata.normalize(
                         'NFKD', phrase) if x in string.printable).lower()         
        
        # words and typos
        for key, value in self.words_to_clean.items():
            phrase = re.sub(r"{}".format(value), r"{}".format(key), phrase)

        if self.greedy_url_regex == True:    
            phrase, num_repl_url = re.subn(
                r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'"]))|(\S+\.com?)''',
                           '', phrase) 
        else:
            phrase, num_repl_url = re.subn(
                    r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'"]))''',
                               '', phrase)   

        phrase, num_repl_at = re.subn(r'@\S+', '', phrase)

        if self.no_nums:
            phrase = re.sub(r'\w*\d\w*', '', phrase)  # remove words with numbers    

        # keep word separated by - (ex.: community-based)
        p = re.compile(r"(\b[-']\b)|[\W_]")
        phrase = p.sub(lambda m: (m.group(1) if m.group(1) else " "), phrase)  
   
        num_links = (num_repl_url, num_repl_at)

        return phrase, num_links


    def _remove_weird_chars(self, data): 
        '''
        Deals with weird ascii artefacts in dataset.
        Turns into lists (1 list per col), appends each statement, 
        no accents, just plain letters
        Then concatenates each columns together, empty str if no words.
        If all cols are empty in a records, it is skipped

        Argument:
         `data`: dataframe of text data 
        Returns:
         A list of strings
         Output example: ['(from 1st record) words from col 1, ..., words from col n',
                          ' . . .',
                          '(from mth record) words from col 1, ..., words from col n']
            for m records and n columns                           
        '''
        all_words = {}
        label_list = []

        count_urls, count_at = (0, 0)

        for col in self.colname:   
            words = []
            for row in data[col]:
                phrase, num_links = self._remove_weird_chars_line(row)
                phrase = strip_multiple_whitespaces(phrase)
                words.append(phrase)
                count_urls += num_links[0]  # count the number of urls found   
                count_at += num_links[1]    # count the number of twitter handles found

            all_words[col] = words   

        for idx, lab in enumerate(data[self.label_colname]):
            label, _ = self._remove_weird_chars_line(lab)
            label = strip_multiple_whitespaces(label)
            label_list.append(label)            

        print('\n{} url(s) removed.'.format(count_urls))
        print('\n{} handle name(s) removed.\n'.format(count_at))

        words_list_tup = list(zip(*[val for val in all_words.values()]))
        # append additional text if more than one column is needed
        joined_words = [' '.join(
                       [word for word in words if word != '']) for words in words_list_tup]
        joined_words = list(filter(None, joined_words)) # remove records with no text after cleaning

        return joined_words, label_list


        

    def data_loader(self, regex_path='helper/cleaning_regex.json',
                          typo_key="clean_words",
                          accent_key="accents"):
        '''
        import and pre-clean data
        Args:
         `get_labels`: bool, whether to get labels for tagging documents
         `regex_path`: str, path to json file with {"corrected_word": "regex"} format
         `typo_key`  :    key for dict of regex of typos, form {keyname:{dict of regex}} 
        Returns:
         A list of strings
         Output example: ['(from 1st record) words from col 1, ..., words from col n',
                          ' . . .',
                          '(from mth record) words from col 1, ..., words from col n']
            for m records and n columns 
        '''
        self.regex_path = regex_path
        self.typo_key = typo_key
        self.accent_key = accent_key

        print('Reading and cleaning data file...\n')
        data = self.read_data()
        data = self._clean_data(data)

        if self.normalize_unicode:

            if len(regex_path):
                self.words_to_clean, self.accents = self._load_regexes()

            data, label_list = self._remove_weird_chars(data)   
        print('Done!\n\n'+'*'*20)

        return data, label_list      



class CleanText(DataLoader):
    '''
    Prepping and cleaning for LDA, LSI or HDP
    '''
    def __init__(self, text, 
                       stop=stopwords,
                       stemmer='Porter',
                       with_stemming=True):
        """
         `text`:    A list of strings, with each strings representing a document
         `stop`:    A list of strings containing the stopwords 
                    ex.: ['english', 'german'] 
         `stemmer`: str, type of stemming process.
                         Choices: 'Porter': often default stemmer
                                  'Lancaster': more conservative (smaller stems) than Porter
                                  'Snowball': language-specific (only implemented in English) 
         `with_stemming`: bool, switch to False if printing full words                                  
        """
        self.text = text
        self.stop = stop
        self.stemmer = stemmer
        self.with_stemming = with_stemming
      
        
    def _lemmatize_stemming(self, word, is_notebook, pos):
        '''
        `word`:        str, single word from a document
        `is_notebook`: bool, is this a jupyter notebook?
        `pos`        : list of word types for tagging
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
        if self.with_stemming:
            stemlemma = stemming.stem(lemmatizer.lemmatize(word, pos=pos)) 
        # lemmatize groups semantics, here pos='v' will take care of verbs
        # for example, 'are', 'were' and 'is' should all be categorized under 'be'
        else:
            stemlemma = lemmatizer.lemmatize(word, pos=pos)
        
        if stemlemma not in self.stop:
            return stemlemma 
        else:
            return ''    

    

    def _get_wordnet_pos(self, word):
        # map POS tag to first character lemmatize() accepts
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"j": wordnet.ADJ,
                    "n": wordnet.NOUN,
                    "v": wordnet.VERB,
                    "r": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)


    def preprocess(self,
                   min_len = 3, 
                   as_series = False,
                   custom_pos = True,
                   is_notebook = True):
        '''
        Full preprocess to prepare for LDA
         `min_len`:      int, minimum number of letters in words to keep
         `as_series`:    bool: returns pandas series (T) or list (F)?
         `custom_pos`:   bool: customize word type in lemmatization. True is preferred but slow.
        '''

        result=[]
        shortword = re.compile(r"\W*\b\w{{1,{}}}\b".format(min_len))   # identify words large enough

        docs = [shortword.sub('', to_unicode(x.strip())).split() for x in self.text ]
        if custom_pos:
            print('\nGetting customized POS tags from WordNet may take some time...\n')

        for phrase in docs:
            if custom_pos:
                phrase = [self._lemmatize_stemming(w, is_notebook, self._get_wordnet_pos(w)) for w in phrase
                                                     if w not in self.stop]                
            else:
                phrase = [self._lemmatize_stemming(w, is_notebook, pos='v') for w in phrase
                                                     if w not in self.stop]                                                                          
            
            if as_series:
                result.append(' '.join(phrase))
            else: 
                result.append(phrase)

        if as_series:
            result = pd.Series(result) 

        return result   


class LemmaCountVectorizer(CountVectorizer):
    '''
    for use when sklearn paradigm is used (with vectorizers/compressed sparse matrix as input)
    '''
    def build_analyzer(self):
        lemm = WordNetLemmatizer()
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))



class DimReducePlot:
    '''
    Using sklearn, with vectorizers/compressed sparse matrix as input
    '''
    def __init__(self, text, model_type='LDA', num_topics=4):

        '''
        Args:
         `text`: dataframe with rows of text data
         `model_type`: clustering method. For now, only LDA is implemented
        '''
        self.text = text
        self.model_type = model_type
        self.cleaned = CleanText(text).preprocess(as_series=True, custom_pos=False)
        self.tf = self._vectorize_lemma()

        if self.model_type == 'LDA':
            self.X_topics = self.withLDA(n_components=num_topics)
            self.tsne_model = self.LDA_TSNE()
        else:
            print('\nModel type not implemented\n')    


    def _vectorize_lemma(self):
        # Calling Count vectorizer
        self.tf_vectorizer = LemmaCountVectorizer(max_df=0.95, 
                                                  min_df=2,
                                                  stop_words='english',
                                                  decode_error='ignore')
        
        tf = self.tf_vectorizer.fit_transform(self.cleaned)      
        return(tf)



    def withLDA(self, n_components, max_iter=500):
        '''
        for sklearn LDA 

        `n_components`: int, number of clusters
        `max_iter`:     int, max iterations before stopping algorithm
        '''
        lda_model = LatentDirichletAllocation(n_components=n_components, max_iter=max_iter)
        lda_fit = lda_model.fit_transform(self.tf)
        self.n_components = n_components
        self.model = lda_model
        return(lda_fit)



    def LDA_TSNE(self, threshold=0.25,
                       **kwargs):
        '''
        Prepare for bokeh t-sne rendering 
        '''
        self.threshold = threshold
        model = self.model
        X_topics = self.X_topics

        _idx = np.amax(X_topics, axis=1) > threshold  # idx of doc that > threshold
        _topics = X_topics[_idx]
        self.num_example = len(_topics)    

        print("Running t-sne with {} method...".format(self.model_type))
        tsne_model = TSNE(**kwargs)

        # Number of dims specified in kwargs 
        tsne_lda = tsne_model.fit_transform(X_topics)
        return(tsne_lda)

    
    def plotTSNE(self, n_top_words = 8,   # number of keywords we show
                       save_dir='visualization',
                       filename='',
                       ext='.html'):
        '''
        Dimension reduction plots using T-SNE 
        Automatically saves - the plot is not displayed automatically
        Output is a html file with the plot
        '''

        # 20 colors
        colormap = np.array([
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
            "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
            "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
            "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"])

        X_topics = self.X_topics
        num_example = self.num_example
        tsne_model = self.tsne_model
        topic_word = self.model.components_  # all topic words
        vocab = self.tf_vectorizer.get_feature_names()
        cleaned = self.cleaned

        _model_keys = []
        for i in range(X_topics.shape[0]):
            _model_keys.append(X_topics[i].argmax())

        topic_summaries = []
        for i, topic_dist in enumerate(topic_word):
            # get topic keywords and append
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
            topic_summaries.append(' '.join(topic_words)) 

        dict_df = {'content':cleaned[:num_example], 
                  'topic_key':_model_keys[:num_example]}
        df = pd.DataFrame(data=dict_df)

        source = bp.ColumnDataSource(df)

        num_example = len(X_topics)

        # plot
        title = "[t-SNE visualization of LDA model trained on {} statements, " \
                "{} topics, thresholding at {} topic probability, ({} data " \
                "points and top {} words)".format(X_topics.shape[0], 
                                                  self.n_components, 
                                                  self.threshold, 
                                                  num_example, 
                                                  n_top_words)
        
        plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                             title=title,
                             tools="pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                             x_axis_type=None, y_axis_type=None, min_border=1)
        
        plot_lda.scatter(x=tsne_model[:, 0], y=tsne_model[:, 1],
                         color=colormap[_model_keys][:num_example])


        # randomly choose a text (within a topic) coordinate as the crucial words coordinate
        topic_coord = np.empty((X_topics.shape[1], 2)) * np.nan
        for topic_num in _model_keys:
            if not np.isnan(topic_coord).any():
                break
            topic_coord[topic_num] = tsne_model[_model_keys.index(topic_num)]
        
        # plot crucial words
        for i in range(X_topics.shape[1]):
            plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])
        
        # hover tools
        hover = plot_lda.select(dict(type=HoverTool))
        hover.tooltips = {"content": "@content - topic: @topic_key"}
  
        if len(filename) == 0:
            filename = "{}_statements_" \
                       "{}_topics_{}_topic_prob_threshold_" \
                       "{}_data_pts_and_top_{}_words".format(X_topics.shape[0], 
                                                             self.n_components, 
                                                             self.threshold, 
                                                             num_example, 
                                                             n_top_words)     

        full_path = save_folder_file(save_dir, filename, ext=ext)
        print('T-SNE html output saved to `{}`.\n'.format(full_path))
        
        # save the plot
        save(plot_lda, full_path)        



class WordCount(CleanText):
    """
    Deals with word frequency related output.
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

        for num, text in enumerate(self.data):
            count_list = Counter(text)
            if verbose:
                print('\ndocument #{}:\n {}\n'.format(num + 1, count_list))
                verbose -= 1
            for token_ in count_list:
                count[token_] += count_list[token_]
        return count             
                
    def count_words(self):
        '''
        Count the total number of words
        '''
        num = 0
        for text in self.data:
            phrase = text.split()
            num += len(phrase)
        return num       


class PlotWords(WordCount):
    '''
    Build interactive plots on the local server and word cloudplots
    '''
    def __init__(self, count_dict):
        '''
         `count_dict`: from WordCount
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
                        is_notebook = True,
                        **kwargs):
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
        fig = go.Figure(data=data_word, layout=layout, **kwargs)

        if is_notebook:
            iplot(fig, filename=plotname, image=image_format)

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
                         is_notebook=True,
                         **kwargs):
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
        
        wc.generate(self.text_cloud, **kwargs);
        plt.title(plot_title, fontsize=plot_fontsize);
        plt.margins(x = 0.5, y = 0.25);
        plt.axis('off');

        plt.imshow(wc, interpolation=interpolation);
        if is_notebook:
            plt.show();

        if save_plot:   
            if len(filename) == 0:
                filename = 'wordcloud_plot_'    

            full_path = save_folder_file(save_dir, filename, ext= image_format)
       
            print('Wordcloud plot saved to `{}`.\n'.format(full_path))
            # store to file
            wc.to_file(full_path);
            plt.savefig(full_path);



    
     


