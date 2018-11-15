import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel

from os import path, getcwd
import re
from pathlib import Path
from datetime import datetime
import errno

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from helper.helper_funs import time_filename, save_folder_file


class GensimPrep:
    def __init__(self, processed_data, cur_dir=getcwd()):

        self.data = processed_data 
        self.cur_dir = cur_dir
        self.bow = self.gensimBOW(save_matrix=False)

    def gensimDict(self, remove_extreme=True,
                         min_word_len=3,
                         prop_docs=0.7,
                         compact=True,
                         save_dict=True,
                         save_dir='corpus_data',
                         filename='',
                         keep_n = 0):
        
            
        dict_words = corpora.Dictionary(self.data)
        if keep_n == 0:
            keep_n = len(self.data) + 1000
        if remove_extreme:
            print('Removing words of less than {} characters, and ' \
                         'words present in at least {}% of documents\n'.format(
                                                                      min_word_len, prop_docs))
            dict_words.filter_extremes(no_below=min_word_len, no_above=prop_docs, keep_n=keep_n)
        if compact:
            dict_words.compactify()  # remove gaps in id sequence after words that were removed 
            print('Removing gaps in indices caused by preprocessing...\n')  

        if save_dict:   
            if len(filename) == 0:
                filename = 'Gensim_dict_Params_RE{}_MWL{}_PD{}_'.format(remove_extreme,
                                                                           min_word_len,
                                                                           prop_docs)     

            full_path = save_folder_file(save_dir, filename, ext='.dict')
            dict_words.save(full_path)  # store the dictionary for future reference        
            print('Saving gensim dictionary to {}\n'.format(full_path))

        return dict_words  
    
    def gensimBOW(self, save_matrix=True,
                        save_dir='corpus_data',
                        filename=''):

        gensim_dict = self.gensimDict(save_dict=False)   
        bow_corpus = [gensim_dict.doc2bow(text) for text in self.data]

        if save_matrix:   
            if len(filename) == 0:
                filename = 'BOWmat'                                                     

            full_path = save_folder_file(save_dir, filename, ext='.mm')

            corpora.MmCorpus.serialize(full_path, bow_corpus)  # store to disk, for later use
            print('Saving .mm matrix to {}\n'.format(full_path))
        return bow_corpus

    def loadBOW(self, filename, saved_dir='corpus_data'):   
        full_path = path.join(saved_dir, filename) 
        bow_corpus = corpora.MmCorpus(full_path) 
        print('Loading BOW matrix from {}...'.format(full_path))
        return bow_corpus

    def printBOWfreq(self,gensim_dict, num=30):
        bow_doc_num = self.bow[num-1]
        if len(bow_doc_num) < 1:
            print('The {}th statement has no words after preprocess.\n'.format(num))
        else:    
            print('In the {}th statement, the word... \n'.format(num))
            for word in range(len(bow_doc_num)):
                print("\"{}\" appears {} time(s).".format(gensim_dict[bow_doc_num[word][0]], 
                                                          bow_doc_num[word][1]))

    def tfidf_trans(self):
        '''
        idf = offset + log_2(number_of_docs/doc_frequency)
        '''
        tfidf_mod = models.TfidfModel(self.bow)
        corpus_tfidf = tfidf_mod[self.bow]
        return corpus_tfidf
                                         


class GenMod(GensimPrep):
    def __init__(self, bow, 
                       gensim_dict):
        tfidf = False
        if type(bow) == gensim.interfaces.TransformedCorpus:
            tfidf = True
        self.tfidf = tfidf    
        self.bow = bow
        self.gensim_dict = gensim_dict

    def LDA(self, num_topics=10,
                  update_every=1, # number of chunks to process prior to moving onto the M step of EM
                  chunksize=50, # Number of documents to load into memory at a time and process E step of EM
                  iterations=10000,
                  passes=10, # Number of passes through the entire corpus
                  eval_every=5,  # lower this value is the better resolution the plot will have.
                  alpha='auto',
                  per_word_topics=True,
                  print_params=True,
                  save_model=True,
                  save_dir='saved_models',
                  filename=''):

        lda_model = models.LdaModel(self.bow, 
                           id2word=self.gensim_dict, 
                           num_topics=num_topics,
                           update_every=update_every,
                           chunksize=chunksize,
                           iterations=iterations,
                           passes=passes,
                           alpha=alpha,
                           per_word_topics=per_word_topics,
                           eval_every=eval_every)

        print('Running LDA model...\n')

        if print_params:
            print('Parameters used in model: ')
            print('Number of topics: {}\nTFIDF transformation: {}\n'\
                         'Number of iterations: {}\nBatch size: {}\n' \
                         'Update every {} pass\nNumber of passes: {}\n' \
                         'Topic inference per word: {}\nAlpha: {}\n'\
                         'Evaluate every: {}\n'.format(num_topics,
                                                       self.tfidf,
                                                       iterations,
                                                       chunksize,
                                                       update_every,
                                                       passes,
                                                       per_word_topics,
                                                       alpha,
                                                       eval_every))
        if save_model:
            if len(filename) == 0:
                filename = 'LDA_Params_NT{}_TFIDF{}_Iter{}_Chunk{}_Update{}_Pass{}_'\
                           'Per_word_topic{}_Alpha{}_Eval{}_'.format(num_topics,
                                                                               self.tfidf,
                                                                               iterations,
                                                                               chunksize,
                                                                               update_every,
                                                                               passes,
                                                                               per_word_topics,
                                                                               alpha,
                                                                               eval_every)                                                      
            
            full_path = save_folder_file(save_dir, filename, ext='.model', optional_folder='LDA')


            print('Saving LDA model to: \n{}'.format(full_path))   
            lda_model.save(full_path) 

        return lda_model


    def print_model(self, model_name):
        '''    
        For printing LDA, LSI output
        '''
        for idx, topic in model_name.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx+1, topic))
            

    def loadLDA(self, filename, saved_dir='saved_models'):   
        '''
        Load a potentially pretrained model from disk.
        '''
        full_path = path.join(saved_dir, filename) 
        lda_model = LdaModel.load(full_path) 
        print('Pretrained LDA model loaded from: {}'.format(self.filename))
        return lda_model   

    def updateLDA(self, new_corpus, unseen_doc):    
        lda.update(new_corpus)
        new_lda_model = lda[unseen_doc]
        print('Updating LDA model with new documents...')
        return new_lda_model

    def LSI(self, num_topics=10, 
                  print_params=True, 
                  save_model=True,
                  save_dir='saved_models',
                  filename=''):
        lsi_model = models.LsiModel(self.bow, 
                                  id2word=self.gensim_dict, 
                                  num_topics=num_topics)
        print('Running LSI model...\n')

        if print_params:
            print('Parameters used in model:')
            print('Number of topics: {}\nTFIDF transformation: {}\n'.format(num_topics,
                                                                          self.tfidf))

        if save_model:  
            if len(filename) == 0:
                time, date = time_filename() 
                filename = 'LSI_Params_NT{}_TFIDF{}_'.format(num_topics,
                                                                         self.tfidf)                                                      
            
            full_path = save_folder_file(save_dir, filename, ext='.model', optional_folder='LSI')

            lsi_model.save(full_path) 
            print('Saving LSI model to: \n{}\n'.format(full_path))             


    def HDP(self, print_params=True, 
                  save_model=True,
                  save_dir='saved_models',
                  filename=''):

        hdp_model = models.HdpModel(self.bow, 
                                  id2word=self.gensim_dict)
        print('Inferring number of topics with Hierarchical Dirichlet Process...\n')

        if print_params:
            print('Parameters used in model:')
            print('TFIDF transformation: {}\n'.format(self.tfidf))

        if save_model:  
            if len(filename) == 0:
                time, date = time_filename() 
                filename = 'HDP_Params_TFIDF{}_'.format(self.tfidf)                                                      
            
            full_path = save_folder_file(save_dir, filename, ext='.model', optional_folder='HDP')
            hdp_model.save(full_path) 
            print('Saving HDP model to: \n{}\n'.format(full_path))  

   

class Diagnostics(GenMod):
    def __init__(self, gensim_dict,
                       bow, 
                       text):
        self.bow = bow
        self.text = text
        self.gensim_dict = gensim_dict
        # self.models, self.coherence, self.perplexity = self.compare_scores(max_num_topics=20, 
        #                                                                    start=2, 
        #                                                                    step=2)

    def compare_scores(self, max_num_topics=20, 
                             start=2, 
                             step=2, 
                             path= getcwd() + 'results/'):
        """
        Compute c_v coherence for various number of topics
    
        Parameters:
            dictionary : Gensim dictionary
            corpus : Gensim corpus
            texts : List of input texts
            max_num_topics : Max num of topics
    
        Returns:
            model_list : List of LDA topic models
            coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        self.start = start
        self.max_num_topics = max_num_topics
        self.step = step

        coherence_values = []
        perplexity_values = []
        model_list = []

        for num_topics in range(start, max_num_topics, step):
            print('Testing with {} topics...\n'.format(num_topics))
            model = self.LDA(print_params=False, save_model=False)
            model_list.append(model)
            
            coherencemodel = CoherenceModel(model=model, 
                                            corpus=self.bow,
                                            texts=self.text,
                                            coherence='c_v')

            coherence_values.append(coherencemodel.get_coherence())
            perplexity_values.append(model.log_perplexity(self.bow)) 
            res_dict = {'coherence': coherence_values, 'perplexity': perplexity_values}   
    
        return model_list, res_dict


    def score_plot(self, res_dict,
                         plot_save=True,
                         save_file='results/visualization/',
                         size=(12, 9)):

        x = np.arange(self.start, self.max_num_topics, self.step)

        for key, value in res_dict.items():

            ytitle = key + ' score'
            title = ytitle + ' for Tuning Topic Number'
            
            plt.figure(figsize=size)
            plt.plot(x, value)
            plt.xlabel("Number of Topics")
            plt.ylabel(ytitle)
            plt.title("Topic Model Scores")
            plt.grid() 
            plt.show()    


        if plot_save:
            filename = '{}_from{}_to{}_by{}.pdf'.format(plot_save, start, max_num_topics, step)
            plt.savefig(save_file)    
    
    def print_scores(self, res_dict):
        x = np.arange(self.start, self.max_num_topics, self.step)
        for key, val in res_dict.items():
            print('{} values:\n'.format(key))
            for top, v in zip(x, val):
                print("{} topics: {}".format( top, round(v, 4)))
            print('\t***\n')    

    def print_model(self, model_name):
        '''    
        For printing optimal model output
        '''
        for idx, topic in model_name.print_topics(-1):
            print('Topic: {} \nWords: \n{}\n'.format(idx+1, topic))
            



    def convergence_plot(self, log_file='log/gensim.log',
                               eval_every=5,
                               save_plot=True, 
                               save_file='results/visualization/convergence_likelihood.pdf',
                               size=(15, 12),
                               show_plot=True):

        pattern = re.compile(r'(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity')
        matches = [pattern.findall(log) for log in open(log_file)]
        matches_pos = [match for match in matches if len(match) > 0]
        tuples = [tup[0] for tup in matches_pos]
        perplexity = [float(tup[1]) for tup in tuples]
        likelihood = [float(tup[0]) for tup in tuples]
        iterations = list(range(0, len(tuples)*eval_every, eval_every))
        plt.figure(figsize=size)
        plt.plot(iterations, likelihood)
        plt.ylabel("Log-Likelihood")
        plt.xlabel("Iteration")
        plt.title("Topic Model Convergence")
        plt.grid()  

        if save_plot:
            plt.savefig(save_file)
        
        if show_plot==True:
            plt.show() 
        else:
            plt.close()   

class ModelResults(GenMod):
    def __init__(self, model, corpus, texts):
        self.model = model
        self.corpus = corpus
        self.texts = texts

    def format_topics_sentences(self):
        # Init output
        sent_topics_df = pd.DataFrame()
    
        # Get main topic in each document
        for i, row in enumerate(self.model[self.corpus]):
            row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic, for more topics increase range
                    wp = self.model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), 
                                                                      round(prop_topic, 4), 
                                                                      topic_keywords]), 
                                                                      ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    
        # Add original text to the end of the output
        sent_topics_df = pd.concat([sent_topics_df, self.texts], axis=1)
        # Format
        dominant_topic = sent_topics_df.reset_index()
        dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']


        return dominant_topic               

    def top5_texts_per_topic(self):

        # Group top 5 sentences under each topic
        topics_sorted = pd.DataFrame()
        
        sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
        
        for i, grp in sent_topics_outdf_grpd:
            topics_sorted = pd.concat([topics_sorted, 
                                       grp.sort_values(['Perc_Contribution'], 
                                       ascending=[0]).head(1)], axis=0)
        
        # Reset Index    
        topics_sorted.reset_index(drop=True, inplace=True)
        
        # Format
        topics_sorted.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
        
        return topics_sorted
        
        





