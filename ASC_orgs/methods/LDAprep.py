import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel

from os import path, getcwd
import re
import warnings

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import pyLDAvis
import pyLDAvis.gensim as gensimvis

from helper.helper_funs import time_filename, save_folder_file

from tmtoolkit.topicmod.visualize import plot_eval_results
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod import tm_gensim



class GensimPrep:
    """
     Class to prepare data specifically for gensim models
    """
    def __init__(self, processed_data, cur_dir=getcwd()):

        self.data = processed_data 
        self.cur_dir = cur_dir

    def gensimDict(self, min_word_len=3,
                         prop_docs=0.8,
                         compact=True,
                         save_dict=True,
                         save_dir='corpus_data',
                         filename='',
                         keep_n = 0):
        '''
         `min_word_len`: int, remove words smaller than min_word_len (should already be done)
         `prop_docs`:    float (0 to 1), max proportion of docs a word can appear before being removed
         `compact`:      bool, Do we reset the index after some rows were deleted in preprocess?
         `save_dict`:    bool, Are we saving this object
         `save_dir`:     str, folder to save the dictionary, child of the current dir
                         will be created if it doesn't exists
         `filename`:     str, filename. If empty string, a new folder name will be created  
         `keep_n`:       int, maximum number of words to keep during filtering  
        '''

        dict_words = corpora.Dictionary(self.data)
        if keep_n == 0:
            keep_n = len(self.data) + 1000

        print('Removing words of less than {} characters, and ' \
                     'words present in at least {}% of documents\n'.format(
                                                                      min_word_len, prop_docs))
        dict_words.filter_extremes(no_below=min_word_len, no_above=prop_docs, keep_n=keep_n)
        if compact:
            dict_words.compactify()  # remove gaps in id sequence after words that were removed 
            print('Removing gaps in indices caused by preprocessing...\n')  

        if save_dict:   
            if len(filename) == 0:
                filename = 'Gensim_dict_Params_MWL{}_PD{}_'.format(min_word_len,
                                                                   prop_docs)     

            full_path = save_folder_file(save_dir, filename, ext='.dict')
            dict_words.save(full_path)  # store the dictionary for future reference        
            print('Saving gensim dictionary to {}\n'.format(full_path))

        return dict_words  
    

    def gensimBOW(self, gensim_dict,
                        save_matrix=True,
                        save_dir='corpus_data',
                        filename=''):
        '''
         Make a gensim Bag-of-Words representation matrix
        '''
        bow_corpus = [gensim_dict.doc2bow(text) for text in self.data]

        if save_matrix:   
            if len(filename) == 0:
                filename = 'BOWmat'                                                     

            full_path = save_folder_file(save_dir, filename, ext='.mm')
            corpora.MmCorpus.serialize(full_path, bow_corpus)  # store to disk, for later use
            print('Saving .mm matrix to {}\n'.format(full_path))

        return bow_corpus


    def loadBOW(self, filename, saved_dir='corpus_data'):  
        '''
         Load a saved gensim Bag-of-Words representation matrix
        ''' 
        full_path = path.join(saved_dir, filename) 
        bow_corpus = corpora.MmCorpus(full_path) 
        print('Loading BOW matrix from {}...'.format(full_path))

        return bow_corpus


    def printBOWfreq(self, gensim_dict, bow, num=30):
        '''
         print a gensim Bag-of-Words representation matrix in the form
              'for document number num, 'word1' appears 1 times, 'word2' appears 3 times, etc

         `num`: int, document number to view
        '''
        bow_doc_num = bow[num-1]
        if len(bow_doc_num) < 1:
            print('The {}th statement has no words after preprocess.\n'.format(num))
        else:    
            print('In the {}th statement, the word... \n'.format(num))
            for word in range(len(bow_doc_num)):
                print("\"{}\" appears {} time(s).".format(gensim_dict[bow_doc_num[word][0]], 
                                                          bow_doc_num[word][1]))

    def tfidf_trans(self, bow, normalize=True):
        '''
        Compute TF-IDF by multiplying a local component (term frequency) 
        with a global component (inverse document frequency), 
        and (if normalize = True) normalizing the resulting documents to unit length.

        `bow`:       gensim bow type, A Bag-of-Words corpus
        `normalize`: bool, Normalize document vectors to unit euclidean length? 
                          You can also inject your own function into normalize.
        '''
        
        tfidf_mod = models.TfidfModel(bow, normalize=normalize)
        corpus_tfidf = tfidf_mod[bow]

        return corpus_tfidf
                                         

class GenMod(GensimPrep):
    '''
     The basic class to run all LDA, LSI and HDP models
    '''
    def __init__(self, bow, 
                       gensim_dict):
        '''
        `bow`:          gensim Bag-of-Words representation matrix
        `gensim_dict`:  from GensimPrep.gensimDict()
        '''
        tfidf = False
        if type(bow) == gensim.interfaces.TransformedCorpus: # see if tfidf transformed
            tfidf = True
            
        self.tfidf = tfidf    
        self.bow = bow
        self.gensim_dict = gensim_dict

    def LDA(self, num_topics=10, 
                  update_every=1, 
                  chunksize=100, 
                  full_data_chunk=True,
                  iterations=10000,
                  passes=10,
                  eval_every=5,  
                  alpha='auto',
                  eta='auto',
                  decay=0.8,
                  minimum_probability = 0.05,
                  minimum_phi_value = 0.02,
                  per_word_topics=True,
                  print_params=True,
                  save_model=True,
                  save_dir='saved_models',
                  filename='',
                  random_state=919):
        '''
         `num_topics`:    int, Number of latent topics (clusters) extracted from training corpus (bow)
         `update_every`:  int, Number of chunks to process prior to moving 
                               onto the M step of EM.
         `chunksize`:     int, Number of documents to load into memory at a time 
                               and process E step of EM
         `full_data_chunk=`: bool, Overrides chunksize. Load all docs into memory at once?
         `iterations`:       int, Maximum number of training iterations through the corpus.
         `passes`:           int, Number of passes through the entire corpus for training
         `eval_every`:       int, the smaller the number, the finer grained is convergence plot
         `alpha='auto',      str, number of expected topics that expresses our a-priori belief 
                                  for the each topics' probability. 
                                  Choices: 'auto': Learns an asymmetric prior from the corpus.
                                           'asymmetric': Fixed normalized asymmetric prior of 1.0 / topicnum.

         `eta`:   prior on word probability, can be:
                          scalar for a symmetric prior over topic/word probability,
                          vector of length num_words for user defined prob for each word,
                          matrix (num_topics x num_words) to assign prob to word-topic combinations,
                          or str 'auto' to learn the asymmetric prior from the data.
         `decay`:               float, Number between (0.5, 1] how much past documents are forgotten when new document is seen    
         `minimum_probability`: float, Topics with a prob lower than this are filtered out.
         `minimum_phi_value`:   float, lower bound on the term probabilities (when `per_word_topics` = True)
         `per_word_topics`:     bool, sorts topics in descending order (from most likely topics for each word) 
         `print_params`:        bool, are the parameters printed?
         `save_model`:          bool, save model?
         `save_dir`:            str, folder to save the model, child of the current dir
                                     will be created if it doesn't exists
         `filename`:            str, filename. If empty string, a new folder name will be created 
         `random_state`:        int, seed to reproduce
        '''

        warnings.filterwarnings("ignore", category = DeprecationWarning) 

        if full_data_chunk:
            chuncksize = len(self.bow)

        lda_model = models.LdaModel(self.bow, 
                           id2word=self.gensim_dict, 
                           num_topics=num_topics,
                           update_every=update_every,
                           chunksize=chunksize,
                           iterations=iterations,
                           passes=passes,
                           alpha=alpha,
                           eta=eta,
                           decay=decay,
                           minimum_probability=minimum_probability,
                           minimum_phi_value=minimum_phi_value,
                           per_word_topics=per_word_topics,
                           eval_every=eval_every,
                           random_state=random_state)

        if print_params:
            print('Parameters used in model: ')
            model_pars = 'Number of topics: {}\nTFIDF transformation: {}\n'\
                         'Number of iterations: {}\nBatch size: {}\n' \
                         'Update every {} pass\nNumber of passes: {}\n' \
                         'Topic inference per word: {}\nAlpha: {}\n'\
                         'Eta: {}\nDecay: {}\nMinimum probability: {}\n' \
                         'Minimum_phi_value: {}\nEvaluate every: {}\n' \
                         'Random seed: {}\n'.format(num_topics,
                                                       self.tfidf,
                                                       iterations,
                                                       chunksize,
                                                       update_every,
                                                       passes,
                                                       per_word_topics,
                                                       alpha,
                                                       eta,
                                                       decay,
                                                       minimum_probability,
                                                       minimum_phi_value,
                                                       eval_every,
                                                       random_state)
            print(model_pars)    

        if save_model:
            if len(filename) == 0:
                filename = 'LDA_Params_NT{}_TFIDF{}'\
                           'Per_word_topic{}'.format(num_topics,
                                                            self.tfidf,
                                                            per_word_topics)                                                      
            
            full_path = save_folder_file(save_dir, filename, ext='.model', optional_folder='LDA')
            full_path_txt = save_folder_file(save_dir, filename, ext='.txt', optional_folder='LDA')
 
            
            print('Saving LDA model to: \n{}'.format(full_path))   
            lda_model.save(full_path) 

            f = open(full_path_txt,'w')  # write down corresponding parameters
            f.write(model_pars)
            f.close()

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
                filename = 'LSI_Params_NT{}_TFIDF{}_'.format(num_topics,
                                                                         self.tfidf)                                                      
            
            full_path = save_folder_file(save_dir, filename, ext='.model', optional_folder='LSI')

            lsi_model.save(full_path) 
            print('Saving LSI model to: \n{}\n'.format(full_path))  

        return(lsi_model)              


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
                filename = 'HDP_Params_TFIDF{}_'.format(self.tfidf)                                                      
            
            full_path = save_folder_file(save_dir, filename, ext='.model', optional_folder='HDP')
            hdp_model.save(full_path) 
            print('Saving HDP model to: \n{}\n'.format(full_path))  

        return hdp_model
   

class Diagnostics(GenMod):
    '''
     Use for diagnostic plots, validation and parameter tuning
    '''
    def __init__(self, gensim_dict,
                       bow, 
                       text):
        '''
         `gensim_dict` : Gensim dictionary (from GensimPrep)
         `bow` :         Gensim bow corpus (from GensimPrep)
         `texts`:        List of str, each element is a document
        '''
        self.bow = bow
        self.text = text
        self.gensim_dict = gensim_dict

    def compare_scores(self, max_num_topics = 20, 
                             start = 2, 
                             step = 2, 
                             etas = 'auto',
                             decays = 0.7,
                             random_state=919,
                             save_output=True,
                             save_dir='model_validation',
                             **kwargs):
        """
        Compute c_v coherence and perplexity for various number of topics
    
          `max_num_topics` :  int, Max number of topics to test
          `start`:            int, Min number of topics to test
          `step`:             int, increased by stepsize
          `save_output`:      bool, save output?
          `save_dir`:         str, folder to save the results, child of the current dir
                                  will be created if it doesn't exists 
          `random_state`:        int, seed to reproduce

    
        Returns:
          `model_list` :  list of LDA topic models used for tuning
          `score_dict`,:    dict with {`key`: value}: 
                          `coherence_values` : Coherence values corresponding to the LDA
                                               model with respective number of topics
                          `perplexity_values`: kl-divergence between theoretical and empirical distribution                         
        `score_df`,:    DataFrame with a column for each tuning parameters, coherence and perplexity  

        to do: implement with kwargs
        """
        warnings.filterwarnings("ignore", category = DeprecationWarning) 

        self.start = start
        self.max_num_topics = max_num_topics
        self.step = step
        

        model_list = []
        eta_list = []
        decay_list = []
        num_topics_list = []
        p_score = []
        c_score = []    
        score_dict = {}    

        print('\nTesting topics {} to {} for:\n'.format(start,
                                                        (max_num_topics - step)))


        for eta in etas:
            for decay in decays:

                print('\n {} eta and {} decay...\n'.format(eta,
                                                           decay))
                for num_topics in range(start, max_num_topics, step):
                    params = "topics{}_eta{}_decay{}".format(num_topics,
                                                             eta,
                                                             decay)
                    
                    model = self.LDA(print_params = False, 
                                     num_topics = num_topics, 
                                     eta = eta, 
                                     decay = decay,
                                     save_model=False,
                                     random_state=random_state)
    
                    model_list.append(model)
                    
                    coherencemodel = CoherenceModel(model=model, 
                                                    corpus=self.bow,
                                                    texts=self.text,
                                                    coherence='c_v')
    
                    coherent = coherencemodel.get_coherence()
                    perplex = model.log_perplexity(self.bow) 
    
                    eta_list.append(eta)
                    decay_list.append(decay)
                    num_topics_list.append(num_topics)
                    p_score.append(coherent)
                    c_score.append(perplex)

        score_df = pd.DataFrame({'eta':eta_list, 'decay':decay_list, 'topic_num':num_topics_list,
                                 'coherence':c_score, 'perplexity':p_score})        
        
        score_df.replace(to_replace=[None], value='none', inplace=True)


        if save_output:  
            filename = 'Coherence_Perplexity_from{}_to{}_by{}'.format(start,
                                                                      max_num_topics,
                                                                      step)                                                                                                                                   
            full_path = save_folder_file(save_dir, filename, ext='.csv', optional_folder='scores')
            score_df.to_csv(full_path, index=False)

        score_dict['perplexity'] = p_score
        score_dict['coherence'] = c_score
        
        return model_list, score_df, score_dict



    def score_plot(self, tuning_df,
                         save_plot=True,
                         save_dir='model_validation',
                         ext='.pdf', 
                         size=(12, 5),
                         is_notebook=True,
                         tune_params=['eta', 'decay'],
                         score = ['coherence', 'perplexity'],
                         pref = ['higher', 'lower']):
        
        fig, axes = plt.subplots(1, len(score), sharex=True, figsize=size);
        
        # create a color palette
        palette = plt.get_cmap('Set1');
        
        params = []
        
        for param1_name, param1_df in tuning_df.groupby(tune_params[0]):
            for param2_name, param2_df in param1_df.groupby(tune_params[1]):
        
                for i, ax in enumerate(axes.flatten()):
                    ax.plot(param2_df["topic_num"], param2_df[score[i]]);
                    
                    ax.set_xlabel('Number of Topics', fontsize=15);
                    ax.set_ylabel('{}'.format(score[i]), fontsize=15);
        
                    ax.spines[ "top" ].set_visible( False );
                    ax.spines[ "right" ].set_visible( False );
            
                    ax.tick_params(axis='both', which='major', labelsize=15 );
                    ax.set_title('{} ({} is better)'.format(score[i], pref[i]));
                    fig.text( 0.5, -0.03,
                              'Note the different y axes',
                              ha='center', va='center',
                              fontsize = 14);
                    ax.grid(True);
                    params.append('{}: {}, {}: {}'.format(tune_params[0], param1_name, 
                                                          tune_params[1], param2_name));
                    
        axes[0].legend( params,
                        loc='upper center',
                        bbox_to_anchor=(1.1, 1.35),
                        shadow=True,
                        ncol=4 );
        
        
        plt.suptitle( 'Validation score plots', fontsize = 20 );
        

        if is_notebook:
            plt.show();    

        if save_plot:
            filename = 'validation_from{}_to{}_by{}_'.format(self.start,
                                                             self.max_num_topics,
                                                             self.step)                                                      
        
        full_path = save_folder_file(save_dir, filename, 
                                               ext=ext, 
                                               optional_folder='CV_score_plots')
      
        plt.savefig(full_path)            
    

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
            



    def convergence_plot(self, log_file,
                               eval_every=5,
                               save_plot=True,
                               save_dir='model_validation',
                               filename='',
                               ext='.pdf', 
                               size=(12, 9),
                               show_plot=True):

        pattern = re.compile(r'(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity')
        matches = [pattern.findall(log) for log in open(log_file)]
        matches_pos = [match for match in matches if len(match) > 0]
        scores = [pos[0] for pos in matches_pos]
        perplexity = [float(score[1]) for score in scores]
        likelihood = [float(score[0]) for score in scores]
        iterations = list(range(0, len(scores)*eval_every, eval_every))
        plt.figure(figsize=size)
        plt.plot(iterations, likelihood)
        plt.ylabel("Log-Likelihood")
        plt.xlabel("Iteration")
        plt.title("Topic Model Convergence")
        plt.grid()  

        if save_plot:
            filename = 'Likelihood-based_convergence_plot_'                                                     
            full_path = save_folder_file(save_dir, filename, ext=ext, optional_folder='convergence_plots')
      
            plt.savefig(full_path)
        
        if show_plot==True:
            plt.show() 
        else:
            plt.close()   



    def toolkit_cv_plot(self, varying_params, 
                         constant_params,
                         save_plot=True,
                         save_dir='model_validation',
                         filename='',
                         ext='.pdf', 
                         size=(20, 15)):

        warnings.filterwarnings("ignore", category = UserWarning)   

        print('evaluating {} topic models'.format(len(varying_params)))
        eval_results = tm_gensim.evaluate_topic_models((self.gensim_dict, 
                                                        self.bow), 
                                                        varying_params, 
                                                        constant_params,
                                               coherence_gensim_texts=self.text)  

        results_by_n_topics = results_by_parameter(eval_results, 'num_topics')
        plot_eval_results(results_by_n_topics, xaxislabel='num topics',
                  title='Evaluation results', figsize=size);

        if save_plot:
            filename = 'tmtoolkit_CV_'                                                     
            full_path = save_folder_file(save_dir, filename, ext=ext, 
                                         optional_folder='convergence_plots')
      
            plt.savefig(full_path)
        return(results_by_n_topics)    




    def LDAvis(self, model,
                     save_plot=True,
                     save_dir='results',
                     filename='',
                     ext='.html',
                     show_plot=True,
                     is_notebook=True):

        print('Rendering visualization...')

  

        vis = gensimvis.prepare(model, self.bow, self.gensim_dict, sort_topics=False)
        
        if save_plot:
            if len(filename) == 0:
                filename = 'LDAvis_plot_'                                                     
                full_path = save_folder_file(save_dir, filename, ext=ext, 
                                             optional_folder='LDAvis_plots')
            if ext == '.html':
                pyLDAvis.save_html(vis, full_path)
            else:
                print('File extension not supported')  
        
        if show_plot:              
            if is_notebook:
                return(vis)  # show          
            else:
                pyLDAvis.show(vis)  





class ModelResults(GenMod):
    def __init__(self, model, 
                       corpus, 
                       texts):

        self.model = model
        self.corpus = corpus
        self.texts = pd.DataFrame(data={'Mission': texts})

    def format_topics_sentences(self,
                                save_output=True,
                                save_dir='results',
                                filename=''):
        '''
        Find the dominant topic in each statement
        Topic with highest percentage contribution in each statement
        '''


        # Init output
        sent_topics_df = pd.DataFrame()
    
        # Get main topic in each document
        for i, row in enumerate(self.model[self.corpus]):
            row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                                                 pd.Series(
                                                       [int(topic_num), 
                                                        round(prop_topic,4), 
                                                        topic_keywords]), 
                                                        ignore_index=True)
                else:
                    break  # break to only get the top topic

        sent_topics_df.columns = ['Dominant_Topic', 'Percent_Contribution', 'Important_Keywords']
    
        # Add original text to the end of the output
        sent_topics = pd.concat([sent_topics_df, self.texts], axis=1)

        topics_df = sent_topics.reset_index()

        if save_output:  
            if len(filename) == 0: 
                filename = 'dominant_topic_per_text_'                                                     
          
            full_path = save_folder_file(save_dir, filename, ext='.csv')
            print('Saving the table to: {}'.format(full_path))
            topics_df.to_csv(full_path, index=False)      

        return topics_df
    

    def top_texts_per_topic(self, df_dominant_topic,
                                   save_output=True,
                                   save_dir='results',
                                   filename=''):

        '''
        Most representative statements for each topic
        Helps to make sense of each topic (for labeling)
        '''

        sent_topics_sorteddf = pd.DataFrame()
        
        sent_topics_outdf_grpd = df_dominant_topic.groupby('Dominant_Topic')
        
        for i, grp in sent_topics_outdf_grpd:
            sent_topics_sorteddf = pd.concat([sent_topics_sorteddf, 
                                              grp.sort_values(['Percent_Contribution'], 
                                              ascending=[0]).head(1)], 
                                              axis=0)

        sent_topics_sorteddf.reset_index(drop=True, inplace=True)

        if save_output:  
            if len(filename) == 0: 
                filename = 'top_texts_per_topic'                                                     
          
            full_path = save_folder_file(save_dir, filename, ext='.csv')
            print('Saving the table to: {}'.format(full_path))
            sent_topics_sorteddf.to_csv(full_path, index=False)            
  
        return sent_topics_sorteddf


    def print_top_sentences_for_topic(self, sent_topics_sorteddf, colname='Mission', topic_num=-1,
                                            contrib = 'Percent_Contribution'):
        top_phrase = [" ".join(re.findall(r"[a-zA-Z.,!?]{1,}", 
                                          phrases)) for phrases in sent_topics_sorteddf[colname]]  
        if topic_num >= 0:
            print('Top entry for topic {} (with {} contribution): \n{}\n'.format(topic_num, 
                                                                                 round(sent_topics_sorteddf[contrib][topic_num], 4),
                                                                                 top_phrase[topic_num]))                                            
        else:
            print('Top entry for all topics {}: \n{}\n'.format(topic_num, top_phrase))                                            


    def topic_distribution(self, df_dominant_topic,
                                 top_text_topic,
                                 save_output=True,
                                 save_dir='results',
                                 filename=''):
        '''
        Topic distribution across statements
        Volume and distribution of topics to see how spread out it is
        '''
        # Number of Documents for Each Topic
        topic_counts = df_dominant_topic['Dominant_Topic'].value_counts()

        # Percentage of Documents for Each Topic
        topic_contribution = round(topic_counts/topic_counts.sum(), 5)
        
        # Topic Number and Keywords
        topic_num_keywords = top_text_topic[['Dominant_Topic', 'Important_Keywords']]
        
        # Concatenate Column wise
        df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], 
                                        axis=1, 
                                        ignore_index=True)

        df_dominant_topics.fillna(0, inplace=True)
        # Change Column names
        df_dominant_topics.columns = ['Dominant_Topic', 
                                      'Important_Keywords', 
                                      'Num_Documents', 
                                      'Perc_Documents']

        df_dominant_topics.reset_index()

        if save_output:  
            if len(filename) == 0: 
                filename = 'doc_distribution_in_topics'                                                     
          
            full_path = save_folder_file(save_dir, filename, ext='.csv')
            print('Saving the table to: {}'.format(full_path))
            df_dominant_topics.to_csv(full_path, index=False)            
  
        return df_dominant_topics


    def per_word_contrib(self, num_words=8):

        for idx, topic in self.model.show_topics(formatted=False, num_words=num_words):
            print('Topic: {}'.format(idx + 1))
            print('\tWord,\t Probability')
            print('\t----     -----------\n')

            for word, prob in topic:
                print('\t\"{}\":  {}'.format(word, round(float(prob), 3)))
            print('\t--------------------\n')

        
    def topic_keywords(self, num_words=8):
        
        for idx, topic in self.model.show_topics(formatted=False, num_words=num_words):
            print('Topic: {}'.format(idx))
            top = []
            for word, _ in topic:
                top.append(word)
            print(top)    
            print('\t---------------\n')

              






