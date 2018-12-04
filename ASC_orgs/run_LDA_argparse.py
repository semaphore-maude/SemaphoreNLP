
import pandas as pd
import numpy as np

import random
import logging
import warnings
warnings.filterwarnings("ignore")
import argparse

from itertools import product

from nltk.corpus import stopwords

# local modules
from helper.helper_funs import time_filename, save_folder_file, logged
from helper.word_preprocess import DataLoader, CleanText, WordCount , PlotWords
from methods.LDAprep import GensimPrep, GenMod, Diagnostics, ModelResults 


def run_LDA(args, seed_num=10019):
    '''
    Full model, plots, validation and diagnostics for Topic Modelling.
    '''
    
    # writes a log file for debugging and plotting convergence
    # creates a `log` folder and timestamped filename if necessary
    if args.logged:

        full_path_log = logged()


    
        logging.basicConfig(filename=full_path_log, 
                          format='%(asctime)s : %(levelname)s : %(message)s', 
                          level=logging.INFO)   



    random.seed(seed_num)

    start, max_num_topics, step, eval_every = args.tuneLDA

    col = args.col

    stop = set(stopwords.words(args.stopwords))

    stop.update(args.stop_update)
    stop.update(args.add_stop_update)

    data_file = args.data_file
    data_dir = args.data_dir

    stemmer = args.stemmer

    seed = 919
    # small params to be set
    verbose = 1

    # top n most frequent words to plot
    top_n = 60

    # Loading data and cleaning out accents and remove and replace wrong characters 
    arts_data = DataLoader(filename=data_file, 
                           data_folder=data_dir,
                           colname=col, # column(s) where weird chars should be removed
                           rm_NAs=False,
                           removed_language=['french']) # list of languages to remove (lowercase)
    
    donnee = arts_data.data_loader()  


    clean = CleanText(donnee,
                      stop=stop,
                      stemmer=stemmer,
                      remove_urls=True)
    


    # fully cleaned data, stemmed and lemmatized 
    # all lower case and URLs removed, words with 1 or 2 chars also removed
    processed_data = clean.preprocess()


    # pd_data will be used later, for using the unstemmed rendition of the text 
    # Note: column name is set here, but should be changed if there are multiple columns
    pd_data = pd.DataFrame({'Mission': donnee}) 

    wp = WordCount(processed_data)
    # verbose number is just the no. of documents to show in output
    word_count = wp.dict_count(verbose=verbose) 


    print('These are the 10 most common terms:\n{}\n'.format(word_count.most_common(10)))
    # print('Some of the 10 least common (to check for oddities):\n{}\n'.format(word_count.most_common()[:-10-1:-1]))
    


    raw_count = WordCount(donnee).count_words()
    # Note: list() below may need to be removed, depending on python version 
    num_words = sum(list(word_count.values())) 
    
    
    print("There are {} words in the combination of all mission statements.".format(raw_count))
    print("There are {} words in all statements given (after preprocess).".format(num_words))

    print(list(word_count.values()))

    # Preparing word plots
    view_data = PlotWords(word_count)
    
    # Frequency plot
    view_data.freq_plot(top_n=top_n, 
                        save_plot=False,
                        is_notebook=False) # top_n number of top words to show

    # # wordcloud plot
    # view_data.cloud_plot(size=(15,12), 
    #                      max_font_size=60,
    #                      min_font_size=5,
    #                      is_notebook=False);


    # Prepare for Topic Modelling
    gp = GensimPrep(processed_data)

    lda_dict = gp.gensimDict()   # Make a gensim dictionary and tidy up words
    lda_id = lda_dict.token2id   # Make a {'word': id} dict

    # Make Bag of Word representation: list of (doc) lists of tuples (id, frequency)
    bow_rep = gp.gensimBOW(lda_dict)     

    # print info
    print('Number of words processed: {}\n'.format(lda_dict.num_pos))
    print('Number of unique words: {}\n'.format(max(lda_id.values())))


    # transform and topic model
    tfidfTrans = gp.tfidf_trans(bow_rep) 

    get_model = GenMod(bow_rep, lda_dict)  # prep regular LDA
    get_tfidf = GenMod(tfidfTrans, lda_dict)  # prep LDA with tfidf transformation
    
    lda_model = get_model.LDA(eval_every=eval_every,
                              random_state=seed) # run LDA
    lsi_model = get_model.LSI()                      # run LSI   
    hdp_model = get_model.HDP()                      # run HDP for topic number inference

    # prep diagnostics
    diag = Diagnostics(lda_dict,
                       bow_rep, 
                       processed_data)

    if args.logged == True:
        diag.convergence_plot(show_plot=False, 
                              eval_every=eval_every, 
                              log_file=full_path_log)

    # run all given parameters permutations with gensim metrics (naive coherence, perplexity)
    model_list, res_dict, all_params = diag.compare_scores(max_num_topics=max_num_topics, 
                                               start=start, 
                                               step=step,
                                               decays = np.arange(0.6, 1.2, 0.2),
                                               etas = ['auto'],
                                               random_state=seed)
    

    print(res_dict)

    for key, value in res_dict.items():
        print('key: {}'.format(key))
        print('value: {}'.format(value))

    # diag.score_plot(res_dict=res_dict, 
    #                 save_plot=True, 
    #                 size=(10, 8),
    #                 is_notebook=False)     # plot coherence/perplexity vs number of topics

    diag.print_scores(res_dict)       # print results

    # best model in terms of coherence
    max_idx = res_dict['coherence'].index(max(res_dict['coherence']))
    
    # Select the best model and print the topics
    optimal_model = model_list[max_idx]   

    diag.print_model(optimal_model) 


    # evaluate topic models with different parameters
    const_params = dict(update_every=0, passes=10, alpha='auto', decay=0.6, random_state=seed)
    d = {'num_topics': list(range(5,12,1))}
    varying_params = [dict(zip(d, v)) for v in product(*d.values())]
    
    tmtool_diag = diag.toolkit_cv_plot(varying_params, const_params, size=(30, 15))   

    # pyLDAvis interactive plots
    diag.LDAvis(optimal_model, 
                show_plot=True,
                is_notebook=False) # clustering/lower dimensional proj of words/topics


    # get optimal model result tables and plots
    mod_res = ModelResults(optimal_model, bow_rep, donnee)


    df_dominant_topic = mod_res.format_topics_sentences()
    reduced_dom_topic = pd.concat([df_dominant_topic.Dominant_Topic, 
                                   df_dominant_topic.Percent_Contribution, 
                                   df_dominant_topic.Topic_Keywords], axis = 1)
    
    reduced_dom_topic.columns = ['Dominant_Topic', 'Percent_Contribution', 'Topic_Keywords']
    
    print(reduced_dom_topic)

    top_text_tab = mod_res.top_texts_per_topic(df_dominant_topic)
    print(top_text_tab) 

    # print the top mission statements for topic 2
    mod_res.print_top_sentences_for_topic(top_text_tab, topic_num=2)   

    # see (num_words) words with most contribution per topic
    mod_res.per_word_contrib(num_words=5)  

    # see distribution of words for each topic
    top_dist = mod_res.topic_distribution(df_dominant_topic, top_text_tab)
    print(top_dist)




def main(): 
    # Parse CLI arguments 
    parser = argparse.ArgumentParser(
                        description = "Set LDA Options for "\
                                      "Topic Modelling and Dimension Reduction.",
                        epilog = "Default Arguments Are Set.", 
                        prefix_chars = "-",
                        add_help = True,
                        allow_abbrev = True)
              
    # parser.print_help() # uncomment to print

    '''
    Args: 
        Command-Line flag name
        Number of arguments in the list
        Data type
        Text that will show up in CL for help with flag --help
    '''
    
    # Writes a log file
    parser.add_argument(
        '-log', '--logged', 
        nargs = 1, 
        default = True, 
        type = bool, 
        help = "True/False: Produces a log file.\n"\
               "NOTE: Must be set to True for convergence_plot method "\
               "in Diagnostics class, `LDAprep` module.")


    # LDA tuning parameters 
    parser.add_argument(
        '-tl', '--tuneLDA', 
        nargs = 1, 
        default = [8, 12, 1, 5], 
        type = list, 
        help = "Tuning Parameters for Number of Topics:\n"\
               "set as [min_number, max_number, by, eval_every], "\
               "where number is the number of topics, by is the skipping step, "\
               "eval_every is the granularity of plot (smaller number = more precise plot)")  


    parser.add_argument(
        '-sw', '--stopwords',  
        default = ["english", "french"], 
        type = str, 
        choices = ['danish', 'dutch', 'english', 
                   'finnish', 'french', 'german', 
                   'hungarian', 'italian', 'norwegian', 
                   'portuguese', 'russian', 'spanish', 
                   'swedish', 'turkish'],
        help = "Languages used to remove words deemed unimportant by nltk.corpus.\n" \
               "NOTE: all letters should be lower case.")  



    parser.add_argument(
        '-su', '--stop_update',  
        default = ['art', 
                   'arts', 
                   'artist', 
                   'artists', 
                   'community', 
                   'use', 
                   'make', 
                   'organization'], 
        type = str, 
        help = "Hand picked words to remove words too universal or vague.\n" \
               "NOTE: all letters should be lower case.") 


    parser.add_argument(
        '-asu', '--add_stop_update',  
        default = [""], 
        type = str, 
        help = "Words to append to the `stop_update` list of strings.\n" \
               "NOTE: all letters should be lower case.")  


    parser.add_argument(
        '-rm_l', '--removed_language',  
        default = ["french"], 
        type = str, 
        help = "List of language(s) to remove. "\
               "Deletes records containing this language, " \
               "specified in the `Languages` column of the dataset.")     


    parser.add_argument(
        '-df', '--data_file',  
        default = "artbridges_profiles.csv", 
        type = str, 
        help = "Name of csv file containing text data of interest.")  


    parser.add_argument(
        '-dd', '--data_dir',  
        default = "data", 
        type = str, 
        help = "Name of folder containing data.\n" \
               "NOTE: Working directory must be parent directory.")  


    parser.add_argument(
        '-stem', '--stemmer',  
        default = "Porter", 
        choices = ["Porter", "Lancaster"],
        type = str, 
        help = "Stemming technique.")                                    


    args = parser.parse_args()
    args.col = ['Mandate/Mission', 
                'Main Community-Engaged Arts / Arts for Social Change Activities', 
                'Additional Info']

    run_LDA(args)



if __name__ == '__main__':

    main()


























