from gensim.utils import to_unicode
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec, keyedvectors
from gensim.models.phrases import Phraser, Phrases

from sklearn import metrics
from sklearn.cluster import Birch, KMeans
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import silhouette_samples, silhouette_score

import numpy as np
import pandas as pd
from collections import Counter

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors as mcolors
from adjustText import adjust_text

from helper.helper_funs import time_filename, save_folder_file



class LabeledLineSentence:
    '''
    data generator + tagger
    Tags can be ID numbers, names, etc
    '''
    def __init__(self, doc_list, labels_list = [], train_set = True):
        self.labels_list = labels_list
        self.doc_list = doc_list
        self.train_set = train_set

    def __iter__(self):
        if self.train_set:
            if len(self.labels_list):
                for idx, doc in enumerate(self.doc_list):
                    yield TaggedDocument(words=to_unicode(str.encode(' '.join(doc))).split(), tags=[self.labels_list[idx]])
            else:        
                for idx, doc in enumerate(self.doc_list):
                    yield TaggedDocument(words=to_unicode(str.encode(' '.join(doc))).split(), tags=['SENT_{}'.format(idx)])       
        else: # test set, no labels
            for idx, doc in enumerate(self.doc_list):
                yield doc


def doc2vec_model(texts, 
                  labels=[],
                  epochs=10,
                  workers=3,
                  lr_reduce=0.02,
                  rm_training_data = False,
                  save_model=True,
                  save_dir='saved_models',
                  filename='',
                  save_as_word2vec = True,
                  **kwargs):

    
    it = LabeledLineSentence(texts, labels)
    model = Doc2Vec(workers=workers, **kwargs)  # use fixed learning rate
    model.build_vocab(it)
    
    for epoch in range(epochs):
        model.train(it, 
                    total_examples=model.corpus_count, 
                    epochs=1, 
                    start_alpha=model.alpha)
        model.alpha -= lr_reduce         # decrease the learning rate
        texts, labels = shuffle(texts, labels)
        it = LabeledLineSentence(texts, labels)
    if rm_training_data:
        print('Deleting training data - keeping doctag vectors and inference...')    
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    if save_model:
        if len(filename) == 0:
            filename = 'doc2vec_{}epochs_'.format(epochs)                                                      
        
        full_path = save_folder_file(save_dir, filename, ext='.model', optional_folder='WordEmbeddings')
            
        if save_as_word2vec:
            filename_w2v = 'doc_to_word2vec_{}epochs_'.format(epochs) 
            full_path_w2v = save_folder_file(save_dir, filename_w2v, ext='.word2vec', optional_folder='WordEmbeddings')
            model.save_word2vec_format(full_path_w2v)
        model.save(full_path)

    return model


def pretrained_doc2vec(texts, 
                   labels = [],
                   pretrained_emb = "saved_models/apnews_dbow/doc2vec.bin",
                   epochs = 10,
                   workers = 3, 
                   lr_reduce = 0.002,
                   rm_training_data = False,
                   save_model=True,
                   save_dir='saved_models',
                   filename='',
                   save_as_word2vec = True,
                   **kwargs):    
    

    it = LabeledLineSentence(texts, labels)
    
    pretrained_d2v = Doc2Vec(pretrained_emb=pretrained_emb, workers=workers, **kwargs) 
    pretrained_d2v.build_vocab(it)
    
    for epoch in range(epochs):
        pretrained_d2v.train(it, 
                             total_examples=pretrained_d2v.corpus_count, 
                             epochs=1, 
                             start_alpha=pretrained_d2v.alpha)
        pretrained_d2v.alpha -= lr_reduce         # decrease the learning rate
        texts, labels = shuffle(texts, labels)
        it = LabeledLineSentence(texts, labels)   

    if rm_training_data:
        print('Deleting training data - keeping doctag vectors and inference...')    
        pretrained_d2v.delete_temporary_training_data(keep_doctags_vectors=True, 
                                                      keep_inference=True)

    if save_model:
        if len(filename) == 0:
            filename = 'pretrained_d2v_{}epochs_'.format(epochs)                                                      
        
        full_path = save_folder_file(save_dir, filename, ext='.model', optional_folder='WordEmbeddings')
            
        if save_as_word2vec:
            filename_w2v = 'pretrained_d2v_to_w2v_{}epochs_'.format(epochs) 
            full_path_w2v = save_folder_file(save_dir, filename_w2v, ext='.word2vec', optional_folder='WordEmbeddings')
            pretrained_d2v.save_word2vec_format(full_path_w2v)
        pretrained_d2v.save(full_path)

    return pretrained_d2v


def word2vec_model(texts, workers=3, **kwargs):

    w2v_model = Word2Vec(workers=workers, **kwargs)    
    w2v_model.build_vocab(texts)
    w2v_model.train(texts, total_examples=len(texts), epochs=w2v_model.epochs)    

    return w2v_model

def pretrained_word2vec(texts, trained_model_name='text8w2v_model',
                               workers=3,
                               **kwargs):
    print('Loading pretrained model...')
    try:
        modtext8 = keyedvectors.KeyedVectors.load_word2vec_format(trained_model_name)
    except OSError:
        print('Error opening {}. Make sure it is in working dir.'.format(trained_model_name))

    premodel = Word2Vec(workers=workers, **kwargs)
    premodel.build_vocab(texts)
    training_examples_count = premodel.corpus_count
    
    premodel.build_vocab([list(modtext8.wv.vocab.keys())], update=True)
    premodel.intersect_word2vec_format(trained_model_name, binary=True)
    premodel.train(texts, total_examples=training_examples_count, epochs=premodel.iter)
    return premodel


def plot_w2v(w2v, 
             size=(18,10), 
             max_idx=200, 
             xlim=(-0.005,0.005), 
             ylim=(-0.005,0.005),
             title= '2D PCA Projection of Word Embeddings'):

 
    words_np = []
    #a list of labels (words)
    words_label = []
    for word in w2v.vocab.keys():
        words_np.append(w2v[word])
        words_label.append(word)
    print('Added {} words. Shape {}'.format(len(words_np), np.shape(words_np)))
 
    pca = PCA(n_components=2)
    pca.fit(words_np)
    reduced = pca.transform(words_np)
    plt.figure(figsize=size)

    for index,vec in enumerate(reduced):
        if index < max_idx:
            x,y=vec[0],vec[1]
            plt.scatter(x,y, s=100)
            plt.annotate(words_label[index], xy=(x,y), size=25)
            plt.ylim(ylim)
            plt.xlim(xlim)
            plt.tick_params(labelsize = 15)
            plt.xticks( rotation=45)
            plt.title(title, fontsize=20)

    plt.show()


def make_bigrams(texts,
                 common_terms = ["of", "with", "without", "and", "or", "the", "a",
                                 "per", "as", "yet"], # words to retain for n-grams
                 series_input = False):
    
    # Create the relevant phrases from the list of sentences:
    phrases = Phrases(texts, common_terms = common_terms)
    # The Phraser object is used from now on to transform sentences
    bigram = Phraser(phrases)
    # Applying the Phraser to transform our sentences is simply
    if series_input:
        all_sentences = pd.Series(bigram[texts])    
    else:
        all_sentences = list(bigram[texts]) 
    return all_sentences


def count_bigrams(bigram, top_n=20):

    bigram_counter = Counter()
    for line in bigram:
        for word in line:
            if len(str(word).split("_")) > 1:
                bigram_counter[word] += 1

    for key, counts in bigram_counter.most_common(top_n):
        print('{} {}'.format(key, counts))                

    return bigram_counter

        
def birch_clusters(textdata, trained_doc2vec, n_clusters, start_alpha = 0.025, 
                                                          infer_epoch = 100, 
                                                          branching_factor = 10,
                                                          threshold=0.01, 
                                                          compute_labels=True,
                                                          metric='cosine',
                                                          **kwargs):        
    infer_list = []
    
    for doc in textdata:
        infer_list.append( trained_doc2vec.infer_vector(doc, 
                                                        alpha=start_alpha, 
                                                        steps=infer_epoch, 
                                                        **kwargs) )
    brc = Birch(branching_factor=branching_factor, 
                n_clusters=int(n_clusters), 
                threshold=threshold, 
                compute_labels=compute_labels)
    
    brc.fit(infer_list)
    clusters = brc.predict(infer_list)
    birch_labels = brc.labels_
    
    silhouette_score = metrics.silhouette_score(infer_list, birch_labels, metric=metric)
     
    return silhouette_score, clusters
 

def plot_d2v(trained_doc2vec,
                doc_tags=[], 
                size=(18,10), 
                n_clusters = 4,
                max_iter = 500,
                init = 'k-means++',
                max_idx=200, 
                title= 'K-Means Clustering with 2D PCA Rendition (triangles represent cluster centroids)',
                random_state=0,
                xlim=(-0.005,0.005), 
                ylim=(-0.005,0.005),                
                with_adjust_text = False,
                group_color_list = None,
                **kwargs):

    words_np = trained_doc2vec.docvecs.doctag_syn0
    # Apply K-means clustering on the model
    kmeans_model = KMeans(n_clusters=n_clusters, 
                          init=init, 
                          max_iter=max_iter,
                          **kwargs)  

    X = kmeans_model.fit(words_np)
    labels=kmeans_model.labels_.tolist()
    l = kmeans_model.fit_predict(words_np)
 
    words_np, labels, doc_tags = shuffle(words_np, labels, doc_tags, n_samples=max_idx, random_state=random_state) 
    pca = PCA(n_components=2)
    pca.fit(words_np)
    datapoint = pca.transform(words_np)
    centroids = kmeans_model.cluster_centers_
    centroidpoint = pca.transform(centroids)

    plt.figure(figsize=size)

    # dict_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    if not group_color_list:
        # default to using Tableau colors - could get fancier with CSS4 colours too
        if n_clusters < 11:
            color_list = [val for key, val in mcolors.TABLEAU_COLORS.items()] 
        else:    
            color_list = [val for key, val in mcolors.CSS4.items()]             
        group_color_list = np.random.choice(color_list, n_clusters, replace=False)
    color = [group_color_list[lab] for lab in labels]    
    texts = []

    for index, vec in enumerate(datapoint):
        # if index < max_idx:
        x, y = vec[0], vec[1]

        if x > xlim[0] and x < xlim[1] and y > ylim[0] and y < ylim[1]:     
            plt.scatter(x, y, c=color[index], edgecolors='#000000', lw=1.5, s=70)
            if len(doc_tags):
                if with_adjust_text:
                    texts.append(plt.annotate(doc_tags[index], xy=(x, y), size=10))
                else:
                    plt.annotate(doc_tags[index], xy=(x, y), size=10)        
    
    plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], 
                marker='^', s=550, c='#000000', edgecolors='#ffffff', lw=2.5)

    if with_adjust_text:
        adjust_text(texts)    

    plt.tick_params(labelsize = 15)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xticks(rotation=45)
    plt.title(title, fontsize=20)
    plt.show()


def plot_groups_w2v(w2v, 
                    size=(18,10), 
                    n_clusters = 4,
                    max_iter = 100,
                    init = 'k-means++',
                    max_idx = 200, 
                    title= '2D Rendition of Keywords, by Category',
                    random_state = 0,
                    with_adjust_text = False,
                    group_color_list = None,
                    **kwargs):

    words_np = []
    #a list of labels (words)
    words_label = []
    for word in w2v.vocab.keys():
        words_np.append(w2v[word])
        words_label.append(word)
    print('Added {} words. Shape {}'.format(len(words_np), np.shape(words_np)))


    # Apply K-means clustering on the model
    kmeans_model = KMeans(n_clusters=n_clusters, 
                          init=init, 
                          max_iter=max_iter,
                          **kwargs)  

    X = kmeans_model.fit(words_np)
    labels=kmeans_model.labels_.tolist()
    l = kmeans_model.fit_predict(words_np)

    words_np, labels, words_label = shuffle(words_np, labels, words_label, n_samples=max_idx, random_state=random_state)
 
    pca = PCA(n_components=2)
    pca.fit(words_np)
    datapoint = pca.transform(words_np)
    centroids = kmeans_model.cluster_centers_
    centroidpoint = pca.transform(centroids)

    # dict_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    if not group_color_list:
        # default to using Tableau colors - could get fancier with CSS4 colours too
        if n_clusters < 11:
            color_list = [val for key, val in mcolors.TABLEAU_COLORS.items()] 
        else:    
            color_list = [val for key, val in mcolors.CSS4.items()]             
        group_color_list = np.random.choice(color_list, n_clusters, replace=False)

    color = [group_color_list[lab] for lab in labels]
    plt.figure(figsize=size)
    texts = []
    for index, vec in enumerate(datapoint):
        x,y=vec[0],vec[1]
        plt.scatter(x,y, s=100, c=color[index], edgecolors='#000000')
        if with_adjust_text:
            texts.append(plt.annotate(words_label[index], xy=(x,y), size=15))
        else:
            plt.annotate(words_label[index],xy=(x,y), size=25)

    plt.tick_params(labelsize = 15)
    plt.xticks( rotation=45)
    plt.title(title, fontsize=20)    

    if with_adjust_text:
        adjust_text(texts)

    filename = 'class_w2v'                                                     
    
    full_path = save_folder_file('results/model_validation', filename, 
                                           ext='.pdf', 
                                           optional_folder='CV_score_plots')
    
    plt.savefig(full_path)   
    plt.show()


def silhouette_plots(trained_doc2vec, clusters_from=3, 
                                      clusters_to=8,
                                      xlim = [-0.1, 1],
                                      seed=10,
                                      title="The silhouette plot for the various clusters",
                                      title2="The visualization of the clustered data",
                                      xlab="The silhouette coefficient values",
                                      xlab2="Feature space for the 1st feature",
                                      ylab="Cluster label",
                                      ylab2="Feature space for the 2nd feature",
                                      metric="l2",  #‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
                                      **kwargs):

    range_n_clusters = [c for c in range(clusters_from, clusters_to)]
    
    X = trained_doc2vec.docvecs.doctag_syn0
    
    for n_clus in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, **kwargs)
        fig.set_size_inches(18, 7)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 
        ax1.set_xlim(xlim)
        # The (n_clus+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clus + 1) * 10])
    
        # Initialize the clusterer with n_clus value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clus, random_state=seed)
        cluster_labels = clusterer.fit_predict(X)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels, metric = metric)
        print("For n_clusters =", n_clus,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        y_lower = 10
        for i in range(n_clus):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clus)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title(title)
        ax1.set_xlabel(xlab)
        ax1.set_ylabel(ylab)
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks(np.arange(xlim[0], xlim[1], 0.2))
    
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clus)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
    
        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
    
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
    
        ax2.set_title(title2)
        ax2.set_xlabel(xlab2)
        ax2.set_ylabel(ylab)
    
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clus),
                     fontsize=14, fontweight='bold')

    plt.show()
  


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
