<hr/>

# Unsupervised Learning for Natural Language Processing

Demos and modules to help implement Latent Dirichlet Allocation for Topic Modeling, Word and Paragraph Embeddings to compare word or document similarity and data reduction schemes for visualizing high dimensional data. See [overview](#view) and notebooks for details.

[![Follow](https://img.shields.io/twitter/follow/maude_ll.svg?style=social&label=Follow)](https://twitter.com/maude_ll)


## :clipboard: Table of Contents

* [Overview](#view)

* [Environment](#env)

* [Launching a Demo](#launch)

* [Data description](#data)

* [Methods Overview](#meth)
   * [LDA for Topic Modeling](#lda)
   * [Word and Paragraph Embeddings](#w2v)
   
* [Contact Info](#fin)   

<hr/>

## :eye: Overview <a name="view"></a>
This repository is the result of an exploration of unsupervised learning for text data for SFU's Big Data Hub/KEY during the Fall 2018 and Spring 2019. The goal was to assess potential use for our client, [Artbridges](http://artbridges.ca/), as well as a reference for future projects. The demos were built using data from Artbridges, but are designed to be easily to use for other projects.

Methods used for this project: extensive text preprocessing, Latent Dirichlet Allocation for Topic Modeling, Word2Vec, Doc2Vec, visualization and clustering methods such as t-sne or PCA, tuning and diagnostics. Hierarchical Dirichlet Process and Latent Semantic Analysis are also implemented in the [methods](notebook_examples/methods/) directory. All demos use Python 3.7 and make extensive use of the [Gensim](https://radimrehurek.com/gensim/) library. See [below](#env) for complete requirements. 

Written by BDH RA Maude Lachaine under the supervision of Dr. Steven Bergner.

<hr/>

## :seedling: Environment <a name="env"></a>

All demos are implemented in python 3.7. If using [Pipenv](https://pipenv.readthedocs.io/en/latest/), use the Pipfile to set up your environment. If you prefer not using Pipenv, use the requirement.txt file to set up an environment or for documentation about working library versions and requirements.

## :rocket: Launching a Demo <a name="launch"></a>

**Note:** The notebook was intended to be viewed using the [`jupyterthemes` package](https://github.com/dunovank/jupyter-themes).

Once in the proper environment, type on the CL:

```unix
jt -t grade3 -f anonymous -fs 13 -cellw 100% -T -N
```

* **Running a notebook:** start an environment from a shell, then type:

```unix

jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10

```

The flag may be necessary to view some interactive plots, depending on your machine.



## :bar_chart: Data description <a name="data"></a>

To start, this project was built using 3 main datasets:

   * [`artbridges_profiles.csv`](data/artbridges_profiles.csv), private dataset supplied by the client containing information about community art centers:
      * **22 Fields:** Additional Info, Address, ArtBridging Section, Arts Focus, Facebook, Languages, Linkedin, Main Catchment Area, Main Communities Served, Main Community-Engaged Arts/Arts for Social Change Activities, Main Contact Email, Main Contact Name, Mandate/Mission, Organization Name, Phone, Profile URL, Secondary Address, Secondary Contact, Status, Twitter, Website
      
      * **Features of interest:** Mandate/Mission, Main Communities Served, Languages
      
      * **Optional features** Main Community-Engaged Arts/Arts for Social Change Activities, Additional Info, ArtBridging Section and Arts Focus can be used with Mandate/Mission to pad this small dataset with more text.
      
      
   * [`artbridges_test.csv`](artbridges_test.csv) Smaller dataset similar to the first one, containing information about BC community art centers. This data was hand curated to reduce the number of categories in Main Community Served.
   
      * **27 Fields:** Organization Name, Address, City, Region, Region(Cleaned), Website, Website https, Arts discipline, Facebook, Languages, LinkedIn, Main Communities Served, Main Communities Served - Cultural, Main Communities Served - Gender, Main Communities Served - Age, Social Change Focus, Main Community-Engaged Arts, Funding Sources, Programs and Projects, Additional Info, Mandate/Mission, Status, Twitter, Artbridges Profile URL, Main Contact Email, Main Contact Name, Consolidated Organization Status
   
      * **Features of interest:** Mandate/Mission, Main Communities Served, Main Communities Served - Cultural, Main Communities Served - Gender, Main Communities Served - Age, Languages
      
      * **Optional features** Additional Info can be used with Mandate/Mission to pad the text data
      
   * [`world_arts.csv`](data/world_arts.csv) (data obtained from [here](https://www.icasc.ca/directory))
   
      * **3 Fields:** Name, Mission, Country
      
      * **Features of interest:** Mission. No Language specified as all entries are in English.
      
Other datasets and pretrained models are available in the [data](data/) directory.      
   
## :hammer: Methods<a name="meth"></a>  

For details about methods and implemention, view the notebook directory.

### [Latent Dirichlet Allocation](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) for Topic Modeling <a name="lda"></a>
   
**Idea:** Latent Dirichlet Allocation is a generative model equivalent to a hierarchical mixture Bayesian model. If we view text as information generated by an underlying mechanism governed by fixed 'topics' (formally: distribution of words), we can simulate the text generation according to some basic rules and use Bayes' theorem to approximate a distribution over the topic data generation process, conditional on the documents at hand. If we model topic assignment for each document as a Dirichlet distribution (multivariate Beta distribution), topic assignment to words under a Multinomial Distribution (Multivariate Binomial distribution), the Multinomial parameter as Dirichlet, and the latent process as continuous, we can take advantage of the Beta-Binomial conjucacy to approximate a joint posterior distribution. The marginal distributions of the parameters of interest remain intractable, but can be approximated with variational methods.

The main assumptions is conditional independence of words w.r.t. the underlying generation process. Conditional independence is necessary for factorization of marginal distributions into joint distribution, and is assumed in LDA's Bag-of-Word representation, where texts are represented as (orderless) sets of words, but keeping track of word count.  

This quick overview summmarizes the simplest LDA model, for more details and complex implementations see [this technical paper](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) (Blei et al 2003) and [this simplified explanation](http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf) (Blei 2012).

Gensim shows [several](https://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html) [excellent](https://radimrehurek.com/gensim/models/ldamodel.html) [tutorials](https://radimrehurek.com/gensim/wiki.html) for topic modeling with LDA.
 

### Word and Paragraph Embeddings for semantic comparison between texts: [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) and [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html) <a name="w2v"></a>

**Idea:** These embeddings are mappings of words or phrases into numerical vectors, based on the idea that words appearing in similar contexts tend to be semantically similar. Their purpose is different than LDA, as the vector representation makes geometric interpretations that can be viewed as semantic distance between words or texts, but were not designed for topic modeling or clustering. The distributed representations (embeddings) are 'learned' as the weights of a small neural network, and those weights can be subsequently used for understanding relationships between words without human supervision. It allows for a basic learning of analogies and semantic equations of the form `house + car - people = garage`. These equations are represented in the `word2vec_demo.ipynb` notebook file as 'positive' and 'negative' word relationships. Since the word2vec algorithm deals with words as autonomous units, we can learn more complex representations with paragraph mappings (Doc2Vec). 

In our demos, word embeddings were used as a replacement for classification, since classification was not possible with our unlabeled datasets. Instead, the words representing the classes of interest were compared to each document to see how 'close' each class was to each document.

For more details, see the [`word2vec_demo.ipynb`](notebook_examples/word2vec_demo.ipynb) and [`doc2vec_demo.ipynb`](notebook_examples/doc2vec_demo.ipynb) notebooks. For technical details, see [Mikolov 2013](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf).

Great tutorials of [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) and [Doc2Vec](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb) can also be seen on the Gensim library page.


## :warning: Potential Issues

Any unsupervised methods for complex data requires a large enough dataset to capture subtle patterns in the data. The datasets used for this project were atypically small. After word preprocessing: `artbridges_profiles.csv` had 231 short texts for the main dataset, `artbridges_tests.csv` 50 short texts for the curated dataset and `world_arts.csv` 500 short texts for the padding dataset.

To address this problem, we compared models pretrained on generic datasets to the models trained strictly on the supplied datasets. Details in the notebooks.

A goal of this project was to automatically predict the types of clientele associated with each art centers, based on their mission statement. Given that the data was manually entered with no finite choices of types of clientele, there are too many groupings to have any meaningful classification based on such a small dataset. This could be an interesting problem to look into in the future, but our preliminary efforts were not fruitful in that respect.

<hr/>
<hr/>

###  :space_invader: Contact Info <a name="fin"></a>

For questions or details, contact me at `mlachain (at) sfu (dot) ca`

