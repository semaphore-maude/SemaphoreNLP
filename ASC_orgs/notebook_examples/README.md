
<hr/>

## Unsupervised Learning Demos

<hr/>

### Running Demos:

While the folders all contain the necessary data to run each demo independently, they can also be run from scratch (with only the datasets and the local modules). 

### First, the datafiles may need to be unzipped

### To run from scratch:

The demos are typically run in this order:

* `word_preprocess.ipynb`
* `LDA_demo.ipynb`
* `word2vec_demo.ipynb`
* `Doc2Vec_demo.ipynb`

Each run will produce new dated files and directories to store the results. 

All notebooks depend on the output of `word_preprocess.ipynb`, and `Doc2Vec_demo.ipynb` depends on the output of `LDA_demo.ipynb`.

### Setting parameters for tuning:

To enable easy tuning, the `changeable_parameters` directory contains two editable `.json` files. For each run, a file will be created storing all parameters and variables used.

`tuning_params.json` contains:

* The names of the text columns for each dataset

* the `kwargs` parameters for each method (view the `help_kwargs` fields for details about each parameters)

* curated `stopwords` (words deemed unimportant for these purposes, manually removed iteratively)

* the names of client groups, with a list of 'positively' and 'negatively' correlated words. 
   * For example, for category "youth": ["youth", "senior"]  

`cleaning_regex.json` contains:

* word patterns to remove with regular expressions that are quirky to the dataset in use. For example, the `word_preprocess` demo automatically removes all urls and email addresses, but the dataset used in this project had very specific types of words to remove. Example: there were many ways to spell 'lgbt' (lgbtq, ltgb2qqs', 'gblt', etc.). These were included manually to the `cleaning_regex.json` file for easier manipulation.

<hr/>
<hr/>

View each notebooks for details of methods and results.

<hr/>
<hr/>
