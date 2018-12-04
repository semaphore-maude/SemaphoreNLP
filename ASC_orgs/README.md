## Results from the ASC project - for 4 topic and 7 topic cross-validated model

### To view result tables, load the following files from the `results/2018-12-4/` folder:

For table showing the distribution of documents across topics, 
   * `doc_dist_4topics_tfidf.csv`
   * `doc_dist_7topics.csv`
   
For table showing the main topic inferred for each document (1 doc = all texts from 1 organization)  
   * `dominant_topic_per_text_4topics_tfidf.csv`
   * `dominant_topic_per_text_7topics.csv`
   
For table showing the most relevant document in the dataset for each topic:
   * `top_texts_per_topic_4topics_tfidf.csv`
   * `top_texts_per_topic_7topics.csv`

<hr>
   
### To view simplified notebooks, download:

    * both datasets in the `data` folder
    * all files in the `helper` and `method` folders
    
#### Then run the `run_LDA_validated_main.ipynb` file for the main plots and tables with the smaller model (4 topics)

#### Or the `run_LDA_validated_side.ipynb` file for a shortened notebook for a larger model (7 topics)

**Both notebooks included an updated result with the BC dataset (from `data/artbridges_test.csv`), but both were inconclusive**

**The tuning in both notebooks are the result of extensive hyperparameter search, and from hand picking words to remove (words with little meaning in this context)**

The notebooks were written to hide the text, view the header of both files to see how to view the code (press toggle on/off)

If needed, the file `run_LDA_argparse.py` can be run from the command line.

All classes, methods and functions are described in details in the files from the `helper` and `method` folders.
         
