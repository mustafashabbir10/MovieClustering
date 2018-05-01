# Clustering of the Top 250 movies from IMDB

## Introduction
What makes a good movie? Most of the top-rated movies in the International movie database (IMDB) are critically acclaimed and are generally a safe bet in terms of commercial success. Naturally, it would be interesting to investigate if these top movies have some distinct features responsible for their high ratings. This project aims to find out the type of natural cluster that exists among the top 250 movies from IMDB. Unsupervised machine learning techniques will be employed, more specifically, clustering algorithms. Hopefully, these clusters will give us information to observe the recurrent pattern.
    To build our dataset we used OMDB's web API which is RESTful web service to obtain movie information. For plot summaries we scrapped a movie's plot from IMDB's website using BeutifulSoup Library in python. Our final extracted dataframe had 250 rows and 113 columns. The inputs to our dataframe were all categorical features which were one hot encoded. We first used dimensionality reduction techniques such as PCA which was followed by K-Means and DB-SCAN clustering to find inherent clusters in the data.
    
## Dataset and Features
Our dataset is composed of the top 250 movies based on IMDB rating. We used data directly from OMBD's API which provided every detail about the more than 5000 movies. From this dataset we subsetted the top movies based on IMDB's list. The movie plot in the OMDB was just a 2 line synopses so we decided to scrape movie summary of every movie from IMDB's website. This was done to include important and frequent words of a plot and make them as features for clustering. From the data obtained from querying OMDB's API we cleaned every column and finally decided to include Runtime, Plot, Genre, Actors, Director, Writer, Year of release, Country and Language since the rest of the columns were not relevant such as IMDB rating, IMDB ID.
        ![OMDB DataFrame](https://github.com/mustafashabbir10/MovieClustering/blob/master/images/omdbdf.PNG=centerme)

                    The above dataframe shows the features that we got from querying OMDB's web API.      
The main challenge was to extract relevant words from the plots which could be used as features for clustering. First step in plot preprocessing was to tokenize the text and remove punctuations and other noise from it. Next, was to remove all the stopwords which do not provide any meaningful information. After the initial cleaning of the text, TF-IDF technique was to used to get important and relevant features from the text. TF-IDF is a term frequency-inverse document frequency is a statistic that reflects how important a word is to a document in a collection of other documents. We chose TF-IDF statistic because it acts as a content descriptive mechanism for the plots. The term TF is the number of times a term appears in a movie plot and IDF measures the rarity of the term in the whole corpus of plots. Thus, TF-IDF provides a measure of uniqueness and will output words that are somewhat unique for plots. We used Tf-IDF because it gives high weights to words that are representative of a movie plot but do not occur too often across all the plots. This gives a uniqueness indicator and thus plots which have words having similar uniqueness indicators must be clustered together. Based on the list that TF-IDF gave us we chose the top important words which are-
           ![Top 20 words](https://github.com/mustafashabbir10/MovieClustering/blob/master/images/top20words.PNG)

All of these words were important according to tf-idf statistic. We can immediately see that there were some usual themes such as war, love, murder. One another important observation is that all of the words have a male connotation boy, man, son which represents the backlash that academy has been facing for under representation of females and minorities. After identification of the important words, one hot encoding was performed for every word and respective dummy columns were created where if a movie had that word it was assigned a value of 1 otherwise 0.

For the rest of the columns such as Actors, Directors, Language and country we took the top 20 Actors, Directors, Language and Country according to frequency and created dummy columns for them. This left us with a dataframe of 250 datapoints and 112 columns. 
Modelling such a dataset is very difficult because of curse of dimensionality which renders the standard computational and statistical techniques useless. To counter this we used Principal Component Analysis to eliminate the irrelevant dimensions which are not useful.

## Methods
### Principal Component Analysis
Principal Component Analysis was used to counter curse of dimensionality. Escaping the curse was crucial for our clustering algorithms because reduced dimension have three fold benefits :- 1) Lighter Computational Workload  2)Less Dimensional redundancy  3) More Effective Distance Metrics. PCA is dimension-reduction tool that is used to reduce a large set of variables to a small number of factors while preserving the variance between the data points. 

### K-Means

k means clustering is an unsupervised clustering algorithm that groups n different observations into k unique clusters where k < n. 
The algorithm clusters data points of similar features based on some parameter, usually Euclidean distance. Clusters are formed 
in a way to maximize inter cluster similarity. This can be achieved by minimizing within cluster sum of squares,defined as
 
 
K means algorithm requires the practitioner to enter the value of parameter k. The optimum value of k can be chosen based on sum 
of squared error (SSE) or advanced statistical measures such as gap statistic. 

Nonetheless, analytical methods often result in complicated clusters. Since, clustering requires human interpretation to make 
conclusions
The k means algorithm presented in the paper employed euclidean distance for clustering. The parameter k was set to 5 for two 
reasons. First, DBSCAN algorithm gave the best results for k = 5. Second, cluster sizes were comparable making it easier to interpret 
similarities. The dataset used is very sparse and high dimensional in nature. Generally, distance based clustering algorithms 
performs quite poorly on such datasets. One reasonable explanation for the former statement is the exponential increase 
in euclidean distance as dimension grows. Principal component analysis was used for dimension reduction to minimize the effect 
of curse of dimensionality. However, it was still difficult to understand the association between movies. Some observable patterns 
exist between movies belonging to the same cluster but these patterns cannot be generalized for all the data points in the cluster. 



### DB-SCAN Clustering
DB-SCAN is a clustering algorithm that focuses on separating clusters of high density from low density. Thus, it is a **D**ensity **B**ased **C**lusetring **A**lgorithm which sorts the data into clusters of varying shape. The basic idea is that if a point belongs to a cluster then it must be near to a lot of other points which are also in that cluster. DBSCAN Algorithm takes up two parameters- first is a positive number epsilon and second is the minimum number of points. Randomly points are picked from the dataset and if there are more than minimum number of points within a distance a of epsilon from the selected point then they all are assigned to one same cluster. Then other points are selected checking whether they too have more than minimum number of points within an epsilon distance so that they could be added to the cluster. When all of the points are exhausted, a point is again selected at random and the whole process is repeated again. Psuedo Code of DBSCAN is as follows:-


One advantage of DBSCAN algorithm is that unlike K-means it does not require the user to provide the number of clusters beforehand. It gives the user the number of inherent cluster which are present in the data as an output to the user. To get a sense of consistency of the clusters that are formed by DBSCAN we have used a silhouette score as a metric. Silhouette score measures the amount of cohesion within a cluster as comapred to other clusters(separation). Silhouette score's value is between -1 to +1 and a higher value indicates that a datapoint is very similar to its own cluster and dissimilar to other neighbouring clusters. This silhouette score is calculated by Eucledian distance metric.



