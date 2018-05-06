# Clustering of the Top 250 movies from IMDB

## Introduction
What makes a good movie? Most of the top-rated movies in the International movie database (IMDB) are critically acclaimed and are generally a safe bet in terms of commercial success. Naturally, it would be interesting to investigate if these top movies have some distinct features responsible for their high ratings. This project aims to find out the type of natural cluster that exists among the top 250 movies from IMDB. Unsupervised machine learning techniques will be employed, more specifically, clustering algorithms. Hopefully, these clusters will give us information to observe the recurrent pattern.
    To build our dataset we used OMDB's web API which is RESTful web service to obtain movie information. For plot summaries we scrapped a movie's plot from IMDB's website using BeautifulSoup Library in python. Our final extracted dataframe had 250 rows and 113 columns. The inputs to our dataframe were all categorical features which were one hot encoded. We first used dimensionality reduction techniques such as PCA which was followed by K-Means and DB-SCAN clustering to find inherent clusters in the data.
    
## Dataset and Features
Our dataset is composed of the top 250 movies based on IMDB rating. We used data directly from OMBD's API which provided every detail for more than 5000 movies. From this dataset we subsetted the top movies based on IMDB's list. The movie plot in the OMDB was just a 2 line synopsis so we decided to scrape movie summary of every movie from IMDB's website. This was done to include important and frequent words of a plot and make them as features for clustering. From the data obtained from querying OMDB's API we cleaned every column and finally decided to include Runtime, Plot, Genre, Actors, Director, Writer, Year of release, Country and Language since the rest of the columns were not relevant such as IMDB rating, IMDB ID.
        ![OMDB DataFrame](https://github.com/mustafashabbir10/MovieClustering/blob/master/Images/omdb.PNG)

                    The above dataframe shows the features that we got from querying OMDB's web API.      
The main challenge was to extract relevant words from the plots which could be used as features for clustering. First step in plot preprocessing was to tokenize the text and remove punctuations and other noise from it. Next, was to remove all the stopwords which do not provide any meaningful information. After the initial cleaning of the text, TF-IDF technique was to used to get important and relevant features from the text. TF-IDF is a statistic that reflects how important a word is to a document in a collection of other documents. We chose TF-IDF statistic because it acts as a content descriptive mechanism for the plots. The term TF is the number of times a term appears in a movie plot and IDF measures the rarity of the term in the whole corpus of plots. Thus, TF-IDF provides a measure of uniqueness and will output words that are somewhat unique for plots. We used Tf-IDF because it gives high weights to words that are representative of a movie plot but do not occur too often across all the plots. This gives a uniqueness indicator and thus plots which have words having similar uniqueness indicators must be clustered together. Based on the list that TF-IDF gave us we chose the top important words which are-
           ![Top 20 words](https://github.com/mustafashabbir10/MovieClustering/blob/master/Images/top20words.PNG)

All of these words were important according to tf-idf statistic. We can immediately see that there were some usual themes such as war, love, murder. One another important observation is that all of the words have a male connotation boy, man, son which represents the backlash that academy has been facing for under representation of females and minorities. After identification of the important words, one hot encoding was performed for every word and respective dummy columns were created where if a movie had that word it was assigned a value of 1 otherwise 0.

For the rest of the columns such as Actors, Directors, Language and country we took the top 20 Actors, Directors, Language and Country according to frequency and created dummy columns for them. This left us with a dataframe of 250 datapoints and 112 columns. 
Modelling such a dataset is very difficult because of curse of dimensionality which renders the standard computational and statistical techniques useless. To counter this we used Principal Component Analysis to eliminate the irrelevant dimensions which are not useful.

## Methods
### Principal Component Analysis
Principal Component Analysis was used to counter curse of dimensionality. Escaping the curse was crucial for our clustering algorithms because reduced dimension have three fold benefits :- 1) Lighter Computational Workload  2)Less Dimensional redundancy  3) More Effective Distance Metrics. PCA is dimension-reduction tool that is used to reduce a large set of variables to a small number of factors while preserving the variance between the data points. 

### K-Means

k means clustering is an unsupervised clustering algorithm that groups n different observations into k unique clusters where k < n. 
The algorithm clusters data points of similar features based on some parameter, usually Euclidean distance. Clusters are formed 
in a way to maximize inter cluster similarity. This can be achieved by minimizing within cluster sum of squares,  
defined as
 ![kmeans](https://github.com/mustafashabbir10/MovieClustering/blob/master/Images/kmeaneq.PNG)
 
K means algorithm requires the practitioner to enter the value of parameter k. The optimum value of k can be chosen based on sum 
of squared error (SSE) or advanced statistical measures such as gap statistic. 

Nonetheless, analytical methods often result in complicated clusters. Since, clustering requires human interpretation to make 
conclusions its easier if the number of instances in a particular cluster is less.
The k means algorithm presented in the paper employed euclidean distance for clustering. The parameter k was set to 13 for two 
reasons. First, DBSCAN algorithm gave the best results for k = 13. Generally, distance based clustering algorithms 
performs quite poorly on such datasets. One reasonable explanation for the former statement is the exponential increase 
in euclidean distance as dimension grows. Principal component analysis was used for dimension reduction to minimize the effect 
of curse of dimensionality. However, it was still difficult to understand the association between movies. Some observable patterns 
exist between movies belonging to the same cluster but these patterns cannot be generalized for all the data points in the cluster. 



### DB-SCAN Clustering
DB-SCAN is a clustering algorithm that focuses on separating clusters of high density from low density. Thus, it is a **D**ensity **B**ased **C**lusetring **A**lgorithm which sorts the data into clusters of varying shape. The basic idea is that if a point belongs to a cluster then it must be near to a lot of other points which are also in that cluster. DBSCAN Algorithm takes up two parameters- first is a positive number epsilon and second is the minimum number of points. Randomly points are picked from the dataset and if there are more than minimum number of points within a distance a of epsilon from the selected point then they all are assigned to one same cluster. Then other points are selected checking whether they too have more than minimum number of points within an epsilon distance so that they could be added to the cluster. When all of the points are exhausted, a point is again selected at random and the whole process is repeated again. Psuedo Code of DBSCAN is as follows (https://algorithmicthoughts.wordpress.com/2013/05/29/machine-learning-dbscan/):-
![dbscan_algorithm](https://github.com/mustafashabbir10/MovieClustering/blob/master/Images/dbscan_algorithm.PNG)

One advantage of DBSCAN algorithm is that unlike K-means it does not require the user to provide the number of clusters beforehand. It gives the user the number of inherent cluster which are present in the data as an output to the user. To get a sense of consistency of the clusters that are formed by DBSCAN we have used a silhouette score as a metric. Silhouette score measures the amount of cohesion within a cluster as comapred to other clusters(separation). Silhouette score's value is between -1 to +1 and a higher value indicates that a datapoint is very similar to its own cluster and dissimilar to other neighbouring clusters. This silhouette score is calculated by Eucledian distance metric.

## Results and Discussion
#### A. *Principal Component Analysis*
Prior to a fitting a clustering algorithm we used PCA to counter the curse of dimensionality. We decided to include those components which explain more than 1% variance in the data. Through this we got 39 components which explain 75% variation in the dataset. We further decided to take a look in the first principal component to check which features were most important in our principal component.
![PCA](https://github.com/mustafashabbir10/MovieClustering/blob/master/Images/PCA.PNG)

We see from the above figure that movies with genre Action and Family have the highest weights. Thus, it is important for a movie to be Action or Family to be in the Top 250. 

#### B. *DBSCAN Clustering*
After performing PCA we fit a DBSCAN clustering algorithm on the data. Since we do not know any inherent number of clusters in the data, we use dbscan to find them. To find the optimal clusters we cycled through a number of values of epsilon and min_samples and evaluated a corresponding silhouette score. The pair of value having the best and most interpretable silhouette score was chosen. This best performing DBSCAN algorithm gave 13 clusters. This number of clusters obtained from DBSCAN was used in K-Means algorithm. 

Since the data is sparse we observe that DBSCAN clustering does not perform well in clustering similar movies. The value of epsilon which was chosen was 6 and minimum samples 3. Due to lack of dense data dbscan made 10 clusters of 3 movies which were similar but did not provide any compelling evidence to our case. Some of the clusters are subsets of the K means clusters.So we haven't reproduced the output of dbscan in the report.The cluster results can be printed from the ipynb notebook.Thus, dbscan was only used to find the optimal number of clusters in our dataset. 

#### C. *K-Means Clustering*
After getting the number of clusetrs from DBSCAN we used K-Means to cluster our dataset. Inspite of our categorical we still used K-Means to cluster our dataset rather than K-Modes because K-Modes forces the centroid to take on majority features value without indicating wether the datapoints in the cluster are in strong agreement. Thus, to avoid this we proceed with K-Means clusetring only. Thirteen cluster of movies were created. However, we are just mentioning the top five clusters that had a clear association. The outputs of all 13 clusters can be retrieved from the python notebook.

##### *Manual Labeling*
We attempt to determine a common theme of every movie cluster that is formed by examining and interpreting the attributes of the movies that have been clustered together.

##### Group 1:
![Cluster0](https://github.com/mustafashabbir10/MovieClustering/blob/master/Images/Cluster2.PNG)
Based on the movies in this cluster we can name this group as **Action/Adventure Conflicts**. These movies are based on the theme of War, Action, Violence and Adventure and include a mix of real and fantasy movies for example Avengers, Pirates of the Carribean, Inglourius Basterds, The Matrix, V for Vandetta, Terminator etc . The main theme of many movies in this cluster is real or fantasy Violence, vengence or War. This cluster has also grouped long biographical epics from different eras and different countries together for example Gandhi, Lawrence of Arabia and Barry Lyndon. All three of these epics are based on the theme of war and happen in different countries. 

##### Group 2:
![Cluster0](https://github.com/mustafashabbir10/MovieClustering/blob/master/Images/Cluster3.PNG)
Based on the movies in this cluster we can name this group as **Animation** Group. All of these movies are animated having a genre of Family and Film-Noir. This cluster is dominated by Hayao Miyazaki a Japanese film director and Manga Artist. Majority of the animated movies are directed by Hayao Miyazaki and involve the theme of adventure by a child either to free or in search of its parent or describes a struggle for survival. 

##### Group 3:
![Cluster0](https://github.com/mustafashabbir10/MovieClustering/blob/master/Images/CLuster6.PNG)
Based on the movies of this group we can name it **Criminal Thriller** This group has movies which have a theme on drug lords, violence, prison, crime and courtroom drama for example Shawshank Redemption, Godfather, Angry Men, Kill Bill, Cool Hand Luke, City of God, Leon:The Professional, Touch of Evil, American History X etc. All of these movies are long and not recent (except Kill Bill) and revolve around the aforementioned theme. This cluster groups all movies which primarily focus on crime and murder. This is a very pure cluster as one can see if they want to make a criminal thriller they can analyze and include all of the characteristics of such movies.

##### Group 4:
![Cluster0](https://github.com/mustafashabbir10/MovieClustering/blob/master/Images/Cluster4.PNG)
This is a pure Christopher Nolan cluster. He works majorly with Christian Bale and has recent Sci-fi movies.

##### Group 5:
![Cluster0](https://github.com/mustafashabbir10/MovieClustering/blob/master/Images/Cluster11.PNG)
Similarly this is a pure Martin Scorsese cluster. His work focus majorly on the theme of a successful man taking a downward spiral in his career beacuse of crime, love, money or drugs.

Group 4 and Group 5 have been monopolized by two directors Nolan and Scorsese. These amazing creators have their own clusters and theme which have become a recepie of success for many other directors.


## Conclusion
In this project, we scrapped our dataset using OMDB's API and IMDB. This dataset was sparse and contained 100 categorical features and 1 numerical feature. This was a very daunting task for any modeler to work with such a dataset which is full of dummy varaibles. CLustering on a such a dataset was a very challenging part. Two different clustering algorithms DBSCAN and K-means were applied and studied. PCA was also performed on the dataset to reduce the dimension and escape the curse of dimensionality. 

DBSCAN gave us the optimum number of clusters in our dataset and then K-Means was applied to get really interpreatble cluster. We have only shown 5 out of a total of 13 clusters. The output of the analysis can be used and cross referenced against large dataset of movie to see what better clusters can be obtained. We see a similar algorithm for movie prediction in Netflix and other webservices which use recommender system. Thus, this concept can be further extended to create efficient recommender systems

## References
1) Rose, B. (n.d.). Document clustering with python. Retrieved from http://brandonrose.org/clustering
2) Data Mining Algorithms In R/Clustering/K-Means. (n.d.). Retrieved from Wikibooks: https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Clustering/K-Means
3) Man,Han. Data Clustering Using Unsupervised Learning, 2017
4) Huang, A., 2008. Similarity measures for text document clustering. In: Proceedings of the
Sixth New Zealand Computer Science Research Student Conference, pp. 49–56.





