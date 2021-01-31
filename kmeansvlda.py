# importing necessary libraries
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
from wordcloud import WordCloud, STOPWORDS
import pyLDAvis
import pyLDAvis.sklearn


# getting the dataset
categories = ['rec.motorcycles', 'comp.sys.mac.hardware', 'sci.space']
corpus = fetch_20newsgroups(
            subset='train', remove=('headers', 'footers', 'quotes'),
            categories=categories)

""" Converting the dataset into bag of words model"""
print("calculating tf-idf values\n")
cv = CountVectorizer(stop_words='english') # Converting into vector space
tf = cv.fit_transform(corpus.data) # calculating the TF values
tfidf_transformer=TfidfTransformer(smooth_idf=True,).fit(tf) # initializing tf_idf transformer
tf_idf = tfidf_transformer.transform(tf) # Fitting the data to the transformer
fn = cv.get_feature_names() # getting the terms in corpus
x = tf_idf.toarray()
# Plotting the TF-IDF score
print("plotting tf-idf vector space..\n")
plt.imshow(x, aspect='auto', cmap='pink')
plt.show()

"""
    K-KMeans
    We will calculate the number of clusters that are optimal for our datasets
    based on tf_idf values. Even though the optimal clusters might be different
    We will put clusters as 3 as we are comparing K-Means with LDA
"""

# calculating the optimal number of clusters for K-Means
print("Plotting Elbow method for K-Means...\n")
K = range(1, 15)
SSE = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(tf_idf)
    SSE.append(kmeans.inertia_)
plt.plot(K, SSE, 'bx-')
plt.title('Elbow Method')
plt.xlabel('cluster numbers')
plt.show() # plotting the results i.e. Elbow Method

print("Running K-Means Algorithm")
k = 3 # defining number of clusters as three because we have data of three topics
kmeans = KMeans(n_clusters = k)
kmeans.fit(tf_idf) # fitting the tfidf values to cluster the documents
centers = kmeans.cluster_centers_.argsort()[:,::-1] # getting the cluster centers
word_list_kmeans = [] # initializing list to get all the terms based on each topic
labels = kmeans.labels_ # getting labels of each word


# printing the top words in each cluster and getting words of each cluster
print("The top words for each cluster when K-means was ran\n")
for i in range(0,k):
    # Printing the top words
    word_list=[]
    print("cluster%d:"% i)
    for j in centers[i,:10]:
        word_list.append(fn[j])
    print(word_list)

    # getting the word list according to clusters
    for j in centers[i]:
        word_list.append(fn[j])
    word_list_kmeans.append(word_list)

#separating values of each clusters into different arrays for easy plotting
# initializing lists for each cluster to plot
print("plotting K-means Clusters...\n")
c1 =[]
c2=[]
c3 =[]
for i in range(len(labels)):
    if labels[i] == 0:
        c1.append(x[i])
    if labels[i] == 1:
        c2.append(x[i])
    if labels[i] == 2:
        c3.append(x[i])
c1 = np.array(c1)
c2 = np.array(c2)
c3 = np.array(c3)

# plotting the three clusters
plt.plot(c1, '*', c='green')
plt.plot(c2, '^', c='blue')
plt.plot(c3, 'x', c= 'red',)
plt.show()

# As the data is of high dimensions, we will reduce the dimensions using t-SNE
# and plot the graph for k-means
print("Plotting K-Means Clusters using t-SNE...\n")
tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=300)
c1_tsne= tsne.fit_transform(c1)
c2_tsne= tsne.fit_transform(c2)
c3_tsne = tsne.fit_transform(c3)

# getting the x and y values for each cluster
c1_x = c1_tsne[:, 0]
c1_y = c1_tsne[:, 1]
c2_x = c2_tsne[:, 0]
c2_y = c2_tsne[:, 1]
c3_x = c3_tsne[:, 0]
c3_y = c3_tsne[:, 1]
# plotting the results
plt.scatter(c1_x,c1_y, c='green', alpha=0.7)
plt.scatter(c2_x,c2_y, c='blue', alpha=0.5)
plt.scatter(c3_x,c3_y, c='red', alpha=0.3)
plt.show()

"""
    LDA - Latent Dirichlet Allocation
    We will fit and transform the data and plot the data using t-SNE
"""
# initializing lda
print("Running LDA algorithm...\n")
lda  = LatentDirichletAllocation(n_components = 3  , max_iter=5,
                learning_method='online', learning_offset=50.,random_state=0)
lda.fit(tf)

# displaying top words of each topic
def display_topics(model, feature_names, no_top_words):
    topics = []
    topic_tfidf = []
    print("Top words for each topic when LDA was ran\n")
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        topics.append(" ".join([feature_names[i]
                        for i in topic.argsort()]))
    return topics

lda_topics= display_topics(lda, fn, 15)


# creating values to plot the data using t-SNE
print("Plotting the LDA findings usign t-SNE\n")
lda_embedding = lda.transform(tf)
lda_embedding = (lda_embedding - lda_embedding.mean(axis=0))/lda_embedding.std(axis=0)
tsne = TSNE(random_state=3211) # initializing t-SNE
l_tsne_embedding = tsne.fit_transform(lda_embedding)
l_tsne_embedding = pd.DataFrame(l_tsne_embedding,columns=['x','y'])
l_tsne_embedding['hue'] = lda_embedding.argmax(axis=1)
# Plotting the findings from LDA
plt.style.use('ggplot')
plt.scatter(data=l_tsne_embedding, x='x', y='y',c=l_tsne_embedding['hue'], cmap= 'spring')
plt.show()

# Visulaizing LDA using pyLDAvis which will be saved as an HTML file to be viewed later
# vis = pyLDAvis.sklearn.prepare(lda, tf, cv, mds='tsne')
# pyLDAvis.display(vis)
# pyLDAvis.save_html(vis, './ldavis_prepared_'+'.html') To save to a HTML file


# Function for generating wordcloud
def word_cloud(data, title, c):

    cluster = str(title) + str(c)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=1500, height=1500,
                        background_color='white',
                        stopwords=stopwords,
                        min_font_size=10).generate(data)

    # plottin the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, cmap='binary')
    plt.axis("off")
    plt.title(cluster)
    plt.tight_layout(pad=0)
    plt.show()

    # to save the wordcloud to a location
    # path = r""
    # os.path.join(path, cluster)
    # plt.savefig(os.path.join(path, cluster))


# Creating wordcloud for LDA
c = 0
print("Generating word clouds for LDA\n")
for topic in lda_topics:
    word_cloud(topic, 'lda_', c)
    c+=1
# Creating wordcloud for K-Means
print("Generating word clouds for K-Means\n")
c = 0
for cluster in word_list_kmeans:
    word_cloud(str(cluster), 'cluster',c)
    c+=1
