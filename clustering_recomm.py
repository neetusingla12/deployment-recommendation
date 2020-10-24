# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 18:51:31 2020

@author: Neetu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 00:59:10 2020

@author: Neetu
"""


##################### clustering docs with tfidf and kmeans 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import nltk
import pickle
#data = pd.read_csv("old_with_date.csv")
data = pd.read_excel("C:\\Users\\Neetu\\Desktop\\study material DS\\sabudh_assignment\\machine learning\\news recommender system group assignment\\scrapednews.xlsx")
data.head()
article_id=list(range(1,data.shape[0]+1))
data['article_id']=article_id

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

import re
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
   
    lemmas= [lemmatizer.lemmatize(t) for t in filtered_tokens]
    return lemmas


from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_vectorizer = TfidfVectorizer(stop_words='english',tokenizer=tokenize_and_stem)
X = tfidf_vectorizer.fit_transform(data.title) #fit the vectorizer to title
print(X.shape)
terms = tfidf_vectorizer.get_feature_names()

#finding optimal clusters
def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    
    
find_optimal_clusters(X, 15)
clusters = MiniBatchKMeans(n_clusters=6, init_size=1024, batch_size=2048, random_state=20).fit_predict(X)
data['cluster']=clusters

##top keywords in each cluster
def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
            
get_top_keywords(X, clusters, tfidf_vectorizer.get_feature_names(), 10)

data['cluster'].value_counts()

data['date'] = pd.to_datetime(data['date'])
data['date'] = pd.to_datetime(data['date'], utc=True)
data = data.sort_values(by='date',ascending=False)

#data.groupby('cluster',as_index=False).cluster.count()
#data.shape

new=data.groupby('cluster', group_keys=False,as_index=False).apply(lambda x: x.sort_values('date', ascending=False)).groupby('cluster').head(2)
#new
news_recommended=new[['article_id','link','text','title','date','cluster']]
text1=list(news_recommended['title'])
nn=[]
nn.append(text1)
nn
#print("news recommended are",text1)
pickle.dump(nn,open('model1.pkl','wb'))

