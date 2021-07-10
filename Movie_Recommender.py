import nltk
#nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
df=pd.read_csv('tmdb_5000_movies.csv')
df.describe()

df=df[['title','tagline','overview','popularity']]
df.tagline = df.tagline.fillna('',inplace=True)
#df.dropna(inplace=True)
df['description']=df['tagline'].map(str)+' '+df['overview']
df=df.sort_values(by='popularity', ascending=False)

import re
import numpy as np
stop_words=nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', str(doc), re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    #filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus=np.vectorize(normalize_document)
norm_corpus=normalize_corpus(list(df['description']))
len(norm_corpus)

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(ngram_range=(1,2), min_df=2)
tfidf_matrix = tf.fit_transform(norm_corpus)

#compute pairwise document similarity
from sklearn.metrics.pairwise import cosine_similarity
doc_sim=cosine_similarity(tfidf_matrix)
doc_sim_df=pd.DataFrame(doc_sim)
#doc_sim_df.head()
#building a movie recommender function to recommend top 5 similar movies
#movie title, movies title list and document similarities matrix will be given as imput
movies_list=df['title'].values
# def movie_recommender(movie_title, movies=movies_list, doc_sims=doc_sim_df):
#     #find movie id
#     if movie_title not in '':
#         movie_idx=np.where(movies==movie_title)[0][0]
#         #get movie similarities
#         movie_similarities=doc_sims.iloc[movie_idx].values
#         #similar_movies ids top 5
#         similar_movie_ids=np.argsort(-movie_similarities)[1:6]
#         #top five movies
#         similar_movies=movies[similar_movie_ids]
#         #return 5 similar movies
#         return similar_movies
#     else:
#         print('Movie Not in List')
    
#popular_movies = ['Minions', 'Interstellar', 'Deadpool', 'Jurassic World',  'The Lord of the Rings: The Fellowship of the Ring',  
#              'Harry Potter and the Chamber of Secrets']

#for movie in popular_movies:
#    print('Movie : ', movie)
#    print('Top 5 recommended movies: ', movie_recommender(movie_title=movie,
#                                                          movies=movies_list, doc_sims=doc_sim_df))
    
#    print("_______________")




#print(movie_recommender('Minions',movies_list, doc_sim_df))




