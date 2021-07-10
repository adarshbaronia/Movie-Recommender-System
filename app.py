from flask import Flask, render_template, request
#import requests
import Movie_Recommender
from Movie_Recommender import doc_sim_df, movies_list
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/movie_recommender", methods=['POST'])

def movie_recommender( movies=movies_list, doc_sims=doc_sim_df):
    #find movie id
    if request.method == "POST":    
        movie = request.form['movie']
    if movie not in '':
        movie_idx=np.where(movies==movie)[0][0]
        #get movie similarities
        movie_similarities=doc_sims.iloc[movie_idx].values
        #similar_movies ids top 5
        similar_movie_ids=np.argsort(-movie_similarities)[1:6]
        #top five movies
        similar_movies=movies[similar_movie_ids]
        
        #return 5 similar movies
        return render_template('result.html',result=list(similar_movies))   
    else:
        return render_template('index.html')
#@app.route('/submit', methods =['POST','GET'])


if __name__=="__main__":
    app.run(debug=True)
    
    