import pandas as pd
import numpy as np
movies=pd.read_csv("./tmdb_5000_movies.csv")
credits=pd.read_csv("./tmdb_5000_credits.csv")
movies=movies.merge(credits,on="title")
movies=movies[['movie_id','title', 'overview', 'genres', 'keywords','cast','crew']]
movies.dropna(inplace=True)
print(movies.head())
import ast as ast

def convert(obj):
  L=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L

movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
def convert3(obj):
  L=[]
  counter=0
  for i in ast.literal_eval(obj):
    if counter != 3:
      L.append(i['name'])
      counter+=1
    else: 
      break
  return L

movies['cast']=movies['cast'].apply(convert3)

def fetch_director(obj):
  L=[]
  for i in ast.literal_eval(obj):
    if i['job']=='Director':
      L.append(i['name'])
      break
  return L

movies['crew']=movies['crew'].apply(fetch_director)

movies['overview']=movies['overview'].apply(lambda x: x.split())

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

new_df=movies[['movie_id','title','tags']]

new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stem(text):
  L=[]
  for i in text.split():
    L.append(ps.stem(i))
  return " ".join(L)

new_df['tags']=new_df['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5000, stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)



def recommend(movie):
  L=[]
  movie_index_tuple = new_df[new_df['title']==movie].index
  movie_index=movie_index_tuple[0]
  distances=similarity[movie_index]
  movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  for i in movies_list:
    moviee = {"movie_id":new_df.iloc[i[0]].movie_id,
           "original_title":new_df.iloc[i[0]].title}
    L.append(moviee)

  return L

import json
class NpEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return super(NpEncoder, self).default(obj)

#json_object = json.dumps(L, cls=NpEncoder) 
#print(json_object)

from flask import Flask,request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
#run_with_ngrok(app)

@app.route("/")
@cross_origin()
def index():
  return "<h1>Hello World!</h1>"
@app.route('/search', methods=['GET'])
@cross_origin()
def search():
  args = request.args
  title=args.get("title", default="", type=str)
  L=recommend(title)
  jj=json.dumps(L, cls=NpEncoder) 
  print(jj)
  return jj
#app.run()