import pickle
import numpy as np
#json_object = json.dumps(L, cls=NpEncoder) 
#print(json_object)


model = pickle.load(open('model.pkl','rb'))
new_df = pickle.load(open('model2.pkl','rb'))

#print(model.predict("Avatar"))
#run_with_ngrok(app)
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
    
from flask import Flask,request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def index():
  return "<h1>Hello World!</h1>"
@app.route('/search', methods=['GET'])
@cross_origin()
def search():
  args = request.args
  title=args.get("title", default="", type=str)
  L=[]
  movie_index_tuple = new_df[new_df['title']==title].index
  movie_index=movie_index_tuple[0]
  distances=model[movie_index]
  movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  for i in movies_list:
    moviee = {"movie_id":new_df.iloc[i[0]].movie_id,
           "original_title":new_df.iloc[i[0]].title}
    L.append(moviee)

  jj=json.dumps(L, cls=NpEncoder) 
  print(jj)
  return jj
#app.run()