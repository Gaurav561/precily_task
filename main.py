# from flask import Flask, request, render_template
# from distutils.log import debug
# import json
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# import flask


# app = flask.Flask(__name__)

# model = SentenceTransformer('stsb-roberta-large')


# @app.route('/',methods=["GET","POST"])
# def index():
#     return flask.render_template("index.html")

# @app.route('/score',methods=['GET'])
# def sim_score():

#     text1 = flask.request.form.get('text1')
#     text2 = flask.request.form.get('text2')
#     print("Retrieved Text"+text1)
#     embedding1 = model.encode(text1, convert_to_tensor=True)
#     embedding2 = model.encode(text2, convert_to_tensor=True)
#     cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
#     print("Similarity score:", cosine_scores.item())


#     return {"score" :cosine_scores.item() }


# @app.route('/', methods=["GET", "POST"])
# def gfg():
#     if flask.request.method == "GET":
#         text1 = flask.request.form.get('text1')
#         text2 = flask.request.form.get('text2')
#         print("Retrieved Text"+str(text1))
#         embedding1 = model.encode(str(text1), convert_to_tensor=True)
#         embedding2 = model.encode(str(text2), convert_to_tensor=True)
#         cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
#         print("Similarity score:", cosine_scores.item())

#         return {"score": cosine_scores.item()}

#     return flask.render_template("index.html")


# if __name__ == '__main__':
#     app.run(debug=True)


# importing Flask and other modules
from asyncio.windows_events import NULL
from flask import Flask, request, render_template 
from sentence_transformers import SentenceTransformer, util

# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function
model = SentenceTransformer('stsb-roberta-large')


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template("index.html")



@app.route('/score', methods=["GET", "POST"])
def gfg():
    # text1 = " "
    # text2 = " "
    # if(request.method == "post"):
    text1 = request.form.get("text1")
    text2 = request.form.get("text2") 
    print(text1)
    print(type(text1))
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    print("Similarity score:", cosine_scores.item())
    # return {"score":cosine_scores.item()}
    return {"similarity score":cosine_scores.item()}
    # return render_template("index.html")
    
if __name__ == '__main__':
    app.run(debug=True)
