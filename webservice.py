#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:42:16 2016

@author: srikant
"""

import email_classifier
import json
import pymongo
from collections import OrderedDict
from flask import Flask, request, jsonify

MONGO_DB = pymongo.MongoClient().email_classifier_db

app = Flask(__name__)
# export FLASK_APP=/home/srikant/Workspace/Email_Classifier/webservice.py


@app.route('/')
def hello_world():
    message =   '''
                <!doctype html>
                <title>Email Classifier</title>
                <b>To Classify an email:</b>
                POST request to http://127.0.0.1:5000/ with {'user_id':id,
                                                             'email_text':txt}
                <p></p>
                <b>To Train the classifier:</b>
                POST request to http://127.0.0.1:5000/ with {'user_id':id,
                                                             'email_text':txt,
                                                             'category':category}
                '''
    return message


@app.route('/classify', methods=['POST']) # Post [user_id, email_txt]
def classify():
    content = request.get_json()
    # get user_id
    user_id = content['user_id']
    # get email_text
    email_text = content['email_text']
    # response json
    response_json = OrderedDict([('pred_categories', []),
                                 ('e_init','-'),
                                 ('e_load','-'),
                                 ('e_classify','-'),
                                 ('e_save','-')])
    # initialize classifier calls for user_id
    try:
        cl = email_classifier.LogLikelihoodClassifier(user_id)
        response_json['e_init'] = 'classifier initialisation done'
    except Exception as e:
        response_json['e_init'] = str(e)
        return jsonify(response_json)
    # load previous training data from db
    try:
        cl.load_data_from_mongodb(MONGO_DB)
        response_json['e_load'] = 'loaded training data from mongo'
    except Exception as e:
        response_json['e_load'] = str(e)
        return jsonify(response_json)
    # classify email
    try:
        response_json['pred_categories'] = cl.classify(email_text,n_multi=2)
        response_json['e_classify'] = 'classification done'
    except Exception as e:
        response_json['e_classify'] = str(e)
        return jsonify(response_json)
    # save training data to db - not required during classification. just testing
    try:
        cl.save_data_to_mongodb(MONGO_DB)
        response_json['e_save'] = 'saved data back to mongo_db'
    except Exception as e:
        response_json['e_save'] = str(e)
        return jsonify(response_json)
    # return category and success/failure notification as response
    return jsonify(response_json)


@app.route('/train', methods=['POST']) # Post [user_id, email_txt, email_category]
def train():
    # get user_id
    # get email text
    # get email_category
    # initialize classifier calls for user_id
    # load previous training data from db
    # train on email
    # save training data to db
    # return success/failure notification as response
    return #response