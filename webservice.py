#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:42:16 2016

@author: srikant
"""

import email_classifier
import pymongo
import json
import time
import requests
from flask import Flask, request, jsonify


MONGO_DB = pymongo.MongoClient().email_classifier_db

GA_URL = 'http://www.google-analytics.com/collect'
GA_ID = 'UA-91609567-1'

app = Flask(__name__)
# export FLASK_APP=/home/srikant/Workspace/Email_Classifier/webservice.py


@app.route('/')
def hello_world():
    message =   '''
                <!doctype html>
                <title>Email Classifier</title>
                <b>To Classify an email:</b>
                POST request to http://skunkworks.ignitesol.com:5000/ with
                {'user_id':id, 'text':txt, 'n_multi':n_multi}
                <p></p>
                <b>To Train the classifier:</b>
                POST request to http://skunkworks.ignitesol.com:5000/ with 
                {'user_id':id, 'text':txt, 'categories':[categories]}
                '''
    return message


@app.route('/classify', methods=['POST']) # Post [user_id, email_txt, n_multi]
def classify():
    t1=time.time()
    content = request.get_json()
    # get user_id
    user_id = content['user_id']
    # get email_text
    email_text = content['text']
    # get number of predicted catergories
    n_multi = content['n_multi']
    # response json
    response_dict = {'predicted_categories':[],'response_message':[]}
    # ga payload
    ga_payload = {'v':'1',
                  't':'event',
                  'tid': GA_ID,
                  'cid': '1',
                  'ec': 'classifier_log',
                  'ea': 'classification',
                  'el': '-'}
    # initialize classifier calls for user_id
    try:
        cl = email_classifier.LogLikelihoodClassifier(user_id)
        response_dict['response_message'].append('Initialized Classifier')
    except Exception as e:
        response_dict['response_message'].append(str(e))
        ga_payload['el'] = str(e)
        r = requests.post(GA_URL, data = ga_payload)
        return jsonify(response_dict)
    # load previous training data from db
    try:
        cl.load_data_from_mongodb(MONGO_DB)
        response_dict['response_message'].append('Loaded Training Data')
    except Exception as e:
        response_dict['response_message'].append(str(e))
        ga_payload['el'] = str(e)
        r = requests.post(GA_URL, data = ga_payload)
        return jsonify(response_dict)
    # classify email text
    try:
        response_dict['predicted_categories'] = cl.classify(email_text, n_multi=n_multi)
        response_dict['response_message'].append('Classified Text')
    except Exception as e:
        response_dict['response_message'].append(str(e))
        ga_payload['el'] = str(e)
        r = requests.post(GA_URL, data = ga_payload)
        return jsonify(response_dict)
    # return category and success/failure notification as response
    t2=time.time()
    ga_payload['el'] = 'successful'
    ga_payload['ev'] = (t2-t1)*1000
    r = requests.post(GA_URL, data = ga_payload)
    print('Processing Time: {:0.3f} sec'.format(t2-t1))
    return jsonify(response_dict)


@app.route('/train', methods=['POST']) # Post [user_id, email_txt, email_category]
def train():
    t1=time.time()
    content = request.get_json()
    # get user_id
    user_id = content['user_id']
    # get email_text
    email_text = content['text']
    # get email_category
    email_categories = content['categories']
    # response json
    response_dict = {'response_message':[], 'n_items_pre':0, 'n_items_post':0}
    # ga payload
    ga_payload = {'v':'1',
                  't':'event',
                  'tid': GA_ID,
                  'cid': '1',
                  'ec': 'classifier_log',
                  'ea': 'training',
                  'el': '-'}
    # initialize classifier calls for user_id
    try:
        cl = email_classifier.LogLikelihoodClassifier(user_id)
        response_dict['response_message'].append('Initialized Classifier')
    except Exception as e:
        response_dict['response_message'].append(str(e))
        ga_payload['el'] = str(e)
        r = requests.post(GA_URL, data = ga_payload)
        return jsonify(response_dict)
    # load previous training data from db
    try:
        cl.load_data_from_mongodb(MONGO_DB)
        response_dict['response_message'].append('Loaded Training Data')
        response_dict['n_items_pre'] = cl.ds_category_count.sum()
    except Exception as e:
        response_dict['response_message'].append(str(e))
    # train on email text
    try:
        cl.train(email_text, email_categories)
        response_dict['response_message'].append('Trained Classifier')
        response_dict['n_items_post'] = cl.ds_category_count.sum()
    except Exception as e:
        response_dict['response_message'].append(str(e))
        ga_payload['el'] = str(e)
        r = requests.post(GA_URL, data = ga_payload)
        return jsonify(response_dict)
    # save updated training data to db
    try:
        cl.save_data_to_mongodb(MONGO_DB)
        response_dict['response_message'].append('Updated Training Data')
    except Exception as e:
        response_dict['response_message'].append(str(e))
        ga_payload['el'] = str(e)
        r = requests.post(GA_URL, data = ga_payload)
        return jsonify(response_dict)
    # return success/failure notification as response
    t2=time.time()
    ga_payload['el'] = 'successful'
    ga_payload['ev'] = (t2-t1)*1000
    r = requests.post(GA_URL, data = ga_payload)
    print('Processing Time: {:0.3f} sec'.format(t2-t1))
    return jsonify(response_dict)