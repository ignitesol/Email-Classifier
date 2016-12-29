#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 11:03:02 2016

@author: srikant
"""
import pandas as pd
import os
import email_classifier
import cl_tester
import requests
import json
import time

EMAIL_DIR = 'Enron/'

def list_users_n_categories(EMAIL_DIR):
    data_folder = 'DataSets/' + EMAIL_DIR
    ds_n_cat = pd.Series()
    users_list = os.listdir(data_folder)
    for user_name in users_list:
        ds_n_cat.loc[EMAIL_DIR + user_name] = len(os.listdir(data_folder + user_name))
    return ds_n_cat

user_id = list_users_n_categories(EMAIL_DIR).sample().index[0]
email_details = cl_tester.list_files_paths(user_id).sample()
email_txt_path = email_details['filepath'].iloc[0]
email_text = open(email_txt_path, 'r').read()
email_true_category = email_details['category'].iloc[0]

def classify(user_id, email_text):
    classify_request_json = {'user_id':user_id,
                             'email_text':email_text}
    classify_response = requests.post('http://127.0.0.1:5000/classify', json=classify_request_json)
    print('Predicted Categories:', classify_response.json()['pred_categories'])
    return classify_response.json()

print('User ID:', user_id)
print('Email Path:', email_txt_path)
print('True Category:', email_true_category)
t1 = time.time()
classify_response = classify(user_id, email_text)
t2 = time.time()
print('\n', classify_response)
print('Response Time: {:0.3f}'.format(t2-t1))