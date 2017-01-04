#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 11:03:02 2016

@author: srikant
"""
import pandas as pd
import os
import cl_tester
import requests
import json
import time

EMAIL_DIR = 'Enron/'
IGNORE_CATS =   ['untitled',
                 'all_documents',
                 'inbox',
                 'notes_inbox',
                 'sent',
                 'sent_items',
                 '_sent_mail',
                 '_sent',
                 'deleted_items',
                 'junk',
                 'enron_mentions',
                 'misc',
                 'misc_',
                 'archive',
                 'archives']
IGNORE_CATS += ['discussion_threads']

CLASSIFY_URL = 'http://skunkworks.ignitesol.com:5000/classify'

def list_users_n_categories(EMAIL_DIR):
    data_folder = 'DataSets/' + EMAIL_DIR
    ds_n_cat = pd.Series()
    users_list = os.listdir(data_folder)
    for user_name in users_list:
        ds_n_cat.loc[EMAIL_DIR + user_name] = len(os.listdir(data_folder + user_name))
    return ds_n_cat


email_details = pd.DataFrame()
while email_details.empty:
    try:
        user_id = list_users_n_categories(EMAIL_DIR).sample().index[0]
        email_details = cl_tester.list_files_paths(user_id, ignore_list=IGNORE_CATS,
                                                   threshold=10).sample()
    except ValueError:
        user_id = list_users_n_categories(EMAIL_DIR).sample().index[0]

email_txt_path = email_details['filepath'].iloc[0]
email_text = open(email_txt_path, 'r').read()
email_true_category = email_details['category'].iloc[0]

def classify(user_id, email_text):
    classify_request = {'user_id':user_id,
                        'text':email_text,
                        'n_multi':3}
    classify_response = requests.post(CLASSIFY_URL, json=classify_request)
    return classify_response.json()

print('User ID:', user_id)
print('Email Path:', email_txt_path)
print('True Category:', email_true_category)

t1 = time.time()
classify_response = classify(user_id, email_text)
t2 = time.time()

print('Response:', classify_response)
print('Response Time: {:0.3f} sec'.format(t2-t1))