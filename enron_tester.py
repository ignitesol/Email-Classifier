#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 12:46:39 2016

@author: srikant
"""

import cl_tester
import email_classifier
import pymongo
import pandas as pd
import os
from importlib import reload
reload(cl_tester)
reload(email_classifier)

dir_name = 'Enron/'

ignore_list =   ['untitled',
                 'all_documents',
                 'inbox',
                 'notes_inbox',
                 'sent',
                 'sent_items',
                 '_sent_mail',
                 'deleted_items',
                 'junk',
                 'enron_mentions',
                 'misc',
                 'misc_',
                 'discussion_threads']

                 
def list_users_n_categories(dir_name, min_n_cats=10):
    data_folder = 'DataSets/' + dir_name
    ds_n_cat = pd.Series()
    users_list = os.listdir(data_folder)
    for user_name in users_list:
        ds_n_cat.loc[dir_name + user_name] = len(os.listdir(data_folder + user_name))
    return ds_n_cat


def save_to_db(cl):
    db = pymongo.MongoClient().email_classifier_db
    cl.save_data_to_mongodb(db)
    cl.save_data_to_sqlite()
    

def test_one_user(user_id, ignore_list=ignore_list, threshold=10, n_cat_limit=20, save_db=False,
                  rndseed=1):
    user_dict = {}
    print('\n\nUser Name:', user_id,'\n')
    cat_count_all = cl_tester.list_files_paths(user_id)['category'].value_counts()
    n_cat_all = cl_tester.list_files_paths(user_id)['category'].nunique()
    n_cat = cl_tester.list_files_paths(user_id, ignore_list=ignore_list,
                                       threshold=threshold)['category'].nunique()
    if n_cat <=5:
        threshold = 5
    while n_cat > n_cat_limit:
        threshold += 1
        n_cat = cl_tester.list_files_paths(user_id, ignore_list=ignore_list,
                                           threshold=threshold)['category'].nunique()
        print('With threshold of',threshold,', number of categories:',n_cat)
    n_cat = cl_tester.list_files_paths(user_id, ignore_list=ignore_list,
                                       threshold=threshold)['category'].nunique()
    cat_count = cl_tester.list_files_paths(user_id, ignore_list=ignore_list,
                                          threshold=threshold)['category'].value_counts()
    user_dict['n_cat_all'] = n_cat_all
    user_dict['n_cat_used'] = n_cat
    user_dict['cat_count_all'] = cat_count_all
    user_dict['cat_count_used'] = cat_count
    print('\nNumber of categories (including inbox, sent, deleted, etc...)\t:', n_cat_all)
    print('Number of categories with email count >= threshold ({:0.0f})\t\t: {:0.0f}'\
          .format(threshold, n_cat))
    print('Categories and email count in categories being tested:')
    print(cat_count)
    if n_cat <=1:
        print('Ignoring user. Number of valid categories <= 1.')
        return {}
    cl = email_classifier.LogLikelihoodClassifier(user_id)
    df_test, df_accu = cl_tester.train_test_on_datadir(cl, n_multi=2, frac=1.0, rndseed=rndseed,
                                                       tfrac=0.2, ignore_list=ignore_list,
                                                       threshold=threshold)
    user_dict['classifier'] = cl
    user_dict['test_results'] = df_test
    user_dict['test_accuracy'] = df_accu
    print('\n')
    print(df_accu.map(lambda x: '{:0.2f}'.format(x)))
    print('\n')
    print(df_test)
    if save_db:
        print('\nSaving to db(s) ... ', end='')
        save_to_db(cl)
        print('done.')
    return user_dict


def test_rand_users(n_users=5, save_db=False, rndseed=1):
    all_users = list_users_n_categories(dir_name)
    rand_n_users = all_users.sample(n_users, random_state = rndseed).index
    users_dicts = {user_id:{} for user_id in rand_n_users}
    for user_id in rand_n_users:
        users_dicts[user_id] = test_one_user(user_id, save_db=save_db, rndseed=rndseed)
    return users_dicts

    
def test_top_users(n_users=5, save_db=False, rndseed=1):
    all_users = list_users_n_categories(dir_name)
    top_n_users = all_users.nlargest(n_users).index
    users_dicts = {user_id:{} for user_id in top_n_users}
    for user_id in top_n_users:
        users_dicts[user_id] = test_one_user(user_id, save_db=save_db, rndseed=rndseed)
    return users_dicts

    
def test_top_nth_user(n_th, save_db=False, rndseed=1):
    all_users = list_users_n_categories(dir_name)
    top_nth_user = all_users.sort_values(ascending=False).index[n_th-1]
    users_dicts = {top_nth_user:test_one_user(top_nth_user, save_db=save_db, rndseed=rndseed)}
    return users_dicts
    
    
def random_test(user_id, n_items=10, n_multi=2, ignore_list=ignore_list, threshold=10):
    cl_tester.random_test(n_items=n_items, user_id=user_id, n_multi=n_multi,
                          ignore_list=ignore_list, threshold=threshold)
    
