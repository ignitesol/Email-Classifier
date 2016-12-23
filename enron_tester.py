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
from random import randint
from importlib import reload
reload(cl_tester)
reload(email_classifier)

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

MONGO_DB = pymongo.MongoClient().email_classifier_db

def list_users_n_categories(EMAIL_DIR):
    data_folder = 'DataSets/' + EMAIL_DIR
    ds_n_cat = pd.Series()
    users_list = os.listdir(data_folder)
    for user_name in users_list:
        ds_n_cat.loc[EMAIL_DIR + user_name] = len(os.listdir(data_folder + user_name))
    return ds_n_cat


def test_one_user(user_id, ignore_list=IGNORE_CATS, threshold=10, n_cat_limit=25, save_db=False,
                  rndseed=1):
    user_dict = {}
    print('\n\nUser Name:', user_id)
    cat_count_all = cl_tester.list_files_paths(user_id)['category'].value_counts()
    n_cat_all = cl_tester.list_files_paths(user_id)['category'].nunique()
    n_cat = cl_tester.list_files_paths(user_id, ignore_list=ignore_list,
                                       threshold=threshold)['category'].nunique()
    if n_cat <=5:
        print('Too few categories, setting base threshold to 5.')
        threshold = 5
    while n_cat > n_cat_limit:
        threshold += 1
        n_cat = cl_tester.list_files_paths(user_id, ignore_list=ignore_list,
                                           threshold=threshold)['category'].nunique()
        print('\rWith threshold of',threshold,', number of categories:',n_cat, end='')
    n_cat = cl_tester.list_files_paths(user_id, ignore_list=ignore_list,
                                       threshold=threshold)['category'].nunique()
    cat_count = cl_tester.list_files_paths(user_id, ignore_list=ignore_list,
                                          threshold=threshold)['category'].value_counts()
    user_dict['n_cat_all'] = n_cat_all
    user_dict['n_cat_used'] = n_cat
    user_dict['cat_count_all'] = cat_count_all
    user_dict['cat_count_used'] = cat_count
    user_dict['threshold'] = threshold
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
        cl.save_data_to_mongodb(MONGO_DB)
        cl.save_data_to_sqlite()
        print('done.')
    return user_dict


def test_rand_users(n_users=5, base_threshold=10, n_cat_limit=25, save_db=False, rndseed=1):
    all_users = list_users_n_categories(EMAIL_DIR)
    rand_n_users = all_users.sample(n_users, random_state = rndseed).index
    users_dicts = {user_id:{} for user_id in rand_n_users}
    users_df = pd.DataFrame(index=rand_n_users, columns=['N_categories_all','N_categories_used',
                                                         'Threshold',
                                                         'N_emails_all','N_emails_used',
                                                         'Accuracy_One_Category',
                                                         'Accuracy_Multi_Category'], dtype='float')
    for user_id in rand_n_users:
        user_dict = test_one_user(user_id, threshold=base_threshold, n_cat_limit=n_cat_limit,
                                  save_db=save_db, rndseed=rndseed)
        users_dicts[user_id] = user_dict
        if user_dict == {}:
            continue
        else:
            users_df.loc[user_id,'N_categories_all'] = user_dict['n_cat_all']
            users_df.loc[user_id,'N_categories_used'] = user_dict['n_cat_used']
            users_df.loc[user_id,'Threshold'] = user_dict['threshold']
            users_df.loc[user_id,'N_emails_all'] = user_dict['cat_count_all'].sum()
            users_df.loc[user_id,'N_emails_used'] = user_dict['cat_count_used'].sum()
            users_df.loc[user_id,'Accuracy_One_Category'] = user_dict['test_accuracy']['Accuracy_One_Category']
            users_df.loc[user_id,'Accuracy_Multi_Category'] = user_dict['test_accuracy']['Accuracy_Multi_Category']
    return users_df, users_dicts


def test_top_users(n_users=5, base_threshold=10, n_cat_limit=25, save_db=False, rndseed=1):
    all_users = list_users_n_categories(EMAIL_DIR)
    top_n_users = all_users.nlargest(n_users).index
    users_dicts = {user_id:{} for user_id in top_n_users}
    users_df = pd.DataFrame(index=top_n_users, columns=['N_categories_all','N_categories_used',
                                                        'Threshold',
                                                         'N_emails_all','N_emails_used',
                                                         'Accuracy_One_Category',
                                                         'Accuracy_Multi_Category'], dtype='float')
    for user_id in top_n_users:
        user_dict = test_one_user(user_id, threshold=base_threshold, n_cat_limit=n_cat_limit,
                                  save_db=save_db, rndseed=rndseed)
        users_dicts[user_id] = user_dict
        if user_dict == {}:
            continue
        else:
            users_df.loc[user_id,'N_categories_all'] = user_dict['n_cat_all']
            users_df.loc[user_id,'N_categories_used'] = user_dict['n_cat_used']
            users_df.loc[user_id,'Threshold'] = user_dict['threshold']
            users_df.loc[user_id,'N_emails_all'] = user_dict['cat_count_all'].sum()
            users_df.loc[user_id,'N_emails_used'] = user_dict['cat_count_used'].sum()
            users_df.loc[user_id,'Accuracy_One_Category'] = user_dict['test_accuracy']['Accuracy_One_Category']
            users_df.loc[user_id,'Accuracy_Multi_Category'] = user_dict['test_accuracy']['Accuracy_Multi_Category']
    return users_df, users_dicts


def test_top_nth_user(nth_user, base_threshold=10, n_cat_limit=25, save_db=False, rndseed=1):
    all_users = list_users_n_categories(EMAIL_DIR)
    top_nth_user = all_users.sort_values(ascending=False).index[nth_user-1]
    user_df = pd.DataFrame(index=[top_nth_user], columns=['N_categories_all','N_categories_used',
                                                          'Threshold',
                                                          'N_emails_all','N_emails_used',
                                                          'Accuracy_One_Category',
                                                          'Accuracy_Multi_Category'], dtype='float')
    user_dict = test_one_user(top_nth_user, threshold=base_threshold,
                              n_cat_limit=n_cat_limit, save_db=save_db, rndseed=rndseed)
    if user_dict == {}:
        return user_df
    else:
        user_df.loc[top_nth_user,'N_categories_all'] = user_dict['n_cat_all']
        user_df.loc[top_nth_user,'N_categories_used'] = user_dict['n_cat_used']
        user_df.loc[top_nth_user,'Threshold'] = user_dict['threshold']
        user_df.loc[top_nth_user,'N_emails_all'] = user_dict['cat_count_all'].sum()
        user_df.loc[top_nth_user,'N_emails_used'] = user_dict['cat_count_used'].sum()
        user_df.loc[top_nth_user,'Accuracy_One_Category'] = user_dict['test_accuracy']['Accuracy_One_Category']
        user_df.loc[top_nth_user,'Accuracy_Multi_Category'] = user_dict['test_accuracy']['Accuracy_Multi_Category']
    return user_df, user_dict


def random_test(nth_user, n_items=10, n_multi=2, ignore_list=IGNORE_CATS, threshold=25):
    all_users = list_users_n_categories(EMAIL_DIR)
    top_nth_user = all_users.sort_values(ascending=False).index[nth_user-1]
    print('\n\nUser Name:', top_nth_user,'\n')
    cl = cl_tester.random_test(n_items=n_items, user_id=top_nth_user, n_multi=n_multi,
                               ignore_list=ignore_list, threshold=threshold)
    return cl

#test_top_nth_user(randint(1,50), base_threshold=10, n_cat_limit=25, rndseed=1)
