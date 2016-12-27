# -*- coding: utf-8 -*-
"""
@author: srikant bondugula

"""
import pandas as pd
import numpy as np
import json
import os
import shutil
from sklearn.model_selection import train_test_split
import time
import email_classifier
import joblib
from importlib import reload
import random
import sqlite3
from sqlite3 import DatabaseError
import pymongo


reload(email_classifier)

MONGO_DB = pymongo.MongoClient().email_classifier_db


# train it ########################################################################################
def train_classifier(cl, X_train, y_train):
    t=0
    for i, (rowidx, file_path) in enumerate(X_train.iteritems()):
        try:
            with open(file_path, 'r') as txt_file:
                txt = txt_file.read()
        except UnicodeDecodeError:
            continue
        category = y_train[rowidx]
        t1 = time.time()
        cl.train(txt, [category])
        t2 = time.time()
        t += t2-t1
        if not bool((i + 1) % 100):
            print('\rTrained {:5.0f} files\t{:0.3f} sec/file'.format(i + 1, t/(i+1)),
                  end='', flush=True)
    print('\rTrained {:5.0f} files\t{:0.3f} sec/file'.format(i + 1, t/(i+1)),
          end='', flush=True)
    return


# predict_categories ##############################################################################
def predict_categories(cl, X_test, n_multi):
    t=0
    df_test = pd.DataFrame(index=X_test.index)
    df_test['Pred_One_Category'] = '-'
    df_test['Pred_Multi_Category'] = '-'
    df_test['Accuracy_One_Category'] = 0
    df_test['Accuracy_Multi_Category'] = 0
    for i, (rowidx, file_path) in enumerate(X_test.iteritems()):
        try:
            with open(file_path, 'r') as txt_file:
                txt = txt_file.read()
        except UnicodeDecodeError:
            continue
        t1 = time.time()
        top_categories = cl.classify(txt, n_multi)
        t2 = time.time()
        t += t2-t1
        df_test.set_value(rowidx, 'Pred_Multi_Category', top_categories)
        df_test.set_value(rowidx, 'Pred_One_Category', top_categories[0])
        if not bool((i + 1) % 100):
            print('\rTested {:5.0f} files\t{:0.3f} sec/file'.format(i + 1, t/(i+1)),
                  end='', flush=True)
    print('\rTested {:5.0f} files\t{:0.3f} sec/file'.format(i + 1, t/(i+1)),
          end='', flush=True)
    return df_test


# accuracy of prediction ##########################################################################
def prediction_accuracy(df_pred, y_test):
    df_pred['True_Category'] = y_test
    def check_single_accuracy(row):
        return int( row['True_Category'] == row['Pred_One_Category'] )
    def check_multi_accuracy(row):
        return int( row['True_Category'] in row['Pred_Multi_Category'] )
    df_pred['Accuracy_One_Category'] = df_pred.apply(check_single_accuracy, axis=1)
    df_pred['Accuracy_Multi_Category'] = df_pred.apply(check_multi_accuracy, axis=1)
    column_order = ['True_Category','Pred_One_Category','Accuracy_One_Category',
                    'Pred_Multi_Category','Accuracy_Multi_Category']
    df_accu = df_pred[['Accuracy_One_Category','Accuracy_Multi_Category']].sum()/len(df_pred)
    return df_pred[column_order], df_accu


# walk through and list files in DataSet folder ###################################################
def list_files_paths(dir_name, ignore_list=[], threshold=0):
    data_dir = './DataSets/' + dir_name
    df_list = []
    for dirpath,dirnames,filenames in os.walk(data_dir):
        category = dirpath.split(sep='/')[-1].strip().replace('.','_').replace('-','_')
        if category not in ignore_list:
            df = pd.DataFrame(filenames,columns=['filename'])
            df['filepath'] = df['filename'].map(lambda fname: dirpath + '/' + fname)
            df['category'] = category
            df_list.append(df)
    df_items_categories = pd.concat(df_list, axis=0, ignore_index=True)
    ds_nitems = df_items_categories['category'].value_counts()
    category_list = ds_nitems[ds_nitems >= threshold].index
    return df_items_categories[df_items_categories['category'].isin(category_list)]


# train and test on datadir #######################################################################
def train_test_on_datadir(cl,n_multi=2,frac=0.2,rndseed=42,tfrac=0.2,ignore_list=[],threshold=0):
    dir_name = cl.user_id
#    print('\nTotal number of items in persistent training data:', cl.ds_category_count.sum())
#    print('Number of items in persistent training data, by category:')
#    print(cl.ds_category_count)

    df_items_categories = list_files_paths(dir_name, ignore_list=ignore_list, threshold=threshold)\
                                        .sample(frac=frac,replace=False,random_state = rndseed*42)
    X = df_items_categories['filepath']
    y = df_items_categories['category']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=tfrac,random_state=rndseed)
    n_train = X_train.size
    n_test = X_test.size
    n_total = n_train + n_test
    print('\nTotal Size:\t', n_total,'\t({:3.0f}% of available data)'.format(frac*100))
    print('Training Size:\t', n_train,'\t({:3.0f}% of {:3.0f} )'.format((1-tfrac)*100, n_total))
    print('Testing Size:\t', n_test,'\t({:3.0f}% of {:3.0f} )'.format(tfrac*100, n_total))

    # train the classifier on training dataset
    print('\nTraining :')
    t1 = time.time()
    train_classifier(cl, X_train, y_train)
    t2 = time.time()
    print('\nFinished Training on {:0.0f} items in in {:0.0f} sec - {:0.0f} items per sec.'\
          .format(n_train, t2-t1, n_train/(t2-t1)) )
#    print('\nTotal number of items used:', cl.ds_category_count.sum())
#    print('Number of items trained, by category:')
#    print(cl.ds_category_count)

    # test the classifier on testing dataset
    print('\nTesting :')
    t1 = time.time()
    df_prediction = predict_categories(cl, X_test, n_multi)
    t2 = time.time()
    print('\nFinished Classification of {:0.0f} items in {:0.0f} sec - {:0.0f} items per sec.'\
          .format(n_test,t2-t1, n_test/(t2-t1)) )

    # find accuracy of prediction
    df_test, df_accu = prediction_accuracy(df_prediction, y_test)
    # return accuracy and predictions
    return df_test, df_accu


# test on N random items ##########################################################################
def random_test(n_items, user_id='20_newsgroup', n_multi=2, ignore_list=[], threshold=0):
    datadir = user_id
    df_items = list_files_paths(datadir, ignore_list=ignore_list, threshold=threshold)
    cl_ll = email_classifier.LogLikelihoodClassifier(user_id = user_id)
    try:
        cl_ll.load_data_from_mongodb(MONGO_DB)
    except:
        print('Looks like there is no training data for this user')
        return cl_ll
    try:
        items = df_items.sample(n_items)
    except ValueError:
        print('Sample size = 0, try incresing threshold')
        return cl_ll
    items_paths = items.filepath
    items_cats = items.category

    print('\nTest Sample :\n')
    print(items)
    print('\nLog-Likelihood Classifier : ')
    df_test, df_accu = prediction_accuracy(predict_categories(cl_ll,items_paths,n_multi=n_multi),
                                           items_cats)
    print('\n')
    print(df_test)
    print('\n')
    print(df_accu)
    print('\n')
    return cl_ll


###################################################################################################
def user_session(uid, n_ops):
    global users_dict
    cl = email_classifier.LogLikelihoodClassifier(user_id = users_dict[uid]['user_id'])
    cl.load_data_from_mongodb(MONGO_DB)
    print('UserID\t', cl.user_id, '\tLoading training data\t',cl.ds_category_count.sum(),'items')
    for i_op in range(n_ops):
        item = None
        while item is None:
            try:
                item_details = items_list.sample(random_state=random.randint(0,100000))
                item = open(item_details.filepath.values[0],'r').read()
            except UnicodeDecodeError:
                continue
        category = item_details.category.values[0]
        if random.randint(0,1):
            cl.train(item,[category])
            print('UserID\t', cl.user_id, '\tTraining on\t\t', item_details.filepath.values[0])
        else:
            cl.classify(item,n_multi=2)
            print('UserID\t', cl.user_id,'\tClassification of\t',item_details.filepath.values[0])
    print('UserID\t', cl.user_id, '\tSaving training data\t',cl.ds_category_count.sum(),'items')
    cl.save_data_to_mongodb(MONGO_DB)
    return cl


###################################################################################################
def load_test(n_users, n_ops, id_suffix='20_newsgroup'):
    global user_id
    global users_dict
    global items_list
    user_id = id_suffix
    items_list = list_files_paths(user_id, ignore_list=[], threshold=0)
    db_dir = './sqlite_db/'
    print('\n\nChecking and replicating main_db if a user_db doesnt exist ... ',end='')
    users_dict = {uid+1:{'user_id':id_suffix + '_' + str(uid+1),
                        'db':db_dir+id_suffix+'_'+str(uid+1)+'.sqlite'} for uid in range(n_users)}
    uids = users_dict.keys()
    for uid in uids:
        if os.path.isfile(users_dict[uid]['db']):
            continue
        else:
            shutil.copyfile(db_dir+id_suffix+'.sqlite', users_dict[uid]['db'])
    print('done.\n\n')
    # using joblib to simulate parallel training and classification
    njobs = 10
    parallelizer = joblib.Parallel(n_jobs=njobs)
    task_iterator = (joblib.delayed(user_session)(uid, n_ops) for uid in uids)
    cl_list = parallelizer(task_iterator)
    for cl in cl_list:
        users_dict[int(cl.user_id.split('_')[-1])]['cl'] = cl
    return

