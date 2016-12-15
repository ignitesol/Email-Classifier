# -*- coding: utf-8 -*-
"""
@author: srikant

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


reload(email_classifier)


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
            print('Trained on {:0.0f} files @ {:0.3f} sec/file'.format(i + 1, t/(i+1)),flush=True)
    return


# predict_categories ##############################################################################
def predict_categories(cl, X_test, n_multi):
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
        top_categories = cl.classify(txt, n_multi)
        df_test.set_value(rowidx, 'Pred_Multi_Category', top_categories)
        df_test.set_value(rowidx, 'Pred_One_Category', top_categories[0])
        # try:
        #     df_test.set_value(rowidx, 'Pred_One_Category', top_categories[0])
        # except IndexError:
        #     df_test.set_value(rowidx, 'Pred_One_Category', '-')
        #     print('Error with' + file_path)
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
    print(df_pred[['Accuracy_One_Category','Accuracy_Multi_Category']].sum()/len(df_pred),'\n')
    return df_pred[column_order]


# walk through and list files in DataSet folder ###################################################
def list_files_paths(dir_name):
    data_dir = './DataSets/' + dir_name
    df_list = []
    for dirpath,dirnames,filenames in os.walk(data_dir):
        df = pd.DataFrame(filenames,columns=['filename'])
        df['filepath'] = df['filename'].map(lambda fname: dirpath + '/' + fname)
        df['category'] = dirpath.split(sep='/')[-1].strip().replace('.','_').replace('-','_')
        df_list.append(df)
    df_items_categories = pd.concat(df_list, axis=0, ignore_index=True)
    return df_items_categories


# train and test on datadir #######################################################################
def train_test_on_datadir(cl, n_multi=3, samplefrac=0.2, randstate=42, testsize=0.2):
    dir_name = cl.user_id
    print('\nTotal number of items in persistent training data:', cl.ds_category_count.sum())
    print('Number of items in persistent training data, by category:')
    print(cl.ds_category_count)

    df_items_categories = list_files_paths(dir_name).sample(frac=samplefrac, replace=False,
                                                            random_state = randstate*42)
    X = df_items_categories['filepath']
    y = df_items_categories['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testsize,
                                                        random_state = randstate)
    n_train = X_train.size
    n_test = X_test.size
    print('\nTotal Sample Size:', n_train+n_test)
    print('Training Sample Size:', n_train)
    print('Testing Sample Size:', n_test)

    # train the classifier on training dataset
    print('\nTraining on new data ...')
    t1 = time.time()
    train_classifier(cl, X_train, y_train)
    t2 = time.time()
    print('\nFinished Training on {:0.0f} items in in {:0.0f} sec - {:0.0f} items per sec.'\
          .format(n_train, t2-t1, n_train/(t2-t1)) )
    print('\nTotal number of items used:', cl.ds_category_count.sum())
    print('Number of items trained, by category:')
    print(cl.ds_category_count)

    # test the classifier on testing dataset
    print('\nTesting ...')
    t1 = time.time()
    df_prediction = predict_categories(cl, X_test, n_multi)
    t2 = time.time()
    print('\nFinished Classification of {:0.0f} items in {:0.0f} sec - {:0.0f} items per sec.'\
          .format(n_test,t2-t1, n_test/(t2-t1)) )

    # find accuracy of prediction
    df_test = prediction_accuracy(df_prediction, y_test)
    # return accuracy and predictions
    return df_test


# test on N random items ##########################################################################
def random_test(n_items, user_id='20_newsgroup', n_multi=2):
    datadir = user_id
    df_items = list_files_paths(datadir)
    items = df_items.sample(n_items)
    items_paths = items.filepath
    items_cats = items.category
    cl_ll = email_classifier.LogLikelihoodClassifier(user_id = user_id)
    # cl_nb = email_classifier.BernoulliNBclassifier(user_id = user_id)
    cl_ll.load_data_from_hdf5()
    # cl_nb.load_data_from_hdf5()
    print('\n\nTest Sample :\n')
    print(items)
    # print('\n\nBernoulli Naive Bayes Classifier : ')
    # print(prediction_accuracy(predict_categories(cl_nb,items_paths,n_multi=n_multi),items_cats))
    print('\n\nLog-Likelihood Classifier : ')
    print(prediction_accuracy(predict_categories(cl_ll,items_paths,n_multi=n_multi),items_cats))
    return cl_ll


###################################################################################################


items_list = list_files_paths('20_newsgroup')
users_dict = None


###################################################################################################
def user_session(uid, n_ops):
    global users_dict
    cl = users_dict[uid]['cl']
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
            print('UserID\t', cl.user_id, '\t- Training on\t\t', item_details.filepath.values[0])
        else:
            cl.classify(item,n_multi=2)
            print('UserID\t', cl.user_id,'\t- Classification of\t',item_details.filepath.values[0])
    return cl


###################################################################################################
def load_test_hdf5db(n_users, n_ops, id_suffix='20_newsgroup'):
    global users_dict
    hdf_db = './hdf5_db/'
    print('\n\nChecking and replicating main_db if a user_db doesnt exist ... ',end='')
    users_dict = {uid+1:{'user_id':id_suffix + '_' + str(uid+1),
                        'hdf_db':hdf_db+id_suffix+'_'+str(uid+1)+'.h5'} for uid in range(n_users)}
    for id in users_dict.keys():
        if os.path.isfile(users_dict[id]['hdf_db']):
            continue
        else:
            shutil.copyfile(hdf_db+id_suffix+'.h5', users_dict[id]['hdf_db'])
    print('done.')
    # initialising classes for each
    print('\nCreating instances of LogLikelihoodClassifier for each user ... ', end='')
    uids = users_dict.keys()
    for uid in uids:
        cl_ll = email_classifier.LogLikelihoodClassifier(user_id = users_dict[uid]['user_id'])
        users_dict[uid]['cl'] = cl_ll
        cl_ll.load_data_from_hdf5()
    print('done.')
    # using joblib to simulate parallel training and classification
    njobs = 10
    parallelizer = joblib.Parallel(n_jobs=njobs) # backend='threading')
    task_iterator = (joblib.delayed(user_session)(uid, n_ops) for uid in uids)
    cl_list = parallelizer(task_iterator)
    for cl in cl_list:
        users_dict[int(cl.user_id.split('_')[-1])]['cl'] = cl
    return

