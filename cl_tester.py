# -*- coding: utf-8 -*-
"""
@author: srikant
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import time
import email_classifier
from importlib import reload

reload(email_classifier)

pd.set_option('display.max_columns',5)
pd.set_option('display.max_rows',24)
pd.set_option('expand_frame_repr',False)


# train it ########################################################################################
def train_classifier(cl, X_train, y_train):
    for i, (rowidx, file_path) in enumerate(X_train.iteritems()):
        try:
            with open(file_path, 'r') as txt_file:
                txt = txt_file.read()
        except UnicodeDecodeError:
            continue
        category = y_train[rowidx]
        cl.train(txt, [category])
        if not bool((i + 1) % 100):
            print('Trained on', i + 1, 'files', flush=True)
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
        # if not bool((i + 1) % 100):
        #     print('Tested on', i + 1, 'files', flush=True)
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
    print('\n')
    print(df_pred[['Accuracy_One_Category','Accuracy_Multi_Category']].sum()/len(df_pred),'\n')
    return df_pred[column_order]


# walk through and list files in DataSet folder ###################################################
def list_files_paths(dir_name):
    data_dir = './DataSets/' + dir_name
    df_list = []
    for dirpath,dirnames,filenames in os.walk(data_dir):
        df = pd.DataFrame(filenames,columns=['filename'])
        df['filepath'] = df['filename'].map(lambda fname: dirpath + '/' + fname)
        df['category'] = dirpath.split(sep='/')[-1]
        df_list.append(df)
    df_items_categories = pd.concat(df_list, axis=0, ignore_index=True)
    return df_items_categories


# train and test on datadir #######################################################################
def train_test_on_datadir(cl, dir_name='20_newsgroup', n_multi=3,
                          samplefrac=0.2, randstate=42, testsize=0.2):

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
    n_jobs = 4 # number of parallel jobs
    parallelizer = Parallel(n_jobs)
    parts_X_test = np.array_split(X_test,n_jobs)
    tasks_iterator = ( delayed(predict_categories)(cl,part_X,n_multi) for part_X in  parts_X_test)
    list_df_test = parallelizer(tasks_iterator)
    df_prediction = pd.concat(list_df_test,axis=0)
    # df_test = predict_categories(cl,X_test)
    t2 = time.time()
    print('\nFinished Classification of {:0.0f} items in {:0.0f} sec - {:0.0f} items per sec.'\
                .format(n_test,t2-t1, n_test/(t2-t1)) )

    # find accuracy of prediction
    df_test = prediction_accuracy(df_prediction, y_test)
    # return accuracy and predictions
    return df_test


# test on N random items ##########################################################################
def random_test(n_items, datadir='20_newsgroup', n_multi=2):
    df_items = list_files_paths(datadir)
    items = df_items.sample(n_items)
    items_paths = items.filepath
    items_cats = items.category
    cl_ll = email_classifier.LogLikelihoodClassifier(email_classifier.get_unique_tokens)
    cl_nb = email_classifier.BernoulliNBclassifier(email_classifier.get_unique_tokens)
    cl_ll.load_data(datadir + '.h5')
    cl_nb.load_data(datadir + '.h5')
    print('\nBernoulli Naive Bayes Classifier : ')
    print(prediction_accuracy(predict_categories(cl_nb,items_paths,n_multi=n_multi),items_cats))
    print('\nLog-Likelihood Classifier : ')
    print(prediction_accuracy(predict_categories(cl_ll,items_paths,n_multi=n_multi),items_cats))


# train on sentences ##############################################################################
def train_on_sentences(cl):
    '''
    train on example texts
    '''
    cl.train('Nobody owns the water.',['good'])
    cl.train('the quick rabbit jumps fences',['good'])
    cl.train('buy pharmaceuticals now',['bad'])
    cl.train('make quick money at the online casino',['bad'])
    cl.train('the quick brown fox jumps',['good'])
