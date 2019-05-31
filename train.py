# -- coding:utf-8 --

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sklearn.linear_model import Ridge, Lasso, LassoCV, RidgeCV, LinearRegression,SGDClassifier,BayesianRidge,LogisticRegressionCV,ElasticNetCV
from sklearn.svm import SVR,SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, roc_auc_score,accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

# import tensorflow as tf
import numpy as np
import pandas as pd
from features import uid_features, author_features, music_features, normalize_features,ucity_features
from data_io import read_final_track2_train, read_final_track2_test, read_track2_title
from data_io import map_title, timer
from data_io import read_track2_face_attrs, read_track2_video_features, read_track2_audio_features

SEED = 2019


def train_ridge(x_train, x_valid, y_train, y_valid,classifier):
    # print('linear_model')
    preds = []
    if classifier == 'RidgeCV':
        clf = RidgeCV(alphas=[1, 0.1, 0.01, 0.001]) #Ridge regression with built-in cross-validation.
    if classifier == 'LassoCV':
        clf = LassoCV(alphas = [1, 0.1, 0.01, 0.001])
    if classifier == 'LR':
        clf = LinearRegression()
    if classifier == 'BAY':
        clf = BayesianRidge()
    if classifier == 'ElaNet':
        clf = ElasticNetCV(cv=5, random_state=0)
    if classifier == 'SVM': # Linear Support Vector Regression, no better than chance
        clf = SVC(gamma = 'scale', tol=1e-5)
    if classifier == 'SGD': # no better than chance
        clf = SGDClassifier(loss='log',max_iter=1000000, tol=1e-3)
    if classifier == 'RF': # no better than chance
        clf = RandomForestClassifier(n_estimators=1000, max_depth=5,random_state = 0)
    if classifier == 'LR': # no better than chance
        clf = LogisticRegressionCV(cv=5, random_state=0,multi_class='multinomial')

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_valid)
    preds.append(y_pred.reshape(-1, 1))

    # preds = np.hstack(preds)
    # print(roc_auc_score(y_valid, preds.mean(1)))
    # print('roc_auc_score:',roc_auc_score(y_valid, y_pred))

    return clf


def draw_roc(y_valid, y_pred,title):

    fpr, tpr, _ = roc_curve(y_valid, y_pred)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('output/{}-{}.png'.format(classifier,title))


if __name__ == "__main__":
    chunk = 4000000
    col_feat = []
    modalities = ['Train_audio','Train_video','Train_title','Train_face_atts','Train_all','Train_main']
    modality = modalities[0] # train_all takes 572s in the final training step
    print('This task is to {}'.format(modality))

    with timer("1. reading final track2 data(main)"):
        df = read_final_track2_train(chunk)
        df_test = read_final_track2_test(chunk)

        df_feat, df_model = train_test_split(
            df, random_state=SEED, shuffle=True, test_size=0.3
        )  # some for generating feature, some for traing model

    with timer("2. creating features of uid and author"):
        df_uid_feature = uid_features(df_feat)
        df_author_feature = author_features(df_feat)
        df_music_feature = music_features(df_feat)
        df_city_feature = ucity_features(df_feat)

    with timer("3. normalizing the features"):
        df_model, df_test = normalize_features(df_model,
                                               df_test)  # normalize and map'duration time', 'music id', 'device' and 'channel'.
        # df_model = normalize_features(df_model)  # we don't use test set

    if modality == 'Train_main' or modality == 'Train_all':
        df_model = df_model.merge(df_uid_feature, on='uid', how='left')
        # df_test = df_test.merge(df_uid_feature, on='uid', how='left')
        df_model = df_model.merge(df_author_feature, on='author_id', how='left')
        # df_test = df_test.merge(df_author_feature, on='author_id', how='left')
        df_model = df_model.merge(df_music_feature, on='music_id', how='left')
        # df_test = df_test.merge(df_music_feature, on='music_id', how='left')
        df_model = df_model.merge(df_city_feature, on='user_city', how='left')
        # df_test = df_test.merge(df_city_feature, on='user_city', how='left')
        col_feat = col_feat + [
            'duration_time', 'uid_view', 'uid_finish', 'uid_like',
            'author_view', 'author_finish', 'author_like']

    if modality== 'Train_title' or modality== 'Train_all':
        with timer(">>>  read_track2_title"):
            item_id, seq = read_track2_title(chunk)
            df_itemid = pd.DataFrame(item_id,columns = ['item_id'])
            df_seq = pd.DataFrame(seq,columns = ['title_{}'.format(i) for i in range(10)])
            df_title = pd.concat([df_itemid,df_seq],axis = 1)

        with timer(">>> map title with train data"):
            seq_order = map_title(df, item_id, seq) # not sure why do this, and we didnot use this later
            df_model = df_model.merge(df_title, on='item_id', how='left')
            col_feat = col_feat + ['title_{}'.format(i) for i in range(10)]  # title training features

    if modality == 'Train_face_atts' or modality == 'Train_all':
        with timer(">>> read_track2_face"):
            df_face = read_track2_face_attrs(chunk)
            df_model = df_model.merge(df_face, on='item_id', how='left')
            col_feat = col_feat + ['face_attrs_{}'.format(i + 1) for i in range(128)]

    if modality == 'Train_video' or modality == 'Train_all':
        with timer(">>> read_track2_video"):
            df_video = read_track2_video_features(chunk)
            df_model = df_model.merge(df_video, on='item_id', how='left')
            col_feat = col_feat + ['video_feature_dim_128_{}'.format(i + 1) for i in range(128)]

    if modality == 'Train_audio' or modality == 'Train_all':
        with timer(">>> read_track2_audio"):
            df_audio = read_track2_audio_features(chunk)
            df_model = df_model.merge(df_audio, on='item_id', how='left')
            col_feat = col_feat + ['audio_feature_128_dim_{}'.format(i + 1) for i in range(128)]

    with timer("3.x filling na and spliting the df_model"):
        df_model = df_model.fillna(df_model.mean())  # fill nan as mean
        # df_test = df_test.fillna(df_test.mean())

        df_train, df_valid = train_test_split(
            df_model, random_state=SEED, shuffle=False, test_size=0.2)

    with timer("4. training the main model"):
        classifier = 'SVM'

        x_train = df_train.loc[:, col_feat].values # Return a Numpy representation of the DataFrame.
        x_valid = df_valid.loc[:, col_feat].values

        # \ is to change to another line after '='
        finish_train, finish_valid = \
            df_train['finish'].values, df_valid['finish'].values
        like_train, like_valid = \
            df_train['like'].values, df_valid['like'].values
        print('features preparation done')

        y_train_finish, y_valid_finish = finish_train, finish_valid
        model_finish = train_ridge(x_train, x_valid, y_train_finish, y_valid_finish, classifier)

        y_train_like, y_valid_like = like_train, like_valid
        model_like = train_ridge(x_train, x_valid, y_train_like, y_valid_like, classifier)

        y_pred_finish = model_finish.predict(x_valid)
        #print('finish ROC ACC:', roc_auc_score(finish_valid, y_pred_finish))
        print('finish confusion_matrix:', confusion_matrix(y_valid_finish, y_pred_finish))
        print('finish accuracy_score:', accuracy_score(y_valid_finish, y_pred_finish))
        #draw_roc(finish_valid, y_pred_finish,'Receiver operating characteristic Curve for finish')

        y_pred_like = model_like.predict(x_valid)
        #print('like ROC ACC:', roc_auc_score(like_valid, y_pred_like))
        print('like confusion_matrix:', confusion_matrix(y_valid_like, y_pred_like))
        print('like accuracy_score:', accuracy_score(y_valid_like, y_pred_like))
        #draw_roc(like_valid, y_pred_like, 'Receiver operating characteristic Curve for like')
