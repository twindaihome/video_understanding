# -- coding:utf-8 --

import pandas as pd
import numpy as np
import json
from contextlib import contextmanager
import time
import sys
import features

paths = {'video_path': 'input/1w/track2_video_features_sorted_1w.txt',
         'audio_path':'input/1w/track2_audio_features_sorted_1w.txt',
         'face_atts_path': 'input/1w/track2_face_attrs_sorted_1w.txt',
         'final_path':'input/1w/final_track2_train_sorted_1w.txt',
         'title_path':'input/1w/track2_title_sorted_1w.txt',
         'final_test_path':'input/final_track2_test_no_anwser_100000.txt'}
'''
paths = {'video_path': 'input/50w/track2_video_features_sorted_50w.txt',
         'audio_path':'input/50w/track2_audio_features_sorted_50w.txt',
         'face_atts_path': 'input/50w/track2_face_attrs_sorted_50w.txt',
         'final_path':'input/50w/final_track2_train_sorted_50w.txt',
         'title_path':'input/50w/track2_title_sorted_50w.txt',
         'final_test_path':'input/final_track2_test_no_anwser_100000.txt'}
'''

def read_list_feature(path, keyname, chunkSize=4000000, dimLength=128, primaryKey='item_id'):
    rows = list()
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= chunkSize:
                break
            row = [0] * (dimLength + 1)
            content = json.loads(line)
            item_id = int(content[primaryKey])
            row[0] = item_id
            key_data = content[keyname]
    
            if keyname == 'face_attrs':
                for idx, attr in enumerate(key_data) :
                    if (idx*6+7) > dimLength:
                        break
                    # each face attr has 6 dims, the 1st of row in item_id
                    row[idx*6+1 : idx*6+7] = [attr['gender'], attr['beauty']] + attr['relative_position']
            else:
                col_num = min(dimLength, len(key_data))
                row[1 : col_num+1] = key_data[:col_num]

            rows.append(row)
    data = pd.DataFrame(rows, columns=[keyname + '_' + str(x) for x in range(dimLength+1)])
    data.rename(columns={keyname + '_0' : 'item_id'}, inplace=True)
    return data

def read_track2_video_features(chunkSize=4000000, maxlen=128):
    return read_list_feature(paths['video_path'], 'video_feature_dim_128', chunkSize, maxlen)

def read_track2_audio_features(chunkSize=4000000, maxlen=128):
    return read_list_feature(paths['audio_path'], 'audio_feature_128_dim', chunkSize, maxlen)

def read_track2_face_attrs(chunkSize=4000000, maxlen=128):
    return read_list_feature(paths['face_atts_path'], 'face_attrs', chunkSize, maxlen)

def read_chunk(reader, chunkSize):
    chunks = []
    while True:
    # for i in range(10): # simple data for test
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            # print("Iteration is stopped.")
            break
    df = pd.concat(chunks, ignore_index=True)

    return df


def read_final_track2_train(chunkSize=100000):
    loop = True
    path = paths['final_path']

    cols = [
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
        'finish', 'like', 'music_id', 'device', 'time', 'duration_time'
    ]
    reader = pd.read_csv(path, iterator=True, sep='\t')
    df = read_chunk(reader, chunkSize)
    df.columns = cols
    return df

    
def read_final_track2_test(chunkSize=100000):
    loop = True
    path = paths['final_test_path']

    cols = [
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
        'finish', 'like', 'music_id', 'device', 'time', 'duration_time'
    ]
    reader = pd.read_csv(path, iterator=True, sep='\t')
    df = read_chunk(reader, chunkSize)
    df.columns = cols
    return df    
    

def read_track2_title(chunkSize=4000000, maxlen=10):

    path = paths['title_path'] # 3114071 rows

    item_id = np.zeros((chunkSize, 1)).astype(np.int)
    seq = np.zeros((chunkSize, maxlen)).astype(np.int)

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= chunkSize:
                break
            content = json.loads(line)
            words = list(map(int, list(content['title_features'].keys())))
            col_num = min(10, len(words))
            item_id[i] = int(content['item_id'])
            seq[i, :col_num] = words[:col_num]
    return item_id[:i], seq[:i]


def map_title(df, item_id, seq):

    match = pd.DataFrame({
        'item_id': item_id.reshape(-1),
        'idx': range(item_id.shape[0])
    })
    match = df.merge(match, on='item_id', how='left')
    idx_null = match['idx'].isnull()
    idx = match[~idx_null]['idx'].astype(int).values
    return seq[idx]

def list_intersection(title_table, word_dims, dimLength=16):
    result = [0] * len(title_table.values)
    for ti, title_words in enumerate(title_table.values):
        result[ti] = [0] * dimLength
        for title_word in title_words: #title words length is 8, the result df have no this dimension
            for di, dim in enumerate(word_dims):
                if di >= dimLength:
                    break
                if dim == title_word:
                    result[ti][di] = 1

    return result


def concat_title_dim(df_title, dimLength=16):
    df_combined = df_title['title_0'].map(str)
    one_col = [df_title['title_0']]
    for x in range(9):
        df_combined += '-'
        df_combined += df_title['title_'+ str(x)].map(str)
        one_col.append(df_title['title_' + str(x)])

    group = pd.value_counts(pd.concat(one_col))
    clip_75 = group.describe()['75%']
    df_view = group.reset_index().astype('int')
    df_view = df_view.rename(columns={
        'index': 'word_ids',
        0: 'word_impression'
    })
    df_words = df_view.head(dimLength)
    d1 = list_intersection(df_title, df_words['word_ids'], dimLength)
    return d1

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


if __name__ == "__main__":

    with timer("read_final_track2_train"):
        df = read_final_track2_train(100000)
        df_test = read_final_track2_test(chunkSize=100000)
        
        df_feat, df_model = train_test_split(
            df, random_state=SEED, shuffle=False,
            test_size=0.5)  # half for feature, half for traing model

    with timer("creating features of uid and author"):
        df_uid_feature = uid_features(df_feat)
        df_author_feature = author_features(df_feat)

    # with timer("read_track2_title"):
    # item_id, seq = read_track2_title()

    # with timer("map title with train data"):
    # seq_order = map_title(df, item_id, seq)

    with timer("normalize the features"):
        df_model = normalize_features(df_model)
        df_model = df_model.merge(df_uid_feature, on='uid', how='left')
        df_model = df_model.merge(
            df_author_feature, on='author_id', how='left')
        df_model = df_model.fillna(df_model.mean())

    df_train, df_valid = train_test_split(
        df_model, random_state=SEED, shuffle=False, test_size=0.2)

    col_feat = [
        'duration_time', 'uid_view', 'uid_finish', 'uid_like', 'author_view',
        'author_finish', 'author_like'
    ]
    x_train, x_valid = df_train.loc[:, col_feat].values, df_valid.loc[:, col_feat].values
    finish_train, finish_valid = df_train['finish'].values, df_valid['finish'].values
    like_train, like_valid = df_train['like'].values, df_valid['like'].values
    # for col in df.columns:
    # print(df[col].drop_duplicates().count(), df[col].min(), df[col].max())
    # print(df[col].astype(str).describe())

    # df.groupby('uid').count()['user_city'].describe()
    # df['user_city'].astype(str).value_counts().describe()
    # df.loc[:,['uid','author_id']].drop_duplicates()
