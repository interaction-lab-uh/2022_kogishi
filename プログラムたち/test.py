# %%
from cmath import nan
from importlib.resources import contents
from pydoc import describe
from re import A
from tracemalloc import start
from cv2 import merge
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import os
import seaborn as sns
import matplotlib as mpl
import datetime
import pickle
from sympy import content
# %%
import time
import lightgbm as lgb
from regex import D
from sklearn.model_selection import KFold, train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
# import math
# import collections
# %%
import transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers import models

transformers.BertTokenizer = transformers.BertJapaneseTokenizer

from transformers import BertJapaneseTokenizer, BertModel
from sentence_transformers import InputExample,losses
from sentence_transformers.losses import TripletDistanceMetric, TripletLoss
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.readers import TripletReader
from sentence_transformers.datasets import SentencesDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from keras_bert import load_trained_model_from_checkpoint
import sentencepiece as spm
import copy

# %%
img_dir = './img_text'
png_dir = './png'
# %%
# データの読み込み
df = pd.read_csv('./reading_v2.csv')
df
# %%
# contents_id + page_no
df['cid_page'] = df['contents_id'] + '_' + df['page_no'].astype(str).str.zfill(3)
df['cid_page']
# %%
# コンテンツidとページ数のcsv
df_contentspage = pd.read_csv('./contents_page_v2.csv')
df_contentspage
# %%
# コンテンツID
contents_0 = 'd1f255a373a3cef72e03aa9d980c7eca' #0
contents_1 = '7eacb532570ff6858afd2723755ff790' #1
contents_2 = 'b6f0479ae87d244975439c6124592772' #2
contents_3 = 'e0c641195b27425bb056ac56f8953d24' #3
contents_4 = 'f85454e8279be180185cac7d243c5eb3' #4
contents_5 = 'faa9afea49ef2ff029a833cccc778fd0' #5
contents_6 = '3c7781a36bcd6cf08c11a970fbe0e2a6' #6
contents_7 = '25b2822c2f5a3230abfadd476e8b04c9' #7
contents_plus = '877a9ba7a98f75b90a9d49f53f15a858' #plus

# %%
with open('./pickle/contents1_dic.pickle', mode='rb') as f:
    contents1_dic = pickle.load(f)
with open('./pickle/contents2_dic.pickle', mode='rb') as f:
    contents2_dic = pickle.load(f)
with open('./pickle/contents3_dic.pickle', mode='rb') as f:
    contents3_dic = pickle.load(f)
with open('./pickle/contents4_dic.pickle', mode='rb') as f:
    contents4_dic = pickle.load(f)
with open('./pickle/contents5_dic.pickle', mode='rb') as f:
    contents5_dic = pickle.load(f)
with open('./pickle/contents6_dic.pickle', mode='rb') as f:
    contents6_dic = pickle.load(f)
with open('./pickle/contents7_dic.pickle', mode='rb') as f:
    contents7_dic = pickle.load(f)
with open('./pickle/contents0_dic.pickle', mode='rb') as f:
    contents0_dic = pickle.load(f)
with open('./pickle/contentsplus_dic.pickle', mode='rb') as f:
    contentsplus_dic = pickle.load(f)
with open('./pickle/quiz1_l_dic.pickle', mode='rb') as f:
    quiz1_l_dic = pickle.load(f)
with open('./pickle/quiz2_1_l_dic.pickle', mode='rb') as f:
    quiz2_1_l_dic = pickle.load(f)
with open('./pickle/quiz2_2_l_dic.pickle', mode='rb') as f:
    quiz2_2_l_dic = pickle.load(f)
with open('./pickle/quiz3_l_dic.pickle', mode='rb') as f:
    quiz3_l_dic = pickle.load(f)
with open('./pickle/quiz4_l_dic.pickle', mode='rb') as f:
    quiz4_l_dic = pickle.load(f)
with open('./pickle/quiz5_l_dic.pickle', mode='rb') as f:
    quiz5_l_dic = pickle.load(f)
with open('./pickle/quiz6_l_dic.pickle', mode='rb') as f:
    quiz6_l_dic = pickle.load(f)
with open('./pickle/quiz7_l_dic.pickle', mode='rb') as f:
    quiz7_l_dic = pickle.load(f)
    

condic_list = [contents0_dic, contentsplus_dic, contents1_dic, contents2_dic, contents3_dic, contents4_dic, contents5_dic, contents6_dic, contents7_dic]



# %%
# 答えあり小テスト文章のdicをlistへ
quizd_l_list = [quiz1_l_dic, quiz2_1_l_dic, quiz2_2_l_dic, quiz3_l_dic, quiz4_l_dic, quiz5_l_dic, quiz6_l_dic, quiz7_l_dic]
# %%
# コンテンツを操作した日付のリスト
date_list = []
for n in df['operation_date'].tolist():
    if n[:10] not in date_list:
        date_list.append(n[:10])
print(date_list)
# %%
# 小テストデータの読み込み
df_quiz = pd.read_csv('./quiz.csv')
df_quiz
# %%
df_read = df.copy()
# 操作日時datetime型へ
df_read['operation_date'] = pd.to_datetime(df_read['operation_date'])
df_read
# %%
# df_quizへのいろいろな変更
df_quiz['question_title'] = np.nan
df_quiz['respons_time'] = np.nan
df_quiz['respons_day'] = np.nan

for i in range(len(df_quiz)):
    df_quiz['question_title'][i] = df_quiz['questionsummary'][i].split(':')[0]
df_quiz['r_or_w'] = np.nan
for i in range(len(df_quiz)):
    if df_quiz['state'][i] == 'gradedright':
        df_quiz['r_or_w'][i] = 1
    else:
        df_quiz['r_or_w'][i] = 0
for i in range(len(df_quiz)):
    df_quiz['timecreated'][i] = datetime.datetime.fromtimestamp(df_quiz['timecreated'][i])

for i in range(len(df_quiz)):
    if (str(df_quiz['timecreated'][i].year) + '-' + str(df_quiz['timecreated'][i].month) + '-' + str(df_quiz['timecreated'][i].day)) == '2020-5-19':
        if (df_quiz['question_title'][i]=='パスワードの仕組みについて、正しいものを一つ選んで下さい。') | (df_quiz['question_title'][i]=='多要素認証・多段階認証について、正しいものを一つ選んで下さい。') | (df_quiz['question_title'][i]=='パスワードが他人に知られる理由，知られにくくできる理由について，正しいものを一つ選んで下さい。') | (df_quiz['question_title'][i]=='無線LAN あるいはその使用方法ついて、正しいものを一つ選んで下さい。') | (df_quiz['question_title'][i]=='使用すべきではない無線LANサービスへの対応として、正しいものを一つ選んで下さい。'):
            df_quiz['respons_day'][i] = '2020-5-19-1'
        else:
            df_quiz['respons_day'][i] = '2020-5-19-2'
    else:
        df_quiz['respons_day'][i] = str(df_quiz['timecreated'][i].year) + '-' + str(df_quiz['timecreated'][i].month) + '-' + str(df_quiz['timecreated'][i].day)
for i in range(len(df_quiz)):
    df_quiz['respons_time'][i] = str(df_quiz['timecreated'][i].hour) + ':' + str(df_quiz['timecreated'][i].minute)

df_quiz


# %%
# とりあえず各週のstart_timeとend_timeをだしとく
start_time_1 = datetime.datetime(2020, 5, 12, 14, 50, 00)
end_time_1 = datetime.datetime(2020, 5, 12, 16, 20, 00)
start_time_2 = datetime.datetime(2020, 5, 19, 14, 50, 00)
end_time_2 = datetime.datetime(2020, 5, 19, 16, 20, 00)
start_time_3 = datetime.datetime(2020, 5, 26, 14, 50, 00)
end_time_3 = datetime.datetime(2020, 5, 26, 16, 20, 00)
start_time_4 = datetime.datetime(2020, 6, 2, 14, 50, 00)
end_time_4 = datetime.datetime(2020, 6, 2, 16, 20, 00)
start_time_5 = datetime.datetime(2020, 6, 9, 14, 50, 00)
end_time_5 = datetime.datetime(2020, 6, 9, 16, 20, 00)
start_time_6 = datetime.datetime(2020, 6, 16, 14, 50, 00)
end_time_6 = datetime.datetime(2020, 6, 16, 16, 20, 00)
start_time_7 = datetime.datetime(2020, 6, 23, 14, 50, 00)
end_time_7 = datetime.datetime(2020, 6, 23, 16, 20, 00)
# test_timeもだしておくよ
test_time_1 = datetime.datetime(2020, 5, 12, 16, 10, 00)
test_time_2 = datetime.datetime(2020, 5, 19, 16, 10, 00)
test_time_3 = datetime.datetime(2020, 5, 26, 16, 10, 00)
test_time_4 = datetime.datetime(2020, 6, 2, 16, 10, 00)
test_time_5 = datetime.datetime(2020, 6, 9, 16, 10, 00)
test_time_6 = datetime.datetime(2020, 6, 16, 16, 10, 00)
test_time_7 = datetime.datetime(2020, 6, 23, 16, 10, 00)
# %%
# いろんなとこで回した時に使えるようにlist化
# contents_list = ['d1f255a373a3cef72e03aa9d980c7eca', '877a9ba7a98f75b90a9d49f53f15a858', '7eacb532570ff6858afd2723755ff790', 'b6f0479ae87d244975439c6124592772', 'e0c641195b27425bb056ac56f8953d24', 'f85454e8279be180185cac7d243c5eb3', 'faa9afea49ef2ff029a833cccc778fd0', '3c7781a36bcd6cf08c11a970fbe0e2a6', '25b2822c2f5a3230abfadd476e8b04c9']
contents_list = [contents_0, contents_plus, contents_1, contents_2, contents_3, contents_4, contents_5, contents_6, contents_7]
start_time_list = [start_time_1, start_time_2, start_time_3, start_time_4, start_time_5, start_time_6, start_time_7]
test_time_list = [test_time_1, test_time_2, test_time_3, test_time_4, test_time_5, test_time_6, test_time_7]
end_time_list = [end_time_1, end_time_2, end_time_3, end_time_4, end_time_5, end_time_6, end_time_7]
start_leclist = [start_time_1, start_time_2, start_time_2, start_time_3, start_time_4, start_time_5, start_time_6, start_time_7]
test_leclist =[test_time_1, test_time_1, test_time_2, test_time_2, test_time_3, test_time_4, test_time_5, test_time_6, test_time_7]
end_leclist = [end_time_1, end_time_2, end_time_2, end_time_3, end_time_4, end_time_5, end_time_6, end_time_7]

# %%
# 講義時間＋前後1時間ふくめたのスタートタイム（st）、エンドタイム（et）
st_1 = datetime.datetime(2020, 5, 12, 13, 50, 00)
et_1 = datetime.datetime(2020, 5, 12, 17, 20, 00)
st_2 = datetime.datetime(2020, 5, 19, 13, 50, 00)
et_2 = datetime.datetime(2020, 5, 19, 17, 20, 00)
st_3 = datetime.datetime(2020, 5, 26, 13, 50, 00)
et_3 = datetime.datetime(2020, 5, 26, 17, 20, 00)
st_4 = datetime.datetime(2020, 6, 2, 13, 50, 00)
et_4 = datetime.datetime(2020, 6, 2, 17, 20, 00)
st_5 = datetime.datetime(2020, 6, 9, 13, 50, 00)
et_5 = datetime.datetime(2020, 6, 9, 17, 20, 00)
st_6 = datetime.datetime(2020, 6, 16, 13, 50, 00)
et_6 = datetime.datetime(2020, 6, 16, 17, 20, 00)
st_7 = datetime.datetime(2020, 6, 23, 13, 50, 00)
et_7 = datetime.datetime(2020, 6, 23, 17, 20, 00)

stw_list = [st_1, st_2, st_2, st_3, st_4, st_5, st_6, st_7]
etw_list = [et_1, et_2, et_2, et_3, et_4, et_5, et_6, et_7]
resday_list = ['2020-5-12', '2020-5-19-1', '2020-5-19-2', '2020-5-26', '2020-6-2', '2020-6-9', '2020-6-16', '2020-6-23']

# %%
# 各ページの滞在時間をだす
df_page = df_read[(df_read['operation_name'] == 'OPEN')|(df_read['operation_name'] == 'NEXT')|(df_read['operation_name'] == 'PREV')|(df_read['operation_name'] == 'CLOSE')|(df_read['operation_name'] == 'PAGE_JUMP')|(df_read['operation_name'] == 'SEARCH_JUMP')|(df_read['operation_name'] == 'BOOKMARK_JUMP')].reset_index()
df_page['open_time'] = 0
df_page = df_page.sort_values(['userid', 'operation_date']).reset_index(drop=True)
for i in range(1, len(df_page)):
    if df_page['operation_name'][i] == 'OPEN':
        df_page['open_time'][i] = 0
    else :
        df_page['open_time'][i] = (df_page['operation_date'][i] - df_page['operation_date'][i-1]).total_seconds()
df_page

# %%
# operation_nameのlist化
operation_name_list = df_read['operation_name'].unique()
operation_name_list




# %%
# 以前使ったLightBGMを関数にしておく（今まで）
# 基本は先生のコード,ところどころhttps://rightcode.co.jp/blog/information-technology/lightgbm-useful-for-kaggler
# def LightGBM(df_youso):
#     df_youso = df_youso.fillna(0)
#     # 訓練データとテストデータに分ける
#     train_set, test_set = train_test_split(df_youso, test_size = 0.2, random_state = 30)
#     # 説明変数と目的変数に分ける
#     x_train = train_set.drop('r_or_w', axis = 1)
#     y_train = train_set['r_or_w']
#     x_test = test_set.drop('r_or_w', axis = 1)
#     y_test = test_set['r_or_w']
    
#     x_tr = x_train.drop('userid', axis = 1)
#     x_tt = x_test.drop('userid', axis = 1)
    
#     y_tr = y_train.values.flatten().astype(np.float32)
#     y_tt = y_test.values.flatten().astype(np.float32)
#     X_tr = x_tr.values.astype(np.float32)
#     X_tt = x_tt.values.astype(np.float32)
#     # feature_dim = x_tr.shape[1]

#     # 正規化
#     # スケール変換
#     yscale_mean = y_tr.mean()
#     yscale_std = y_tr.std()
#     scaler = StandardScaler()
#     scaler.fit(X_tr)
#     X_tr = scaler.transform(X_tr)
#     y_tr = (y_tr - yscale_mean) / yscale_std

#     # 評価基準を設定する
#     param = {"objective": "regression",
#         "metric": "rmse",
#         "verbosity": -1,
#         "boosting_type": "gbdt",
#         "min_child_samples": 20,
#         "bagging_freq": 0,
#         "bagging_fraction": 0.8,
#         "feature_fraction": 0.9,
#         "num_leaves": 50,  # 100
#         # "lambda_l1": 1e-5,
#         # "lambda_l2": 1e-5,
#         "feature_pre_filter": False,
#         "saved_feature_importance_type": 1,
#         "seed": 0
#         }
#     # LightGBM用のデータセットに入れる
#     dtrain = lgb.Dataset(X_tr, label=y_tr)
#     # dtest = lgb.Dataset(X_tt, y_tt, reference=dtrain)

#     # 訓練データから回帰モデルを作る
#     reg = lgb.train(param, dtrain)
#     # lgb.test = lgb.Dataset(x_test, y_test)


#     # テストデータを用いて予測精度を確認する
#     X_tt = scaler.transform(X_tt)
#     y_pred = reg.predict(X_tt)
#     y_pred = np.clip(y_pred * yscale_std + yscale_mean, 0, 100)

#     # rmseを求める
#     rmse = mean_squared_error(y_tt, y_pred, squared=False)
#     print(f"RMSE: {rmse}")
#     print(f'正解値:{y_tt}')
#     print(f'予測値:{y_pred}')

#     # importanceを表示する
#     importance = pd.DataFrame(reg.feature_importance(), index=x_tr.columns, columns=['importance'])
#     display(importance)
#     display(importance[importance['importance'] != 0].sort_values('importance', ascending=False))

#     attr_impt = reg.feature_importance(importance_type='gain')
#     print('重要性')
#     print(attr_impt)

#     lgb.plot_importance(reg, height = 0.5, figsize = (8,16))
#     # print(df_youso.columns)
    
#     # 正方形にしような
#     max_value = y_test.max()
#     plt.figure(figsize=(8,8))
#     x_line = np.linspace(0, max_value+5, 100)
#     plt.plot(x_line, x_line, color = "red")
#     plt.xlabel('y_tt')
#     plt.ylabel('y_pred')
#     plt.scatter(y_tt, y_pred)


# %%
# df_wy = df_wy.fillna(0)
# X = df_wy.drop('r_or_w', axis = 1)
# y = df_wy['r_or_w']
# skf = StratifiedKFold(n_splits=5)
# for train_index, test_index in skf.split(X, y):
#     X.iloc[train_index]

# %%
# 5-fold CVにしたほう
def LightGBM(df_youso):
    df_youso = df_youso.fillna(0)
    X = df_youso.drop('r_or_w', axis = 1)
    y = df_youso['r_or_w']
    rmse_list = []
    # 訓練データとテストデータに分ける
    skf = StratifiedKFold(n_splits=5)
    
    for train_index, test_index in skf.split(X, y):
        
        # 説明変数と目的変数に分ける
        x_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        x_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        x_tr = x_train.drop('userid', axis = 1)
        x_tt = x_test.drop('userid', axis = 1)
        
        y_tr = y_train.values.flatten().astype(np.float32)
        y_tt = y_test.values.flatten().astype(np.float32)
        X_tr = x_tr.values.astype(np.float32)
        X_tt = x_tt.values.astype(np.float32)

        # 正規化
        # スケール変換
        yscale_mean = y_tr.mean()
        yscale_std = y_tr.std()
        scaler = StandardScaler()
        scaler.fit(X_tr)
        X_tr = scaler.transform(X_tr)
        y_tr = (y_tr - yscale_mean) / yscale_std

        # 評価基準を設定する
        param = {"objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "min_child_samples": 20,
            "bagging_freq": 0,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.9,
            "num_leaves": 50,  # 100
            "feature_pre_filter": False,
            "saved_feature_importance_type": 1,
            "seed": 0
            }
        # LightGBM用のデータセットに入れる
        dtrain = lgb.Dataset(X_tr, label=y_tr)

        # 訓練データから回帰モデルを作る
        reg = lgb.train(param, dtrain)


        # テストデータを用いて予測精度を確認する
        X_tt = scaler.transform(X_tt)
        y_pred = reg.predict(X_tt)
        y_pred = np.clip(y_pred * yscale_std + yscale_mean, 0, 5)

        # rmseを求める
        rmse = mean_squared_error(y_tt, y_pred, squared=False)
        rmse_list.append(rmse)
        
        # print(f"RMSE: {rmse}")
        # print(f'正解値:{y_tt}')
        # print(f'予測値:{y_pred}')

    print(f"RMSE(5FoldCV): {np.mean(rmse_list)}")
    print(f"std: {np.std(rmse_list)}")



# %%
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score, f1_score

def LightGBM_2(df_youso, ofname):
    df_youso = df_youso.fillna(0)
    X = df_youso.drop('r_or_w', axis = 1)
    y = df_youso['r_or_w']
    rmse_list = []
    # 訓練データとテストデータに分ける
    skf = StratifiedKFold(n_splits=5)

    y_pred_list = []
    y_tt_list = []
    test_index_list = []
    for train_index, test_index in skf.split(X, y):
        
        # 説明変数と目的変数に分ける
        x_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        x_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        x_tr = x_train.drop('userid', axis = 1)
        x_tt = x_test.drop('userid', axis = 1)
        
        y_tr = y_train.values.flatten().astype(np.float32)
        y_tt = y_test.values.flatten().astype(np.float32)
        X_tr = x_tr.values.astype(np.float32)
        X_tt = x_tt.values.astype(np.float32)

        # 正規化
        # スケール変換
        yscale_mean = y_tr.mean()
        yscale_std = y_tr.std()
        scaler = StandardScaler()
        scaler.fit(X_tr)
        X_tr = scaler.transform(X_tr)
        y_tr = (y_tr - yscale_mean) / yscale_std

        # 評価基準を設定する
        param = {"objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "min_child_samples": 20,
            "bagging_freq": 0,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.9,
            "num_leaves": 50,  # 100
            "feature_pre_filter": False,
            "saved_feature_importance_type": 1,
            "seed": 0
            }
        # LightGBM用のデータセットに入れる
        dtrain = lgb.Dataset(X_tr, label=y_tr)

        # 訓練データから回帰モデルを作る
        reg = lgb.train(param, dtrain)


        # テストデータを用いて予測精度を確認する
        X_tt = scaler.transform(X_tt)
        y_pred = reg.predict(X_tt)
        y_pred = np.clip(y_pred * yscale_std + yscale_mean, 0, 5)

        # rmseを求める
        rmse = mean_squared_error(y_tt, y_pred, squared=False)
        rmse_list.append(rmse)
        test_index_list.append(test_index)
        y_pred_list.append(y_pred)
        y_tt_list.append(y_tt)
        # print(f"RMSE: {rmse}")
        # print(f'正解値:{y_tt}')
        # print(f'予測値:{y_pred}')
    rmse_list = np.array(rmse_list)
    print('my cross validation result(RMSE):', rmse_list.mean())
    pd.DataFrame({
        'test_index': np.concatenate(test_index_list),
        'y_pred': np.concatenate(y_pred_list),
        'y_truth':np.concatenate(y_tt_list)
    }).to_csv(ofname, index=False)

    
    
# %%
# 各週の小テストの点数（timecreatedでgroupbyしたおかげで2週目が分かれた）
df_quiz_score_w = df_quiz.groupby(['userid', 'timecreated', 'respons_day'], as_index=False)['r_or_w'].sum()
df_quiz_score_w
# %%
# 各学生の各ページをみている時間をだしてるよ
# 一旦セッションタイムアウト(何秒以上は無視)とか無視してやってみる
df_open = df_page.groupby(['userid', 'contents_id', 'cid_page'], as_index=False)['open_time'].sum()
df_open
# %%
# 要素のためのuseridとcid_pageのリストを作成している（見ていない人もだすため）
us_list = []
con_list = []
cid_list = []
for i in range(100):
    for j in range(385):
        us_list.append(i)
for k in range(100):
    for i in range(len(contents_list)):
        pnum = df_contentspage['page'][i]
        for j in range(1,pnum+1):
            con_list.append(contents_list[i])
            cid_list.append(contents_list[i] + '_' + str(j).zfill(3))

# %%
# 各ユーザーが各ページで各行動をどれくらいとっているかの確認
df_opeu_t = pd.DataFrame({'userid':us_list, 'contents_id':con_list, 'cid_page':cid_list})
for i in range(len(operation_name_list)):
    df_ope = df_read[df_read['operation_name'] == operation_name_list[i]].reset_index()
    df_ope_user = df_ope.groupby(['userid', 'contents_id', 'cid_page'])['operation_name'].count().reset_index()
    df_ope_user = df_ope_user.rename({'operation_name':operation_name_list[i] + '_count'}, axis=1)
    df_opeu_t = pd.merge(df_opeu_t, df_ope_user, on=['userid', 'contents_id', 'cid_page'], how='left')
df_opeu_t = df_opeu_t.fillna(0)
df_opeu_t
# %%
# df_opeuserの中で無視していい列を消しておく（無視していいって言われた部分）
df_opeu_t = df_opeu_t.drop(columns=['TIMER_STOP_count', 'TIMER_PAUSE_count', 'MEMO_TEXT_CHANGE_HISTORY_count'])
df_opeu_t



# %%
# パスを通す
bertPath = 'c:/Users/fgobs/bert/'
config_path = bertPath + 'bert-wiki-ja/bert_config.json'
checkpoint_path = bertPath + 'bert-wiki-ja/model.ckpt-1400000'

bert = load_trained_model_from_checkpoint(config_path, checkpoint_path)
bert.summary()

# %%
spp = spm.SentencePieceProcessor()
spp.Load(bertPath + 'bert-wiki-ja/wiki-ja.model')


# %%
# https://qiita.com/jjking/items/a9fe907f992ccaefbd2a
# https://www.enoki-inc.co.jp/tech-5/
# 512次元の入力ベクトルを作成、そのベクトルを関数bert.predictに渡し768次元の文章ベクトルを作成
# kerasで学習させたもの

def text2vector(text):
    maxlen = 512 # from BERT config
    common_seg_input = np.zeros((1, maxlen), dtype = np.float32)
    matrix = np.zeros((1, maxlen), dtype = np.float32)
    token = [w for w in spp.encode_as_pieces(text.replace(" ", ""))]
    if token and len(token) <= maxlen:
        tokens = []
        tokens.append('[CLS]')
        tokens.extend(token)
        tokens.append('[SEP]')
        for t, token in enumerate(tokens):
            try:
                matrix[0, t] = spp.piece_to_id(token)
            except:
                print(token+"is unknown")
                matrix[0, t] = spp.piece_to_id('<unk>')
    return bert.predict([matrix, common_seg_input])[:,0] # embedding of [CLS]





# %%
# 学生ごとのページ毎での行動の回数と閲覧時間をあわせる
df_youso_cid = pd.merge(df_opeu_t, df_open, on=['userid', 'contents_id', 'cid_page'], how='left')
df_youso_cid

# %%
# コンテンツ文章のベクトルいれるdfの枠作成
df_npvec = pd.DataFrame({'cid_page':cid_list[:385]})
df_npvec['page_vec'] = np.NaN
df_npvec['page_vec'] = df_npvec['page_vec'].astype('O')
df_npvec




# %%
# 文章のベクトル化
for i in range(len(df_npvec)):
    p = int(df_npvec['cid_page'][i][-3:])
    conname = df_npvec['cid_page'][i][:-4]
    if conname == 'd1f255a373a3cef72e03aa9d980c7eca':
        context = ' '.join(contents0_dic[p]) 
        context = context.replace('•', ' ')
        vect = text2vector(context)
    elif conname == '877a9ba7a98f75b90a9d49f53f15a858':
        context = ' '.join(contentsplus_dic[p]) 
        context = context.replace('•', ' ')
        context = context.replace('-', ' ')
        vect = text2vector(context)
    elif conname == '7eacb532570ff6858afd2723755ff790':
        context = ' '.join(contents1_dic[p])
        context = context.replace('•', ' ')
        vect = text2vector(context)
    elif conname == 'b6f0479ae87d244975439c6124592772':
        context = ' '.join(contents2_dic[p])
        context = context.replace('•', ' ')
        vect = text2vector(context)
    elif conname == 'e0c641195b27425bb056ac56f8953d24':
        context = ' '.join(contents3_dic[p])
        context = context.replace('•', ' ')
        vect = text2vector(context)
    elif conname == 'f85454e8279be180185cac7d243c5eb3':
        context = ' '.join(contents4_dic[p])
        context = context.replace('•', ' ')
        vect = text2vector(context)
    elif conname == 'faa9afea49ef2ff029a833cccc778fd0':
        context = ' '.join(contents5_dic[p])
        context = context.replace('•', ' ') 
        vect = text2vector(context)
    elif conname == '3c7781a36bcd6cf08c11a970fbe0e2a6':
        context = ' '.join(contents6_dic[p])
        context = context.replace('•', ' ') 
        vect = text2vector(context)
    elif conname == '25b2822c2f5a3230abfadd476e8b04c9':
        context = ' '.join(contents7_dic[p])
        context = context.replace('•', ' ') 
        vect = text2vector(context)
    df_npvec.iat[i, 1] = vect
df_npvec

# %%
for i in range(len(df_npvec)):
    v = df_npvec['page_vec'][i][0]
    normalized_v = v/np.linalg.norm(v)
    df_npvec.iat[i, 1] = normalized_v
df_npvec



# %%
# コサイン類似度の実装
def cos_similarity(x, y, eps=1e-8):
    # コサイン類似度を計算:式(2.1)
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)

# %%
# 小テスト問題のベクトル化のdfの枠組み
df_quiz1_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec']) #1
df_quiz2_1_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec']) #2-1
df_quiz2_2_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec']) #2-2
df_quiz3_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec']) #3
df_quiz4_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec']) #4
df_quiz5_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec']) #5
df_quiz6_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec']) #6
df_quiz7_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec']) #7

df_qvec_l_list = [df_quiz1_l_vec, df_quiz2_1_l_vec, df_quiz2_2_l_vec, df_quiz3_l_vec, df_quiz4_l_vec, df_quiz5_l_vec, df_quiz6_l_vec, df_quiz7_l_vec] 

# %%
# 小テスト問題（答え含む）をベクトル化
for i in range(8):
    df_qvec_l_l = copy.copy(df_qvec_l_list)
    df_qvec_l = df_qvec_l_l[i]
    quiz_l_dic = quizd_l_list[i]
    for j in range(len(df_qvec_l)):
        quiztext = ' '.join(quiz_l_dic[j+1])
        vect = text2vector(quiztext)
        v = vect[0]
        normalized_v = v/np.linalg.norm(v)
        df_qvec_l['quiz_vec'][j+1] = normalized_v
    display(df_qvec_l)


# %%
# コンテンツをベクトル化したdfにcontents_id追加
df_npvec['contents_id'] = 0
for i in range(len(df_npvec)):
    df_npvec['contents_id'][i] = df_npvec['cid_page'][i][:-4]
df_npvec


# %%
df_cosim1_l = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim2_1_l = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim2_2_l = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim3_l = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim4_l = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim5_l = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim6_l = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim7_l = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
cosimd_l_list = [df_cosim1_l, df_cosim2_1_l, df_cosim2_2_l, df_cosim3_l, df_cosim4_l, df_cosim5_l, df_cosim6_l, df_cosim7_l]

# 各コンテンツと小テスト内容のコサイン類似度をだしてる
for k in range(8):
    df_cosim_l = cosimd_l_list[k]
    df_x = df_qvec_l_list[k]
    for i in range(5):
        if k == 0:
            df_wvec = df_npvec[(df_npvec['contents_id']==contents_list[0])|(df_npvec['contents_id']==contents_list[1])].reset_index(drop=True)
        else:
            df_wvec = df_npvec[df_npvec['contents_id']==contents_list[k+1]].reset_index(drop=True)
        for j in range(len(df_wvec)):
            x = df_x['quiz_vec'][i+1]
            y = df_wvec['page_vec'][j]
            cosim = cos_similarity(x, y)
            if i == 0:
                df_cosim_l = df_cosim_l.append({'cid_page': df_wvec['cid_page'][j], 'cos_sim1': cosim}, ignore_index=True)
            else:
                df_cosim_l['cos_sim'+str(i+1)][j] = cosim
    cosimd_l_list[k] = df_cosim_l
    display(df_cosim_l)




# %%
# ★手法1


# %%
# ★★学生ごとによく見ているページほど重要度が高くする
# とりあえず一回に見ている時間1020秒（17分）以下でやってみた、正直長い気がする
con_list_t = ['OPEN_count', 'NEXT_count', 'PREV_count', 'CLOSE_count', 'PAGE_JUMP_count', 'GETIT_count', 'OPEN_RECOMMENDATION_count', 'CLOSE_RECOMMENDATION_count', 'NOTGETIT_count', 'ADD MARKER_count', 'DELETE MARKER_count', 'CLICK_RECOMMENDATION_count', 'open_time_count']

# df_po = df_page[df_page['open_time'] <= 1020].reset_index(drop=False)
df_po = df_page[df_page['open_time'] <= 300].reset_index(drop=False)
# df_po = df_page.copy()
# df_po.loc[df_po['open_time'] > 300, 'open_time'] = 300

df_po


# %%
# 重要度を連続的にする
# TODO:maxをはずす->結果かわる？
# 理屈的にやるばあいはある
# df_page_operation_timeとかって書きたかったけど長いので略した結果df_potになったはず、最後のsは本当に分からないです。識別記号みたいなものです。
df_pot_s = df_po.groupby(['userid', 'contents_id', 'cid_page'], as_index=False)['open_time'].sum().sort_values('open_time')
df_pot_s['weit_p'] = 0.0
for i in range(len(contents_list)):
    df_potw = df_pot_s[df_pot_s['contents_id'] == contents_list[i]].reset_index(drop=True)
    df_potw['weit_p'] = df_potw['open_time'] # / max(df_potw['open_time'])
    for j in range(len(df_potw)):
        index_n = df_pot_s[(df_pot_s['cid_page'] == df_potw['cid_page'][j])&(df_pot_s['userid'] == df_potw['userid'][j])].index
        df_pot_s['weit_p'][index_n] = df_potw['weit_p'][j]
df_pot_s
# %%
# 重要度を連続的にする
# 馬鹿なことしてほんとすみませんでした。。。
# まじで反省、ちゃんと説明する。。。。。。。
# わからなすぎて、頭が働かないです
# df_page_operation_timeとかって書きたかったけど長いので略した結果df_potになったはず、最後のsは本当に分からないです。識別記号みたいなものです。
df_pot_t = df_po.groupby(['userid', 'contents_id', 'cid_page'], as_index=False)['open_time'].sum().sort_values('open_time')
df_pot_t['weit_p'] = 0.0
for i in range(len(contents_list)):
    for k in range(100):
        df_potw = df_pot_t[(df_pot_t['contents_id'] == contents_list[i])&(df_pot_t['userid'] == k)].reset_index(drop=True)
        df_potw['weit_p'] = df_potw['open_time'] / sum(df_potw['open_time'])
        for j in range(len(df_potw)):
            index_n = df_pot_t[(df_pot_t['cid_page'] == df_potw['cid_page'][j])&(df_pot_t['userid'] == df_potw['userid'][j])].index
            df_pot_t['weit_p'][index_n] = df_potw['weit_p'][j]
df_pot_t
# %%
# ページベクトルとくっつけて重み付きベクトルweit_vecをつくっている
df_povet_s = pd.merge(df_pot_s, df_npvec, on=['contents_id', 'cid_page'])
# df_povet_s = pd.merge(df_pot_t, df_npvec, on=['contents_id', 'cid_page'])

df_povet_s['weit_vec'] = df_povet_s['weit_p'] * df_povet_s['page_vec']
df_povets = df_povet_s.copy()
df_povets = df_povets[['userid', 'cid_page', 'contents_id', 'weit_vec']]
df_wsum = df_povets.groupby(['userid', 'contents_id'], as_index=False)['weit_vec'].sum()

# 消す？
# ちゃんと考えていませんでした。。
for i in range(len(df_wsum)):
    v = df_wsum['weit_vec'][i]
    normalized_v = v/np.linalg.norm(v)
    df_wsum.iat[i, 2] = normalized_v
df_wsum



# %%
# 関数化をすすめる（1個目）（手法1）
# ベクトルをひろげるだけの関数
def make_vec_df(df_wsum, contentsnum):
    df_wsumw = df_wsum.copy()
    if contentsnum == 1:
        df_wsumw = df_wsumw[(df_wsumw['contents_id'] == contents_list[0]) | (df_wsumw['contents_id'] == contents_list[1])].reset_index(drop=True)
    else:
        df_wsumw = df_wsumw[df_wsumw['contents_id'] == contents_list[contentsnum]].reset_index(drop=True)
    df_wsumw = df_wsumw.groupby('userid', as_index=False)['weit_vec'].sum()
    for i in range(len(df_wsumw)):
        for j in range(len(df_wsumw['weit_vec'][i])):
            if i == 0:
                df_wsumw['vec_'+str(j)] = np.nan
            if df_wsumw['weit_vec'][i][j].dtype == 'float32':
                df_wsumw['vec_'+str(j)][i] = df_wsumw['weit_vec'][i][j].astype(np.float64)
            else:
                df_wsumw['vec_'+str(j)][i] = df_wsumw['weit_vec'][i][j]
    df_wsumw = df_wsumw.fillna(0)
    return df_wsumw



# %%
# 関数化しようね（2個目）(手法1、手法2)
# 行動とか小テスト点数以外を整理する
def make_ope_vec_df(df_youso_cid, contentsnum):
    con_list_t = ['OPEN_count', 'NEXT_count', 'PREV_count', 'CLOSE_count', 'PAGE_JUMP_count', 'GETIT_count', 'OPEN_RECOMMENDATION_count', 'CLOSE_RECOMMENDATION_count', 'NOTGETIT_count', 'ADD MARKER_count', 'DELETE MARKER_count', 'CLICK_RECOMMENDATION_count', 'open_time_count']
    df_w_s = pd.DataFrame({'userid':[i for i in range(100)]})
    if contentsnum == 1:
        df_cid_w_s = df_youso_cid[(df_youso_cid['contents_id'] == contents_list[0]) | (df_youso_cid['contents_id'] == contents_list[1])].reset_index(drop=True)
        for i in range(2):
            pnum = df_contentspage['page'][i]
            for j in range(1,pnum+1):
                for k in range(len(con_list_t)):
                    df_w_s[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
    else:
        df_cid_w_s = df_youso_cid[df_youso_cid['contents_id'] == contents_list[contentsnum]].reset_index(drop=True)
        pnum = df_contentspage['page'][contentsnum]
        for j in range(1,pnum+1):
            for k in range(len(con_list_t)):
                df_w_s[[con_list_t[k] + '_' + str(j)]] = np.nan

    for i in range(len(df_w_s)):
        df_w_po_user = df_cid_w_s[df_cid_w_s['userid'] == i].reset_index()
        df_w_po_user = df_w_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        if 'weit_p' in df_w_po_user.columns:
            df_w_po_user = df_w_po_user.drop('weit_p', axis=1)
        df_w_po_user = df_w_po_user.fillna(0)
        for j in range(len(df_w_po_user)):
            for k in range(len(df_w_po_user.columns)):
                df_w_s.iat[i, (len(df_w_po_user.columns)*j)+k+1] = df_w_po_user.iat[j, k]
    return df_w_s


# %%
# 予測できる形にする（小テストの点数とくっつける）関数（手法1）
def make_youso_df(df_wsumw, df_w_s, contentsnum):
    df_wsumw = df_wsumw.drop(['weit_vec'], axis=1)
    df_wy = pd.merge(df_w_s, df_wsumw, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[contentsnum-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    return df_wy



# %%
# 手法1でつかう関数をまとめてみる
def make_youso_all(df_youso_cid, df_wsum, contentsnum):
    df_wsumw = make_vec_df(df_wsum, contentsnum)
    df_w_s = make_ope_vec_df(df_youso_cid, contentsnum)
    df_wy = make_youso_df(df_wsumw, df_w_s, contentsnum)
    display(df_w_s)
    return df_wy


# %%
contents_names = ['1', '2-1', '2-2', '3', '4', '5', '6', '7']

# %%
# すべての小テストに対して予測する
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wy = make_youso_all(df_youso_cid, df_wsum, o)
    LightGBM(df_wy)





# %%
# 講義内時間（前後1時間含む）に絞ろうの回
# 各学生の各ページをみている時間をだしてるよ
df_lec_p = df_page[((df_page['operation_date'] >= st_1) & (df_page['operation_date'] <= et_1)) | ((df_page['operation_date'] >= st_2) & (df_page['operation_date'] <= et_2)) | ((df_page['operation_date'] >= st_3) & (df_page['operation_date'] <= et_3)) | ((df_page['operation_date'] >= st_4) & (df_page['operation_date'] <= et_4)) | ((df_page['operation_date'] >= st_5) & (df_page['operation_date'] <= et_5)) | ((df_page['operation_date'] >= st_6) & (df_page['operation_date'] <= et_6)) | ((df_page['operation_date'] >= st_7) & (df_page['operation_date'] <= et_7))].reset_index(drop=True)
# df_inlec_p = df_lec_p[df_lec_p['open_time'] <= 1020].reset_index(drop=True)
df_inlec_p = df_lec_p[df_lec_p['open_time'] <= 300].reset_index(drop=True)
# df_inlec_p = df_lec_p.copy()
# df_inlec_p.loc[df_inlec_p['open_time'] > 300, 'open_time'] = 300

df_inlec_p
# %%
df_lec = df_read[((df_read['operation_date'] >= st_1) & (df_read['operation_date'] <= et_1)) | ((df_read['operation_date'] >= st_2) & (df_read['operation_date'] <= et_2)) | ((df_read['operation_date'] >= st_3) & (df_read['operation_date'] <= et_3)) | ((df_read['operation_date'] >= st_4) & (df_read['operation_date'] <= et_4)) | ((df_read['operation_date'] >= st_5) & (df_read['operation_date'] <= et_5)) | ((df_read['operation_date'] >= st_6) & (df_read['operation_date'] <= et_6)) | ((df_read['operation_date'] >= st_7) & (df_read['operation_date'] <= et_7))].reset_index(drop=True)
df_lec
# %%
df_open_in = df_inlec_p.groupby(['userid', 'contents_id', 'cid_page'], as_index=False)['open_time'].sum()
df_open_in['weit_p'] = 0.0
for i in range(len(contents_list)):
    df_potw = df_open_in[df_open_in['contents_id'] == contents_list[i]].reset_index(drop=True)
    df_potw['weit_p'] = df_potw['open_time'] #/ max(df_potw['open_time'])
    for j in range(len(df_potw)):
        index_n = df_open_in[(df_open_in['cid_page'] == df_potw['cid_page'][j])&(df_open_in['userid'] == df_potw['userid'][j])].index
        df_open_in['weit_p'][index_n] = df_potw['weit_p'][j]
df_open_in
# %%
df_open_int = df_inlec_p.groupby(['userid', 'contents_id', 'cid_page'], as_index=False)['open_time'].sum()
df_open_int['weit_p'] = 0.0
for i in range(len(contents_list)):
    for k in range(100):
        df_potw = df_open_int[(df_open_int['contents_id'] == contents_list[i])&(df_open_int['userid'] == k)].reset_index(drop=True)
        df_potw['weit_p'] = df_potw['open_time'] / sum(df_potw['open_time'])
        for j in range(len(df_potw)):
            index_n = df_open_int[(df_open_int['cid_page'] == df_potw['cid_page'][j])&(df_open_int['userid'] == df_potw['userid'][j])].index
            df_open_int['weit_p'][index_n] = df_potw['weit_p'][j]
df_open_int

# %%
# df_invec = pd.merge(df_open_in, df_npvec, on=['contents_id', 'cid_page'])
df_invec = pd.merge(df_open_int, df_npvec, on=['contents_id', 'cid_page'])

df_invec['weit_vec'] = df_invec['weit_p'] * df_invec['page_vec']
df_invecs = df_invec.copy()
df_invecs = df_invecs[['userid', 'cid_page', 'contents_id', 'weit_vec']]
df_wsum_in = df_invecs.groupby(['userid', 'contents_id'], as_index=False)['weit_vec'].sum()
for i in range(len(df_wsum_in)):
    v = df_wsum_in['weit_vec'][i]
    normalized_v = v/np.linalg.norm(v)
    df_wsum_in.iat[i, 2] = normalized_v
df_wsum_in

# %%
# 各ユーザーが各ページで各行動をどれくらいとっているかの確認
df_opeu_in = pd.DataFrame({'userid':us_list, 'contents_id':con_list, 'cid_page':cid_list})
for i in range(len(operation_name_list)):
    df_ope = df_lec[df_lec['operation_name'] == operation_name_list[i]].reset_index(drop=True)
    df_ope_user = df_ope.groupby(['userid', 'contents_id', 'cid_page'])['operation_name'].count().reset_index()
    df_ope_user = df_ope_user.rename({'operation_name':operation_name_list[i] + '_count'}, axis=1)
    df_opeu_in = pd.merge(df_opeu_in, df_ope_user, on=['userid', 'contents_id', 'cid_page'], how='left')
df_opeu_in = df_opeu_in.fillna(0)
# df_opeuserの中で無視していい列を消しておく
df_opeu_in = df_opeu_in.drop(columns=['TIMER_STOP_count', 'TIMER_PAUSE_count', 'MEMO_TEXT_CHANGE_HISTORY_count'])
df_opeu_in

# %%
df_youso_in = pd.merge(df_opeu_in, df_open_in, on=['userid', 'contents_id', 'cid_page'], how='left')
# df_youso_in = pd.merge(df_opeu_in, df_open_int, on=['userid', 'contents_id', 'cid_page'], how='left')

df_youso_in

# %%
# 講義時間内に絞ったほうで求める
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wy = make_youso_all(df_youso_in, df_wsum_in, o)
    LightGBM(df_wy)





# %%
# ★手法2（一旦忘れる）
# ★★ページごとのコサイン類似度を足してそれを重みとして各要素に掛け合わせている
# 閾値処理として0.4以下は消す
cosimd_z_list = []
for i in range(len(cosimd_l_list)):
    cosimd_l_l = cosimd_l_list.copy()
    df_cosim = cosimd_l_l[i]
    df_cosim['cos_sum_4'] = 0
    for j in range(len(df_cosim)):
        sim1 = np.nan
        sim2 = np.nan
        sim3 = np.nan
        sim4 = np.nan
        sim5 = np.nan
        if df_cosim['cos_sim1'][j] > 0.4:
            sim1 = df_cosim['cos_sim1'][j]
        if df_cosim['cos_sim2'][j] > 0.4:
            sim2 = df_cosim['cos_sim2'][j]
        if df_cosim['cos_sim3'][j] > 0.4:
            sim3 = df_cosim['cos_sim3'][j]
        if df_cosim['cos_sim4'][j] > 0.4:
            sim4 = df_cosim['cos_sim4'][j]
        if df_cosim['cos_sim5'][j] > 0.4:
            sim5 = df_cosim['cos_sim5'][j]
        cosum = sim1 + sim2 + sim3 + sim4 + sim5
        df_cosim.iat[j, 6] = cosum.astype(np.float64)
    cosimd_z_list.append(df_cosim)
cosimd_z_list


# %%
df_p2 = cosimd_z_list[0]
for i in range(1, len(cosimd_z_list)):
    df_p2 = pd.concat([df_p2, cosimd_z_list[i]], axis=0, ignore_index=True)
df_p2['contents_id'] = 0
for i in range(len(df_p2)):
    df_p2['contents_id'][i] = df_p2['cid_page'][i][:-4]
df_p2




# %%
# 重みをもとめて掛け合わせる関数（手法2）
def weit_vec_2(df_w_s, df_page_test_cosim, contentsnum):
    w = 0
    cnt = 0
    for i in range(1, len(df_w_s.columns)):
        df_p2w = df_page_test_cosim.copy()
        if contentsnum == 1:
            df_p2w = df_p2w[(df_p2w['contents_id'] == contents_list[0]) | (df_p2w['contents_id'] == contents_list[1])].reset_index(drop=True)
        else:
            df_p2w = df_p2w[df_p2w['contents_id'] == contents_list[contentsnum]].reset_index(drop=True)
        weit = df_p2w['cos_sum_4'][w]
        df_w_s[df_w_s.columns[i]] = df_w_s[df_w_s.columns[i]] * weit
        cnt += 1
        if cnt == len(con_list_t):
            w += 1
            cnt = 0
        return df_w_s

# %%
# 予測できる形にする関数（手法2）
def make_youso_df2(df_w_s, contentsnum):
    df_w_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[contentsnum-1]]
    df_w_youso = pd.merge(df_w_s, df_w_q, on='userid', how='left')
    df_w_youso = df_w_youso.drop(['timecreated', 'respons_day'], axis=1)
    return df_w_youso
    
# %%
# 手法2で使う関数をまとめる
def make_youso_all2(df_youso_cid, df_page_test_cosim, contentsnum):
    df_w_s = make_ope_vec_df(df_youso_cid, contentsnum)
    df_w_s = weit_vec_2(df_w_s, df_page_test_cosim, contentsnum)
    df_w_youso2 = make_youso_df2(df_w_s, contentsnum)
    return df_w_youso2

# %%
# まわす
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wy2 = make_youso_all2(df_youso_cid, df_p2, o)
    LightGBM(df_wy2)



# %%
# 講義時間内に絞ってやるよ（手法2）
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wy2in = make_youso_all2(df_youso_in, df_p2, o)
    LightGBM(df_wy2in)



# %%
# https://github.com/sonoisa/sentence-transformers
# Sentence-BERTでやったver.ノート変えようかと思いましたが色々写すの面倒なのでこのままの.pyです
class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)

# %%
MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"  # <- v2です。
# MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
model = SentenceBertJapanese(MODEL_NAME)


# %%
# Sentense-BERTでベクトル化
df_npvec_n = pd.DataFrame({'cid_page':cid_list[:385]})
df_npvec_n['page_vec'] = np.NaN
df_npvec_n['page_vec'] = df_npvec_n['page_vec'].astype('O')
contentsdic_list = [contents0_dic, contentsplus_dic, contents1_dic, contents2_dic, contents3_dic, contents4_dic, contents5_dic, contents6_dic, contents7_dic]
page_n = 0
contentsdic_l = copy.copy(contentsdic_list)
for i in range(len(contentsdic_list)):
    contents_dic = contentsdic_l[i]
    contentstext_list = []
    for j in range(len(contents_dic)):
        context = ' '.join(contents_dic[int(df_npvec_n['cid_page'][page_n+j][-3:])])
        context = context.replace('•', ' ')
        contentstext_list.append(context)
    sentence_embeddings = model.encode(contentstext_list, batch_size=8)
    print("Sentence embeddings:", sentence_embeddings)
    for k in range(len(sentence_embeddings)):
        num = page_n + k
        df_npvec_n.iat[num, 1] = sentence_embeddings[k]
    page_n = page_n + len(contentstext_list)
    
for i in range(len(df_npvec_n)):
    v = df_npvec_n['page_vec'][i]
    normalized_v = v/torch.norm(v)
    df_npvec_n.iat[i, 1] = normalized_v
    
df_npvec_n['contents_id'] = 0
for i in range(len(df_npvec_n)):
    df_npvec_n['contents_id'][i] = df_npvec_n['cid_page'][i][:-4]
df_npvec_n
# %%
# 小テスト問題（答え含む）をベクトル化(snetence-BERT)
qvec_l_l = copy.copy(df_qvec_l_list)
for i in range(len(qvec_l_l)):
    df_qvec_l = qvec_l_l[i]
    quiz_l_dic = quizd_l_list[i]
    quiz_list = []
    for j in range(len(df_qvec_l)):
        quiztext = ' '.join(quiz_l_dic[j+1])
        quiz_list.append(quiztext)
    sentence_embeddings = model.encode(quiz_list, batch_size=8)
    print("Sentence embeddings:", sentence_embeddings)
    for k in range(len(sentence_embeddings)):
        df_qvec_l['quiz_vec'][k+1] = sentence_embeddings[k]

for i in range(len(qvec_l_l)):
    df_qvec_l = qvec_l_l[i]
    for j in range(1, len(df_qvec_l)):
        v = df_qvec_l['quiz_vec'][j]
        normalized_v = v/torch.norm(v)
        df_qvec_l['quiz_vec'][j] = normalized_v
    display(df_qvec_l)




# %%
df_cosim1_s = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim2_1_s = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim2_2_s = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim3_s = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim4_s = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim5_s = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim6_s = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
df_cosim7_s = pd.DataFrame(columns=['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5'])
cosimd_s_list = [df_cosim1_s, df_cosim2_1_s, df_cosim2_2_s, df_cosim3_s, df_cosim4_s, df_cosim5_s, df_cosim6_s, df_cosim7_s]


# 各コンテンツと小テスト内容のコサイン類似度をだしてる(Sentents-BERT)
for k in range(8):
    df_cosim_s = cosimd_s_list[k]
    df_x = qvec_l_l[k]
    for i in range(5):
        if k == 0:
            df_wvec = df_npvec_n[(df_npvec_n['contents_id']==contents_list[0])|(df_npvec_n['contents_id']==contents_list[1])].reset_index(drop=True)
        else:
            df_wvec = df_npvec_n[df_npvec_n['contents_id']==contents_list[k+1]].reset_index(drop=True)
        for j in range(len(df_wvec)):
            t1 = df_x['quiz_vec'][i+1]
            t2 = df_wvec['page_vec'][j]
            x = torch.FloatTensor(t1)
            y = torch.FloatTensor(t2)
            cosim = F.cosine_similarity(x, y, dim=0)
            if i == 0:
                df_cosim_s = df_cosim_s.append({'cid_page': df_wvec['cid_page'][j], 'cos_sim1': cosim.item()}, ignore_index=True)
            else:
                df_cosim_s['cos_sim'+str(i+1)][j] = cosim.item()
    cosimd_s_list[k] = df_cosim_s
    display(df_cosim_s)


# %%
# ★手法1
# サチラセル＝飽和させる
df_povet_sn = pd.merge(df_pot_s, df_npvec_n, on=['contents_id', 'cid_page'])
# df_povet_sn = pd.merge(df_pot_t, df_npvec_n, on=['contents_id', 'cid_page'])

df_povet_sn
# %%
df_povet_sn['weit_vec'] = df_povet_sn['weit_p'] * df_povet_sn['page_vec']
df_povetf_n = df_povet_sn.copy()
df_povetf_n = df_povetf_n[['userid', 'cid_page', 'contents_id', 'weit_vec']]
df_wsumf_n = df_povetf_n.groupby(['userid', 'contents_id'], as_index=False)['weit_vec'].sum()
for i in range(len(df_wsumf_n)):
    v = df_wsumf_n['weit_vec'][i]
    normalized_v = v/np.linalg.norm(v)
    df_wsumf_n.iat[i, 2] = normalized_v
df_wsumf_n


# %%
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wy = make_youso_all(df_youso_cid, df_wsumf_n, o)
    LightGBM(df_wy)
    

# %%
# for o in range(1, len(contents_list)):
#     n = o-1
#     filename = f'.\\score\\1_all\\ypred_c{contents_names[n]}.csv'
#     print(contents_list[o])
#     df_wy = make_youso_all(df_youso_cid, df_wsumf_n, o)
#     LightGBM_2(df_wy, filename)

# %%
# 講義時間内
df_invec_n = pd.merge(df_open_in, df_npvec_n, on=['contents_id', 'cid_page'])
# df_invec_n = pd.merge(df_open_int, df_npvec_n, on=['contents_id', 'cid_page'])

df_invec_n['weit_vec'] = df_invec_n['weit_p'] * df_invec_n['page_vec']
df_invecs_n = df_invec_n.copy()
df_invecs_n = df_invecs_n[['userid', 'cid_page', 'contents_id', 'weit_vec']]
df_wsumn_in = df_invecs_n.groupby(['userid', 'contents_id'], as_index=False)['weit_vec'].sum()
for i in range(len(df_wsumn_in)):
    v = df_wsumn_in['weit_vec'][i]
    normalized_v = v/np.linalg.norm(v)
    df_wsumn_in.iat[i, 2] = normalized_v
df_wsumn_in
# %%
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wyn = make_youso_all(df_youso_in, df_wsumn_in, o)
    LightGBM(df_wyn)


# %%
# for o in range(1, len(contents_list)):
#     n = o-1
#     filename = f'.\\score\\1_inlec\\ypred_c{contents_names[n]}.csv'
#     print(contents_list[o])
#     df_wyn = make_youso_all(df_youso_in, df_wsumn_in, o)
#     LightGBM_2(df_wyn, filename)

# %%
# ★手法2
cosimd_zs_list = []
cosimd_s = copy.copy(cosimd_s_list)
for i in range(len(cosimd_s_list)):
    df_cosim = cosimd_s[i]
    df_cosim['cos_sum_4'] = 0
    for j in range(len(df_cosim)):
        sim1 = 0
        sim2 = 0
        sim3 = 0
        sim4 = 0
        sim5 = 0
        if df_cosim['cos_sim1'][j] > 0.4:
            sim1 = df_cosim['cos_sim1'][j]
        if df_cosim['cos_sim2'][j] > 0.4:
            sim2 = df_cosim['cos_sim2'][j]
        if df_cosim['cos_sim3'][j] > 0.4:
            sim3 = df_cosim['cos_sim3'][j]
        if df_cosim['cos_sim4'][j] > 0.4:
            sim4 = df_cosim['cos_sim4'][j]
        if df_cosim['cos_sim5'][j] > 0.4:
            sim5 = df_cosim['cos_sim5'][j]
        cosum = sim1 + sim2 + sim3 + sim4 + sim5
        df_cosim['cos_sum_4'][j] = cosum
    cosimd_zs_list.append(df_cosim)
cosimd_zs_list

# %%
df_p2_n = cosimd_zs_list[0]
df_p2_n['contents_id'] = 0
for i in range(1, len(cosimd_zs_list)):
    df_p2_n = pd.concat([df_p2_n, cosimd_zs_list[i]], axis=0, ignore_index=True)
for i in range(len(df_p2_n)):
    df_p2_n['contents_id'][i] = df_p2_n['cid_page'][i][:-4]
df_p2_n

# %%
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wy2 = make_youso_all2(df_youso_cid, df_p2_n, o)
    LightGBM(df_wy2)
    
    
    
# %%
# 講義時間内にしぼる
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wy2nin = make_youso_all2(df_youso_in, df_p2_n, o)
    LightGBM(df_wy2nin)

    

# %%
# ここから試しにやってみようのつもりだったので関数化を全然してないです


# 各ページ間のコサイン類似度をヒートマップで表す。
for i in range(len(contents_list)):
# for i in range(1):
    print(contents_list[i])
    df_con = df_npvec_n[df_npvec_n['contents_id'] == contents_list[i]].reset_index(drop=False)
    df_convec = pd.DataFrame(index=([o for o in range(1, len(df_con['cid_page']))]), columns=list([o for o in range(1, len(df_con['cid_page']))]))
    for j in range(len(df_convec)):
        for k in range(len(df_convec)):
            t1 = df_con['page_vec'][j]
            t2 = df_con['page_vec'][k]
            x = torch.FloatTensor(t1)
            y = torch.FloatTensor(t2)
            cosim = F.cosine_similarity(x, y, dim=0)
            df_convec.iat[j,k] = cosim.item()
    # plt.figure()
    # plt.subplots(figsize=(18, 15))
    plt.figure(figsize=(8, 6.75))
    sns.heatmap(df_convec.fillna(0), cmap='GnBu')
    plt.xlabel('page_number', fontsize=12)
    plt.ylabel('page_number', fontsize=12)    
    # plt.yticks([i for i in range(1,len(df_con)+1, 5)])
    # plt.xticks([i for i in range(1,len(df_con)+1, 5)],rotation=90)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()
    # for j in range(len(df_convec)):
    #     display(df_convec.sort_values(df_convec.index[j], ascending=False)[df_convec.index[j]].head(3))




# %%
# 情報ないページ絞って予測したらどうなる？
# 情報ないと判断したページ全抜きver
dcpage_list = []
for i in [1,2,3,4,5,6,7,8]:
    dcpage_list.append(contents_0 + '_' + str(i).zfill(3))
for i in [1,2,5,6,7,9,14,16,17,20,21,22,25,26,27,28,29,30,31,35,36,37,38,39,43,47,48,49,50,51,53,54,56,57,62,63,64,65,66]:
    dcpage_list.append(contents_1 + '_' + str(i).zfill(3))
for i in [1,2,4,5,7,19,24,28,30,37,41,48,49,50,51,52,53,54,55,59,62]:
    dcpage_list.append(contents_2 + '_' + str(i).zfill(3))
for i in [1,2,3,4]:
    dcpage_list.append(contents_3 + '_' + str(i).zfill(3))
for i in [1,2,3,4,11,13,15,19,37,38,39,40,41,48,49]:
    dcpage_list.append(contents_4 + '_' + str(i).zfill(3))
for i in [1,2,4]:
    dcpage_list.append(contents_5 + '_' + str(i).zfill(3))
for i in [1,2,3,13,28,30]:
    dcpage_list.append(contents_6 + '_' + str(i).zfill(3))
for i in [1,2,3,4,7,9,10,11,13,14,15,19,20,23,26,27,31,33,35,36,42,44,46,55,56,63,64,65,67,69,70,72,73,79,80,81,82]:
    dcpage_list.append(contents_7 + '_' + str(i).zfill(3))
# %%
dlist = []
df_pot_d = df_po.groupby(['userid', 'contents_id','cid_page'], as_index=False)['open_time'].sum().sort_values('open_time')
for i in range(len(df_pot_d)):
    if df_pot_d['cid_page'][i] in dcpage_list:
        dlist.append(i)
df_pt_d = df_pot_d.drop(dlist).reset_index(drop=False)
df_pt_d
# %%
# 手法1
df_pt_d['weit_p'] = 0.0
for i in range(1, len(contents_list)):
    df_potw = df_pt_d[df_pt_d['contents_id'] == contents_list[i]].reset_index(drop=True)
    df_potw['weit_p'] = df_potw['open_time'] / max(df_potw['open_time'])
    for j in range(len(df_potw)):
        index_n = df_pt_d[(df_pt_d['cid_page'] == df_potw['cid_page'][j])&(df_pt_d['userid'] == df_potw['userid'][j])].index
        df_pt_d['weit_p'][index_n] = df_potw['weit_p'][j]
df_pt_d

# %%
df_pvet_d = pd.merge(df_pt_d, df_npvec_n, on=['contents_id', 'cid_page'])
df_pvet_d['weit_vec'] = df_pvet_d['weit_p'] * df_pvet_d['page_vec']
df_pvetd = df_pvet_d.copy()
df_pvetd = df_pvetd[['userid', 'cid_page', 'contents_id', 'weit_vec']]
df_wsumd = df_pvetd.groupby(['userid', 'contents_id'], as_index=False)['weit_vec'].sum()
for i in range(len(df_wsumd)):
    v = df_wsumd['weit_vec'][i]
    normalized_v = v/np.linalg.norm(v)
    df_wsumd.iat[i, 2] = normalized_v
df_wsumd


# %%
# 試しに1週目だけでやってみる
df_wsumdw = make_vec_df(df_wsumd, 1)
df_wsumdw

# %%
# 全ての学生に対して全てのページの行動を数えているdf_youso_cid自体からも抜いちゃおう
df_youso_cidd = df_youso_cid.copy()
dlist = []
for i in range(len(df_youso_cidd)):
    if df_youso_cidd['cid_page'][i] in dcpage_list:
        dlist.append(i)
df_youso_cidd = df_youso_cidd.drop(dlist).reset_index(drop=True)
df_youso_cidd


# %%
df_w_d = pd.DataFrame({'userid':[i for i in range(100)]})
df_cid_w_d = df_youso_cidd[df_youso_cidd['contents_id'] == contents_list[1]].reset_index(drop=True)
pnum = df_contentspage['page'][1]
for j in range(1,pnum+1):
    for k in range(len(con_list_t)):
        df_w_d[[con_list_t[k] + '_' + str(j)]] = np.nan

for i in range(len(df_w_d)):
    df_w_p_user = df_cid_w_d[df_cid_w_d['userid'] == i].reset_index(drop=True)
    df_w_p_user = df_w_p_user.drop(['userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    if 'weit_p' in df_w_p_user.columns:
        df_w_p_user = df_w_p_user.drop('weit_p', axis=1)
    df_w_p_user = df_w_p_user.fillna(0)
    for j in range(len(df_w_p_user)):
        for k in range(len(df_w_p_user.columns)):
            df_w_d.iat[i, (len(df_w_p_user.columns)*j)+k+1] = df_w_p_user.iat[j, k]
df_w_d



# %%
# いけたっぽい？
df_wyd = make_youso_df(df_wsumdw, df_w_d, 1)
LightGBM(df_wyd)


# %%
# 2週目以降もやってみよう
del_list = [[1,2,3,4,5,6,7,8], [1,2,5,6,7,9,14,16,17,20,21,22,25,26,27,28,29,30,31,35,36,37,38,39,43,47,48,49,50,51,53,54,56,57,62,63,64,65,66], [1,2,4,5,7,19,24,28,30,37,41,48,49,50,51,52,53,54,55,59,62], [1,2,3,4], [1,2,3,4,11,13,15,19,37,38,39,40,41,48,49], [1,2,4], [1,2,3,13,28,30], [1,2,3,4,7,9,10,11,13,14,15,19,20,23,26,27,31,33,35,36,42,44,46,55,56,63,64,65,67,69,70,72,73,79,80,81,82]]
for o in range(2, len(contents_list)):
    print(contents_list[o])
    df_wsumdw = make_vec_df(df_wsumd, o)
    df_w_d = pd.DataFrame({'userid':[i for i in range(100)]})
    df_cid_w_d = df_youso_cidd[df_youso_cidd['contents_id'] == contents_list[o]].reset_index(drop=True)
    pnum = df_contentspage['page'][o]
    n_list = list(set([j for j in range(1,pnum+1)]) ^ set(del_list[o-1]))
    n_list.sort(reverse=False)
    for j in n_list:
        for k in range(len(con_list_t)):
            df_w_d[[con_list_t[k] + '_' + str(j)]] = np.nan

    for i in range(len(df_w_d)):
        df_w_p_user = df_cid_w_d[df_cid_w_d['userid'] == i].reset_index(drop=True)
        df_w_p_user = df_w_p_user.drop(['userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        if 'weit_p' in df_w_p_user.columns:
            df_w_p_user = df_w_p_user.drop('weit_p', axis=1)
        df_w_p_user = df_w_p_user.fillna(0)
        for j in range(len(df_w_p_user)):
            for k in range(len(df_w_p_user.columns)):
                df_w_d.iat[i, (len(df_w_p_user.columns)*j)+k+1] = df_w_p_user.iat[j, k]
    display(df_w_d)
    
    df_wyd = make_youso_df(df_wsumdw, df_w_d, 1)
    LightGBM(df_wyd)



# %%
# 講義時間内に絞る
dlist = []
df_inlec_pd = df_inlec_p.groupby(['userid', 'contents_id', 'cid_page'], as_index=False)['open_time'].sum().sort_values('open_time')
for i in range(len(df_inlec_pd)):
    if df_inlec_pd['cid_page'][i] in dcpage_list:
        dlist.append(i)
df_inpt_d = df_inlec_pd.drop(dlist).reset_index(drop=False)
df_inpt_d['weit_p'] = 0.0
for i in range(1, len(contents_list)):
    df_potw = df_inpt_d[df_inpt_d['contents_id'] == contents_list[i]].reset_index(drop=True)
    df_potw['weit_p'] = df_potw['open_time'] / max(df_potw['open_time'])
    for j in range(len(df_potw)):
        index_n = df_inpt_d[(df_inpt_d['cid_page'] == df_potw['cid_page'][j])&(df_inpt_d['userid'] == df_potw['userid'][j])].index
        df_inpt_d['weit_p'][index_n] = df_potw['weit_p'][j]
df_inpt_d

# %%
df_inpvet_d = pd.merge(df_inpt_d, df_npvec_n, on=['contents_id', 'cid_page'])
df_inpvet_d['weit_vec'] = df_inpvet_d['weit_p'] * df_pvet_d['page_vec']
df_inpvetd = df_inpvet_d.copy()
df_inpvetd = df_inpvetd[['userid', 'cid_page', 'contents_id', 'weit_vec']]
df_wsumd_in = df_inpvetd.groupby(['userid', 'contents_id'], as_index=False)['weit_vec'].sum()
for i in range(len(df_wsumd_in)):
    v = df_wsumd_in['weit_vec'][i]
    normalized_v = v/np.linalg.norm(v)
    df_wsumd_in.iat[i, 2] = normalized_v
df_wsumd_in


# %%
# とりあえず1週目やる
df_wsumdw_in = make_vec_df(df_wsumd_in, 1)
df_wsumdw_in

# %%
# 全ての学生に対して全てのページの行動を数えているdf_youso_in自体からも抜いちゃおう
df_youso_ind = df_youso_in.copy()
dlist = []
for i in range(len(df_youso_ind)):
    if df_youso_ind['cid_page'][i] in dcpage_list:
        dlist.append(i)
df_youso_ind = df_youso_ind.drop(dlist).reset_index(drop=True)
df_youso_ind


# %%
df_w_ind = pd.DataFrame({'userid':[i for i in range(100)]})
df_cid_w_ind = df_youso_ind[df_youso_ind['contents_id'] == contents_list[1]].reset_index(drop=True)
pnum = df_contentspage['page'][1]
for j in range(1,pnum+1):
    for k in range(len(con_list_t)):
        df_w_ind[[con_list_t[k] + '_' + str(j)]] = np.nan
for i in range(len(df_w_ind)):
    df_w_p_user = df_cid_w_ind[df_cid_w_ind['userid'] == i].reset_index(drop=True)
    df_w_p_user = df_w_p_user.drop(['userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    if 'weit_p' in df_w_p_user.columns:
        df_w_p_user = df_w_p_user.drop('weit_p', axis=1)
    df_w_p_user = df_w_p_user.fillna(0)
    for j in range(len(df_w_p_user)):
        for k in range(len(df_w_p_user.columns)):
            df_w_ind.iat[i, (len(df_w_p_user.columns)*j)+k+1] = df_w_p_user.iat[j, k]
df_w_ind



# %%
df_wyd = make_youso_df(df_wsumdw_in, df_w_ind, 1)
LightGBM(df_wyd)


# %%
# 2週目以降もやる
del_list = [[1,2,3,4,5,6,7,8], [1,2,5,6,7,9,14,16,17,20,21,22,25,26,27,28,29,30,31,35,36,37,38,39,43,47,48,49,50,51,53,54,56,57,62,63,64,65,66], [1,2,4,5,7,19,24,28,30,37,41,48,49,50,51,52,53,54,55,59,62], [1,2,3,4], [1,2,3,4,11,13,15,19,37,38,39,40,41,48,49], [1,2,4], [1,2,3,13,28,30], [1,2,3,4,7,9,10,11,13,14,15,19,20,23,26,27,31,33,35,36,42,44,46,55,56,63,64,65,67,69,70,72,73,79,80,81,82]]
for o in range(2, len(contents_list)):
    print(contents_list[o])
    df_wsumdw_in = make_vec_df(df_wsumd_in, o)
    df_w_ind = pd.DataFrame({'userid':[i for i in range(100)]})
    df_cid_w_ind = df_youso_ind[df_youso_ind['contents_id'] == contents_list[o]].reset_index(drop=True)
    pnum = df_contentspage['page'][o]
    n_list = list(set([j for j in range(1,pnum+1)]) ^ set(del_list[o-1]))
    n_list.sort(reverse=False)
    for j in n_list:
        for k in range(len(con_list_t)):
            df_w_ind[[con_list_t[k] + '_' + str(j)]] = np.nan

    for i in range(len(df_w_ind)):
        df_w_p_user = df_cid_w_ind[df_cid_w_ind['userid'] == i].reset_index(drop=True)
        df_w_p_user = df_w_p_user.drop(['userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        if 'weit_p' in df_w_p_user.columns:
            df_w_p_user = df_w_p_user.drop('weit_p', axis=1)
        df_w_p_user = df_w_p_user.fillna(0)
        for j in range(len(df_w_p_user)):
            for k in range(len(df_w_p_user.columns)):
                df_w_ind.iat[i, (len(df_w_p_user.columns)*j)+k+1] = df_w_p_user.iat[j, k]
    display(df_w_ind)
    
    df_wyd = make_youso_df(df_wsumdw_in, df_w_ind, 1)
    LightGBM(df_wyd)





# %%
# 手法2のほうでも絞ってみる？
dlist = []
df_p2_d = df_p2_n.copy()
for i in range(len(df_p2_d)):
    if df_p2_n['cid_page'][i] in dcpage_list:
        dlist.append(i)
df_pt2_d = df_p2_d.drop(dlist).reset_index(drop=True)
df_pt2_d


# %%
df_w_d = pd.DataFrame({'userid':[i for i in range(100)]})
df_cid_w_d = df_youso_cidd[df_youso_cidd['contents_id'] == contents_list[1]].reset_index(drop=True)
pnum = df_contentspage['page'][1]
for j in range(1,pnum+1):
    for k in range(len(con_list_t)):
        df_w_d[[con_list_t[k] + '_' + str(j)]] = np.nan

for i in range(len(df_w_d)):
    df_w_p_user = df_cid_w_d[df_cid_w_d['userid'] == i].reset_index(drop=True)
    df_w_p_user = df_w_p_user.drop(['userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    if 'weit_p' in df_w_p_user.columns:
        df_w_p_user = df_w_p_user.drop('weit_p', axis=1)
    df_w_p_user = df_w_p_user.fillna(0)
    for j in range(len(df_w_p_user)):
        for k in range(len(df_w_p_user.columns)):
            df_w_d.iat[i, (len(df_w_p_user.columns)*j)+k+1] = df_w_p_user.iat[j, k]
df_w_d


# %%
df_w_d = weit_vec_2(df_w_d, df_pt2_d, 1)
df_wy2_d = make_youso_df2(df_w_d, 1)
LightGBM(df_wy2_d)


# %%
for o in range(2, len(contents_list)):
    print(contents_list[o])
    df_w_d = pd.DataFrame({'userid':[i for i in range(100)]})
    df_cid_w_d = df_youso_cidd[df_youso_cidd['contents_id'] == contents_list[o]].reset_index(drop=True)
    pnum = df_contentspage['page'][o]
    n_list = list(set([j for j in range(1,pnum+1)]) ^ set(del_list[o-1]))
    n_list.sort(reverse=False)
    for j in n_list:
        for k in range(len(con_list_t)):
            df_w_d[[con_list_t[k] + '_' + str(j)]] = np.nan

    for i in range(len(df_w_d)):
        df_w_p_user = df_cid_w_d[df_cid_w_d['userid'] == i].reset_index(drop=True)
        df_w_p_user = df_w_p_user.drop(['userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        if 'weit_p' in df_w_p_user.columns:
            df_w_p_user = df_w_p_user.drop('weit_p', axis=1)
        df_w_p_user = df_w_p_user.fillna(0)
        for j in range(len(df_w_p_user)):
            for k in range(len(df_w_p_user.columns)):
                df_w_d.iat[i, (len(df_w_p_user.columns)*j)+k+1] = df_w_p_user.iat[j, k]
    display(df_w_d)
    
    df_w_d = weit_vec_2(df_w_d, df_pt2_d, o)
    df_wy2_d = make_youso_df2(df_w_d, o)
    LightGBM(df_wy2_d)



# %%
# いちおう時間内もやる
df_w_d = pd.DataFrame({'userid':[i for i in range(100)]})
df_cid_w_d = df_youso_ind[df_youso_ind['contents_id'] == contents_list[1]].reset_index(drop=True)
pnum = df_contentspage['page'][1]
for j in range(1,pnum+1):
    for k in range(len(con_list_t)):
        df_w_d[[con_list_t[k] + '_' + str(j)]] = np.nan

for i in range(len(df_w_d)):
    df_w_p_user = df_cid_w_d[df_cid_w_d['userid'] == i].reset_index(drop=True)
    df_w_p_user = df_w_p_user.drop(['userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    if 'weit_p' in df_w_p_user.columns:
        df_w_p_user = df_w_p_user.drop('weit_p', axis=1)
    df_w_p_user = df_w_p_user.fillna(0)
    for j in range(len(df_w_p_user)):
        for k in range(len(df_w_p_user.columns)):
            df_w_d.iat[i, (len(df_w_p_user.columns)*j)+k+1] = df_w_p_user.iat[j, k]
df_w_d


# %%
df_w_d = weit_vec_2(df_w_d, df_pt2_d, 1)
df_wy2_d = make_youso_df2(df_w_d, 1)
LightGBM(df_wy2_d)


# %%
for o in range(2, len(contents_list)):
    print(contents_list[o])
    df_w_d = pd.DataFrame({'userid':[i for i in range(100)]})
    df_cid_w_d = df_youso_ind[df_youso_ind['contents_id'] == contents_list[o]].reset_index(drop=True)
    pnum = df_contentspage['page'][o]
    n_list = list(set([j for j in range(1,pnum+1)]) ^ set(del_list[o-1]))
    n_list.sort(reverse=False)
    for j in n_list:
        for k in range(len(con_list_t)):
            df_w_d[[con_list_t[k] + '_' + str(j)]] = np.nan

    for i in range(len(df_w_d)):
        df_w_p_user = df_cid_w_d[df_cid_w_d['userid'] == i].reset_index(drop=True)
        df_w_p_user = df_w_p_user.drop(['userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        if 'weit_p' in df_w_p_user.columns:
            df_w_p_user = df_w_p_user.drop('weit_p', axis=1)
        df_w_p_user = df_w_p_user.fillna(0)
        for j in range(len(df_w_p_user)):
            for k in range(len(df_w_p_user.columns)):
                df_w_d.iat[i, (len(df_w_p_user.columns)*j)+k+1] = df_w_p_user.iat[j, k]
    display(df_w_d)
    
    df_w_d = weit_vec_2(df_w_d, df_pt2_d, o)
    df_wy2_d = make_youso_df2(df_w_d, o)
    LightGBM(df_wy2_d)



# %%
# 上のやつ意味が分からないけど精度があがっている時があるので
# なにか一貫性のないキメラがうまれている可能性がある
# ほぼ表紙しか抜いていないような回の時も微妙やからあんまり期待できひんよなあ
# 微妙っつーかあがっていない



# %%
# 文章ベクトルをいれるだけではどうなるのかやってみるよ
# 手法1
df_wsuma = df_page.groupby(['userid', 'contents_id', 'cid_page'], as_index=False)['open_time'].sum().sort_values('open_time')
df_wsuma = pd.merge(df_wsuma, df_npvec_n, on=['contents_id', 'cid_page'])
df_wsuma = df_wsuma.groupby(['userid', 'contents_id'], as_index=False)['page_vec'].sum()
df_wsuma = df_wsuma.rename(columns={'page_vec':'weit_vec'})
df_wsuma
# %%
# みんなにおんなじベクトルあたえるんだからいちいち分解必要ないじゃんと思ってやったら見ていないコンテンツのベクトルは含まれていないことがわかって大ショック
# とりあえず、見てないコンテンツは含めないものでやります
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wsumaw = make_vec_df(df_wsuma, o)
    df_w_s = make_ope_vec_df(df_youso_cid, o)
    df_wy = make_youso_df(df_wsumaw, df_w_s, o)
    display(df_w_s)
    LightGBM(df_wy)



# %%
# 時間内もやっておくよ
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wsumaw = make_vec_df(df_wsuma, o)
    df_w_s = make_ope_vec_df(df_youso_in, o)
    df_wy = make_youso_df(df_wsumaw, df_w_s, o)
    display(df_w_s)
    LightGBM(df_wy)




# %%
# コンテンツベクトルだけで予測？（重みとかないver.）(講義時間内外も関係ない)
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wsumaw = make_vec_df(df_wsuma, o)
    df_wsumaw = df_wsumaw.drop(['weit_vec'], axis=1)
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[o-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wsumaw, df_wq, on='userid', how='left')
    display(df_wy)
    LightGBM(df_wy)


# %%
for o in range(1, len(contents_list)):
    n = o-1
    filename = f'.\\score\\veconly_all\\ypred_c{contents_names[n]}.csv'
    print(contents_list[o])
    df_wsumaw = make_vec_df(df_wsuma, o)
    df_wsumaw = df_wsumaw.drop(['weit_vec'], axis=1)
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[o-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wsumaw, df_wq, on='userid', how='left')
    display(df_wy)
    LightGBM_2(df_wy, filename)
    
# %%
# コンテンツベクトルだけで予測（重みありver）(外)
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wsumaw = make_vec_df(df_wsumf_n, o)
    df_wsumaw = df_wsumaw.drop(['weit_vec'], axis=1)
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[o-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wsumaw, df_wq, on='userid', how='left')
    display(df_wy)
    LightGBM(df_wy)


# %%
for o in range(1, len(contents_list)):
    n = o-1
    filename = f'.\\score\\wveconly_all\\ypred_c{contents_names[n]}.csv'
    print(contents_list[o])
    df_wsumaw = make_vec_df(df_wsumf_n, o)
    df_wsumaw = df_wsumaw.drop(['weit_vec'], axis=1)
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[o-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wsumaw, df_wq, on='userid', how='left')
    display(df_wy)
    LightGBM_2(df_wy, filename)

# %%
# コンテンツベクトルだけで予測（重みありver）(内)
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wsumaw = make_vec_df(df_wsumn_in, o)
    df_wsumaw = df_wsumaw.drop(['weit_vec'], axis=1)
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[o-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wsumaw, df_wq, on='userid', how='left')
    display(df_wy)
    LightGBM(df_wy)
    
    
# %%
for o in range(1, len(contents_list)):
    n = o-1
    filename = f'.\\score\\wveconly_inlec\\ypred_c{contents_names[n]}.csv'
    print(contents_list[o])
    df_wsumaw = make_vec_df(df_wsumn_in, o)
    df_wsumaw = df_wsumaw.drop(['weit_vec'], axis=1)
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[o-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wsumaw, df_wq, on='userid', how='left')
    display(df_wy)
    LightGBM_2(df_wy, filename)


# %%
# 重要度だけで予測するよ
# open_timeだけとりだす方が早いよな
# これミスです全部でやっちゃってる
for o in range(1, len(contents_list)):
    print(contents_list[o])
    contentsnum = o
    df_w_s = pd.DataFrame({'userid':[i for i in range(100)]})
    if contentsnum == 1:
        df_cid_w_s = df_youso_cid[(df_youso_cid['contents_id'] == contents_list[0]) | (df_youso_cid['contents_id'] == contents_list[1])].reset_index(drop=True)
        df_povet = df_povet_sn[(df_povet_sn['contents_id'] == contents_list[0]) | (df_povet_sn['contents_id'] == contents_list[1])].reset_index(drop=True)
        for i in range(2):
            pnum = df_contentspage['page'][i]
            for j in range(1,pnum+1):
                df_w_s[['open_time_' + str(j) + '_' + str(i)]] = np.nan
    else:
        df_cid_w_s = df_youso_cid[df_youso_cid['contents_id'] == contents_list[contentsnum]].reset_index(drop=True)
        df_povet = df_povet_sn[df_povet_sn['contents_id'] == contents_list[contentsnum]].reset_index(drop=True)
        pnum = df_contentspage['page'][contentsnum]
        for j in range(1,pnum+1):
            df_w_s[['open_time_' + str(j)]] = np.nan

    for i in range(len(df_w_s)):
        df_w_po_useru = df_cid_w_s[df_cid_w_s['userid'] == i].reset_index()
        df_povet = df_povet_sn[(df_povet_sn['userid'] == i)]
        df_w_po_useru = df_w_po_useru[['cid_page']].reset_index(drop=True)
        df_w_po_user = pd.merge(df_w_po_useru, df_povet, on='cid_page', how='left')
        df_w_po_user = df_w_po_user.fillna(0)
        df_w_po_user = df_w_po_user[['weit_p']]
        for j in range(len(df_w_po_user)):
            df_w_s.iat[i, j+1] = df_w_po_user.iat[j, 0]
    
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[contentsnum-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_w_s, df_wq, on='userid', how='left')
    display(df_wy)
    LightGBM(df_wy)


# %%
# 重要度だけで予測するよ
# open_timeだけとりだす方が早いよな
# これミスです全部でやっちゃってる
for o in range(1, len(contents_list)):
    print(contents_list[o])
    contentsnum = o
    df_w_s = pd.DataFrame({'userid':[i for i in range(100)]})
    df_invec_in = pd.merge(df_open_in, df_npvec_n, on=['contents_id', 'cid_page'])
    if contentsnum == 1:
        df_cid_w_s = df_youso_in[(df_youso_in['contents_id'] == contents_list[0]) | (df_youso_cid['contents_id'] == contents_list[1])].reset_index(drop=True)
        df_povet = df_invec_in[(df_invec_in['contents_id'] == contents_list[0]) | (df_invec_in['contents_id'] == contents_list[1])].reset_index(drop=True)
        for i in range(2):
            pnum = df_contentspage['page'][i]
            for j in range(1,pnum+1):
                df_w_s[['open_time_' + str(j) + '_' + str(i)]] = np.nan
    else:
        df_cid_w_s = df_youso_cid[df_youso_cid['contents_id'] == contents_list[contentsnum]].reset_index(drop=True)
        df_povet = df_invec_in[df_invec_in['contents_id'] == contents_list[contentsnum]].reset_index(drop=True)
        pnum = df_contentspage['page'][contentsnum]
        for j in range(1,pnum+1):
            df_w_s[['open_time_' + str(j)]] = np.nan

    for i in range(len(df_w_s)):
        df_w_po_useru = df_cid_w_s[df_cid_w_s['userid'] == i].reset_index()
        df_povet = df_invec_in[(df_invec_in['userid'] == i)]
        df_w_po_useru = df_w_po_useru[['cid_page']].reset_index(drop=True)
        df_w_po_user = pd.merge(df_w_po_useru, df_povet, on='cid_page', how='left')
        df_w_po_user = df_w_po_user.fillna(0)
        df_w_po_user = df_w_po_user[['weit_p']]
        for j in range(len(df_w_po_user)):
            df_w_s.iat[i, j+1] = df_w_po_user.iat[j, 0]
    
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[contentsnum-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_w_s, df_wq, on='userid', how='left')
    display(df_wy)
    LightGBM(df_wy)
    
    
# %%
# ベクトル圧縮をしたらどうなる（t-SNE）
from sklearn.manifold import TSNE
# %%
# t-SNE（外）
# perplexity=[2,5,10,50,100]
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wsumw = make_vec_df(df_wsumf_n, o)
    df_w_s = make_ope_vec_df(df_youso_cid, o)
    
    df_wsum_w = df_wsumw.drop(['weit_vec', 'userid'], axis=1)
    model_tsne = TSNE(random_state=0, n_components=2, perplexity=2)
    tsnews_list = model_tsne.fit_transform(df_wsum_w)
    df_tsnews = pd.DataFrame(tsnews_list)
    df_tsnews['userid'] = list(df_wsumw['userid'])  
    df_wy = pd.merge(df_w_s, df_tsnews, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[o-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    display(df_wy)
    LightGBM(df_wy)

# %%
# t-SNE（内）
# perplexity=[2,5,10,50,100]
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wsumw = make_vec_df(df_wsumn_in, o)
    df_w_s = make_ope_vec_df(df_youso_in, o)
    
    df_wsum_w = df_wsumw.drop(['weit_vec', 'userid'], axis=1)
    model_tsne = TSNE(random_state=0, n_components=2, perplexity=2)
    tsnews_list = model_tsne.fit_transform(df_wsum_w)
    df_tsnews = pd.DataFrame(tsnews_list)
    df_tsnews['userid'] = list(df_wsumw['userid'])  
    df_wy = pd.merge(df_w_s, df_tsnews, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[o-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    display(df_wy)
    LightGBM(df_wy)

# %%
# 全体の次元をおとす（外）
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wsumw = make_vec_df(df_wsumf_n, o)
    df_w_s = make_ope_vec_df(df_youso_cid, o)
    
    df_wsum_w = df_wsumw.drop(['weit_vec'], axis=1)
    df_wy = pd.merge(df_w_s, df_wsum_w, on='userid')
    model_tsne = TSNE(random_state=0, n_components=2, perplexity=2)
    df_w_y = df_wy.drop(['userid'],axis=1)
    tsnews_list = model_tsne.fit_transform(df_w_y)
    df_tsnews = pd.DataFrame(tsnews_list)
    df_tsnews['userid'] = list(df_wy['userid'])  
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[o-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_tsnews, df_wq, on='userid', how='left')
    display(df_wy)
    LightGBM(df_wy)

# %%
# 全体の次元をおとす（内）
for o in range(1, len(contents_list)):
    print(contents_list[o])
    df_wsumw = make_vec_df(df_wsumn_in, o)
    df_w_s = make_ope_vec_df(df_youso_in, o)
    
    df_wsum_w = df_wsumw.drop(['weit_vec'], axis=1)
    df_wy = pd.merge(df_w_s, df_wsum_w, on='userid')
    model_tsne = TSNE(random_state=0, n_components=2, perplexity=2)
    df_w_y = df_wy.drop(['userid'],axis=1)
    tsnews_list = model_tsne.fit_transform(df_w_y)
    df_tsnews = pd.DataFrame(tsnews_list)
    df_tsnews['userid'] = list(df_wsum_w['userid'])  
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[o-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_tsnews, df_wq, on='userid', how='left')
    display(df_wy)
    LightGBM(df_wy)

# %%
# べ―スラインの方もこっちでやろう（外）
for l in range(1, len(contents_list)):
    print(contents_list[l])
    if l == 1:
        df_cid_w = df_youso_cid[(df_youso_cid['contents_id'] == contents_list[0]) | (df_youso_cid['contents_id'] == contents_list[1])]
        df_w = pd.DataFrame({'userid':[i for i in range(100)]})
        for i in range(2):
            pnum = df_contentspage['page'][i]
            for j in range(1,pnum+1):
                for k in range(len(con_list_t)):
                    df_w[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    else:
        df_cid_w = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]]
        df_w = pd.DataFrame({'userid':[i for i in range(100)]})
        pnum = df_contentspage['page'][l]
        for i in range(1,pnum+1):
            for j in range(len(con_list_t)):
                df_w[[con_list_t[j] + '_' + str(i)]] = np.nan
            
    for i in range(len(df_w)):
        df_w_user = df_cid_w[df_cid_w['userid'] == i].reset_index()
        df_w_user = df_w_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_user = df_w_user.fillna(0)
        for j in range(len(df_w_user)):
            for k in range(len(df_w_user.columns)):
                df_w.iat[i, (len(df_w_user.columns)*j)+k+1] = df_w_user.iat[j, k]
    
    model_tsne = TSNE(random_state=0, n_components=2, perplexity=2)
    df_wy = df_w.drop(['userid'],axis=1)
    tsnews_list = model_tsne.fit_transform(df_wy)
    df_tsnews = pd.DataFrame(tsnews_list)
    df_tsnews['userid'] = list(df_w['userid']) 

    df_w_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_w_youso = pd.merge(df_tsnews, df_w_q, on='userid', how='left')
    df_w_youso = df_w_youso.drop(['timecreated', 'respons_day'], axis=1)
    display(df_w_youso)

    # シード値固定してる
    print(LightGBM(df_w_youso))
    
    
# %%
# べ―スラインの方もこっちでやろう（内）
for l in range(1, len(contents_list)):
    print(contents_list[l])
    if l == 1:
        df_cid_w = df_youso_in[(df_youso_in['contents_id'] == contents_list[0]) | (df_youso_in['contents_id'] == contents_list[1])]
        df_w = pd.DataFrame({'userid':[i for i in range(100)]})
        for i in range(2):
            pnum = df_contentspage['page'][i]
            for j in range(1,pnum+1):
                for k in range(len(con_list_t)):
                    df_w[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    else:
        df_cid_w = df_youso_in[df_youso_in['contents_id'] == contents_list[l]]
        df_w = pd.DataFrame({'userid':[i for i in range(100)]})
        pnum = df_contentspage['page'][l]
        for i in range(1,pnum+1):
            for j in range(len(con_list_t)):
                df_w[[con_list_t[j] + '_' + str(i)]] = np.nan
    
    for i in range(len(df_w)):
        df_w_user = df_cid_w[df_cid_w['userid'] == i].reset_index()
        df_w_user = df_w_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count','weit_p'], axis=1)
        df_w_user = df_w_user.fillna(0)
        for j in range(len(df_w_user)):
            for k in range(len(df_w_user.columns)):
                df_w.iat[i, (len(df_w_user.columns)*j)+k+1] = df_w_user.iat[j, k]
    
    model_tsne = TSNE(random_state=0, n_components=2, perplexity=2)
    df_wy = df_w.drop(['userid'],axis=1)
    tsnews_list = model_tsne.fit_transform(df_wy)
    df_tsnews = pd.DataFrame(tsnews_list)
    df_tsnews['userid'] = list(df_w['userid']) 

    df_w_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_w_youso = pd.merge(df_tsnews, df_w_q, on='userid', how='left')
    df_w_youso = df_w_youso.drop(['timecreated', 'respons_day'], axis=1)
    display(df_w_youso)

    # シード値固定してる
    print(LightGBM(df_w_youso))
    
    
# %%
# そもそも一回も資料見てないやつ消す？
df_cid_c = df_youso_cid.groupby(['userid', 'contents_id'], as_index=False)['OPEN_count'].sum()
df_cid_c
# %%
deluser = []
for i in range(len(df_cid_c)):
    if df_cid_c['OPEN_count'][i] == 0:
        deluser.append((df_cid_c['userid'][i], df_cid_c['contents_id'][i]))
deluser
# %%
# 消すかっておもったけど、一回も見てない=閲覧時間0ってことやからすでに消えているのでは？
# 閲覧時間0=重み0やからベクトル自体はないけど、全て0の列ができているってことよな
# 予測もそのまま0のはずやから大丈夫か、あんま必要なさそう、、と信じる、、
# ただもしかしたらそういうやつの影響が行動はしてるけど0点のやつにでてる可能性
# あとリスクがあるかどうかがわかるかもしれんけど、
# 一番危ないのって予測上は良い点数してるけど、点数取れてないやつじゃね？
# モデルを作らないと意味ないけどモデルはできていないと、なるほどなあ
# モデルの作り方教えてもらっていいですか
# 

# %%
df_cid_w1 = df_youso_cid[(df_youso_cid['contents_id'] == contents_list[0]) | (df_youso_cid['contents_id'] == contents_list[1])]
df_w1 = pd.DataFrame({'userid':[i for i in range(100)]})
con_list_t = ['OPEN_count', 'NEXT_count', 'PREV_count', 'CLOSE_count', 'PAGE_JUMP_count', 'GETIT_count', 'OPEN_RECOMMENDATION_count', 'CLOSE_RECOMMENDATION_count', 'NOTGETIT_count', 'ADD MARKER_count', 'DELETE MARKER_count', 'CLICK_RECOMMENDATION_count', 'open_time_count'] # 'red_per', 'yellow_per'
for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
for i in range(len(df_w1)):
    df_w1_user = df_cid_w1[df_cid_w1['userid'] == i].reset_index()
    df_w1_user = df_w1_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    df_w1_user = df_w1_user.fillna(0)
    for j in range(len(df_w1_user)):
        for k in range(len(df_w1_user.columns)):
            df_w1.iat[i, (len(df_w1_user.columns)*j)+k+1] = df_w1_user.iat[j, k]
df_w1_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1_youso = pd.merge(df_w1, df_w1_q, on='userid', how='left')
df_w1_youso = df_w1_youso.drop(['timecreated', 'respons_day'], axis=1)
df_w1_youso
# %%
LightGBM(df_w1_youso)
# %%
# やりなおしたVer.
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_cid_w = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]]
    df_w = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for i in range(1,pnum+1):
        for j in range(len(con_list_t)):
            df_w[[con_list_t[j] + '_' + str(i)]] = np.nan
            
    for i in range(len(df_w)):
        df_w_user = df_cid_w[df_cid_w['userid'] == i].reset_index()
        df_w_user = df_w_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_user = df_w_user.fillna(0)
        for j in range(len(df_w_user)):
            for k in range(len(df_w_user.columns)):
                df_w.iat[i, (len(df_w_user.columns)*j)+k+1] = df_w_user.iat[j, k]
    display(df_w)

    df_w_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_w_youso = pd.merge(df_w, df_w_q, on='userid', how='left')
    df_w_youso = df_w_youso.drop(['timecreated', 'respons_day'], axis=1)
    display(df_w_youso)

    # シード値固定してる
    print(LightGBM(df_w_youso))
# %%
df_in_w1 = df_youso_in[(df_youso_in['contents_id'] == contents_list[0]) | (df_youso_in['contents_id'] == contents_list[1])]
df_w1_in = pd.DataFrame({'userid':[i for i in range(100)]})
con_list_t = ['OPEN_count', 'NEXT_count', 'PREV_count', 'CLOSE_count', 'PAGE_JUMP_count', 'GETIT_count', 'OPEN_RECOMMENDATION_count', 'CLOSE_RECOMMENDATION_count', 'NOTGETIT_count', 'ADD MARKER_count', 'DELETE MARKER_count', 'CLICK_RECOMMENDATION_count', 'open_time_count'] # 'red_per', 'yellow_per'
for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1_in[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
for i in range(len(df_w1_in)):
    df_w1_user = df_in_w1[df_in_w1['userid'] == i].reset_index()
    df_w1_user = df_w1_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count', 'weit_p'], axis=1)
    df_w1_user = df_w1_user.fillna(0)
    for j in range(len(df_w1_user)):
        for k in range(len(df_w1_user.columns)):
            df_w1_in.iat[i, (len(df_w1_user.columns)*j)+k+1] = df_w1_user.iat[j, k]
df_w1_q_in = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1_youso_in = pd.merge(df_w1_in, df_w1_q_in, on='userid', how='left')
df_w1_youso_in = df_w1_youso_in.drop(['timecreated', 'respons_day'], axis=1)
df_w1_youso_in
# %%
LightGBM(df_w1_youso_in)
# %%
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_cid_w = df_youso_in[df_youso_in['contents_id'] == contents_list[l]]
    df_w = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for i in range(1,pnum+1):
        for j in range(len(con_list_t)):
            df_w[[con_list_t[j] + '_' + str(i)]] = np.nan
            
    for i in range(len(df_w)):
        df_w_user = df_cid_w[df_cid_w['userid'] == i].reset_index()
        df_w_user = df_w_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count', 'weit_p'], axis=1)
        df_w_user = df_w_user.fillna(0)
        for j in range(len(df_w_user)):
            for k in range(len(df_w_user.columns)):
                df_w.iat[i, (len(df_w_user.columns)*j)+k+1] = df_w_user.iat[j, k]
    display(df_w)

    df_w_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_w_youso = pd.merge(df_w, df_w_q, on='userid', how='left')
    df_w_youso = df_w_youso.drop(['timecreated', 'respons_day'], axis=1)
    display(df_w_youso)

    # シード値固定してる
    print(LightGBM(df_w_youso))

