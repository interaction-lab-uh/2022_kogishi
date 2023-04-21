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
import os
import seaborn as sns
import matplotlib as mpl
import datetime
import pickle
from sympy import content
# from datetime import datetime
# %%
import time
import lightgbm as lgb
from regex import D
from sklearn.model_selection import KFold, train_test_split  #, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import math
# %%
import sys
import pyocr.builders
from PIL import Image, ImageEnhance
import pyocr
# from janome.tokenizer import Tokenizer
# from janome.analyzer import Analyzer
# from janome.charfilter import UnicodeNormalizeCharFilter
# from janome.tokenfilter import POSKeepFilter, POSStopFilter, CompoundNounFilter, LowerCaseFilter
# from sklearn.feature_extraction.text import TfidfVectorizer
# from wordcloud import WordCloud
from math import log
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import normalize
# import termextract.janome
# import termextract.mecab
# import termextract.core
# import termextract.japanese_plaintext
import collections
# import MeCab
# %%
import transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers import models

transformers.BertTokenizer = transformers.BertJapaneseTokenizer

from transformers import BertJapaneseTokenizer, BertModel
from sentence_transformers import models,InputExample,losses
from sentence_transformers.losses import TripletDistanceMetric, TripletLoss
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.readers import TripletReader
from sentence_transformers.datasets import SentencesDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from keras_bert import load_trained_model_from_checkpoint
import sentencepiece as spm
import copy
# %%
# 次元削減用
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy.sparse.csgraph import connected_components


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
    print(contents1_dic)
# %%
with open('./pickle/contents2_dic.pickle', mode='rb') as f:
    contents2_dic = pickle.load(f)
    print(contents2_dic)
# %%
with open('./pickle/contents3_dic.pickle', mode='rb') as f:
    contents3_dic = pickle.load(f)
    print(contents3_dic)
# %%
with open('./pickle/contents4_dic.pickle', mode='rb') as f:
    contents4_dic = pickle.load(f)
    print(contents4_dic)
# %%
with open('./pickle/contents5_dic.pickle', mode='rb') as f:
    contents5_dic = pickle.load(f)
    print(contents5_dic)
# %%
with open('./pickle/contents6_dic.pickle', mode='rb') as f:
    contents6_dic = pickle.load(f)
    print(contents6_dic)
# %%
with open('./pickle/contents7_dic.pickle', mode='rb') as f:
    contents7_dic = pickle.load(f)
    print(contents7_dic)
# %%
with open('./pickle/contents0_dic.pickle', mode='rb') as f:
    contents0_dic = pickle.load(f)
    print(contents0_dic)
# %%
with open('./pickle/contentsplus_dic.pickle', mode='rb') as f:
    contentsplus_dic = pickle.load(f)
    print(contentsplus_dic)
    
# %%
with open('./pickle/quiz1_l_dic.pickle', mode='rb') as f:
    quiz1_l_dic = pickle.load(f)
    print(quiz1_l_dic)
# %%
with open('./pickle/quiz2_1_l_dic.pickle', mode='rb') as f:
    quiz2_1_l_dic = pickle.load(f)
    print(quiz2_1_l_dic)
# %%
with open('./pickle/quiz2_2_l_dic.pickle', mode='rb') as f:
    quiz2_2_l_dic = pickle.load(f)
    print(quiz2_2_l_dic)
# %%
with open('./pickle/quiz3_l_dic.pickle', mode='rb') as f:
    quiz3_l_dic = pickle.load(f)
    print(quiz3_l_dic)
# %%
with open('./pickle/quiz4_l_dic.pickle', mode='rb') as f:
    quiz4_l_dic = pickle.load(f)
    print(quiz4_l_dic)
# %%
with open('./pickle/quiz5_l_dic.pickle', mode='rb') as f:
    quiz5_l_dic = pickle.load(f)
    print(quiz5_l_dic)
# %%
with open('./pickle/quiz6_l_dic.pickle', mode='rb') as f:
    quiz6_l_dic = pickle.load(f)
    print(quiz6_l_dic)
# %%
with open('./pickle/quiz7_l_dic.pickle', mode='rb') as f:
    quiz7_l_dic = pickle.load(f)
    print(quiz7_l_dic)    
    
    



# %%
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
# df_readへのいろいろな変更
df_read['day'] = df_read['operation_date'].str[:10]
df_read['operation_date'] = pd.to_datetime(df_read['operation_date'])
df_read['user_contents'] = df_read['userid'].astype(str) + '_' + df_read['contents_id']
df_read['user_contents_date'] = df_read['user_contents'] + '_' + df_read['day']
df_read['operation_minutes'] = np.nan
for i in range(len(df_read)):
    df_read['operation_minutes'][i] = "{0:%Y-%m-%d %H:%M}".format(df_read['operation_date'][i])
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
contents_list = ['d1f255a373a3cef72e03aa9d980c7eca', '877a9ba7a98f75b90a9d49f53f15a858', '7eacb532570ff6858afd2723755ff790', 'b6f0479ae87d244975439c6124592772', 'e0c641195b27425bb056ac56f8953d24', 'f85454e8279be180185cac7d243c5eb3', 'faa9afea49ef2ff029a833cccc778fd0', '3c7781a36bcd6cf08c11a970fbe0e2a6', '25b2822c2f5a3230abfadd476e8b04c9']
start_time_list = [start_time_1, start_time_2, start_time_3, start_time_4, start_time_5, start_time_6, start_time_7]
test_time_list = [test_time_1, test_time_2, test_time_3, test_time_4, test_time_5, test_time_6, test_time_7]
end_time_list = [end_time_1, end_time_2, end_time_3, end_time_4, end_time_5, end_time_6, end_time_7]
start_leclist = [start_time_1, start_time_2, start_time_2, start_time_3, start_time_4, start_time_5, start_time_6, start_time_7]
test_leclist =[test_time_1, test_time_1, test_time_2, test_time_2, test_time_3, test_time_4, test_time_5, test_time_6, test_time_7]
end_leclist = [end_time_1, end_time_2, end_time_2, end_time_3, end_time_4, end_time_5, end_time_6, end_time_7]

# %%
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
# %%
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
operation_name_list = df_read['operation_name'].unique()



# %%
# 使ったLightBGMを関数にしておく
# 基本は先生のコード,ところどころhttps://rightcode.co.jp/blog/information-technology/lightgbm-useful-for-kaggler
def LightGBM(df_youso):
    df_youso = df_youso.fillna(0)
    # 訓練データとテストデータに分ける
    train_set, test_set = train_test_split(df_youso, test_size = 0.2, random_state = 30)
    # 説明変数と目的変数に分ける
    x_train = train_set.drop('r_or_w', axis = 1)
    y_train = train_set['r_or_w']
    x_test = test_set.drop('r_or_w', axis = 1)
    y_test = test_set['r_or_w']
    
    x_tr = x_train.drop('userid', axis = 1)
    x_tt = x_test.drop('userid', axis = 1)
    
    y_tr = y_train.values.flatten().astype(np.float32)
    y_tt = y_test.values.flatten().astype(np.float32)
    X_tr = x_tr.values.astype(np.float32)
    X_tt = x_tt.values.astype(np.float32)
    # feature_dim = x_tr.shape[1]

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
        # "lambda_l1": 1e-5,
        # "lambda_l2": 1e-5,
        "feature_pre_filter": False,
        "saved_feature_importance_type": 1,
        "seed": 0
        }
    # LightGBM用のデータセットに入れる
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    # dtest = lgb.Dataset(X_tt, y_tt, reference=dtrain)

    # 訓練データから回帰モデルを作る
    reg = lgb.train(param, dtrain)
    # lgb.test = lgb.Dataset(x_test, y_test)


    # テストデータを用いて予測精度を確認する
    X_tt = scaler.transform(X_tt)
    y_pred = reg.predict(X_tt)
    y_pred = np.clip(y_pred * yscale_std + yscale_mean, 0, 100)

    # rmseを求める
    rmse = mean_squared_error(y_tt, y_pred, squared=False)
    print(f"RMSE: {rmse}")
    print(f'正解値:{y_tt}')
    print(f'予測値:{y_pred}')

    # importanceを表示する
    importance = pd.DataFrame(reg.feature_importance(), index=x_tr.columns, columns=['importance'])
    display(importance)
    display(importance[importance['importance'] != 0].sort_values('importance', ascending=False))

    attr_impt = reg.feature_importance(importance_type='gain')
    print('重要性')
    print(attr_impt)

    lgb.plot_importance(reg, height = 0.5, figsize = (8,16))
    # print(df_youso.columns)
    
    # 正方形にしような
    max_value = y_test.max()
    plt.figure(figsize=(8,8))
    x_line = np.linspace(0, max_value+5, 100)
    plt.plot(x_line, x_line, color = "red")
    plt.xlabel('y_tt')
    plt.ylabel('y_pred')
    plt.scatter(y_tt, y_pred)





# %%
# 各週の小テストの点数（timecreatedでgroupbyしたおかげで2週目が分かれてしもた）
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
# df_opeuserの中で無視していい列を消しておく
df_opeu_t = df_opeu_t.drop(columns=['TIMER_STOP_count', 'TIMER_PAUSE_count', 'MEMO_TEXT_CHANGE_HISTORY_count'])
df_opeu_t



# %%
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
# 1週にしぼる
df_cid_w1 = df_youso_cid[(df_youso_cid['contents_id'] == contents_list[0]) | (df_youso_cid['contents_id'] == contents_list[1])].reset_index(drop=True)
df_cid_w1

# %%
# dfの枠作成
df_pvec = pd.DataFrame({'cid_page':cid_list[:385]})
df_pvec['page_vec'] = np.NaN
df_pvec['page_vec'] = df_pvec['page_vec'].astype('O')
df_pvec



# %%
# dfの枠作成
df_npvec = pd.DataFrame({'cid_page':cid_list[:385]})
df_npvec['page_vec'] = np.NaN
df_npvec['page_vec'] = df_npvec['page_vec'].astype('O')
df_npvec


# %%
for i in range(len(df_pvec)):
    p = int(df_pvec['cid_page'][i][-3:])
    conname = df_pvec['cid_page'][i][:-4]
    if conname == 'd1f255a373a3cef72e03aa9d980c7eca':
        context = ' '.join(contents0_dic[p]) 
        context = context.replace('•', ' ')
        vect = text2vector(context)
    elif conname == '877a9ba7a98f75b90a9d49f53f15a858':
        context = ' '.join(contentsplus_dic[p]) 
        context = context.replace('•', ' ')
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
    df_pvec.iat[i, 1] = vect
df_pvec

# %%
for i in range(len(df_npvec)):
    v = df_pvec['page_vec'][i][0]
    normalized_v = v/np.linalg.norm(v)
    df_npvec.iat[i, 1] = normalized_v
df_npvec


# %%
# やべえ時間かかるので必要になるまで回さない方がいい
# df_pagevec = df_npvec.copy()
# for i in range(len(df_pagevec)):
#     for j in range(len(df_pagevec['page_vec'][i][0])):
#         if i == 0:
#             df_pagevec['vec_'+str(j)] = np.nan
#         df_pagevec['vec_'+str(j)][i] = df_pagevec['page_vec'][i][0][j].astype(np.float64)
# df_pagevec
# %%
condic_list = [contents0_dic, contentsplus_dic, contents1_dic, contents2_dic, contents3_dic, contents4_dic, contents5_dic, contents6_dic, contents7_dic]


# %%
# コサイン類似度の実装
def cos_similarity(x, y, eps=1e-8):
    # コサイン類似度を計算:式(2.1)
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)

# こっちでも可？
# 正直使っていない
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# %%
df_quiz1_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec'])
df_quiz2_1_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec'])
df_quiz2_2_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec'])
df_quiz3_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec'])
df_quiz4_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec'])
df_quiz5_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec'])
df_quiz6_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec'])
df_quiz7_l_vec = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['quiz_vec'])

# %%
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

# %%
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
            x = df_x['quiz_vec'][i+1][0]
            y = df_wvec['page_vec'][j]
            cosim = cos_similarity(x, y)
            if i == 0:
                df_cosim_l = df_cosim_l.append({'cid_page': df_wvec['cid_page'][j], 'cos_sim1': cosim}, ignore_index=True)
            else:
                df_cosim_l['cos_sim'+str(i+1)][j] = cosim
    cosimd_l_list[k] = df_cosim_l
    display(df_cosim_l)



# %%
# 週ごとのテスト内容（答え付き）とページとのコサイン類似度
for i in range(len(cosimd_l_list)):
    print(i)
    df_cosim_l = cosimd_l_list[i].copy()
    df_cosim_l = df_cosim_l.set_index('cid_page')
    plt.figure()
    plt.subplots(figsize=(15, 15))
    sns.heatmap(df_cosim_l.fillna(0), annot=True, cmap='GnBu')
    plt.show()



# %%
# 各ページのコサイン類似度
# 正直めっちゃ書き方依存な気がする
for i in range(len(contents_list)):
    df_con = df_npvec[df_npvec['contents_id'] == contents_list[i]].reset_index(drop=False)
    df_convec = pd.DataFrame(index=list(df_con['cid_page']), columns=list(df_con['cid_page']))
    for j in range(len(df_convec)):
        for k in range(len(df_convec)):
            x = df_con['page_vec'][j][0]
            y = df_con['page_vec'][k][0]
            cosim = cos_similarity(x, y)
            df_convec.iat[j,k] = cosim
    plt.figure()
    plt.subplots(figsize=(15, 15))
    sns.heatmap(df_convec.fillna(0), annot=True, cmap='GnBu')
    plt.show()    
            
            
            

# %%
# ページのベクトルが今大変なことになっているのですが、圧縮する方法とは
# 元のベクトル生成する時にもう次元削減しちゃあかんのかな
# あんまり詳しくなくなっちゃうかな
# でもなんか主成分分析で削減とかはまた話が変わってくるのでねえか
# 主成分分析で減らしていいらしい





# %%
# ★方針１

# %%
# ★★学生ごとによく見ているページほど重要度が高くする
# 何秒から何秒の間は重要度0.7とか何秒から何秒の間は0.3とかってするか
# ↑多分各学生ごとで違いでないからいいと思うんやけどどうなんやろ
# 学生ごとで秒数に重要度つけてもたら全然みてない学生どうなるんやって感じになるから
# ただ私の独断と偏見で重要度決めてしまってもいいのか感はある
# とりあえず一回に見ている時間1000秒（17分）以下でやってみた、正直長い気がする
df_po = df_page[df_page['open_time'] <= 1020].reset_index(drop=False)
df_po
# %%
# 合計10分以上ならよく見ている（重要度0.9）
# 合計7分以上10分未満ならまあよく見ている（重要度0.7）
# 合計5分以上7分未満ならまあ見ている（重要度0.5）
# 合計3分以上5分未満なら微妙に見ている（重要度0.3）
# 合計1分以上3分未満なら微妙に見ている（重要度0.1）
# 合計0分なら見てない（重要度0.0）
# とか？
df_pot = df_po.groupby(['userid', 'cid_page'], as_index=False)['open_time'].sum().sort_values('open_time')
df_pot['weit_p'] = 0.0
for i in range(len(df_pot)):
    if df_pot['open_time'][i] >= 600:
        df_pot['weit_p'][i] = 0.9
    elif df_pot['open_time'][i] >= 420:
        df_pot['weit_p'][i] = 0.7
    elif df_pot['open_time'][i] >= 300:
        df_pot['weit_p'][i] = 0.5
    elif df_pot['open_time'][i] >= 180:
        df_pot['weit_p'][i] = 0.3
    elif df_pot['open_time'][i] >= 60:
        df_pot['weit_p'][i] = 0.1
df_pot


# %%
df_povet = pd.merge(df_pot, df_npvec, on='cid_page')
df_povet
# %%
df_povet['weit_vec'] = df_povet['weit_p'] * df_povet['page_vec']
df_povet


# %%
df_povetf = df_povet.copy()
df_povetf = df_povetf[['userid', 'cid_page', 'contents_id', 'weit_vec']]
df_povetf
# %%
df_wsumf = df_povetf.groupby(['userid', 'contents_id'], as_index=False)['weit_vec'].sum()
df_wsumf



# %%
# 週に分ける
df_wsumf1 = df_wsumf.copy()
df_wsumf1 = df_wsumf1[(df_wsumf1['contents_id'] == contents_list[0]) | (df_wsumf1['contents_id'] == contents_list[1])]
df_wsumf1 = df_wsumf1.groupby('userid', as_index=False)['weit_vec'].sum()
for i in range(len(df_wsumf1)):
    for j in range(len(df_wsumf1['weit_vec'][i])):
        if i == 0:
            df_wsumf1['vec_'+str(j)] = np.nan
        df_wsumf1['vec_'+str(j)][i] = df_wsumf1['weit_vec'][i][j].astype(np.float64)
df_wsumf1 = df_wsumf1.fillna(0)
df_wsumf1
# %%
df_cid_w1_po = df_youso_cid[(df_youso_cid['contents_id'] == contents_list[0]) | (df_youso_cid['contents_id'] == contents_list[1])].reset_index(drop=True)
df_cid_w1_po
# %%
# df_povet2 = df_povet.copy()
# df_povet2 = df_povet2.drop(columns=['open_time', 'weit_p', 'page_vec'])
# df_cid_w1_po = pd.merge(df_cid_w1_po, df_povet2, on=['userid', 'cid_page', 'contents_id'], how='left')
# df_cid_w1_po
# %%
con_list_t = ['OPEN_count', 'NEXT_count', 'PREV_count', 'CLOSE_count', 'PAGE_JUMP_count', 'GETIT_count', 'OPEN_RECOMMENDATION_count', 'CLOSE_RECOMMENDATION_count', 'NOTGETIT_count', 'ADD MARKER_count', 'DELETE MARKER_count', 'CLICK_RECOMMENDATION_count', 'open_time_count']
df_w1_po = pd.DataFrame({'userid':[i for i in range(100)]})
for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1_po[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
df_w1_po
# %%
for i in range(len(df_w1_po)):
    df_w1_po_user = df_cid_w1_po[df_cid_w1_po['userid'] == i].reset_index()
    df_w1_po_user = df_w1_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    df_w1_po_user = df_w1_po_user.fillna(0)
    for j in range(len(df_w1_po_user)):
        for k in range(len(df_w1_po_user.columns)):
            df_w1_po.iat[i, (len(df_w1_po_user.columns)*j)+k+1] = df_w1_po_user.iat[j, k]
df_w1_po


# %%
df_wsumf1 = df_wsumf1.drop(['weit_vec'], axis=1)
df_w1ypo = pd.merge(df_w1_po, df_wsumf1, on='userid')
df_w1qpo = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1qpo = df_w1qpo.groupby('userid', as_index=False)['r_or_w'].sum()
df_w1ypo = pd.merge(df_w1ypo, df_w1qpo, on='userid', how='left')
df_w1ypo


# %%
LightGBM(df_w1ypo)



# %%
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_wsumw = df_wsumf.copy()
    df_wsum_w = df_wsumw[df_wsumw['contents_id'] == contents_list[l]]
    df_wsum_w = df_wsum_w.groupby('userid', as_index=False)['weit_vec'].sum()
    for i in range(len(df_wsum_w)):
        for j in range(len(df_wsum_w['weit_vec'][i])):
            if i == 0:
                df_wsum_w['vec_'+str(j)] = np.nan
            df_wsum_w['vec_'+str(j)][i] = df_wsum_w['weit_vec'][i][j].astype(np.float64)
    df_wsum_w = df_wsum_w.fillna(0)

    df_cid_w_po = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_w_po = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w_po[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_w_po)):
        df_w_po_user = df_cid_w_po[df_cid_w_po['userid'] == i].reset_index()
        df_w_po_user = df_w_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_po_user = df_w_po_user.fillna(0)
        for j in range(len(df_w_po_user)):
            for k in range(len(df_w_po_user.columns)):
                df_w_po.iat[i, (len(df_w_po_user.columns)*j)+k+1] = df_w_po_user.iat[j, k]
    display(df_w_po)

    df_wsum_w = df_wsum_w.drop(['weit_vec'], axis=1)
    df_wy = pd.merge(df_w_po, df_wsum_w, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    df_wy

    LightGBM(df_wy)





# %%
# 重要度を連続的にする
df_pot_s = df_po.groupby(['userid', 'cid_page'], as_index=False)['open_time'].sum().sort_values('open_time')
df_pot_s['weit_p'] = 0.0
for i in range(len(df_pot_s)):
    bweit = 1 / max(df_pot_s['open_time'])
    weit = bweit * df_pot_s['open_time'][i]
    df_pot_s['weit_p'][i] = weit
df_pot_s
# %%
df_povet_s = pd.merge(df_pot_s, df_npvec, on='cid_page')
df_povet_s
# %%
df_povet_s['weit_vec'] = df_povet_s['weit_p'] * df_povet_s['page_vec']
df_povet_s


# %%
df_povets = df_povet_s.copy()
df_povets = df_povets[['userid', 'cid_page', 'contents_id', 'weit_vec']]
df_povets
# %%
df_wsum = df_povets.groupby(['userid', 'contents_id'], as_index=False)['weit_vec'].sum()
df_wsum




# %%
# ここから週にわけていく
# 1週に絞ってベクトルをひろげている
df_wsum1 = df_wsum.copy()
df_wsum1 = df_wsum1[(df_wsum1['contents_id'] == contents_list[0]) | (df_wsum1['contents_id'] == contents_list[1])]
df_wsum1 = df_wsum1.groupby('userid', as_index=False)['weit_vec'].sum()
for i in range(len(df_wsum1)):
    for j in range(len(df_wsum1['weit_vec'][i])):
        if i == 0:
            df_wsum1['vec_'+str(j)] = np.nan
        df_wsum1['vec_'+str(j)][i] = df_wsum1['weit_vec'][i][j].astype(np.float64)
df_wsum1 = df_wsum1.fillna(0)
df_wsum1
# %%
df_cid_w1_s = df_youso_cid[(df_youso_cid['contents_id'] == contents_list[0]) | (df_youso_cid['contents_id'] == contents_list[1])].reset_index(drop=True)
df_cid_w1_s
# %%
df_w1_s = pd.DataFrame({'userid':[i for i in range(100)]})
for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1_s[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
df_w1_s
# %%
for i in range(len(df_w1_s)):
    df_w1_po_user = df_cid_w1_s[df_cid_w1_s['userid'] == i].reset_index()
    df_w1_po_user = df_w1_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    df_w1_po_user = df_w1_po_user.fillna(0)
    for j in range(len(df_w1_po_user)):
        for k in range(len(df_w1_po_user.columns)):
            df_w1_s.iat[i, (len(df_w1_po_user.columns)*j)+k+1] = df_w1_po_user.iat[j, k]
df_w1_s


# %%
df_wsum1 = df_wsum1.drop(['weit_vec'], axis=1)
df_w1y = pd.merge(df_w1_s, df_wsum1, on='userid')
df_w1q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1q = df_w1q.groupby('userid', as_index=False)['r_or_w'].sum()
df_w1y = pd.merge(df_w1y, df_w1q, on='userid', how='left')
df_w1y


# %%
LightGBM(df_w1y)






# %%
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_wsumw = df_wsum.copy()
    df_wsum_w = df_wsumw[df_wsumw['contents_id'] == contents_list[l]]
    df_wsum_w = df_wsum_w.groupby('userid', as_index=False)['weit_vec'].sum()
    for i in range(len(df_wsum_w)):
        for j in range(len(df_wsum_w['weit_vec'][i])):
            if i == 0:
                df_wsum_w['vec_'+str(j)] = np.nan
            df_wsum_w['vec_'+str(j)][i] = df_wsum_w['weit_vec'][i][j].astype(np.float64)
    df_wsum_w = df_wsum_w.fillna(0)

    df_cid_w_s = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_w_s = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w_s[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_w_s)):
        df_w_po_user = df_cid_w_s[df_cid_w_s['userid'] == i].reset_index()
        df_w_po_user = df_w_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_po_user = df_w_po_user.fillna(0)
        for j in range(len(df_w_po_user)):
            for k in range(len(df_w_po_user.columns)):
                df_w_s.iat[i, (len(df_w_po_user.columns)*j)+k+1] = df_w_po_user.iat[j, k]
    display(df_w_s)

    df_wsum_w = df_wsum_w.drop(['weit_vec'], axis=1)
    df_wy = pd.merge(df_w_s, df_wsum_w, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    df_wy

    LightGBM(df_wy)








# %%
# 講義内時間（前後1時間含む）に絞ろうの回
# 各学生の各ページをみている時間をだしてるよ
df_lec_p = df_page[((df_page['operation_date'] >= st_1) & (df_page['operation_date'] <= et_1)) | ((df_page['operation_date'] >= st_2) & (df_page['operation_date'] <= et_2)) | ((df_page['operation_date'] >= st_3) & (df_page['operation_date'] <= et_3)) | ((df_page['operation_date'] >= st_4) & (df_page['operation_date'] <= et_4)) | ((df_page['operation_date'] >= st_5) & (df_page['operation_date'] <= et_5)) | ((df_page['operation_date'] >= st_6) & (df_page['operation_date'] <= et_6)) | ((df_page['operation_date'] >= st_7) & (df_page['operation_date'] <= et_7))].reset_index(drop=True)
df_inlec_p = df_lec_p[df_lec_p['open_time'] <= 1020].reset_index(drop=True)
df_inlec_p
# %%
df_lec = df_read[((df_read['operation_date'] >= st_1) & (df_read['operation_date'] <= et_1)) | ((df_read['operation_date'] >= st_2) & (df_read['operation_date'] <= et_2)) | ((df_read['operation_date'] >= st_3) & (df_read['operation_date'] <= et_3)) | ((df_read['operation_date'] >= st_4) & (df_read['operation_date'] <= et_4)) | ((df_read['operation_date'] >= st_5) & (df_read['operation_date'] <= et_5)) | ((df_read['operation_date'] >= st_6) & (df_read['operation_date'] <= et_6)) | ((df_read['operation_date'] >= st_7) & (df_read['operation_date'] <= et_7))].reset_index(drop=True)
df_lec
# %%
df_open_in = df_inlec_p.groupby(['userid', 'contents_id', 'cid_page'], as_index=False)['open_time'].sum()
df_open_in
# %%
df_open_in['weit_p'] = 0.0
for i in range(len(df_open_in)):
    bweit = 1 / max(df_open_in['open_time'])
    weit = bweit * df_open_in['open_time'][i]
    df_open_in['weit_p'][i] = weit
df_open_in
# %%
df_invec = pd.merge(df_open_in, df_npvec, on=['contents_id', 'cid_page'])
df_invec['weit_vec'] = df_invec['weit_p'] * df_invec['page_vec']
df_invec


# %%
df_invecs = df_invec.copy()
df_invecs = df_invecs[['userid', 'cid_page', 'contents_id', 'weit_vec']]
df_invecs
# %%
df_wsum_in = df_invecs.groupby(['userid', 'contents_id'], as_index=False)['weit_vec'].sum()
df_wsum_in




# %%
# ここから週にわけていく
df_wsum1_in = df_wsum_in.copy()
df_wsum1_in = df_wsum1_in[(df_wsum1_in['contents_id'] == contents_list[0]) | (df_wsum1_in['contents_id'] == contents_list[1])]
df_wsum1_in = df_wsum1_in.groupby('userid', as_index=False)['weit_vec'].sum()
for i in range(len(df_wsum1_in)):
    for j in range(len(df_wsum1_in['weit_vec'][i])):
        if i == 0:
            df_wsum1_in['vec_'+str(j)] = np.nan
        df_wsum1_in['vec_'+str(j)][i] = df_wsum1_in['weit_vec'][i][j].astype(np.float64)
df_wsum1_in = df_wsum1_in.fillna(0)
df_wsum1_in
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
df_youso_in

# %%
df_in_w1 = df_youso_in[(df_youso_in['contents_id'] == contents_list[0]) | (df_youso_in['contents_id'] == contents_list[1])]
df_in_w1

# %%
df_w1_in = pd.DataFrame({'userid':[i for i in range(100)]})
for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1_in[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
df_w1_in
# %%
for i in range(len(df_w1_in)):
    df_w1_user = df_in_w1[df_in_w1['userid'] == i].reset_index()
    df_w1_user = df_w1_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count', 'weit_p'], axis=1)
    df_w1_user = df_w1_user.fillna(0)
    for j in range(len(df_w1_user)):
        for k in range(len(df_w1_user.columns)):
            df_w1_in.iat[i, (len(df_w1_user.columns)*j)+k+1] = df_w1_user.iat[j, k]
df_w1_in
# %%
df_wsum1_in = df_wsum1_in.drop(['weit_vec'], axis=1)
df_w1y_in = pd.merge(df_w1_in, df_wsum1_in, on='userid')
df_w1_q_in = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1_youso_in = pd.merge(df_w1y_in, df_w1_q_in, on='userid', how='left')
df_w1_youso_in = df_w1_youso_in.drop(['timecreated', 'respons_day'], axis=1)
df_w1_youso_in
# %%
# あってるんか？？
LightGBM(df_w1_youso_in)





# %%
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_wsumw = df_wsum_in.copy()
    df_wsum_w = df_wsumw[df_wsumw['contents_id'] == contents_list[l]]
    df_wsum_w = df_wsum_w.groupby('userid', as_index=False)['weit_vec'].sum()
    for i in range(len(df_wsum_w)):
        for j in range(len(df_wsum_w['weit_vec'][i])):
            if i == 0:
                df_wsum_w['vec_'+str(j)] = np.nan
            df_wsum_w['vec_'+str(j)][i] = df_wsum_w['weit_vec'][i][j].astype(np.float64)
    df_wsum_w = df_wsum_w.fillna(0)

    df_cid_w_s = df_youso_in[df_youso_in['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_w_s = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w_s[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_w_s)):
        df_w_po_user = df_cid_w_s[df_cid_w_s['userid'] == i].reset_index()
        df_w_po_user = df_w_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count', 'weit_p'], axis=1)
        df_w_po_user = df_w_po_user.fillna(0)
        for j in range(len(df_w_po_user)):
            for k in range(len(df_w_po_user.columns)):
                df_w_s.iat[i, (len(df_w_po_user.columns)*j)+k+1] = df_w_po_user.iat[j, k]
    display(df_w_s)

    df_wsum_w = df_wsum_w.drop(['weit_vec'], axis=1)
    df_wy = pd.merge(df_w_s, df_wsum_w, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    df_wy

    LightGBM(df_wy)





# %%
# ★★ページごとで重要なページほど重要度を高くする
# 問題とのコサイン類似度の合計値を重みとしてページベクトルにかけてそれを足す
# 生徒一人一人が別のベクトルを持つわけではない
# （各ページと小テストの内容をベクトル化したもの横方向に全部足す）？
cosimd_r_list = []
for i in range(len(cosimd_l_list)):
    cosimd_l_l = cosimd_l_list.copy()
    df_cosim = cosimd_l_l[i]
    df_cosim['cos_sum'] = 0
    df_cosim['cos_sum'] = df_cosim['cos_sim1'] + df_cosim['cos_sim2'] + df_cosim['cos_sim3'] + df_cosim['cos_sim4'] + df_cosim['cos_sim5']
    cosimd_r_list.append(df_cosim)
cosimd_r_list
# %%
cosimd_r_list[0]

# %%
# すでに週に分かれてる
df_cosimr = cosimd_r_list[0]
df_vecosim1 = pd.merge(df_cosimr, df_npvec, on='cid_page', how='left')
df_vecosim1
# %%
df_vecosim1['weit_vec'] = df_vecosim1['cos_sum'] * df_vecosim1['page_vec']
df_vecosim1


# %%
df_vecosim1 = df_vecosim1[['cid_page', 'contents_id', 'weit_vec']]
df_vecosim1

# %%
df_vecosim1_n = pd.DataFrame(index=[0], columns=['weit_vec'])
df_vecosim1_n['weit_vec'][0] = df_vecosim1['weit_vec'].sum()
df_vecosim1_n
# %%
for j in range(len(df_vecosim1_n['weit_vec'][0])):
    df_vecosim1_n['vec_'+str(j)] = np.nan
    df_vecosim1_n['vec_'+str(j)] = df_vecosim1_n['weit_vec'][0][j].astype(np.float64)
df_vecsum1 = df_vecosim1_n.fillna(0)
df_vecsum1
# %%
df_cidcos1 = df_youso_cid[(df_youso_cid['contents_id'] == contents_list[0]) | (df_youso_cid['contents_id'] == contents_list[1])].reset_index(drop=True)
df_cidcos1
# %%
df_cosw1 = pd.DataFrame({'userid':[i for i in range(100)]})
for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_cosw1[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
df_cosw1
# %%
for i in range(len(df_cosw1)):
    df_w1_user = df_cidcos1[df_cidcos1['userid'] == i].reset_index()
    df_w1_user = df_w1_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    df_w1_user = df_w1_user.fillna(0)
    for j in range(len(df_w1_user)):
        for k in range(len(df_w1_user.columns)):
            df_cosw1.iat[i, (len(df_w1_user.columns)*j)+k+1] = df_w1_user.iat[j, k]
df_cosw1


# %%
df_vecsum1 = df_vecsum1.drop(['weit_vec'], axis=1)
for i in range(len(df_vecsum1.columns)):
    df_cosw1[df_vecsum1.columns[i]] = df_vecsum1[df_vecsum1.columns[i]][0]
df_w1qcos = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1qcos = df_w1qcos.groupby('userid', as_index=False)['r_or_w'].sum()
df_w1ycos = pd.merge(df_cosw1, df_w1qcos, on='userid', how='left')
df_w1ycos


# %%
LightGBM(df_w1ycos)






# %%
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_cosimr = cosimd_r_list[l-1]
    df_vecosim = pd.merge(df_cosimr, df_npvec, on='cid_page', how='left')
    df_vecosim['weit_vec'] = df_vecosim['cos_sum'] * df_vecosim['page_vec']

    df_vecosim_n = pd.DataFrame(index=[0], columns=['weit_vec'])
    df_vecosim_n['weit_vec'][0] = df_vecosim['weit_vec'].sum()

    for j in range(len(df_vecosim_n['weit_vec'][0])):
        df_vecosim_n['vec_'+str(j)] = np.nan
        df_vecosim_n['vec_'+str(j)] = df_vecosim_n['weit_vec'][0][j].astype(np.float64)
    df_vecsum = df_vecosim1_n.fillna(0)

    df_cidcos = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_cosw = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_cosw[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_cosw)):
        df_w_user = df_cidcos[df_cidcos['userid'] == i].reset_index()
        df_w_user = df_w_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_user = df_w_user.fillna(0)
        for j in range(len(df_w_user)):
            for k in range(len(df_w_user.columns)):
                df_cosw.iat[i, (len(df_w_user.columns)*j)+k+1] = df_w_user.iat[j, k]
    
    df_vecsum = df_vecsum.drop(['weit_vec'], axis=1)
    for i in range(len(df_vecsum.columns)):
        df_cosw[df_vecsum.columns[i]] = df_vecsum[df_vecsum.columns[i]][0]
    df_wqcos = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wqcos = df_wqcos.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wycos = pd.merge(df_cosw, df_wqcos, on='userid', how='left')
    display(df_wycos)


    LightGBM(df_wycos)







# %%
# ★方針２
# ★★ページごとのコサイン類似度を足してそれを重みとして各要素に掛け合わせている


# %%
df_cid_w1_r = df_youso_cid[(df_youso_cid['contents_id'] == contents_list[0]) | (df_youso_cid['contents_id'] == contents_list[1])]
df_cid_w1_r
# %%
con_list_t = ['OPEN_count', 'NEXT_count', 'PREV_count', 'CLOSE_count', 'PAGE_JUMP_count', 'GETIT_count', 'OPEN_RECOMMENDATION_count', 'CLOSE_RECOMMENDATION_count', 'NOTGETIT_count', 'ADD MARKER_count', 'DELETE MARKER_count', 'CLICK_RECOMMENDATION_count', 'open_time_count']
df_w1_r = pd.DataFrame({'userid':[i for i in range(100)]})
for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1_r[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
for i in range(len(df_w1_r)):
    df_w1_r_user = df_cid_w1_r[df_cid_w1_r['userid'] == i].reset_index()
    df_w1_r_user = df_w1_r_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    df_w1_r_user = df_w1_r_user.fillna(0)
    for j in range(len(df_w1_r_user)):
        for k in range(len(df_w1_r_user.columns)):
            df_w1_r.iat[i, (len(df_w1_r_user.columns)*j)+k+1] = df_w1_r_user.iat[j, k]
df_w1_r

# %%
w = 0
cnt = 0
for i in range(1, len(df_w1_r.columns)):
    weit = cosimd_r_list[0]['cos_sum'][w]
    df_w1_r[df_w1_r.columns[i]] = df_w1_r[df_w1_r.columns[i]] * weit
    cnt += 1
    if cnt == len(con_list_t):
        w += 1
        cnt = 0
df_w1_r
# %%
df_w1_r_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1_r_q = df_w1_r_q.groupby('userid', as_index=False)['r_or_w'].sum()
df_w1_r_youso = pd.merge(df_w1_r, df_w1_r_q, on='userid', how='left')
df_w1_r_youso
# %%
LightGBM(df_w1_r_youso)




# %%
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
    
    w = 0
    cnt = 0
    for i in range(1, len(df_w.columns)):
        weit = cosimd_r_list[l-1]['cos_sum'][w]
        df_w[df_w.columns[i]] = df_w[df_w.columns[i]] * weit
        cnt += 1
        if cnt == len(con_list_t):
            w += 1
            cnt = 0
    display(df_w)

    df_w_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_w_youso = pd.merge(df_w, df_w_q, on='userid', how='left')
    df_w_youso = df_w_youso.drop(['timecreated', 'respons_day'], axis=1)
    display(df_w_youso)

    # シード値固定してる
    print(LightGBM(df_w_youso))




# %%
# ★★ページごとのコサイン類似度を足してそれを重みとして各要素に掛け合わせている
# 閾値処理して0.4(0.5)以下は消す
# 正直消していいと思うけど、閾値ってどうする？
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
        df_cosim.iat[j, 7] = cosum.astype(np.float64)
    cosimd_z_list.append(df_cosim)
cosimd_z_list


# %%
cosimd_z_list[1]


# %%
df_p2 = cosimd_z_list[0]
for i in range(1, len(cosimd_z_list)):
    df_p2 = pd.concat([df_p2, cosimd_z_list[i]], axis=0, ignore_index=True)
df_p2

# %%
df_w1_4 = pd.DataFrame({'userid':[i for i in range(100)]})

for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1_4[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
for i in range(len(df_w1_4)):
    df_w1_r_user = df_cid_w1_r[df_cid_w1_r['userid'] == i].reset_index()
    df_w1_r_user = df_w1_r_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    df_w1_r_user = df_w1_r_user.fillna(0)
    for j in range(len(df_w1_r_user)):
        for k in range(len(df_w1_r_user.columns)):
            df_w1_4.iat[i, (len(df_w1_r_user.columns)*j)+k+1] = df_w1_r_user.iat[j, k]
df_w1_4

# %%
w = 0
cnt = 0
df_p2['contents_id'] = 0
for i in range(len(df_p2)):
    df_p2['contents_id'][i] = df_p2['cid_page'][i][:-4]
for i in range(1, len(df_w1_4.columns)):
    df_p2w = df_p2.copy()
    df_p2w = df_p2w[(df_p2w['contents_id'] == contents_list[0]) | (df_p2w['contents_id'] == contents_list[1])]
    weit = df_p2w['cos_sum_4'][w]
    df_w1_4[df_w1_4.columns[i]] = df_w1_4[df_w1_4.columns[i]] * weit
    cnt += 1
    if cnt == len(con_list_t):
        w += 1
        cnt = 0
df_w1_4
# %%
df_w1_4_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1_4_q = df_w1_4_q.groupby('userid', as_index=False)['r_or_w'].sum()
df_w1_4_youso = pd.merge(df_w1_4, df_w1_4_q, on='userid', how='left')
df_w1_4_youso
# %%
LightGBM(df_w1_4_youso)




# %%
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
    
    w = 0
    cnt = 0
    for i in range(1, len(df_w.columns)):
        df_p2w = df_p2.copy()
        df_p2w = df_p2w[df_p2w['contents_id'] == contents_list[l]].reset_index(drop=True)
        weit = df_p2w['cos_sum_4'][w]
        df_w[df_w.columns[i]] = df_w[df_w.columns[i]] * weit
        cnt += 1
        if cnt == len(con_list_t):
            w += 1
            cnt = 0
    display(df_w)

    df_w_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_w_youso = pd.merge(df_w, df_w_q, on='userid', how='left')
    df_w_youso = df_w_youso.drop(['timecreated', 'respons_day'], axis=1)
    display(df_w_youso)

    # シード値固定してる
    print(LightGBM(df_w_youso))





# %%
# 講義時間内にする
df_w1_4in = pd.DataFrame({'userid':[i for i in range(100)]})

for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1_4in[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
for i in range(len(df_w1_4in)):
    df_w1_r_user = df_in_w1[df_in_w1['userid'] == i].reset_index()
    df_w1_r_user = df_w1_r_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count', 'weit_p'], axis=1)
    df_w1_r_user = df_w1_r_user.fillna(0)
    for j in range(len(df_w1_r_user)):
        for k in range(len(df_w1_r_user.columns)):
            df_w1_4in.iat[i, (len(df_w1_r_user.columns)*j)+k+1] = df_w1_r_user.iat[j, k]
df_w1_4in

# %%
w = 0
cnt = 0

for i in range(1, len(df_w1_4in.columns)):
    df_p2w = df_p2.copy()
    df_p2w = df_p2w[(df_p2w['contents_id'] == contents_list[0]) | (df_p2w['contents_id'] == contents_list[1])]
    weit = df_p2w['cos_sum_4'][w]
    df_w1_4in[df_w1_4in.columns[i]] = df_w1_4in[df_w1_4in.columns[i]] * weit
    cnt += 1
    if cnt == len(con_list_t):
        w += 1
        cnt = 0
df_w1_4in
# %%
df_w1_4in_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1_4in_q = df_w1_4in_q.groupby('userid', as_index=False)['r_or_w'].sum()
df_w1_4in_youso = pd.merge(df_w1_4in, df_w1_4in_q, on='userid', how='left')
df_w1_4in_youso
# %%
LightGBM(df_w1_4in_youso)




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
    
    w = 0
    cnt = 0
    for i in range(1, len(df_w.columns)):
        df_p2w = df_p2.copy()
        df_p2w = df_p2w[df_p2w['contents_id'] == contents_list[l]].reset_index(drop=True)
        weit = df_p2w['cos_sum_4'][w]
        df_w[df_w.columns[i]] = df_w[df_w.columns[i]] * weit
        cnt += 1
        if cnt == len(con_list_t):
            w += 1
            cnt = 0
    display(df_w)

    df_w_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_w_youso = pd.merge(df_w, df_w_q, on='userid', how='left')
    df_w_youso = df_w_youso.drop(['timecreated', 'respons_day'], axis=1)
    display(df_w_youso)

    # シード値固定してる
    print(LightGBM(df_w_youso))





# %%
# ★方針３
# ★★行動とページの重要度を使って学生ごとに異なるページベクトルを作成
# （★★行動の重要度及びスライドの重要度によって重みを変える）
# self-attentionとかつかうといいらしい
# ページの重要度なら全学生は同じなので。行動の重要度っつうのは閲覧時間でいいのか










# %%
# 次元削減をしよう

# PCA
model_pca = PCA(n_components=2)
df_wsum1_1 = df_wsum1.drop(['userid'], axis=1)
pcaws1_list = model_pca.fit_transform(df_wsum1_1)
print(model_pca.explained_variance_ratio_)
df_pcaws1 = pd.DataFrame(pcaws1_list)
df_pcaws1['userid'] = list(df_wsum1['userid'])
df_pcaws1



# %%
# t-SNE
# perplexity=[2,5,10,50,100]
model_tsne = TSNE(random_state=0, n_components=2, perplexity=2)
tsnews1_list = model_tsne.fit_transform(df_wsum1_1)
df_tsnews1 = pd.DataFrame(tsnews1_list)
df_tsnews1['userid'] = list(df_wsum1['userid'])
df_tsnews1


# %%
# UMAP
# n_neighbors
model_umap = umap.UMAP(random_state=0, n_components=2, n_neighbors=3) 
umapws1_list = model_umap.fit_transform(df_wsum1_1)
df_umapws1 = pd.DataFrame(umapws1_list)
df_umapws1['userid'] = list(df_wsum1['userid'])
df_umapws1


# %%
df_w1y_p = pd.merge(df_w1_s, df_pcaws1, on='userid')
df_w1y_p = pd.merge(df_w1y_p, df_w1q, on='userid', how='left')
LightGBM(df_w1y_p)


# %%
df_w1y_t = pd.merge(df_w1_s, df_tsnews1, on='userid')
df_w1y_t = pd.merge(df_w1y_t, df_w1q, on='userid', how='left')
LightGBM(df_w1y_t)


# %%
df_w1y_u = pd.merge(df_w1_s, df_umapws1, on='userid')
df_w1y_u = pd.merge(df_w1y_u, df_w1q, on='userid', how='left')
LightGBM(df_w1y_u)


# %%
# PCA
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_wsumw = df_wsum.copy()
    df_wsum_w = df_wsumw[df_wsumw['contents_id'] == contents_list[l]]
    df_wsum_w = df_wsum_w.groupby('userid', as_index=False)['weit_vec'].sum()
    for i in range(len(df_wsum_w)):
        for j in range(len(df_wsum_w['weit_vec'][i])):
            if i == 0:
                df_wsum_w['vec_'+str(j)] = np.nan
            df_wsum_w['vec_'+str(j)][i] = df_wsum_w['weit_vec'][i][j].astype(np.float64)
    df_wsum_w = df_wsum_w.fillna(0)

    df_cid_w_s = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_w_s = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w_s[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_w_s)):
        df_w_po_user = df_cid_w_s[df_cid_w_s['userid'] == i].reset_index()
        df_w_po_user = df_w_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_po_user = df_w_po_user.fillna(0)
        for j in range(len(df_w_po_user)):
            for k in range(len(df_w_po_user.columns)):
                df_w_s.iat[i, (len(df_w_po_user.columns)*j)+k+1] = df_w_po_user.iat[j, k]
    display(df_w_s)

    df_wsum_w = df_wsum_w.drop(['weit_vec'], axis=1)
    model_pca = PCA(n_components=2)
    df_wsumw_1 = df_wsum_w.drop(['userid'], axis=1)
    pcaws_list = model_pca.fit_transform(df_wsumw_1)
    print(model_pca.explained_variance_ratio_)
    df_pcaws = pd.DataFrame(pcaws_list)
    df_pcaws['userid'] = list(df_wsum_w['userid'])

    df_wy = pd.merge(df_w_s, df_pcaws, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    df_wy

    LightGBM(df_wy)


# %%
# t-SNE
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_wsumw = df_wsum.copy()
    df_wsum_w = df_wsumw[df_wsumw['contents_id'] == contents_list[l]]
    df_wsum_w = df_wsum_w.groupby('userid', as_index=False)['weit_vec'].sum()
    for i in range(len(df_wsum_w)):
        for j in range(len(df_wsum_w['weit_vec'][i])):
            if i == 0:
                df_wsum_w['vec_'+str(j)] = np.nan
            df_wsum_w['vec_'+str(j)][i] = df_wsum_w['weit_vec'][i][j].astype(np.float64)
    df_wsum_w = df_wsum_w.fillna(0)

    df_cid_w_s = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_w_s = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w_s[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_w_s)):
        df_w_po_user = df_cid_w_s[df_cid_w_s['userid'] == i].reset_index()
        df_w_po_user = df_w_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_po_user = df_w_po_user.fillna(0)
        for j in range(len(df_w_po_user)):
            for k in range(len(df_w_po_user.columns)):
                df_w_s.iat[i, (len(df_w_po_user.columns)*j)+k+1] = df_w_po_user.iat[j, k]
    display(df_w_s)

    df_wsum_w = df_wsum_w.drop(['weit_vec'], axis=1)
    model_tsne = TSNE(random_state=0, n_components=2, perplexity=2)
    df_wsumw_1 = df_wsum_w.drop(['userid'], axis=1)
    tsnews_list = model_tsne.fit_transform(df_wsumw_1)
    df_tsnews = pd.DataFrame(tsnews_list)
    df_tsnews['userid'] = list(df_wsum_w['userid'])

    df_wy = pd.merge(df_w_s, df_tsnews, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    df_wy

    LightGBM(df_wy)


# %%
# UMAP
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_wsumw = df_wsum.copy()
    df_wsum_w = df_wsumw[df_wsumw['contents_id'] == contents_list[l]]
    df_wsum_w = df_wsum_w.groupby('userid', as_index=False)['weit_vec'].sum()
    for i in range(len(df_wsum_w)):
        for j in range(len(df_wsum_w['weit_vec'][i][0])):
            if i == 0:
                df_wsum_w['vec_'+str(j)] = np.nan
            df_wsum_w['vec_'+str(j)][i] = df_wsum_w['weit_vec'][i][0][j].astype(np.float64)
    df_wsum_w = df_wsum_w.fillna(0)

    df_cid_w_s = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_w_s = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w_s[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_w_s)):
        df_w_po_user = df_cid_w_s[df_cid_w_s['userid'] == i].reset_index()
        df_w_po_user = df_w_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_po_user = df_w_po_user.fillna(0)
        for j in range(len(df_w_po_user)):
            for k in range(len(df_w_po_user.columns)):
                df_w_s.iat[i, (len(df_w_po_user.columns)*j)+k+1] = df_w_po_user.iat[j, k]
    display(df_w_s)

    df_wsum_w = df_wsum_w.drop(['weit_vec'], axis=1)
    model_umap = umap.UMAP(random_state=0, n_components=2, n_neighbors=3) 
    df_wsumw_1 = df_wsum_w.drop(['userid'], axis=1)
    umapws1_list = model_umap.fit_transform(df_wsumw_1)
    df_umapws = pd.DataFrame(umapws1_list)
    df_umapws['userid'] = list(df_wsum_w['userid'])

    df_wy = pd.merge(df_w_s, df_umapws, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    df_wy

    LightGBM(df_wy)






# %%
# なんか普通にsentence-BERTできそう
# Sentence-BERTでやったver.ノート変えようかと思いましたが色々写すの面倒なのでこのままです
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


MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"  # <- v2です。
model = SentenceBertJapanese(MODEL_NAME)

# %%
sentences = ["暴走したAI", "暴走した人工知能"]
sentence_embeddings = model.encode(sentences, batch_size=8)

print("Sentence embeddings:", sentence_embeddings)


# %%
print(sentence_embeddings[0].dtype)
# %%
df_pvec_n = pd.DataFrame({'cid_page':cid_list[:385]})
df_pvec_n['page_vec'] = np.NaN
df_pvec_n['page_vec'] = df_pvec_n['page_vec'].astype('O')
df_pvec_n
# %%
df_npvec_n = pd.DataFrame({'cid_page':cid_list[:385]})
df_npvec_n['page_vec'] = np.NaN
df_npvec_n['page_vec'] = df_npvec_n['page_vec'].astype('O')
df_npvec_n
# %%
contentsdic_list = [contents0_dic, contentsplus_dic, contents1_dic, contents2_dic, contents3_dic, contents4_dic, contents5_dic, contents6_dic, contents7_dic]
page_n = 0
for i in range(len(contentsdic_list)):
    contentsdic_l = copy.copy(contentsdic_list)
    contents_dic = contentsdic_l[i]
    contentstext_list = []
    for j in range(len(contents_dic)):
        context = ' '.join(contents_dic[int(df_pvec_n['cid_page'][j][-3:])])
        context = context.replace('•', ' ')
        contentstext_list.append(context)
    sentence_embeddings = model.encode(contentstext_list, batch_size=8)
    print("Sentence embeddings:", sentence_embeddings)
    for k in range(len(sentence_embeddings)):
        num = page_n + k
        df_pvec_n.iat[num, 1] = sentence_embeddings[k]
    page_n = page_n + len(contentstext_list)
df_pvec_n

# %%
for i in range(len(df_npvec_n)):
    v = df_pvec_n['page_vec'][i]
    normalized_v = v/torch.norm(v)
    df_npvec_n.iat[i, 1] = normalized_v
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
    display(df_qvec_l)

# %%
for i in range(len(qvec_l_l)):
    df_qvec_l = qvec_l_l[i]
    for j in range(1, len(df_qvec_l)):
        v = df_qvec_l['quiz_vec'][j]
        normalized_v = v/torch.norm(v)
        df_qvec_l['quiz_vec'][j] = normalized_v
    display(df_qvec_l)
        
# %%
df_npvec_n['contents_id'] = 0
for i in range(len(df_npvec_n)):
    df_npvec_n['contents_id'][i] = df_npvec_n['cid_page'][i][:-4]
df_npvec_n




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

# %%
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
# ★方針１

# %%
# ★★学生ごとによく見ているページほど重要度が高くする
# 合計10分以上ならよく見ている（重要度0.9）
# 合計7分以上10分未満ならまあよく見ている（重要度0.7）
# 合計5分以上7分未満ならまあ見ている（重要度0.5）
# 合計3分以上5分未満なら微妙に見ている（重要度0.3）
# 合計1分以上3分未満なら微妙に見ている（重要度0.1）
# 合計0分なら見てない（重要度0.0）
# とか？


# %%
df_povet_n = pd.merge(df_pot, df_npvec_n, on='cid_page')
df_povet_n
# %%
df_povet_n['weit_vec'] = df_povet_n['weit_p'] * df_povet_n['page_vec']
df_povet_n


# %%
df_povetf_n = df_povet_n.copy()
df_povetf_n = df_povetf_n[['userid', 'cid_page', 'contents_id', 'weit_vec']]
df_povetf_n
# %%
df_wsumf_n = df_povetf_n.groupby(['userid', 'contents_id'], as_index=False)['weit_vec'].sum()
df_wsumf_n



# %%
# 週に分ける
df_wsumf1_n = df_wsumf_n.copy()
df_wsumf1_n = df_wsumf1_n[(df_wsumf1_n['contents_id'] == contents_list[0]) | (df_wsumf1_n['contents_id'] == contents_list[1])]
df_wsumf1_n = df_wsumf1_n.groupby('userid', as_index=False)['weit_vec'].sum()
for i in range(len(df_wsumf1_n)):
    for j in range(len(df_wsumf1_n['weit_vec'][i])):
        if i == 0:
            df_wsumf1_n['vec_'+str(j)] = np.nan
        df_wsumf1_n['vec_'+str(j)][i] = df_wsumf1_n['weit_vec'][i][j]
df_wsumf1_n = df_wsumf1_n.fillna(0)
df_wsumf1_n
# %%
# df_cid_w1_po = df_youso_cid[(df_youso_cid['contents_id'] == contents_list[0]) | (df_youso_cid['contents_id'] == contents_list[1])].reset_index(drop=True)
# df_cid_w1_po
df_w1_po_n = pd.DataFrame({'userid':[i for i in range(100)]})
for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1_po_n[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
df_w1_po_n
# %%
for i in range(len(df_w1_po_n)):
    df_w1_po_user = df_cid_w1_po[df_cid_w1_po['userid'] == i].reset_index()
    df_w1_po_user = df_w1_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    df_w1_po_user = df_w1_po_user.fillna(0)
    for j in range(len(df_w1_po_user)):
        for k in range(len(df_w1_po_user.columns)):
            df_w1_po_n.iat[i, (len(df_w1_po_user.columns)*j)+k+1] = df_w1_po_user.iat[j, k]
df_w1_po_n


# %%
df_wsumf1_n = df_wsumf1_n.drop(['weit_vec'], axis=1)
df_w1ypo_n = pd.merge(df_w1_po_n, df_wsumf1_n, on='userid')
df_w1qpo_n = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1qpo_n = df_w1qpo_n.groupby('userid', as_index=False)['r_or_w'].sum()
df_w1ypo_n = pd.merge(df_w1ypo_n, df_w1qpo_n, on='userid', how='left')
df_w1ypo_n


# %%
LightGBM(df_w1ypo_n)



# %%
vec = df_npvec_n['page_vec'][1]
norm = 0
for n in vec:
    norm += n ** 2
norm
# %%
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_wsumw = df_wsumf_n.copy()
    df_wsum_w = df_wsumw[df_wsumw['contents_id'] == contents_list[l]]
    df_wsum_w = df_wsum_w.groupby('userid', as_index=False)['weit_vec'].sum()
    for i in range(len(df_wsum_w)):
        for j in range(len(df_wsum_w['weit_vec'][i])):
            if i == 0:
                df_wsum_w['vec_'+str(j)] = np.nan
            df_wsum_w['vec_'+str(j)][i] = df_wsum_w['weit_vec'][i][j]
    df_wsum_w = df_wsum_w.fillna(0)

    df_cid_w_po = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_w_po = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w_po[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_w_po)):
        df_w_po_user = df_cid_w_po[df_cid_w_po['userid'] == i].reset_index()
        df_w_po_user = df_w_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_po_user = df_w_po_user.fillna(0)
        for j in range(len(df_w_po_user)):
            for k in range(len(df_w_po_user.columns)):
                df_w_po.iat[i, (len(df_w_po_user.columns)*j)+k+1] = df_w_po_user.iat[j, k]
    display(df_w_po)

    df_wsum_w = df_wsum_w.drop(['weit_vec'], axis=1)
    df_wy = pd.merge(df_w_po, df_wsum_w, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    df_wy

    LightGBM(df_wy)





# %%
# 重要度を連続的にする
df_povet_sn = pd.merge(df_pot_s, df_npvec_n, on='cid_page')
df_povet_sn
# %%
df_povet_sn['weit_vec'] = df_povet_sn['weit_p'] * df_povet_sn['page_vec']
df_povet_sn


# %%
df_povets_n = df_povet_sn.copy()
df_povets_n = df_povets_n[['userid', 'cid_page', 'contents_id', 'weit_vec']]
df_povets_n
# %%
df_wsum_n = df_povets_n.groupby(['userid', 'contents_id'], as_index=False)['weit_vec'].sum()
df_wsum_n




# %%
# ここから週にわけていく
df_wsum1_n = df_wsum_n.copy()
df_wsum1_n = df_wsum1_n[(df_wsum1_n['contents_id'] == contents_list[0]) | (df_wsum1_n['contents_id'] == contents_list[1])]
df_wsum1_n = df_wsum1_n.groupby('userid', as_index=False)['weit_vec'].sum()
for i in range(len(df_wsum1_n)):
    for j in range(len(df_wsum1_n['weit_vec'][i])):
        if i == 0:
            df_wsum1_n['vec_'+str(j)] = np.nan
        df_wsum1_n['vec_'+str(j)][i] = df_wsum1_n['weit_vec'][i][j]
df_wsum1_n = df_wsum1_n.fillna(0)
df_wsum1_n
# %%
df_w1_sn = pd.DataFrame({'userid':[i for i in range(100)]})
for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1_sn[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
df_w1_sn
# %%
for i in range(len(df_w1_sn)):
    df_w1_po_user = df_cid_w1_s[df_cid_w1_s['userid'] == i].reset_index()
    df_w1_po_user = df_w1_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    df_w1_po_user = df_w1_po_user.fillna(0)
    for j in range(len(df_w1_po_user)):
        for k in range(len(df_w1_po_user.columns)):
            df_w1_sn.iat[i, (len(df_w1_po_user.columns)*j)+k+1] = df_w1_po_user.iat[j, k]
df_w1_sn


# %%
df_wsum1_n = df_wsum1_n.drop(['weit_vec'], axis=1)
df_w1y_n = pd.merge(df_w1_sn, df_wsum1_n, on='userid')
df_w1q_n = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1q_n = df_w1q_n.groupby('userid', as_index=False)['r_or_w'].sum()
df_w1y_n = pd.merge(df_w1y_n, df_w1q_n, on='userid', how='left')
df_w1y_n


# %%
LightGBM(df_w1y_n)






# %%
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_wsumw = df_wsum_n.copy()
    df_wsum_w = df_wsumw[df_wsumw['contents_id'] == contents_list[l]]
    df_wsum_w = df_wsum_w.groupby('userid', as_index=False)['weit_vec'].sum()
    for i in range(len(df_wsum_w)):
        for j in range(len(df_wsum_w['weit_vec'][i])):
            if i == 0:
                df_wsum_w['vec_'+str(j)] = np.nan
            df_wsum_w['vec_'+str(j)][i] = df_wsum_w['weit_vec'][i][j]
    df_wsum_w = df_wsum_w.fillna(0)

    df_cid_w_s = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_w_s = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w_s[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_w_s)):
        df_w_po_user = df_cid_w_s[df_cid_w_s['userid'] == i].reset_index()
        df_w_po_user = df_w_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_po_user = df_w_po_user.fillna(0)
        for j in range(len(df_w_po_user)):
            for k in range(len(df_w_po_user.columns)):
                df_w_s.iat[i, (len(df_w_po_user.columns)*j)+k+1] = df_w_po_user.iat[j, k]
    display(df_w_s)

    df_wsum_w = df_wsum_w.drop(['weit_vec'], axis=1)
    df_wy = pd.merge(df_w_s, df_wsum_w, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    df_wy

    LightGBM(df_wy)








# %%
# ★★ページごとで重要なページほど重要度を高くする
# 問題とのコサイン類似度の合計値を重みとしてページベクトルにかけてそれを足す
# 生徒一人一人が別のベクトルを持つわけではない
# （各ページと小テストの内容をベクトル化したもの横方向に全部足す）？
cosimd_rs_list = []
for i in range(len(cosimd_s_list)):
    cosimd_s_l = cosimd_s_list.copy()
    df_cosim = cosimd_s_l[i]
    df_cosim['cos_sum'] = 0
    df_cosim['cos_sum'] = df_cosim['cos_sim1'] + df_cosim['cos_sim2'] + df_cosim['cos_sim3'] + df_cosim['cos_sim4'] + df_cosim['cos_sim5']
    cosimd_rs_list.append(df_cosim)
cosimd_rs_list
# %%
cosimd_rs_list[0]

# %%
# すでに週に分かれてる
df_cosimr_n = cosimd_rs_list[0]
df_vecosim1_s = pd.merge(df_cosimr_n, df_npvec_n, on='cid_page', how='left')
df_vecosim1_s
# %%
df_vecosim1_s['weit_vec'] = df_vecosim1_s['cos_sum'] * df_vecosim1_s['page_vec']
df_vecosim1_s


# %%
df_vecosim1_s = df_vecosim1_s[['cid_page', 'contents_id', 'weit_vec']]
df_vecosim1_s

# %%
df_vecosim1_ns = pd.DataFrame(index=[0], columns=['weit_vec'])
df_vecosim1_ns['weit_vec'][0] = df_vecosim1_s['weit_vec'].sum()
df_vecosim1_ns
# %%
for j in range(len(df_vecosim1_ns['weit_vec'][0])):
    df_vecosim1_ns['vec_'+str(j)] = np.nan
    df_vecosim1_ns['vec_'+str(j)][0] = df_vecosim1_ns['weit_vec'][0][j]
df_vecsum1_s = df_vecosim1_ns.fillna(0)
df_vecsum1_s
# %%
df_cidcos1_s = df_youso_cid[(df_youso_cid['contents_id'] == contents_list[0]) | (df_youso_cid['contents_id'] == contents_list[1])].reset_index(drop=True)
df_cidcos1_s
# %%
df_cosw1_n = pd.DataFrame({'userid':[i for i in range(100)]})
for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_cosw1_n[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
df_cosw1_n
# %%
for i in range(len(df_cosw1_n)):
    df_w1_user = df_cidcos1[df_cidcos1['userid'] == i].reset_index()
    df_w1_user = df_w1_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    df_w1_user = df_w1_user.fillna(0)
    for j in range(len(df_w1_user)):
        for k in range(len(df_w1_user.columns)):
            df_cosw1_n.iat[i, (len(df_w1_user.columns)*j)+k+1] = df_w1_user.iat[j, k]
df_cosw1_n


# %%
df_vecsum1_s = df_vecsum1_s.drop(['weit_vec'], axis=1)
for i in range(len(df_vecsum1_s.columns)):
    df_cosw1_n[df_vecsum1_s.columns[i]] = df_vecsum1_s[df_vecsum1_s.columns[i]][0]
df_w1qcos_n = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1qcos_n = df_w1qcos_n.groupby('userid', as_index=False)['r_or_w'].sum()
df_w1ycos_n = pd.merge(df_cosw1_n, df_w1qcos_n, on='userid', how='left')
df_w1ycos_n


# %%
LightGBM(df_w1ycos_n)






# %%
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_cosimr = cosimd_rs_list[l-1]
    df_vecosim = pd.merge(df_cosimr, df_npvec_n, on='cid_page', how='left')
    df_vecosim['weit_vec'] = df_vecosim['cos_sum'] * df_vecosim['page_vec']

    df_vecosim_n = pd.DataFrame(index=[0], columns=['weit_vec'])
    df_vecosim_n['weit_vec'][0] = df_vecosim['weit_vec'].sum()

    for j in range(len(df_vecosim_n['weit_vec'][0])):
        df_vecosim_n['vec_'+str(j)] = np.nan
        df_vecosim_n['vec_'+str(j)][0] = df_vecosim_n['weit_vec'][0][j]
    df_vecsum = df_vecosim1_n.fillna(0)

    df_cidcos = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_cosw = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_cosw[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_cosw)):
        df_w_user = df_cidcos[df_cidcos['userid'] == i].reset_index()
        df_w_user = df_w_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_user = df_w_user.fillna(0)
        for j in range(len(df_w_user)):
            for k in range(len(df_w_user.columns)):
                df_cosw.iat[i, (len(df_w_user.columns)*j)+k+1] = df_w_user.iat[j, k]
    
    df_vecsum = df_vecsum.drop(['weit_vec'], axis=1)
    for i in range(len(df_vecsum.columns)):
        df_cosw[df_vecsum.columns[i]] = df_vecsum[df_vecsum.columns[i]][0]
    df_wqcos = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wqcos = df_wqcos.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wycos = pd.merge(df_cosw, df_wqcos, on='userid', how='left')
    display(df_wycos)


    LightGBM(df_wycos)







# %%
# ★方針２
# ★★ページごとのコサイン類似度を足してそれを重みとして各要素に掛け合わせている


# %%
# df_cid_w1_r = df_youso_cid[(df_youso_cid['contents_id'] == contents_list[0]) | (df_youso_cid['contents_id'] == contents_list[1])]
# df_cid_w1_r
# %%
# con_list_t = ['OPEN_count', 'NEXT_count', 'PREV_count', 'CLOSE_count', 'PAGE_JUMP_count', 'GETIT_count', 'OPEN_RECOMMENDATION_count', 'CLOSE_RECOMMENDATION_count', 'NOTGETIT_count', 'ADD MARKER_count', 'DELETE MARKER_count', 'CLICK_RECOMMENDATION_count', 'open_time_count']
df_w1_rs = pd.DataFrame({'userid':[i for i in range(100)]})
for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1_rs[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
for i in range(len(df_w1_rs)):
    df_w1_r_user = df_cid_w1_r[df_cid_w1_r['userid'] == i].reset_index()
    df_w1_r_user = df_w1_r_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    df_w1_r_user = df_w1_r_user.fillna(0)
    for j in range(len(df_w1_r_user)):
        for k in range(len(df_w1_r_user.columns)):
            df_w1_rs.iat[i, (len(df_w1_r_user.columns)*j)+k+1] = df_w1_r_user.iat[j, k]
df_w1_rs

# %%
w = 0
cnt = 0
for i in range(1, len(df_w1_rs.columns)):
    weit = cosimd_rs_list[0]['cos_sum'][w]
    df_w1_rs[df_w1_rs.columns[i]] = df_w1_rs[df_w1_rs.columns[i]] * weit.item()
    cnt += 1
    if cnt == len(con_list_t):
        w += 1
        cnt = 0
df_w1_rs
# %%
df_w1_rs_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1_rs_q = df_w1_rs_q.groupby('userid', as_index=False)['r_or_w'].sum()
df_w1_rs_youso = pd.merge(df_w1_rs, df_w1_rs_q, on='userid', how='left')
df_w1_rs_youso
# %%
LightGBM(df_w1_rs_youso)




# %%
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
    
    w = 0
    cnt = 0
    for i in range(1, len(df_w.columns)):
        weit = cosimd_rs_list[l-1]['cos_sum'][w]
        df_w[df_w.columns[i]] = df_w[df_w.columns[i]] * weit.item()
        cnt += 1
        if cnt == len(con_list_t):
            w += 1
            cnt = 0
    display(df_w)

    df_w_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_w_youso = pd.merge(df_w, df_w_q, on='userid', how='left')
    df_w_youso = df_w_youso.drop(['timecreated', 'respons_day'], axis=1)
    display(df_w_youso)

    # シード値固定してる
    print(LightGBM(df_w_youso))




# %%
# ★★ページごとのコサイン類似度を足してそれを重みとして各要素に掛け合わせている
# 閾値処理して0.4(0.5)以下は消す
# 正直消していいと思うけど、閾値ってどうする？
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
cosimd_zs_list[1]

# %%
df_p2_n = cosimd_zs_list[0]
for i in range(1, len(cosimd_zs_list)):
    df_p2_n = pd.concat([df_p2_n, cosimd_zs_list[i]], axis=0, ignore_index=True)
df_p2_n

# %%
df_w1_4n = pd.DataFrame({'userid':[i for i in range(100)]})

for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1_4n[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
for i in range(len(df_w1_4n)):
    df_w1_r_user = df_cid_w1_r[df_cid_w1_r['userid'] == i].reset_index()
    df_w1_r_user = df_w1_r_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
    df_w1_r_user = df_w1_r_user.fillna(0)
    for j in range(len(df_w1_r_user)):
        for k in range(len(df_w1_r_user.columns)):
            df_w1_4n.iat[i, (len(df_w1_r_user.columns)*j)+k+1] = df_w1_r_user.iat[j, k]
df_w1_4n

# %%
w = 0
cnt = 0
df_p2_n['contents_id'] = 0
for i in range(len(df_p2_n)):
    df_p2_n['contents_id'][i] = df_p2_n['cid_page'][i][:-4]
for i in range(1, len(df_w1_4n.columns)):
    df_p2w = df_p2_n.copy()
    df_p2w = df_p2w[(df_p2w['contents_id'] == contents_list[0]) | (df_p2w['contents_id'] == contents_list[1])]
    weit = df_p2w['cos_sum_4'][w]
    df_w1_4n[df_w1_4n.columns[i]] = df_w1_4n[df_w1_4n.columns[i]] * weit
    cnt += 1
    if cnt == len(con_list_t):
        w += 1
        cnt = 0
df_w1_4n
# %%
df_w1_4n_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1_4n_q = df_w1_4n_q.groupby('userid', as_index=False)['r_or_w'].sum()
df_w1_4n_youso = pd.merge(df_w1_4n, df_w1_4n_q, on='userid', how='left')
df_w1_4n_youso
# %%
drop_list = []
for i in range(len(df_w1_4n_youso.columns)-1):
    if sum(df_w1_4n_youso[df_w1_4n_youso.columns[i]]) == 0:
        drop_list.append(df_w1_4n_youso.columns[i])
df_w1_4n_youso = df_w1_4n_youso.drop(drop_list,axis=1)
df_w1_4n_youso
# %%
LightGBM(df_w1_4n_youso)




# %%
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
    
    w = 0
    cnt = 0
    for i in range(1, len(df_w.columns)):
        df_p2w = df_p2_n.copy()
        df_p2w = df_p2w[df_p2w['contents_id'] == contents_list[l]].reset_index(drop=True)
        weit = df_p2w['cos_sum_4'][w]
        df_w[df_w.columns[i]] = df_w[df_w.columns[i]] * weit
        cnt += 1
        if cnt == len(con_list_t):
            w += 1
            cnt = 0
    display(df_w)

    df_w_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_w_youso = pd.merge(df_w, df_w_q, on='userid', how='left')
    df_w_youso = df_w_youso.drop(['timecreated', 'respons_day'], axis=1)
    display(df_w_youso)

    # シード値固定してる
    print(LightGBM(df_w_youso))


# %%
# 週ごとのテスト内容（答え付き）とページとのコサイン類似度
for i in range(len(cosimd_s_list)):
    print(i)
    df_cosim_s = cosimd_s_list[i].copy()
    df_cosim_s = df_cosim_s[['cid_page', 'cos_sim1', 'cos_sim2', 'cos_sim3', 'cos_sim4', 'cos_sim5']]
    df_cosim_s = df_cosim_s.set_index('cid_page')
    plt.figure()
    plt.subplots(figsize=(15, 15))
    sns.heatmap(df_cosim_s.fillna(0), annot=True, cmap='GnBu')
    plt.show()



# %%
# 各ページのコサイン類似度
for i in range(len(contents_list)):
    df_con = df_npvec_n[df_npvec_n['contents_id'] == contents_list[i]].reset_index(drop=False)
    df_convec = pd.DataFrame(index=list(df_con['cid_page']), columns=list(df_con['cid_page']))
    for j in range(len(df_convec)):
        for k in range(len(df_convec)):
            t1 = df_con['page_vec'][j]
            t2 = df_con['page_vec'][k]
            x = torch.FloatTensor(t1)
            y = torch.FloatTensor(t2)
            cosim = F.cosine_similarity(x, y, dim=0)
            df_convec.iat[j,k] = cosim.item()
    plt.figure()
    plt.subplots(figsize=(15, 15))
    sns.heatmap(df_convec.fillna(0), annot=True, cmap='GnBu')
    plt.show()    
            
            
            
            
            
            
# %%
# 講義内時間（前後1時間含む）に絞ろうの回
# 各学生の各ページをみている時間をだしてるよ
df_lec_p = df_page[((df_page['operation_date'] >= st_1) & (df_page['operation_date'] <= et_1)) | ((df_page['operation_date'] >= st_2) & (df_page['operation_date'] <= et_2)) | ((df_page['operation_date'] >= st_3) & (df_page['operation_date'] <= et_3)) | ((df_page['operation_date'] >= st_4) & (df_page['operation_date'] <= et_4)) | ((df_page['operation_date'] >= st_5) & (df_page['operation_date'] <= et_5)) | ((df_page['operation_date'] >= st_6) & (df_page['operation_date'] <= et_6)) | ((df_page['operation_date'] >= st_7) & (df_page['operation_date'] <= et_7))].reset_index(drop=True)
df_inlec_p = df_lec_p[df_lec_p['open_time'] <= 1020].reset_index(drop=True)
df_inlec_p
# %%
df_open_in = df_inlec_p.groupby(['userid', 'contents_id', 'cid_page'], as_index=False)['open_time'].sum()
df_open_in
# %%
df_open_in['weit_p'] = 0.0
for i in range(len(df_open_in)):
    bweit = 1 / max(df_open_in['open_time'])
    weit = bweit * df_open_in['open_time'][i]
    df_open_in['weit_p'][i] = weit
df_open_in
# %%
df_invec = pd.merge(df_open_in, df_npvec_n, on=['contents_id', 'cid_page'])
df_invec['weit_vec'] = df_invec['weit_p'] * df_invec['page_vec']
df_invec


# %%
df_invecs = df_invec.copy()
df_invecs = df_invecs[['userid', 'cid_page', 'contents_id', 'weit_vec']]
df_invecs
# %%
df_wsum_in = df_invecs.groupby(['userid', 'contents_id'], as_index=False)['weit_vec'].sum()
df_wsum_in




# %%
# ここから週にわけていく
df_wsum1_in = df_wsum_in.copy()
df_wsum1_in = df_wsum1_in[(df_wsum1_in['contents_id'] == contents_list[0]) | (df_wsum1_in['contents_id'] == contents_list[1])]
df_wsum1_in = df_wsum1_in.groupby('userid', as_index=False)['weit_vec'].sum()
for i in range(len(df_wsum1_in)):
    for j in range(len(df_wsum1_in['weit_vec'][i])):
        if i == 0:
            df_wsum1_in['vec_'+str(j)] = np.nan
        df_wsum1_in['vec_'+str(j)][i] = df_wsum1_in['weit_vec'][i][j]
df_wsum1_in = df_wsum1_in.fillna(0)
df_wsum1_in
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
df_youso_in

# %%
df_in_w1 = df_youso_in[(df_youso_in['contents_id'] == contents_list[0]) | (df_youso_in['contents_id'] == contents_list[1])]
df_in_w1

# %%
df_w1_in = pd.DataFrame({'userid':[i for i in range(100)]})
for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1_in[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
df_w1_in
# %%
for i in range(len(df_w1_in)):
    df_w1_user = df_in_w1[df_in_w1['userid'] == i].reset_index()
    df_w1_user = df_w1_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count', 'weit_p'], axis=1)
    df_w1_user = df_w1_user.fillna(0)
    for j in range(len(df_w1_user)):
        for k in range(len(df_w1_user.columns)):
            df_w1_in.iat[i, (len(df_w1_user.columns)*j)+k+1] = df_w1_user.iat[j, k]
df_w1_in
# %%
df_wsum1_in = df_wsum1_in.drop(['weit_vec'], axis=1)
df_w1y_in = pd.merge(df_w1_in, df_wsum1_in, on='userid')
df_w1_q_in = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1_youso_in = pd.merge(df_w1y_in, df_w1_q_in, on='userid', how='left')
df_w1_youso_in = df_w1_youso_in.drop(['timecreated', 'respons_day'], axis=1)
df_w1_youso_in
# %%
# あってるんか？？
LightGBM(df_w1_youso_in)





# %%
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_wsumw = df_wsum_in.copy()
    df_wsum_w = df_wsumw[df_wsumw['contents_id'] == contents_list[l]]
    df_wsum_w = df_wsum_w.groupby('userid', as_index=False)['weit_vec'].sum()
    for i in range(len(df_wsum_w)):
        for j in range(len(df_wsum_w['weit_vec'][i])):
            if i == 0:
                df_wsum_w['vec_'+str(j)] = np.nan
            df_wsum_w['vec_'+str(j)][i] = df_wsum_w['weit_vec'][i][j]
    df_wsum_w = df_wsum_w.fillna(0)

    df_cid_w_s = df_youso_in[df_youso_in['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_w_s = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w_s[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_w_s)):
        df_w_po_user = df_cid_w_s[df_cid_w_s['userid'] == i].reset_index()
        df_w_po_user = df_w_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count', 'weit_p'], axis=1)
        df_w_po_user = df_w_po_user.fillna(0)
        for j in range(len(df_w_po_user)):
            for k in range(len(df_w_po_user.columns)):
                df_w_s.iat[i, (len(df_w_po_user.columns)*j)+k+1] = df_w_po_user.iat[j, k]
    display(df_w_s)

    df_wsum_w = df_wsum_w.drop(['weit_vec'], axis=1)
    df_wy = pd.merge(df_w_s, df_wsum_w, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    df_wy

    LightGBM(df_wy)





# %%
# 講義時間内にする
df_w1_4in = pd.DataFrame({'userid':[i for i in range(100)]})

for i in range(2):
    pnum = df_contentspage['page'][i]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w1_4in[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan
for i in range(len(df_w1_4in)):
    df_w1_r_user = df_in_w1[df_in_w1['userid'] == i].reset_index()
    df_w1_r_user = df_w1_r_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count', 'weit_p'], axis=1)
    df_w1_r_user = df_w1_r_user.fillna(0)
    for j in range(len(df_w1_r_user)):
        for k in range(len(df_w1_r_user.columns)):
            df_w1_4in.iat[i, (len(df_w1_r_user.columns)*j)+k+1] = df_w1_r_user.iat[j, k]
df_w1_4in

# %%
w = 0
cnt = 0

for i in range(1, len(df_w1_4in.columns)):
    df_p2w = df_p2_n.copy()
    df_p2w = df_p2w[(df_p2w['contents_id'] == contents_list[0]) | (df_p2w['contents_id'] == contents_list[1])]
    weit = df_p2w['cos_sum_4'][w]
    df_w1_4in[df_w1_4in.columns[i]] = df_w1_4in[df_w1_4in.columns[i]] * weit
    cnt += 1
    if cnt == len(con_list_t):
        w += 1
        cnt = 0
df_w1_4in
# %%
df_w1_4in_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[0]]
df_w1_4in_q = df_w1_4in_q.groupby('userid', as_index=False)['r_or_w'].sum()
df_w1_4in_youso = pd.merge(df_w1_4in, df_w1_4in_q, on='userid', how='left')
df_w1_4in_youso
# %%
LightGBM(df_w1_4in_youso)




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
    
    w = 0
    cnt = 0
    for i in range(1, len(df_w.columns)):
        df_p2w = df_p2_n.copy()
        df_p2w = df_p2w[df_p2w['contents_id'] == contents_list[l]].reset_index(drop=True)
        weit = df_p2w['cos_sum_4'][w]
        df_w[df_w.columns[i]] = df_w[df_w.columns[i]] * weit
        cnt += 1
        if cnt == len(con_list_t):
            w += 1
            cnt = 0
    display(df_w)

    df_w_q = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_w_youso = pd.merge(df_w, df_w_q, on='userid', how='left')
    df_w_youso = df_w_youso.drop(['timecreated', 'respons_day'], axis=1)
    display(df_w_youso)

    # シード値固定してる
    print(LightGBM(df_w_youso))



# %%
# 次元削減をしよう

# PCA
model_pca = PCA(n_components=2)
df_wsum1_n1 = df_wsum1_n.drop(['userid'], axis=1)
pcan1_list = model_pca.fit_transform(df_wsum1_n1)
print(model_pca.explained_variance_ratio_)
df_pcan1 = pd.DataFrame(pcan1_list)
df_pcan1['userid'] = list(df_wsum1_n['userid'])
df_pcan1



# %%
# t-SNE
# perplexity=[2,5,10,50,100]
model_tsne = TSNE(random_state=0, n_components=2, perplexity=2)
tsnen1_list = model_tsne.fit_transform(df_wsum1_n1)
df_tsnen1 = pd.DataFrame(tsnen1_list)
df_tsnen1['userid'] = list(df_wsum1_n['userid'])
df_tsnen1


# %%
# UMAP
# n_neighbors
model_umap = umap.UMAP(random_state=0, n_components=2, n_neighbors=3) 
umapn1_list = model_umap.fit_transform(df_wsum1_n1)
df_umapn1 = pd.DataFrame(umapn1_list)
df_umapn1['userid'] = list(df_wsum1_n['userid'])
df_umapn1


# %%
df_w1y_np = pd.merge(df_w1_sn, df_pcan1, on='userid')
df_w1y_np = pd.merge(df_w1y_np, df_w1q_n, on='userid', how='left')
LightGBM(df_w1y_np)


# %%
df_w1y_nt = pd.merge(df_w1_sn, df_tsnen1, on='userid')
df_w1y_nt = pd.merge(df_w1y_nt, df_w1q_n, on='userid', how='left')
LightGBM(df_w1y_nt)


# %%
df_w1y_nu = pd.merge(df_w1_sn, df_umapn1, on='userid')
df_w1y_nu = pd.merge(df_w1y_nu, df_w1q_n, on='userid', how='left')
LightGBM(df_w1y_nu)

# %%
# PCA
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_wsumw = df_wsum_n.copy()
    df_wsum_w = df_wsumw[df_wsumw['contents_id'] == contents_list[l]]
    df_wsum_w = df_wsum_w.groupby('userid', as_index=False)['weit_vec'].sum()
    for i in range(len(df_wsum_w)):
        for j in range(len(df_wsum_w['weit_vec'][i])):
            if i == 0:
                df_wsum_w['vec_'+str(j)] = np.nan
            df_wsum_w['vec_'+str(j)][i] = df_wsum_w['weit_vec'][i][j]
    df_wsum_w = df_wsum_w.fillna(0)

    df_cid_w_s = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_w_s = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w_s[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_w_s)):
        df_w_po_user = df_cid_w_s[df_cid_w_s['userid'] == i].reset_index()
        df_w_po_user = df_w_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_po_user = df_w_po_user.fillna(0)
        for j in range(len(df_w_po_user)):
            for k in range(len(df_w_po_user.columns)):
                df_w_s.iat[i, (len(df_w_po_user.columns)*j)+k+1] = df_w_po_user.iat[j, k]
    display(df_w_s)

    df_wsum_w = df_wsum_w.drop(['weit_vec'], axis=1)
    model_pca = PCA(n_components=2)
    df_wsumw_1 = df_wsum_w.drop(['userid'], axis=1)
    pcaws_list = model_pca.fit_transform(df_wsumw_1)
    print(model_pca.explained_variance_ratio_)
    df_pcaws = pd.DataFrame(pcaws_list)
    df_pcaws['userid'] = list(df_wsum_w['userid'])

    df_wy = pd.merge(df_w_s, df_pcaws, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    df_wy

    LightGBM(df_wy)


# %%
# t-SNE
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_wsumw = df_wsum_n.copy()
    df_wsum_w = df_wsumw[df_wsumw['contents_id'] == contents_list[l]]
    df_wsum_w = df_wsum_w.groupby('userid', as_index=False)['weit_vec'].sum()
    for i in range(len(df_wsum_w)):
        for j in range(len(df_wsum_w['weit_vec'][i])):
            if i == 0:
                df_wsum_w['vec_'+str(j)] = np.nan
            df_wsum_w['vec_'+str(j)][i] = df_wsum_w['weit_vec'][i][j]
    df_wsum_w = df_wsum_w.fillna(0)

    df_cid_w_s = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_w_s = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w_s[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_w_s)):
        df_w_po_user = df_cid_w_s[df_cid_w_s['userid'] == i].reset_index()
        df_w_po_user = df_w_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_po_user = df_w_po_user.fillna(0)
        for j in range(len(df_w_po_user)):
            for k in range(len(df_w_po_user.columns)):
                df_w_s.iat[i, (len(df_w_po_user.columns)*j)+k+1] = df_w_po_user.iat[j, k]
    display(df_w_s)

    df_wsum_w = df_wsum_w.drop(['weit_vec'], axis=1)
    model_tsne = TSNE(random_state=0, n_components=2, perplexity=2)
    df_wsumw_1 = df_wsum_w.drop(['userid'], axis=1)
    tsnews_list = model_tsne.fit_transform(df_wsumw_1)
    df_tsnews = pd.DataFrame(tsnews_list)
    df_tsnews['userid'] = list(df_wsum_w['userid'])

    df_wy = pd.merge(df_w_s, df_tsnews, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    df_wy

    LightGBM(df_wy)


# %%
# UMAP
for l in range(2, len(contents_list)):
    print(contents_list[l])
    df_wsumw = df_wsum_n.copy()
    df_wsum_w = df_wsumw[df_wsumw['contents_id'] == contents_list[l]]
    df_wsum_w = df_wsum_w.groupby('userid', as_index=False)['weit_vec'].sum()
    for i in range(len(df_wsum_w)):
        for j in range(len(df_wsum_w['weit_vec'][i])):
            if i == 0:
                df_wsum_w['vec_'+str(j)] = np.nan
            df_wsum_w['vec_'+str(j)][i] = df_wsum_w['weit_vec'][i][j]
    df_wsum_w = df_wsum_w.fillna(0)

    df_cid_w_s = df_youso_cid[df_youso_cid['contents_id'] == contents_list[l]].reset_index(drop=True)
    df_w_s = pd.DataFrame({'userid':[i for i in range(100)]})
    pnum = df_contentspage['page'][l]
    for j in range(1,pnum+1):
        for k in range(len(con_list_t)):
            df_w_s[[con_list_t[k] + '_' + str(j) + '_' + str(i)]] = np.nan

    for i in range(len(df_w_s)):
        df_w_po_user = df_cid_w_s[df_cid_w_s['userid'] == i].reset_index()
        df_w_po_user = df_w_po_user.drop(['index', 'userid', 'contents_id', 'cid_page', 'ADD MEMO_count', 'ADD BOOKMARK_count', 'LINK_CLICK_count', 'CHANGE MEMO_count', 'BOOKMARK_JUMP_count', 'DELETE BOOKMARK_count', 'DELETE_MEMO_count', 'SEARCH_count', 'SEARCH_JUMP_count', 'ADD_HW_MEMO_count'], axis=1)
        df_w_po_user = df_w_po_user.fillna(0)
        for j in range(len(df_w_po_user)):
            for k in range(len(df_w_po_user.columns)):
                df_w_s.iat[i, (len(df_w_po_user.columns)*j)+k+1] = df_w_po_user.iat[j, k]
    display(df_w_s)

    df_wsum_w = df_wsum_w.drop(['weit_vec'], axis=1)
    model_umap = umap.UMAP(random_state=0, n_components=2, n_neighbors=3) 
    df_wsumw_1 = df_wsum_w.drop(['userid'], axis=1)
    umapws1_list = model_umap.fit_transform(df_wsumw_1)
    df_umapws = pd.DataFrame(umapws1_list)
    df_umapws['userid'] = list(df_wsum_w['userid'])

    df_wy = pd.merge(df_w_s, df_umapws, on='userid')
    df_wq = df_quiz_score_w[df_quiz_score_w['respons_day'] == resday_list[l-1]]
    df_wq = df_wq.groupby('userid', as_index=False)['r_or_w'].sum()
    df_wy = pd.merge(df_wy, df_wq, on='userid', how='left')
    df_wy

    LightGBM(df_wy)



