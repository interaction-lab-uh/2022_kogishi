# %%
from cmath import nan
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
# データの読み込み
df = pd.read_csv('./reading_v2.csv')
df
# %%
# 生徒数
df['userid'].nunique()
# %%
# contents_id + page_no
df['cid_page'] = df['contents_id'] + '_' + df['page_no'].astype(str).str.zfill(3)
df['cid_page']
# %%
# getit（理解）のdf
df_getit = df[df['operation_name'] == 'GETIT']
df_getit
# %%
# notgetit（未理解）のdf
df_notgetit = df[df['operation_name'] == 'NOTGETIT']
df_notgetit
# %%
# 理解と未理解くっつけようぜ
df_cstat = pd.concat([df_getit, df_notgetit], axis=1).fillna(0, downcast='infer')
df_cstat
# %%
df_get = df_getit.groupby(['cid_page']).count()['operation_name'].reset_index()
df_get
# %%
df_notget = df_notgetit.groupby(['cid_page']).count()['operation_name'].reset_index()
df_notget
# %%
# GETIT, NOTGETITだけを取り出したもの
df_get_notget = pd.merge(df_get, df_notget, on='cid_page', how = 'outer', ).fillna(0, downcast='infer')
df_get_notget
# %%
# GETIT, NOTGETITの合計をだす
df_get_notget['get_sum'] = df_get_notget['operation_name_x'] + df_get_notget['operation_name_y']
df_get_notget
# %%
# なんかNOTGETITがない位置多くないですか？そんなことあります？でも半分ならそうか…そうか…
df_get_notget[df_get_notget['operation_name_y'] == 0]
# %%
# NOTGETの方が多いページ（7ページ）
# 圧倒的に4週目の理解がたりていない
df_get_notget[df_get_notget['operation_name_x'] < df_get_notget['operation_name_y']]








# %%
# 表紙しか情報ないところとりのぞいた
df_drop_title = df_get_notget.drop(df_get_notget.query('cid_page == ["25b2822c2f5a3230abfadd476e8b04c9_001", "25b2822c2f5a3230abfadd476e8b04c9_004", "25b2822c2f5a3230abfadd476e8b04c9_044", "25b2822c2f5a3230abfadd476e8b04c9_065", "f85454e8279be180185cac7d243c5eb3_001", "f85454e8279be180185cac7d243c5eb3_011", "f85454e8279be180185cac7d243c5eb3_015", "f85454e8279be180185cac7d243c5eb3_019", "7eacb532570ff6858afd2723755ff790_001", "7eacb532570ff6858afd2723755ff790_009", "7eacb532570ff6858afd2723755ff790_014", "7eacb532570ff6858afd2723755ff790_043", "3c7781a36bcd6cf08c11a970fbe0e2a6_001", "faa9afea49ef2ff029a833cccc778fd0_001", "b6f0479ae87d244975439c6124592772_001", "b6f0479ae87d244975439c6124592772_004", "b6f0479ae87d244975439c6124592772_030", "e0c641195b27425bb056ac56f8953d24_001"]').index)
df_drop_title
# %%
df_drop_title.sort_values(['get_sum'])
# %%
df_drop_title['get_sum'].describe()
# %%
# 生徒それぞれが理解したコンテンツの数
df_x = df_getit.groupby(['userid']).count()['operation_name'].reset_index()
df_x
# %%
# それぞれのコンテンツの理解した数を数える
df_x_1 = df_getit.groupby(['contents_id', 'cid_page']).count()['operation_name'].reset_index()
df_x_1
# %%
# それぞれの週でのgetit（理解）の合計
df_getit_sum = df_x_1.groupby(['contents_id']).sum()['operation_name'].reset_index()
df_getit_sum










# %%
# markerがつけられたところの抽出
df_marker = df[df['operation_name'] == 'ADD MARKER']
df_marker
# %%
df_marker.groupby('contents_id')['operation_name'].count()
# %%
# markertext存在するところの抽出
df_markertext = df_marker[df_marker['marker_text'].notna()]
df_markertext
# %%
print(len(df_marker))
print(len(df_markertext))
# %%
# marker_textだけあらいだしてみる
df_markertext['marker_text'].unique()










# %%
# 何に使えるかしらんがとりあえず読み込んでおく
df_contentspage = pd.read_csv('./contents_page_v2.csv')
df_contentspage
# %%
img_dir = './img_text'
png_dir = './png'
# %%
contents_0 = 'd1f255a373a3cef72e03aa9d980c7eca' #0
contents_1 = '7eacb532570ff6858afd2723755ff790' #1
contents_2 = 'b6f0479ae87d244975439c6124592772' #2
contents_3 = 'e0c641195b27425bb056ac56f8953d24' #3
contents_4 = 'f85454e8279be180185cac7d243c5eb3' #4
contents_5 = 'faa9afea49ef2ff029a833cccc778fd0' #5
contents_6 = '3c7781a36bcd6cf08c11a970fbe0e2a6' #6
contents_7 = '25b2822c2f5a3230abfadd476e8b04c9' #7
contents_plus = '877a9ba7a98f75b90a9d49f53f15a858' #plus

df_contime_0 = df[df['contents_id'] == contents_0]
df_contime_1 = df[df['contents_id'] == contents_1]
df_contime_2 = df[df['contents_id'] == contents_2]
df_contime_3 = df[df['contents_id'] == contents_3]
df_contime_4 = df[df['contents_id'] == contents_4]
df_contime_5 = df[df['contents_id'] == contents_5]
df_contime_6 = df[df['contents_id'] == contents_6]
df_contime_7 = df[df['contents_id'] == contents_7]
df_contime_plus = df[df['contents_id'] == contents_plus]
# %%
time_list = df['operation_date'].tolist()
# %%
# 日付だけ得ることには成功しましたが、こちらとても不可解ですﾀｽｹﾃ
date_list = []
for n in time_list:
    if n[:10] not in date_list:
        date_list.append(n[:10])
print(date_list)
# %%
# スライスの動きを把握したい
print(df['operation_date'][:10])
print(df['operation_date'][:10][0])
print(df['operation_date'][:10][0][:10])
print(df['operation_date'][0][:10])
# %%
# 操作回数が多いcontents知りたい
# 最終週が多いっぽいけど多分これページ数が多いからだあ
df_tameshi = df.groupby('contents_id').count()['operation_name'].reset_index()
# 各contentsにおける操作回数の平均とってみた
df_opepage = pd.merge( df_contentspage, df_tameshi, on='contents_id', how = 'outer', ).fillna(0, downcast='infer')
df_opepage['ope_mean'] = df_opepage['operation_name'] / df_opepage['page']
df_opepage
# %%
# 授業資料見てないやつおるなあ！！！(100*9=900)
df_userope = df.groupby(['userid', 'contents_id']).count()['operation_name'].reset_index()
df_userope = df_userope.rename({'operation_name':'operation_count'}, axis=1)
df_userope







# %%
# 個々人がどんなページ遷移をしているかってどうすればみれるかな
# NEXT, PREVを行った部分だけ抽出
# df_pagemove = df[(df['operation_name'] == 'NEXT') | (df['operation_name'] == 'PREV')] 
# df_pagemove['NEXT_operation'] = (df_pagemove['operation_name'] == 'NEXT')*1
# df_pagemove['PREV_operation'] = (df_pagemove['operation_name'] == 'PREV')*1
# df_pagemove
# %%
# ページ遷移の方法でクラスタリングとかしてみたい
# それには自分で特徴量を出す必要あるよね
# 日数？一日でどれくらい開いているか？
# コンテンツごと｛開いた日数、（操作回数）、OPEN-CLOSEの時間(の平均？中央値？)、｝
# 開いた日付ごと｛開いた日時、操作回数、OPEN-CLOSEの時間、PREV・NEXT・PAGE_JUMPの回数｝
# 私が知りたいのは各ユーザーの各コンテンツごとの開いた日数と操作回数





# %%
df_c = df.copy()
df_c['day'] = df_c['operation_date'].str[:10]
# userid、contents_id、開いた日付、(開いた日時)、OPEN-CLOSEの時間、NEXT,PREV,PAGE_JUMPのそれぞれの回数
df_c['operation_date'] = pd.to_datetime(df_c['operation_date'])
df_c['operation_date']
# %%
df_c['user_contents'] = df_c['userid'].astype(str) + '_' + df_c['contents_id']
df_c['user_contents_date'] = df_c['user_contents'] + '_' + df_c['day']
# %%
df_c['operation_minutes'] = np.nan
for i in range(len(df_c)):
    df_c['operation_minutes'][i] = "{0:%Y-%m-%d %H:%M}".format(df_c['operation_date'][i])
df_c
# %%
df_c_open = df_c[df_c['operation_name'] == 'OPEN']
df_c_close = df_c[df_c['operation_name'] == 'CLOSE']
df_c_next = df_c[df_c['operation_name'] == 'NEXT']
df_c_prev = df_c[df_c['operation_name'] == 'PREV']
df_c_pagejump = df_c[(df_c['operation_name'] == 'PAGE_JUMP') & (df_c['operation_name'] == 'SERCH_JUMP') & (df_c['operation_name'] == 'BOOKMARK_JUMP')]
# %%
df_c_open.groupby(['user_contents_date']).count()['operation_name']
# %%
df['operation_name'].value_counts()
# %%
date_list.sort()
print(date_list)
# %%
# 学生ごとの日付と操作回数のヒートマップ
# df_dayconope_1 = df_c.groupby(['userid', 'day', 'contents_id']).count()['operation_name'].reset_index()
# for i in range(0,10):
#     print(i)
#     df_c_user = df_dayconope_1[df_dayconope_1['userid'] == i].reset_index()
#     # ヒートマップのためのデータフレーム化
#     df_heat = pd.DataFrame(index = ['d1f255a373a3cef72e03aa9d980c7eca', '877a9ba7a98f75b90a9d49f53f15a858', '7eacb532570ff6858afd2723755ff790', 'b6f0479ae87d244975439c6124592772', 'e0c641195b27425bb056ac56f8953d24', 'f85454e8279be180185cac7d243c5eb3', 'faa9afea49ef2ff029a833cccc778fd0', '3c7781a36bcd6cf08c11a970fbe0e2a6', '25b2822c2f5a3230abfadd476e8b04c9'], columns=date_list)
#     # df_heatにデータを突っ込む
#     for i in range(len(df_heat)):
#         for j in range(len(date_list)):
#             for k in range(len(df_c_user)):
#                 if df_c_user['contents_id'][k] == df_heat.index[i] and df_c_user['day'][k] == df_heat.columns[j]:
#                     df_heat.iloc[i][j] = float(df_c_user['operation_name'][k])
#                     break
#                 else:
#                     df_heat.iloc[i][j] = 0
#     df_heat = df_heat.rename({'d1f255a373a3cef72e03aa9d980c7eca':0, '877a9ba7a98f75b90a9d49f53f15a858':'plus', '7eacb532570ff6858afd2723755ff790':1, 'b6f0479ae87d244975439c6124592772':2, 'e0c641195b27425bb056ac56f8953d24':3, 'f85454e8279be180185cac7d243c5eb3':4, 'faa9afea49ef2ff029a833cccc778fd0':5, '3c7781a36bcd6cf08c11a970fbe0e2a6':6, '25b2822c2f5a3230abfadd476e8b04c9':7},axis=0)
#     df_heat = df_heat.astype(float)
#     plt.figure()
#     sns.heatmap(df_heat.T, annot=True, cmap='GnBu')
#     plt.show()





# %%
import sys
from math import log
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import collections
import regex


# %%
df_marker_t = df_marker.groupby(['cid_page'])['marker_position'].count().reset_index()
df_marker_t['marker_position'].describe()
# %%
df_marker_t = df_marker_t.sort_values(by='marker_position', ascending=False)
df_marker_t['cid_page'][:120]
# %%
df_marker_user = df_marker.groupby(['userid'])['marker_position'].count().reset_index()
df_marker_user
# %%
df_marker_user.describe()
# %%
df_user_color = df_marker.groupby(['userid', 'marker_color'])['marker_position'].count().reset_index()
# %%
# 表示するdfの行の設定（50にしてる）
pd.set_option('display.max_rows', 50)
# %%
display(df_user_color[:78])





# %%
# 次は学生単位に絞って行動の内約をみるよ
df_c_open['page_no'].unique()
# %%
# ADD BOOKMARKだけの行
df_c_book = df_c[df_c['operation_name'] == 'ADD BOOKMARK']
df_c_book
# %%
# 1ページ目からOPENされていないページ
df_c_open[df_c_open['page_no'] != 1].reset_index()
# %%
# DELETE BOOKMARKの数
df_delbook = df_c[df_c['operation_name'] == 'DELETE BOOKMARK']
df_delbook
# %%
# メモの数
df_memo = df_c[df_c['operation_name'] == 'ADD MEMO']
df_memo
# %%
# メモテキストがある場所
df_c[df_c['memo_text'].notna()]
# %%
# メモタイトルがある場所
df_c[df_c['memo_title'].notna()]
# %%
# これが記録されているタイミングでめちゃ長記録がされてる
# おそらくこれはメモテキストにメモした時に残るもので、タイトルにメモしている人はこれが残っていない
# だから余計メモの扱いがわかっていなかった可能性ある。
df_c[df_c['operation_name'] == 'MEMO_TEXT_CHANGE_HISTORY']
# %%
# 各ユーザーが各行動をどれくらいとっているかを見たかった
operation_name_list = df_c['operation_name'].unique()
df_opeuser_c = pd.DataFrame({'userid':[i for i in range(100)]})
for i in range(len(operation_name_list)):
    df_ope = df_c[df_c['operation_name'] == operation_name_list[i]].reset_index()
    df_ope_user = df_ope.groupby('userid')['operation_name'].count().reset_index()
    df_ope_user = df_ope_user.rename({'operation_name':operation_name_list[i] + '_count'}, axis=1)
    df_opeuser_c = pd.merge(df_opeuser_c, df_ope_user, on='userid', how='left')
df_opeuser_c = df_opeuser_c.fillna(0)
df_opeuser_c
# %%
df_opeuser_c.describe()
# %%
df_opeuser_c.plot(kind='scatter', x='PREV_count', y='NEXT_count')







# %%
# ページを開いている時間をだそう
df_page = df_c[(df_c['operation_name'] == 'OPEN') | (df_c['operation_name'] == 'NEXT') | (df_c['operation_name'] == 'PREV') | (df_c['operation_name'] == 'CLOSE') | (df_c['operation_name'] == 'PAGE_JUMP') | (df_c['operation_name'] == 'SEARCH_JUMP') | (df_c['operation_name'] == 'BOOKMARK_JUMP')].reset_index()
df_page['open_time'] = 0
df_page = df_page.sort_values(['userid', 'operation_date']).reset_index()
for i in range(1, len(df_page)):
    if df_page['operation_name'][i] == 'OPEN':
        df_page['open_time'][i] = 0
    else :
        df_page['open_time'][i] = (df_page['operation_date'][i] - df_page['operation_date'][i-1]).total_seconds()
df_page
# %%
# よさげ
df_page['open_time'].describe()
# %%
# あわない…もしかしてOpenLAの方はPAGE_JUMP系のなにかが数えられていないのか？
df_page[(df_page['open_time'] <= 1757) & (df_page['open_time'] > 5)].describe()
# %%
# コンテンツごと｛開いた日数、（操作回数）、OPEN-CLOSEの時間(の平均？中央値？)、｝
# 開いた日付ごと｛開いた日時、操作回数、OPEN-CLOSEの時間、PREV・NEXT・PAGE_JUMPの回数｝
# log_id userid contents_id contents_name device_code day user_contents user_contents_date









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
# %%
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
start_leclist = [start_time_1, start_time_1, start_time_2, start_time_2, start_time_3, start_time_4, start_time_5, start_time_6, start_time_7]
test_leclist =[test_time_1, test_time_1, test_time_2, test_time_2, test_time_3, test_time_4, test_time_5, test_time_6, test_time_7]
end_leclist = [end_time_1, end_time_1, end_time_2, end_time_2, end_time_3, end_time_4, end_time_5, end_time_6, end_time_7]
# %%
# 授業内時間の抽出
df_inlec = df_c[((df_c['operation_date'] >= start_time_1) & (df_c['operation_date'] <= end_time_1)) | ((df_c['operation_date'] >= start_time_2) & (df_c['operation_date'] <= end_time_2)) | ((df_c['operation_date'] >= start_time_3) & (df_c['operation_date'] <= end_time_3)) | ((df_c['operation_date'] >= start_time_4) & (df_c['operation_date'] <= end_time_4)) | ((df_c['operation_date'] >= start_time_5) & (df_c['operation_date'] <= end_time_5)) | ((df_c['operation_date'] >= start_time_6) & (df_c['operation_date'] <= end_time_6)) | ((df_c['operation_date'] >= start_time_7) & (df_c['operation_date'] <= end_time_7))].reset_index()
df_inlec
# %%
# 授業外時間の抽出
df_outlec = df_c[(df_c['operation_date'] < start_time_1) | ((df_c['operation_date'] > end_time_1) & (df_c['operation_date'] < start_time_2)) | ((df_c['operation_date'] > end_time_2) & (df_c['operation_date'] < start_time_3)) | ((df_c['operation_date'] > end_time_3) & (df_c['operation_date'] < start_time_4)) | ((df_c['operation_date'] > end_time_4) & (df_c['operation_date'] < start_time_5)) | ((df_c['operation_date'] > end_time_5) & (df_c['operation_date'] < start_time_6)) | ((df_c['operation_date'] > end_time_6) & (df_c['operation_date'] < start_time_7)) | (df_c['operation_date'] > end_time_7)].reset_index()
df_outlec
# %%
# テスト時間内の抽出
df_testtime = df_c[((df_c['operation_date'] >= test_time_1) & (df_c['operation_date'] <= end_time_1)) | ((df_c['operation_date'] >= test_time_2) & (df_c['operation_date'] <= end_time_2)) | ((df_c['operation_date'] >= test_time_3) & (df_c['operation_date'] <= end_time_3)) | ((df_c['operation_date'] >= test_time_4) & (df_c['operation_date'] <= end_time_4)) | ((df_c['operation_date'] >= test_time_5) & (df_c['operation_date'] <= end_time_5)) | ((df_c['operation_date'] >= test_time_6) & (df_c['operation_date'] <= end_time_6)) | ((df_c['operation_date'] >= test_time_7) & (df_c['operation_date'] <= end_time_7))].reset_index()
df_testtime
# %%
# テストまでの授業時間内の抽出
df_inlec_notest = df_c[((df_c['operation_date'] >= start_time_1) & (df_c['operation_date'] < test_time_1)) | ((df_c['operation_date'] >= start_time_2) & (df_c['operation_date'] < test_time_2)) | ((df_c['operation_date'] >= start_time_3) & (df_c['operation_date'] < test_time_3)) | ((df_c['operation_date'] >= start_time_4) & (df_c['operation_date'] < test_time_4)) | ((df_c['operation_date'] >= start_time_5) & (df_c['operation_date'] < test_time_5)) | ((df_c['operation_date'] >= start_time_6) & (df_c['operation_date'] < test_time_6)) | ((df_c['operation_date'] >= start_time_7) & (df_c['operation_date'] < test_time_7))].reset_index()
df_inlec_notest
# %%
df_inlec_ope = df_inlec.groupby('userid')['operation_name'].count().reset_index()
df_inlec_t_ope = df_inlec_notest.groupby('userid')['operation_name'].count().reset_index()
df_outlec_ope = df_outlec.groupby('userid')['operation_name'].count().reset_index()
df_testtime_ope = df_testtime.groupby('userid')['operation_name'].count().reset_index()
# %%
print(df_inlec[df_inlec['operation_name']=='ADD MARKER']['operation_name'].count())
print(df_outlec[df_outlec['operation_name']=='ADD MARKER']['operation_name'].count())

# %%
# 行：userid　列：時間区分　要素：操作回数のdfをつくってみている
df_lectime = pd.DataFrame(index=[i for i in range(100)], columns=['inlec', 'intest', 'outlec'])
for i in range(len(df_lectime)):
    for j in range(len(df_inlec_t_ope)):
        if df_inlec_t_ope['userid'][j] == i:
            df_lectime['inlec'][i] = df_inlec_t_ope['operation_name'][j]
            break
    for j in range(len(df_outlec_ope)):
        if df_outlec_ope['userid'][j] == i:
            df_lectime['outlec'][i] = df_outlec_ope['operation_name'][j]
            break
    for j in range(len(df_testtime_ope)):
        if df_testtime_ope['userid'][j] == i:
            df_lectime['intest'][i] = df_testtime_ope['operation_name'][j]
            break
df_lectime = df_lectime.fillna(0)
df_lectime
# %%
# df_levtimeでヒートマップだしてみたけど、多分時間的平均とったほうがいい気がする
# あとテスト時間中に誰も悠長にマーカーとかひかんやろ、ページ遷移行動に絞ってみてもいいかも？？
plt.figure()
plt.subplots(figsize=(5, 5))
# sns.heatmap(df_lectime, annot=True, cmap='GnBu')
sns.heatmap(df_lectime, cmap='GnBu')
plt.show()

# %%
# 講義時間＋前後1時間ふくめたのスタートタイム（st）、エンドタイム（et）
st_1 = datetime.datetime(2020, 5, 12, 9, 50, 00)
et_1 = datetime.datetime(2020, 5, 12, 22, 20, 00)
st_2 = datetime.datetime(2020, 5, 19, 9, 50, 00)
et_2 = datetime.datetime(2020, 5, 19, 22, 20, 00)
st_3 = datetime.datetime(2020, 5, 26, 9, 50, 00)
et_3 = datetime.datetime(2020, 5, 26, 22, 20, 00)
st_4 = datetime.datetime(2020, 6, 2, 9, 50, 00)
et_4 = datetime.datetime(2020, 6, 2, 22, 20, 00)
st_5 = datetime.datetime(2020, 6, 9, 9, 50, 00)
et_5 = datetime.datetime(2020, 6, 9, 22, 20, 00)
st_6 = datetime.datetime(2020, 6, 16, 9, 50, 00)
et_6 = datetime.datetime(2020, 6, 16, 22, 20, 00)
st_7 = datetime.datetime(2020, 6, 23, 9, 50, 00)
et_7 = datetime.datetime(2020, 6, 23, 22, 20, 00)

stw_list = [st_1, st_2, st_2, st_3, st_4, st_5, st_6, st_7]
etw_list = [et_1, et_2, et_2, et_3, et_4, et_5, et_6, et_7]
resday_list = ['2020-5-12', '2020-5-19-1', '2020-5-19-2', '2020-5-26', '2020-6-2', '2020-6-9', '2020-6-16', '2020-6-23']

# %%
# 授業後一週間までに絞ってみる（917ログ消えた）
df_front = df_c[df_c['day'] < '2020-06-30']
df_front
# %%
# 絞ったほうで授業外時間の抽出
df_outlec_front = df_front[(df_front['operation_date'] < start_time_1) | ((df_front['operation_date'] > end_time_1) & (df_front['operation_date'] < start_time_2)) | ((df_front['operation_date'] > end_time_2) & (df_front['operation_date'] < start_time_3)) | ((df_front['operation_date'] > end_time_3) & (df_front['operation_date'] < start_time_4)) | ((df_front['operation_date'] > end_time_4) & (df_front['operation_date'] < start_time_5)) | ((df_front['operation_date'] > end_time_5) & (df_front['operation_date'] < start_time_6)) | ((df_front['operation_date'] > end_time_6) & (df_front['operation_date'] < start_time_7)) | (df_front['operation_date'] > end_time_7)].reset_index()
# df_outlec_front = df_front[((df_front['operation_date'] < start_time_1) & (df_front['operation_date'] >= st_1)) | ((df_front['operation_date'] > end_time_1) & (df_front['operation_date'] <= et_1)) | ((df_front['operation_date'] < start_time_2) & (df_front['operation_date'] >= st_2)) | ((df_front['operation_date'] > end_time_2) & (df_front['operation_date'] <= et_2)) | ((df_front['operation_date'] < start_time_3) & (df_front['operation_date'] >= st_3)) | ((df_front['operation_date'] > end_time_3) & (df_front['operation_date'] <= et_3)) | ((df_front['operation_date'] < start_time_4) & (df_front['operation_date'] >= st_4)) | ((df_front['operation_date'] > end_time_4) & (df_front['operation_date'] <= et_4)) | ((df_front['operation_date'] < start_time_5) & (df_front['operation_date'] >= st_5)) | ((df_front['operation_date'] > end_time_5) & (df_front['operation_date'] <= et_5)) | ((df_front['operation_date'] < start_time_6) & (df_front['operation_date'] >= st_6)) | ((df_front['operation_date'] > end_time_6) & (df_front['operation_date'] <= et_6)) | ((df_front['operation_date'] < start_time_7) & (df_front['operation_date'] >= st_7)) | ((df_front['operation_date'] > end_time_7) & (df_front['operation_date'] < et_7))].reset_index()

df_outlec_front
# %%
df_outfront_ope = df_outlec_front.groupby('userid')['operation_name'].count().reset_index()
# %%
# 行：userid　列：時間区分　要素：操作回数のdfをつくってみている
df_lectime_re = pd.DataFrame(index=[i for i in range(100)], columns=['inlec', 'intest', 'outlec'])
for i in range(len(df_lectime_re)):
    for j in range(len(df_inlec_t_ope)):
        if df_inlec_t_ope['userid'][j] == i:
            df_lectime_re['inlec'][i] = df_inlec_t_ope['operation_name'][j]
            break
    for j in range(len(df_outfront_ope)):
        if df_outfront_ope['userid'][j] == i:
            df_lectime_re['outlec'][i] = df_outfront_ope['operation_name'][j]
            break
    for j in range(len(df_testtime_ope)):
        if df_testtime_ope['userid'][j] == i:
            df_lectime_re['intest'][i] = df_testtime_ope['operation_name'][j]
            break
df_lectime_re = df_lectime_re.fillna(0)
df_lectime_re
# %%
df_lec_re = df_lectime_re.copy()
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_sc = sc.fit_transform(df_lec_re)
df_sc = pd.DataFrame(df_sc, columns=df_lec_re.columns)
model = KMeans(n_clusters=4, random_state=1)
model.fit(df_sc)
cluster = model.labels_
df_lec_re['cluster'] = cluster
df_lec_re.groupby('cluster').mean().style.bar(axis=0)
# %%
df_lec_re.groupby('cluster')['inlec'].count()

# %%
df_re = df_lectime_re.copy()
df_re['inlec'] = df_re['inlec'] / 56
df_re['intest'] = df_re['intest'] / 7
df_re['outlec'] = df_re['outlec'] / 63
display(df_re)
plt.figure()
plt.subplots(figsize=(15, 15))
sns.heatmap(df_re, annot=True, cmap='GnBu')
plt.show()

# %%
df_re.describe()

# %%
df_userlec_re = df_re.copy()
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_sc = sc.fit_transform(df_userlec_re)
df_sc = pd.DataFrame(df_sc, columns=df_userlec_re.columns)
model = KMeans(n_clusters=4, random_state=1)
model.fit(df_sc)
cluster = model.labels_
df_userlec_re['cluster'] = cluster
df_userlec_re.groupby('cluster').mean().style.bar(axis=0)
# %%
df_userlec_re.groupby('cluster')['inlec'].count()

# %%
# fig, axes = plt.subplots(1, 3, figsize=(14, 6))
plt.subplots(figsize=(12, 7))

plt.subplot(1, 3, 1)
sns.boxplot(data=df_userlec_re, x='cluster', y='inlec')
plt.legend()

plt.subplot(1, 3, 2)
sns.boxplot(data=df_userlec_re, x='cluster', y='intest')
plt.legend()

plt.subplot(1, 3, 3)
sns.boxplot(data=df_userlec_re, x='cluster', y='outlec')
plt.legend()

# %%
plt.figure(figsize=(5, 10))
sns.boxplot(data=df_userlec_re, x='cluster', y='inlec')
plt.xlabel('cluster', fontsize=15)
plt.ylabel('inlec', fontsize=15)
plt.tick_params(labelsize=15)
plt.legend()

# %%
plt.figure(figsize=(5, 10))
sns.boxplot(data=df_userlec_re, x='cluster', y='intest')
plt.xlabel('cluster', fontsize=15)
plt.ylabel('intest', fontsize=15)
plt.tick_params(labelsize=15)
plt.legend()

# %%
plt.figure(figsize=(5, 10))
sns.boxplot(data=df_userlec_re, x='cluster', y='outlec')
plt.xlabel('cluster', fontsize=15)
plt.ylabel('outlec', fontsize=15)
plt.tick_params(labelsize=15)
plt.legend()

# %%
plt.figure()
plt.subplots(figsize=(5, 5))
# sns.heatmap(df_lectime_re, annot=True, cmap='GnBu')
sns.heatmap(df_lectime_re, cmap='GnBu')
plt.show()
# %%
df_lectime_re[df_lectime_re['inlec'] <= df_lectime_re['outlec']]
# %%
#グループにわけれるかも？
df_lectime_re.plot(kind='scatter', x='inlec', y='intest')
# plt.axis('square')
# %%
df_contime = df_front.groupby(['operation_date', 'contents_id'])['operation_name'].count().reset_index()
df_contime
# %%
# 講義時間前後10日の操作回数の合計のグラフ
# for i in range(len(contents_list)):
#     df_con = df_contime[df_contime['contents_id'] == contents_list[i]]
#     x = df_con['operation_date']
#     y = df_con['operation_name']
#     fig, ax = plt.subplots()
#     plt.xlim(start_leclist[i] + datetime.timedelta(days=-10),end_leclist[i] + datetime.timedelta(days=10))
#     plt.title(contents_list[i])
#     plt.xlabel('time')
#     plt.ylabel('operation_count')
#     ax.plot(x, y)
#     plt.show()
# %%
# 講義時間前後1時間の操作回数の合計のグラフ
# for i in range(len(contents_list)):
#     df_con = df_contime[df_contime['contents_id'] == contents_list[i]]
#     x = df_con['operation_date']
#     y = df_con['operation_name']
#     fig, ax = plt.subplots()
#     plt.xlim(start_leclist[i] + datetime.timedelta(hours=-1), end_leclist[i] + datetime.timedelta(hours=1))
#     plt.title(contents_list[i])
#     plt.xlabel('time')
#     plt.ylabel('operation_count')
#     ax.plot(x, y)
#     plt.show()
# %%
# 各コンテンツの各行動を行った回数（合計）
for n in contents_list:
    df_con = df_front[df_front['contents_id'] == n]
    print(n)
    print(max(df_con['page_no']))
    print(df_con['operation_name'].value_counts())
# %%
# 講義時間前後10分の操作回数の合(df_opeuser, df_ope_user, on='userid', how='left')計のグラフ
# for i in range(len(contents_list)):
#     df_con = df_contime[df_contime['contents_id'] == contents_list[i]]
#     x = df_con['operation_date']
#     y = df_con['operation_name']
#     fig, ax = plt.subplots()
#     plt.xlim(start_leclist[i] + datetime.timedelta(minutes=-10),end_leclist[i] + datetime.timedelta(minutes=10))
#     plt.title(contents_list[i])
#     plt.xlabel('time')
#     plt.ylabel('operation_count')
#     ax.plot(x, y)
#     plt.show()
# %%
# userid0-9の講義時間10分前後のユーザーの操作回数の合計と時間のページ遷移（4週目）
# for i in range(10):
#     df_con = df_front[df_front['contents_id'] == 'f85454e8279be180185cac7d243c5eb3']
#     df_con = df_con[df_con['userid'] == i]
#     x = df_con['operation_date']
#     y = df_con['page_no']
#     fig, ax = plt.subplots()
#     plt.title('f85454e8279be180185cac7d243c5eb3'+ '_' + str(i))
#     print(((df_con['operation_date'] >= start_time_4 + datetime.timedelta(minutes=-10)) & (df_con['operation_date'] <= end_time_4 + datetime.timedelta(minutes=10))).sum())
#     plt.xlim(start_time_4 + datetime.timedelta(minutes=-10),end_time_4 + datetime.timedelta(minutes=10))
#     plt.xlabel('operation_date')
#     plt.ylabel('page_no')
#     ax.plot(x, y)
#     plt.show()







# %%
df_userlec = pd.DataFrame(index=[i for i in range(100)], columns=['before_lec', '1st_lec', '1to2_lec', '2nd_lec', '2to3_lec', '3rd_lec', '3to4_lec', '4th_lec', '4to5_lec', '5th_lec', '5to6_lec', '6th_lec', '6to7_lec', '7th_lec', 'after_lec'])
for i in range(len(df_userlec)):
    df_user = df_front[df_front['userid'] == i]
    df_userlec['before_lec'][i] = int((df_user['operation_date'] < start_time_1).sum())
    df_userlec['1st_lec'][i] = ((df_user['operation_date'] >= start_time_1) & (df_user['operation_date'] <= end_time_1)).sum()
    df_userlec['1to2_lec'][i] = ((df_user['operation_date'] < start_time_2) & (df_user['operation_date'] > end_time_1)).sum()
    df_userlec['2nd_lec'][i] = ((df_user['operation_date'] >= start_time_2) & (df_user['operation_date'] <= end_time_2)).sum()
    df_userlec['2to3_lec'][i] = ((df_user['operation_date'] < start_time_3) & (df_user['operation_date'] > end_time_2)).sum()
    df_userlec['3rd_lec'][i] = ((df_user['operation_date'] >= start_time_3) & (df_user['operation_date'] <= end_time_3)).sum()
    df_userlec['3to4_lec'][i] = ((df_user['operation_date'] < start_time_4) & (df_user['operation_date'] > end_time_3)).sum()
    df_userlec['4th_lec'][i] = ((df_user['operation_date'] >= start_time_4) & (df_user['operation_date'] <= end_time_4)).sum()
    df_userlec['4to5_lec'][i] = ((df_user['operation_date'] < start_time_5) & (df_user['operation_date'] > end_time_4)).sum()
    df_userlec['5th_lec'][i] = ((df_user['operation_date'] >= start_time_5) & (df_user['operation_date'] <= end_time_5)).sum()
    df_userlec['5to6_lec'][i] = ((df_user['operation_date'] < start_time_6) & (df_user['operation_date'] > end_time_5)).sum()
    df_userlec['6th_lec'][i] = ((df_user['operation_date'] >= start_time_6) & (df_user['operation_date'] <= end_time_6)).sum()
    df_userlec['6to7_lec'][i] = ((df_user['operation_date'] < start_time_7) & (df_user['operation_date'] > end_time_6)).sum()
    df_userlec['7th_lec'][i] = ((df_user['operation_date'] >= start_time_7) & (df_user['operation_date'] <= end_time_7)).sum()
    df_userlec['after_lec'][i] = (df_user['operation_date'] > end_time_7).sum()
df_userlec
# %%
df_userlec.sum()
# %%
df_userlec_test = pd.DataFrame(index=[i for i in range(100)], columns=['before_lec', '1st_lec', '1st_test', '1to2_lec', '2nd_lec', '2nd_test', '2to3_lec', '3rd_lec', '3rd_test', '3to4_lec', '4th_lec', '4th_test', '4to5_lec', '5th_lec', '5th_test', '5to6_lec', '6th_lec', '6th_test', '6to7_lec', '7th_lec', '7th_test', 'after_lec'])
for i in range(len(df_userlec)):
    df_user = df_front[df_front['userid'] == i]
    df_userlec_test['before_lec'][i] = (df_user['operation_date'] < start_time_1).sum()
    df_userlec_test['1st_lec'][i] = ((df_user['operation_date'] >= start_time_1) & (df_user['operation_date'] < test_time_1)).sum()
    df_userlec_test['1st_test'][i] = ((df_user['operation_date'] >= test_time_1) & (df_user['operation_date'] <= end_time_1)).sum()
    df_userlec_test['1to2_lec'][i] = ((df_user['operation_date'] < start_time_2) & (df_user['operation_date'] > end_time_1)).sum()
    df_userlec_test['2nd_lec'][i] = ((df_user['operation_date'] >= start_time_2) & (df_user['operation_date'] < test_time_2)).sum()
    df_userlec_test['2nd_test'][i] = ((df_user['operation_date'] >= test_time_2) & (df_user['operation_date'] <= end_time_2)).sum()
    df_userlec_test['2to3_lec'][i] = ((df_user['operation_date'] < start_time_3) & (df_user['operation_date'] > end_time_2)).sum()
    df_userlec_test['3rd_lec'][i] = ((df_user['operation_date'] >= start_time_3) & (df_user['operation_date'] < test_time_3)).sum()
    df_userlec_test['3rd_test'][i] = ((df_user['operation_date'] >= test_time_3) & (df_user['operation_date'] <= end_time_3)).sum()
    df_userlec_test['3to4_lec'][i] = ((df_user['operation_date'] < start_time_4) & (df_user['operation_date'] > end_time_3)).sum()
    df_userlec_test['4th_lec'][i] = ((df_user['operation_date'] >= start_time_4) & (df_user['operation_date'] < test_time_4)).sum()
    df_userlec_test['4th_test'][i] = ((df_user['operation_date'] >= test_time_4) & (df_user['operation_date'] <= end_time_4)).sum()
    df_userlec_test['4to5_lec'][i] = ((df_user['operation_date'] < start_time_5) & (df_user['operation_date'] > end_time_4)).sum()
    df_userlec_test['5th_lec'][i] = ((df_user['operation_date'] >= start_time_5) & (df_user['operation_date'] < test_time_5)).sum()
    df_userlec_test['5th_test'][i] = ((df_user['operation_date'] >= test_time_5) & (df_user['operation_date'] <= end_time_5)).sum()
    df_userlec_test['5to6_lec'][i] = ((df_user['operation_date'] < start_time_6) & (df_user['operation_date'] > end_time_5)).sum()
    df_userlec_test['6th_lec'][i] = ((df_user['operation_date'] >= start_time_6) & (df_user['operation_date'] < test_time_6)).sum()
    df_userlec_test['6th_test'][i] = ((df_user['operation_date'] >= test_time_6) & (df_user['operation_date'] <= end_time_6)).sum()
    df_userlec_test['6to7_lec'][i] = ((df_user['operation_date'] < start_time_7) & (df_user['operation_date'] > end_time_6)).sum()
    df_userlec_test['7th_lec'][i] = ((df_user['operation_date'] >= start_time_7) & (df_user['operation_date'] < test_time_7)).sum()
    df_userlec_test['7th_test'][i] = ((df_user['operation_date'] >= test_time_7) & (df_user['operation_date'] <= end_time_7)).sum()
    df_userlec_test['after_lec'][i] = (df_user['operation_date'] > end_time_7).sum()
df_userlec_test
# %%
df_userlec_test.sum()
# %%
# 縦：ページ番号　横：時間　要素：コンテンツを開いているユーザーの数　のヒートマップ
# 1分ごとでまとめるのはいいけどみんな数秒とかしかみていないからどのページを開いているって難しくない？
# その1分に入った瞬間のページを測定する方式？
# df_page['operation_minutes'] = np.nan
# for i in range(len(df_page)):
#     df_page['operation_minutes'][i] = "{0:%Y-%m-%d %H:%M}".format(df_page['operation_date'][i])
# df_page





# %%
# 講義時間内は8で、講義時間外は90で割って、テスト時間にそろえようとしている
# そろっているかわからない
df_ut = df_userlec_test[['1st_lec', '1st_test', '1to2_lec']]
df_ut = df_ut.astype({"1st_lec": "int64", "1st_test": "int64", "1to2_lec":"int64"})
df_ut['1st_lec'] = df_ut['1st_lec'] / 8
df_ut['1to2_lec'] = df_ut['1to2_lec'] / 12 #なんで90？1,008ならわかるけど
plt.figure()
plt.subplots(figsize=(15, 15))
sns.heatmap(df_ut, annot=True, cmap='GnBu')
plt.show()
# %%
# ページごとの操作回数の合計と講義時間内
for l in range(len(contents_list)):
    df_page_t = df_page[df_page['contents_id'] == contents_list[l]]
    df_page_t = df_page_t[(df_page_t['operation_date'] >= start_leclist[l]) & (df_page_t['operation_date'] <= end_leclist[l])]
    minutes_list = df_page_t['operation_minutes'].unique()
    minutes_list.sort()
    df_heatpage = pd.DataFrame(index=[i for i in range(1, df_contentspage['page'][l]+1)], columns=minutes_list)
    df_pagemin = df_page_t.groupby(['operation_minutes', 'page_no'], as_index=False)['userid'].count()
    for i in range(len(df_heatpage)):
        df_pagenum = df_pagemin[df_pagemin['page_no'] == i+1].reset_index()
        for j in range(len(df_heatpage.columns)):
            date = df_heatpage.columns[j]
            for k in range(len(df_pagenum)):
                if date == df_pagenum['operation_minutes'][k]:
                    df_heatpage[date][i+1] = df_pagenum['userid'][k]
                    break
    df_heatpage = df_heatpage.fillna(0)
    print(contents_list[l])
    display(df_heatpage)
    plt.figure()
    plt.subplots(figsize=(15, 15))
    sns.heatmap(df_heatpage, annot=True, cmap='GnBu')
    plt.show()







# %%
# それぞれの週の講義時間内、テスト時間内、講義時間外で活動している人たちのクラスタリングしてみた
# クラスタ0：全体的に活動していない人
# クラスタ1：全体的に活動している人
# クラスタ2：テスト中にめっちゃ頑張る人
# クラスタ3：微妙な人（後に行くにつれて活動減っている？）
df_userlec_test_c = df_userlec_test.copy()
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_sc = sc.fit_transform(df_userlec_test_c)
df_sc = pd.DataFrame(df_sc, columns=df_userlec_test_c.columns)
model = KMeans(n_clusters=4, random_state=1)
model.fit(df_sc)
cluster = model.labels_
df_userlec_test_c['cluster'] = cluster
df_userlec_test_c.groupby('cluster').mean().style.bar(axis=0)
# %%
df_userlec_test_c.groupby('cluster')['before_lec'].count()
# %%
# クラスターの学生のリスト作ってみる
c0_list = df_userlec_test_c[df_userlec_test_c['cluster'] == 0].index
c1_list = df_userlec_test_c[df_userlec_test_c['cluster'] == 1].index
c2_list = df_userlec_test_c[df_userlec_test_c['cluster'] == 2].index
c3_list = df_userlec_test_c[df_userlec_test_c['cluster'] == 3].index
print(c0_list)
print(c1_list)
print(c2_list)
print(c3_list)
# %%
# 各クラスターの行動に注目してみる？
df_c0 = df_c.query(' userid in (5, 8, 10, 27, 28, 30, 32, 33, 37, 39, 41, 43, 45, 50, 63, 68, 69, 70, 74, 76, 81, 85, 86, 87, 88, 92, 97, 98)').reset_index()
df_c1 = df_c.query(' userid in (6, 25, 46, 66, 80, 99)').reset_index()
df_c2 = df_c.query(' userid in (0,  3,  4, 14, 16, 20, 23, 24, 34, 38, 44, 47, 48, 49, 52, 54, 58, 60, 62, 65, 71, 75, 84, 89, 91, 95)').reset_index()
df_c3 = df_c.query(' userid in (1,  2,  7,  9, 11, 12, 13, 15, 17, 18, 19, 21, 22, 26, 29, 31, 35, 36, 40, 42, 51, 53, 55, 56, 57, 59, 61, 64, 67, 72, 73, 77, 78, 79, 82, 83, 90, 93, 94, 96)').reset_index()
display(df_c0)
display(df_c1)
display(df_c2)
display(df_c3)
# %%
print(df_c0['operation_name'].value_counts())
print(df_c1['operation_name'].value_counts())
print(df_c2['operation_name'].value_counts())
print(df_c3['operation_name'].value_counts())
# %%
df_c0_op = df_c0.groupby('cid_page')['operation_name'].count().reset_index()
df_c0_op['ope_mean'] = df_c0_op['operation_name'] / len(c0_list)
df_c1_op = df_c1.groupby('cid_page')['operation_name'].count().reset_index()
df_c1_op['ope_mean'] = df_c1_op['operation_name'] / len(c1_list)
df_c2_op = df_c2.groupby('cid_page')['operation_name'].count().reset_index()
df_c2_op['ope_mean'] = df_c2_op['operation_name'] / len(c2_list)
df_c3_op = df_c3.groupby('cid_page')['operation_name'].count().reset_index()
df_c3_op['ope_mean'] = df_c3_op['operation_name'] / len(c3_list)
# %%
display(df_c0_op.sort_values(by='operation_name', ascending=False))
display(df_c1_op.sort_values(by='operation_name', ascending=False))
display(df_c2_op.sort_values(by='operation_name', ascending=False))
display(df_c3_op.sort_values(by='operation_name', ascending=False))





# %%
# 主成分分析を行ってみた（遊び）
df_userlec_test_p = df_userlec_test.copy()
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=1)
pca.fit(df_sc)
feature = pca.transform(df_sc)
plt.figure(figsize=(6, 6))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=cluster)
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.show()
# %%
dfs = df_userlec_test_p.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
dfs.head()
# %%
#主成分分析の実行
pca = PCA()
pca.fit(dfs)
# データを主成分空間に写像
feature = pca.transform(dfs)
# %%
# 主成分得点
pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(dfs.columns))]).head()







# %%
# テスト中の行動に関してもっとみるよ
# テスト中にADD MARKERしている人いる
# ただテスト時間というものが正直もうちょい幅があるというか、絶対10分に始まっているわけではないと考えている
df_testtime['operation_name'].value_counts()
# %%
# 日付、ページごとで操作されている回数をしりたい
df_testope = df_testtime.groupby(['day', 'cid_page'])['operation_name'].count().reset_index()
df_testope = df_testope.sort_values(by='operation_name', ascending=False)
df_testope
# %%
# 各学生のテスト時間中の操作回数
df_testtime_ope
# %%
# テスト中4秒以上見られているページを抽出（OPENものこっている）
df_testpage = df_page[((df_page['operation_date'] >= test_time_1) & (df_page['operation_date'] <= end_time_1)) | ((df_page['operation_date'] >= test_time_2) & (df_page['operation_date'] <= end_time_2)) | ((df_page['operation_date'] >= test_time_3) & (df_page['operation_date'] <= end_time_3)) | ((df_page['operation_date'] >= test_time_4) & (df_page['operation_date'] <= end_time_4)) | ((df_page['operation_date'] >= test_time_5) & (df_page['operation_date'] <= end_time_5)) | ((df_page['operation_date'] >= test_time_6) & (df_page['operation_date'] <= end_time_6)) | ((df_page['operation_date'] >= test_time_7) & (df_page['operation_date'] <= end_time_7))]
df_testpageope = df_testpage[(df_testpage['operation_name'] == 'OPEN')|(df_testpage['open_time'] >= 4)]
df_testpageope = df_testpageope.groupby(['cid_page'])['userid'].count().reset_index()
df_testpageope = df_testpageope.sort_values(by='userid', ascending=False)
df_testpageope
# %%
# テスト時間中によく見られているページの確認
df_many_t = df_testpageope[df_testpageope['userid'] >= 10]
# for n in df_many_t['cid_page'].unique():
#     img = Image.open('./png/' + n + '.png')
#     img.save("./many_t_png/" + n + ".png")
# %%
# テストの提出がいつも遅い人と早い人の行動をみる
# 講義終了後1週間に絞って全部の操作が記録されている方
late_res_list = [0, 1, 98, 16, 83, 20, 54, 23, 25, 91]
early_res_list = [96, 33, 65, 99, 7, 72, 74, 43, 77, 18, 55, 88, 26]
df_late_res = df_front.query('userid in (0, 1, 98, 16, 83, 20, 54, 23, 25, 91)')
df_early_res = df_front.query('userid in (96, 33, 65, 99, 7, 72, 74, 43, 77, 18, 55, 88, 26)')
display(df_late_res)
display(df_early_res)
# %%
print(df_late_res['operation_name'].value_counts())
print(df_early_res['operation_name'].value_counts())
# %%
# ページをみている時間が記録されている方
df_latep = df_page.query('userid in (0, 1, 16, 20, 23, 25, 54, 83, 91, 98)')
df_earlyp = df_page.query('userid in (7, 18, 26, 33, 43, 55, 65, 72, 74, 77, 88, 96, 99)')
display(df_latep)
display(df_earlyp)
# %%
df_lateac = df_userlec_test.iloc[[0, 1, 16, 20, 23, 25, 54, 83, 91, 98]]
df_earlyac = df_userlec_test.iloc[[7, 18, 26, 33, 43, 55, 65, 72, 74, 77, 88, 96, 99]]
display(df_lateac)
display(df_earlyac)
# %%
df_contime0 = df_late_res.groupby(['operation_minutes', 'contents_id'])['operation_name'].count().reset_index()
df_contime1 = df_early_res.groupby(['operation_minutes', 'contents_id'])['operation_name'].count().reset_index()
df_contime0['operation_minutes'] = pd.to_datetime(df_contime0['operation_minutes'])
df_contime1['operation_minutes'] = pd.to_datetime(df_contime1['operation_minutes'])
# 講義時間前後1時間の操作回数の合計のグラフ
for i in range(len(contents_list)):
    df_con = df_contime0[df_contime0['contents_id'] == contents_list[i]]
    x = df_con['operation_minutes']
    y = df_con['operation_name']
    fig, ax = plt.subplots()
    plt.xlim(start_leclist[i] + datetime.timedelta(hours=-1),end_leclist[i] + datetime.timedelta(hours=1))
    plt.title(contents_list[i])
    plt.xlabel('time')
    plt.ylabel('operation_count')
    ax.plot(x, y)
    plt.show()
# %%
# 講義時間前後1時間の操作回数の合計のグラフ
for i in range(len(contents_list)):
    df_con = df_contime1[df_contime1['contents_id'] == contents_list[i]]
    x = df_con['operation_minutes']
    y = df_con['operation_name']
    fig, ax = plt.subplots()
    plt.xlim(start_leclist[i] + datetime.timedelta(hours=-1),end_leclist[i] + datetime.timedelta(hours=1))
    plt.title(contents_list[i])
    plt.xlabel('time')
    plt.ylabel('operation_count')
    ax.plot(x, y)
    plt.show()
# %%
# 講義中によく説明されているページってどうはかる
# みられている長さ、操作の数、長さが一番無難？
# テストまでの授業時間内の抽出
del df_page['level_0']
# %%
df_ilec_nt = df_page[(((df_page['operation_date'] >= start_time_1) & (df_page['operation_date'] < test_time_1)) | ((df_page['operation_date'] >= start_time_2) & (df_page['operation_date'] < test_time_2)) | ((df_page['operation_date'] >= start_time_3) & (df_page['operation_date'] < test_time_3)) | ((df_page['operation_date'] >= start_time_4) & (df_page['operation_date'] < test_time_4)) | ((df_page['operation_date'] >= start_time_5) & (df_page['operation_date'] < test_time_5)) | ((df_page['operation_date'] >= start_time_6) & (df_page['operation_date'] < test_time_6)) | ((df_page['operation_date'] >= start_time_7) & (df_page['operation_date'] < test_time_7)))&(df_page['page_no']!=1)].reset_index()
df_ilec_nt
# %%
df_opensum = df_ilec_nt.groupby('cid_page')['open_time'].sum().reset_index()
df_opensum = df_opensum.sort_values(by='open_time', ascending=False)
df_opensum
# %%
# 計6000秒以上開かれているページ
df_opensum[df_opensum['open_time'] > 6000]
# %%
# 講義時間中によく見られているページの確認
df_many = df_opensum[df_opensum['open_time'] > 6000].reset_index()
# for i in range(len(df_many)):
#     img = Image.open('./png/' + df_many['cid_page'][i] + '.png')
#     img.save("./many_png/" + df_many['cid_page'][i] + '_' + str(df_many['open_time'][i]) + ".png")
# %%
# 53行違うページが含まれている
set(df_many['cid_page'][:120]) - set(df_many_t['cid_page'])
# %%
testp_list = [f'{contents_list[1]}_003', f'{contents_list[1]}_006', f'{contents_list[1]}_008', f'{contents_list[1]}_009', f'{contents_list[1]}_010', f'{contents_list[1]}_011', 
              f'{contents_list[2]}_010', f'{contents_list[2]}_011', f'{contents_list[2]}_013', f'{contents_list[2]}_018', f'{contents_list[2]}_023', f'{contents_list[2]}_034', f'{contents_list[2]}_035', f'{contents_list[2]}_039', f'{contents_list[2]}_040', f'{contents_list[2]}_041', f'{contents_list[2]}_042', f'{contents_list[2]}_046', f'{contents_list[2]}_055', f'{contents_list[2]}_058', 
              f'{contents_list[3]}_011', f'{contents_list[3]}_013', f'{contents_list[3]}_025', f'{contents_list[3]}_039', f'{contents_list[3]}_040', f'{contents_list[3]}_058', f'{contents_list[3]}_060',
              f'{contents_list[4]}_005', f'{contents_list[4]}_007', f'{contents_list[4]}_011', f'{contents_list[4]}_005', f'{contents_list[4]}_020', f'{contents_list[4]}_021', f'{contents_list[4]}_025', f'{contents_list[4]}_026', f'{contents_list[4]}_027',
              f'{contents_list[5]}_005', f'{contents_list[5]}_021', f'{contents_list[5]}_023', f'{contents_list[5]}_024',
              f'{contents_list[6]}_005', f'{contents_list[6]}_006', f'{contents_list[6]}_009', f'{contents_list[6]}_010', f'{contents_list[6]}_017', f'{contents_list[6]}_021', f'{contents_list[6]}_022',
              f'{contents_list[7]}_004', f'{contents_list[7]}_007', f'{contents_list[7]}_013', f'{contents_list[7]}_020', f'{contents_list[7]}_036', f'{contents_list[7]}_037', f'{contents_list[7]}_038', f'{contents_list[7]}_039', f'{contents_list[7]}_040', f'{contents_list[7]}_041',
              f'{contents_list[8]}_006', f'{contents_list[8]}_020', f'{contents_list[8]}_047', f'{contents_list[8]}_050', f'{contents_list[8]}_060']
# %%
# 講義中によく見られていたページ,テスト中に確認したページ,テストに出たページのテーブルつくる？
df_inspage = pd.DataFrame(index=df_c['cid_page'].unique(), columns=['inlec', 'intest', 'testp'])
df_inspage
# %%
for i in range(len(df_inspage)):
    if df_inspage.index[i] in list(df_many_t['cid_page']):
        df_inspage['intest'][i] = 1
for j in range(len(df_inspage)):
    if df_inspage.index[j] in list(df_many['cid_page']):
        df_inspage['inlec'][j] = 1
for k in range(len(df_inspage)):
    if df_inspage.index[k] in testp_list:
        df_inspage['testp'][k] = 1
df_inspage
# %%
df_inspage.sum()
# %%
plt.figure(figsize=(8,10))
sns.heatmap(df_inspage.fillna(0), annot=True, cmap='GnBu')
plt.show()

# %%
