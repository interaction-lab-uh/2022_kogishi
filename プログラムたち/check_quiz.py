# %%
from cmath import nan
from re import A
from cv2 import merge
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import matplotlib as mpl
import datetime
# from datetime import datetime
# %%
df_quiz = pd.read_csv('./quiz.csv')
df_quiz
# %%
# さらに減った人数、なぜ
df_quiz['userid'].nunique()
# %%
# 諦めた人あらいだしてやろうか
for i in range(100):
    if i not in set(df_quiz['userid']):
        print(i)
# %%
df_quiz['question_title'] = np.nan
for i in range(len(df_quiz)):
    df_quiz['question_title'][i] = df_quiz['questionsummary'][i].split(':')[0]
# %%
df_quiz['r_or_w'] = np.nan
for i in range(len(df_quiz)):
    if df_quiz['state'][i] == 'gradedright':
        df_quiz['r_or_w'][i] = 1
    else:
        df_quiz['r_or_w'][i] = 0
df_quiz
# %%
df_quiz['question_title'].unique()
# %%
# 正解している人
df_right = df_quiz[df_quiz['state'] == 'gradedright']
df_right
# %%
# 正解していない人
df_wrong = df_quiz[df_quiz['state'] == 'gradedwrong']
df_wrong
# %%
df_quiz.groupby('userid')['state'].count().describe()
# %%
# 小テストの内容学生ごとに変化していることに今気が付きました、草です
# 問題が変化するというより正解の選択肢が変化する感じな気がする、一番初めしかみてないけど
df_quiz['questionid'].unique()
# %%
for i in range(5):
    df_user = df_quiz[df_quiz['userid'] == i]
    print(i)
    print(df_user['questionid'])
# %%
quiz_title = df_quiz['question_title'].unique()
quizid_list = df_quiz['questionid'].unique()
quizid_list
# %%
# 行：学生　列：問題　から成るdf_rwを作ろうとしている
df_rw = pd.DataFrame(index=[i for i in range(100)], columns=quiz_title)
for i in range(100):
    if i in set(df_quiz['userid']):
        df_user = df_quiz[df_quiz['userid'] == i].reset_index()
        for j in range(len(df_user)):
            for k in range(len(quiz_title)):
                if df_user['question_title'][j] == df_rw.columns[k]:
                    df_rw.iloc[i][k] = float(df_user['r_or_w'][j])
                    break
df_rw = df_rw.fillna(0)
df_rw['count'] = np.nan
df_rw['count'] = df_rw.sum(axis=1)
df_rw
# %%
df_rw['count'].describe()
# %%
# timecreatedを日本時間に変更するよ
# df_test = df_quiz.copy()
# df_test['timecreated'] = pd.to_datetime(df_quiz['timecreated'], unit='s', utc=True)
# for i in range(len(df_test)):
#     df_test['timecreated'][i] = df_test['timecreated'][i].tz_convert('Asia/Tokyo')
# df_test
# %%
# 別の方法
df_quiz_n = df_quiz.copy()
for i in range(len(df_quiz_n)):
    df_quiz_n['timecreated'][i] = datetime.datetime.fromtimestamp(df_quiz_n['timecreated'][i])
df_quiz_n
# %%
df_quiz_score = df_rw[['count']].reset_index()
df_quiz_score.rename({'index':'userid', 'count':'score'}, axis=1)
# %%
# df_quiz_score.to_csv("./quiz_score.csv", 
#           index=False   # 行インデックスを削除
#          )
# %%
df_quiztime = df_quiz_n.sort_values(['userid', 'timecreated']).reset_index()
df_quiztime['respons_time'] = np.nan
df_quiztime['respons_day'] = np.nan
for i in range(len(df_quiz_n)):
    df_quiztime['respons_day'][i] = str(df_quiztime['timecreated'][i].year) + '-' + str(df_quiztime['timecreated'][i].month) + '-' + str(df_quiztime['timecreated'][i].day)
for i in range(len(df_quiz_n)):
    df_quiztime['respons_time'][i] = str(df_quiztime['timecreated'][i].hour) + ':' + str(df_quiztime['timecreated'][i].minute)
df_quiztime
# %%
quiz_index = ['16:2', '16:5', '16:6', '16:7', '16:8', '16:10', '16:11', '16:12', '16:13', '16:14', '16:15', '16:16', '16:17', '16:18',
 '16:19', '16:20', '16:21', '16:22', '16:23', '16:24', '16:25', '16:26', '16:27', '16:28', '16:29', '16:31', '16:34', '16:37']
quiz_columns = ['2020-5-12', '2020-5-19', '2020-5-26', '2020-6-2', '2020-6-9', '2020-6-16', '2020-6-23']
df_timebar = pd.DataFrame(index=quiz_index, columns=quiz_columns)
# %%
df_quiztime['userid_c'] = np.nan
df_quiztime['userid_c'][0] = 1
for i in range(1, len(df_quiztime)):
    if df_quiztime['timecreated'][i-1] != df_quiztime['timecreated'][i]:
        df_quiztime['userid_c'][i] = 1
df_quiztime = df_quiztime.fillna(0)
df_quiztime
# %%
df_quiztime[df_quiztime['respons_day'] == '2020-5-19']
# %%
# 行：回答（提出）時間：分　　列：回答（提出）日付
df_quiztime_t = df_quiztime.groupby(['respons_day', 'respons_time'])['userid_c'].sum().reset_index()
for i in range(len(quiz_index)):
    time = quiz_index[i]
    df_time = df_quiztime_t[df_quiztime_t['respons_time'] == time].reset_index()
    for j in range(len(df_timebar.columns)):
        day = df_timebar.columns[j]
        for k in range(len(df_time)):
            if day == df_time['respons_day'][k]:
                df_timebar[day][i] = df_time['userid_c'][k]
                break
df_timebar = df_timebar.fillna(0)
df_timebar
# %%
# 積み上げグラフを描きたい
# 基本的に16:13くらいには提出終わっている感じ、はやい
# fig, ax = plt.subplots(figsize=(15, 8))
# for i in range(len(df_timebar.T)):
#     ax.bar(df_timebar.T.columns, df_timebar.T.iloc[i], bottom=df_timebar.T.iloc[:i].sum())
# ax.set(xlabel='respons_time', ylabel='respons_count')
# ax.legend(df_timebar.T.index)
# plt.show()
# %%
fig, ax = plt.subplots(figsize=(15, 8))
for i in range(len(df_timebar.T)):
    ax.bar(df_timebar.T.columns, df_timebar.T.iloc[i], bottom=df_timebar.T.iloc[:i].sum())
ax.set(xlabel='respons_time', ylabel='respons_count')
ax.legend(['1week', '2week(1)', '2week(2)', '3week', '4week', '5week', '6week', '7week'])
plt.show()
# %%
# 行：userid 列：dayのdf　もしかしたら途中で諦めた人もいるかもしれない
df_res_quiz = df_quiztime.groupby(['userid', 'respons_day'])['state'].count().reset_index()
df_resquiz = pd.DataFrame(index=[i for i in range(100)], columns=quiz_columns)
for i in range(len(df_resquiz)):
    df_usernum = df_res_quiz[df_res_quiz['userid'] == i].reset_index()
    for j in range(len(df_resquiz.columns)):
        day = df_resquiz.columns[j]
        for k in range(len(df_usernum)):
            if day == df_usernum['respons_day'][k]:
                df_resquiz[day][i] = df_usernum['state'][k]
                break
df_resquiz = df_resquiz.fillna(0)
df_resquiz
#%%
# 単純に各週のクイズの正解不正解の分布をだすよ
# やり直しました
# 良い感じにガタガタです
df_time_right = df_quiztime[df_quiztime['state'] == 'gradedright'].reset_index()
df_res_quiz_r = df_time_right.groupby(['userid', 'respons_day'])['state'].count().reset_index()
df_resquiz_r = pd.DataFrame(index=[i for i in range(100)], columns=quiz_columns)
for i in range(len(df_resquiz_r)):
    df_usernum = df_res_quiz_r[df_res_quiz_r['userid'] == i].reset_index()
    for j in range(len(df_resquiz_r.columns)):
        day = df_resquiz_r.columns[j]
        for k in range(len(df_usernum)):
            if day == df_usernum['respons_day'][k]:
                df_resquiz_r[day][i] = df_usernum['state'][k]
                break
print(df_resquiz_r.mean(skipna=True))
df_resquiz_r = df_resquiz_r.fillna(0)
df_resquiz_r
# %%
df_quiz_score['count'].std()
# %%
fig, ax = plt.subplots(figsize=(15, 8))
df_resquiz_rT = df_resquiz_r.T
for i in range(len(df_resquiz_rT)):
    ax.bar(df_resquiz_rT.columns, df_resquiz_rT.iloc[i], bottom=df_resquiz_rT.iloc[:i].sum())
ax.set(xlabel='userid', ylabel='quizscore')
ax.legend(df_resquiz_rT.index)
plt.show()
# %%
fig, ax = plt.subplots(figsize=(15, 8))
df_resquiz_rT = df_resquiz_r.T
for i in range(len(df_resquiz_rT)):
    ax.bar(df_resquiz_rT.columns, df_resquiz_rT.iloc[i], bottom=df_resquiz_rT.iloc[:i].sum())
ax.set(xlabel='userid', ylabel='quizscore')
ax.legend(['1week', '2week(1)', '2week(2)', '3week', '4week', '5week', '6week', '7week'])
plt.show()
# %%
# 最初に提出した人を0秒、最後に提出した人をmaxとして平均をとる？
# 各学生がいつ提出したかの確認(日時)
# 平均とる必要ある？
quiz_columns_n = ['2020-5-12', '2020-5-26', '2020-6-2', '2020-6-9', '2020-6-16', '2020-6-23']
df_restime = df_quiztime[df_quiztime['userid_c'] == 1.0].reset_index()
df_restime['light'] = 0
for i in range(len(quiz_columns_n)):
    for j in range(len(df_resquiz_r[quiz_columns_n[i]])):
        for k in range(len(df_restime)):
            if quiz_columns_n[i] == df_restime['respons_day'][k] and j == df_restime['userid'][k]:
                df_restime['light'][k] = df_resquiz_r[quiz_columns_n[i]][j]
                break
df_restime = df_restime[['userid', 'timecreated', 'respons_day', 'respons_time', 'question_title', 'light']]
df_restime
# %%
# 各週の提出時間の平均をとってみた(2週目以外)
res_time_list = []
quiz_columns_n = ['2020-5-12', '2020-5-26', '2020-6-2', '2020-6-9', '2020-6-16', '2020-6-23']
for i in range(len(quiz_columns_n)):
    df_ret = df_restime[df_restime['respons_day'] == quiz_columns_n[i]]
    df_ret = df_ret.sort_values('timecreated').reset_index()
    total_s = 0
    for j in range(1, len(df_ret)):
        total_s = total_s + ((df_ret['timecreated'][j] - df_ret['timecreated'][0]).total_seconds())
    s = total_s / len(df_ret)
    res_time_list.append(df_ret['timecreated'][0] + datetime.timedelta(seconds=s))
    print(quiz_columns_n[i])
    print(df_ret['timecreated'][0] + datetime.timedelta(seconds=s))
# %%
# 各週の提出時間の四分位数と各週の正解数の分布(2週目以外)
for i in range(len(quiz_columns_n)):
    df_ret = df_restime[df_restime['respons_day'] == quiz_columns_n[i]]
    df_ret = df_ret.sort_values('timecreated').reset_index()
    df_ret['user_s'] = 0
    for j in range(0, len(df_ret)):
        df_ret['user_s'][j] = (df_ret['timecreated'][j] - df_ret['timecreated'][0]).total_seconds()
    # df_ret['restime_s'] = df_ret['timecreated'][0] + datetime.timedelta(seconds=df_ret['restime_s'])
    c_array = np.percentile(df_ret['user_s'], q=[0, 25, 50, 75, 100])
    # print(df_ret['light'].describe())
    # display(df_ret)
    df_ret['timecreated'][0] + datetime.timedelta(seconds=c_array[0])
    print(c_array)
    plt.title(quiz_columns_n[i])
    plt.xlabel('time')
    plt.ylabel('lightnum')
    plt.vlines([df_ret['timecreated'][0] + datetime.timedelta(seconds=c_array[1]), df_ret['timecreated'][0] + datetime.timedelta(seconds=c_array[2]), df_ret['timecreated'][0] + datetime.timedelta(seconds=c_array[3])],0,5,color="red")
    plt.scatter(df_ret['timecreated'], df_ret['light'])
    plt.show()
# %%
# 2週目の提出時間の平均をとってみた
for i in range(2):
    if i == 0:
        df_rest = df_restime[(df_restime['respons_day'] == '2020-5-19') & ((df_restime['question_title'] == 'スマートフォン、タブレット、パソコンなど手元のデバイス(端末)に保存されている情報などの説明で、正しいものを一つ選んで下さい。') | (df_restime['question_title'] == 'デバイスのパスワードに関して、正しいものを一つ選んで下さい。'))]
    if i == 1:
        df_rest = df_restime[(df_restime['respons_day'] == '2020-5-19') & ((df_restime['question_title'] == 'パスワードの仕組みについて、正しいものを一つ選んで下さい。') | (df_restime['question_title'] == '多要素認証・多段階認証について、正しいものを一つ選んで下さい。'))]
    df_rest = df_rest.sort_values('timecreated').reset_index()
    total_s = 0
    for j in range(1, len(df_rest)):
        total_s = total_s + ((df_rest['timecreated'][j] - df_rest['timecreated'][0]).total_seconds())
    s = total_s / len(df_rest)
    res_time_list.append(df_rest['timecreated'][0] + datetime.timedelta(seconds=s))
    print('2020-5-19'+'_'+str(i))
    print(df_rest['timecreated'][0] + datetime.timedelta(seconds=s))
res_time_list.sort()
res_time_list
# %%
df_quiztime['2nd_q'] = np.nan
for i in range(len(df_quiztime)):
    if (df_quiztime['question_title'][i] == 'パスワードの仕組みについて、正しいものを一つ選んで下さい。') | (df_quiztime['question_title'][i] == '多要素認証・多段階認証について、正しいものを一つ選んで下さい。') | (df_quiztime['question_title'][i] == 'パスワードが他人に知られる理由，知られにくくできる理由について，正しいものを一つ選んで下さい。') | (df_quiztime['question_title'][i] == '無線LAN あるいはその使用方法ついて、正しいものを一つ選んで下さい。') | (df_quiztime['question_title'][i] == '使用すべきではない無線LANサービスへの対応として、正しいものを一つ選んで下さい。'):
        df_quiztime['2nd_q'][i] = 2
    else:
        df_quiztime['2nd_q'][i] = 1
df_quiztime
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
# 生徒を二分化する,提出早dfと提出遅dfに分ける
# 早い人
df_early = df_quiztime[(df_quiztime['timecreated']<=res_time_list[0])|((df_quiztime['2nd_q']==1)&((df_quiztime['timecreated']<=res_time_list[1])&(df_quiztime['timecreated']>start_time_2)))|((df_quiztime['2nd_q']==2)&((df_quiztime['timecreated']<=res_time_list[2])&(df_quiztime['timecreated']>start_time_2)))|((df_quiztime['timecreated']>start_time_3)&(df_quiztime['timecreated']<=res_time_list[3]))|((df_quiztime['timecreated']>start_time_4)&(df_quiztime['timecreated']<=res_time_list[4]))|((df_quiztime['timecreated']>start_time_5)&(df_quiztime['timecreated']<=res_time_list[5]))|((df_quiztime['timecreated']>start_time_6)&(df_quiztime['timecreated']<=res_time_list[6]))|((df_quiztime['timecreated']>start_time_7)&(df_quiztime['timecreated']<=res_time_list[7]))].reset_index()
df_early
# %%
# 遅い人
df_late = df_quiztime[((df_quiztime['timecreated']>res_time_list[0])&(df_quiztime['timecreated']<start_time_2))|((df_quiztime['2nd_q']==1)&((df_quiztime['timecreated']>res_time_list[1])&(df_quiztime['timecreated']<start_time_3)))|((df_quiztime['2nd_q']==2)&((df_quiztime['timecreated']>res_time_list[2])&(df_quiztime['timecreated']<start_time_3)))|((df_quiztime['timecreated']>res_time_list[3])&(df_quiztime['timecreated']<start_time_4))|((df_quiztime['timecreated']>res_time_list[4])&(df_quiztime['timecreated']<start_time_5))|((df_quiztime['timecreated']>res_time_list[5])&(df_quiztime['timecreated']<start_time_6))|((df_quiztime['timecreated']>res_time_list[6])&(df_quiztime['timecreated']<start_time_7))|(df_quiztime['timecreated']>res_time_list[7])].reset_index()
df_late
# %%
# そら週によって違う
print(df_early['userid'].unique())
print(df_late['userid'].unique())
print('毎週早い人')
print(set(df_early['userid'].unique())-set(df_late['userid'].unique()))
print('毎週遅い人')
print(set(df_late['userid'].unique())-set(df_early['userid'].unique()))
# %%
# 各回の早い人と遅い人だす？
# 早い人
for i in range(len(start_leclist)):
    if i == 1 or i == 2:
        df_early_c = df_quiztime[(df_quiztime['2nd_q']==i)&((df_quiztime['timecreated']<=res_time_list[i])&(df_quiztime['timecreated']>start_leclist[i]))].reset_index()
    else:
        df_early_c = df_quiztime[(df_quiztime['timecreated']>start_leclist[i])&(df_quiztime['timecreated']<=res_time_list[i])].reset_index()
    print(df_early_c['respons_day'][0])
    print(df_early_c['userid'].unique())
    print(len(df_early_c['userid'].unique()))
# %%
# 遅い人
start_leclistp = [start_time_2, start_time_3, start_time_3, start_time_4, start_time_5, start_time_6, start_time_7, datetime.datetime(2020, 6, 24, 16, 20, 00)]
for i in range(len(start_leclist)):
    if i == 1 or i == 2:
        df_late_c = df_quiztime[(df_quiztime['2nd_q']==i)&((df_quiztime['timecreated']>res_time_list[i])&(df_quiztime['timecreated']<start_leclistp[i]))].reset_index()
    else:
        df_late_c = df_quiztime[(df_quiztime['timecreated']>res_time_list[i])&(df_quiztime['timecreated']<start_leclistp[i])].reset_index()
    print(df_late_c['respons_day'][0])
    print(df_late_c['userid'].unique())
    print(len(df_late_c['userid'].unique()))
# %%
df_resquiz_r.iloc[[0, 1, 98, 16, 83, 20, 54, 23, 25, 91]]
# %%
df_resquiz_r.iloc[[96, 33, 65, 99, 7, 72, 74, 43, 77, 18, 55, 88, 26]]


# %%
# 小テスト内容のとりだし(問題文のみ)
# quiz1_dic = {1:'ICT は進んでいるのになかなかセキュリティインシデントが減らないか、正しいものを一つ選んで下さい。',
#              2:'たくさんある別々のサイトのパスワードの扱いについて、正しいものを一つ選んで下さい。',
#              3:'セキュリティ・バイ・デザインについて、正しいものを一つ選んで下さい。',
#              4:'日本国政府によるわが国へのセキュリティ対策について、正しいものを一つ選んで下さい。',
#              5:'サプライチェーンに関する正しい説明を一つ選んで下さい。'}

# quiz2_1_dic = {1:'スマートフォン、タブレット、パソコンなど手元のデバイス(端末)に保存されている情報などの説明で、正しいものを一つ選んで下さい。',
#                2:'自分のスマートフォン、タブレット、パソコンが、もし、紛失・盗難された時に起こりうる状況で、正しいものを一つ選んで下さい。',
#                3:'デバイスのパスワードに関して、正しいものを一つ選んで下さい。',
#                4:'データのバックアップ・暗号化について、正しいものを一つ選んで下さい。',
#                5:'ソフトウェア更新について、正しいものを一つ選んで下さい。'}

# quiz2_2_dic = {1:'パスワードの仕組みについて、正しいものを一つ選んで下さい。',
#                2:'多要素認証・多段階認証について、正しいものを一つ選んで下さい。',
#                3:'パスワードが他人に知られる理由，知られにくくできる理由について，正しいものを一つ選んで下さい。',
#                4:'無線LAN あるいはその使用方法ついて、正しいものを一つ選んで下さい。',
#                5:'使用すべきではない無線LANサービスへの対応として、正しいものを一つ選んで下さい。'}

# quiz3_dic = {1:'研究における、中立、客観性について、正しいものを一つ選んで下さい。',
#              2:'研究不正となる 行為を一つ選んで下さい。',
#              3:'共同研究について、正しいものを一つ選んで下さい。',
#              4:'利益相反について、正しいものを一つ選んで下さい。',
#              5:'九州大学情報倫理規定・セキュリティポリシーについて、正しいものを一つ選んで下さい。'}

# quiz4_dic = {1:'機密性について、正しいものを一つ選んで下さい。',
#              2:'完全性について、正しいものを一つ選んで下さい。',
#              3:'可用性について、正しいものを一つ選んで下さい。',
#              4:'共通鍵方式について、正しいものを一つ選んで下さい。',
#              5:'公開鍵方式について、正しいものを一つ選んで下さい。'}

# quiz5_dic = {1:'他人にDoS攻撃を行った場合について、正しいものを一つ選んで下さい。',
#              2:'サイバーセキュリティ基本法について、正しいものを一つ選んで下さい。',
#              3:'マルウェアを配布した場合について、正しいものを一つ選んで下さい。',
#              4:'不正アクセス行為の禁止等に関する法律について、正しいものを一つ選んで下さい。',
#              5:'個人情報の保護に関する法律について、正しいものを一つ選んで下さい。'}

# quiz6_dic = {1:'知的財産権と著作権の関係について、正しいものを一つ選んで下さい。',
#              2:'著作者人格権と著作権について、正しいものを一つ選んで下さい。',
#              3:'著作物について、当てはまるものを一つ選んで下さい。',
#              4:'著作権（財産権）について、当てはまるものを一つ選んで下さい。',
#              5:'著作物等の例外的な無断利用について、正しいものを一つ選んで下さい。'}

# quiz7_dic = {1:'ICTを使わない脅威について、正しいものを一つ選んで下さい。',
#              2:'ICTを使わない方法で、ICTに与える脅威について、正しいものを一つ選んで下さい。',
#              3:'個人を特定される可能性が最も高いものを一つ選んで下さい。',
#              4:'ICT環境上で匿名性の高いものを一つ選んで下さい。',
#              5:'ICT環境が乗っ取られた後、対応が困難なものを一つ選んで下さい。'}




# %%
# 内容入り（正解のみ）
# quiz1_l_dic = {1:np.array(['ICT は進んでいるのになかなかセキュリティインシデントが減らないか、正しいものを一つ選んで下さい。', 'メールの文章などで人間をだます攻撃が用いられているから。', 'コンピュータセキュリティについてよく理解していない人が多いから。', 'メール送信のタイミングなどで人間をだます攻撃が用いられているから。']),
#              2:np.array(['たくさんある別々のサイトのパスワードの扱いについて、正しいものを一つ選んで下さい。', 'すべてのサイトで、パスワードが漏えいした時に発生する事態を想定しておく必要がある。', '万が一、パスワードが漏洩しても、そこでしか使われていないので、安心だ。', '一文字でも異なっていれば別の異なったパスワードとして効果的だ。']),
#              3:np.array(['セキュリティ・バイ・デザインについて、正しいものを一つ選んで下さい。', '設計にセキュリティ対策が入っていると、入っていなかった場合よりもその対策コストが格段に安い。', '設計段階からセキュリティ対策をする。', '製品化の段階でセキュリティ対策を行う人材は、セキュリティエンジニアと呼ばれる。']),
#              4:np.array(['日本国政府によるわが国へのセキュリティ対策について、正しいものを一つ選んで下さい。', '内閣サイバーセキュリティセンター (NISC) を事務局として、関連府省庁等と連携し、セキュリティ対策を行っている。', 'サイバーセキュリティ戦略本部でサイバーセキュリティに関わる政策や方針を策定している。', 'サイバーセキュリティ戦略を立案し、本格的に取り組んでいる。']),
#              5:np.array(['サプライチェーンに関する正しい説明を一つ選んで下さい。', '取引の連鎖には、同じ企業・グループではない取引先も含まれる。', 'サプライチェーンは、委託元と委託先だけでなく、再委託先、再々委託先と、ある目的を達成するために必要となる取引関係が連鎖する。', '複数の組織で分担されるシステム・サービスの企画・設計・製造・運用・保守・利用・提供等のプロセスのことである。'])}

# quiz2_1_l_dic = {1:np.array(['スマートフォン、タブレット、パソコンなど手元のデバイス(端末)に保存されている情報などの説明で、正しいものを一つ選んで下さい。', 'スマートフォンにはキャッシュレス決済機能が設定されていることがある。', 'デバイスには他人に使用して欲しくないアプリケーションの設定が保存されていることがある。', 'スマートフォンには他人に見られたくないメールや SNSの設定が保存されていることがある。']),
#                2:np.array(['自分のスマートフォン、タブレット、パソコンが、もし、紛失・盗難された時に起こりうる状況で、正しいものを一つ選んで下さい。', 'スマートフォンが盗難にあうと、保存されている他人の個人情報が悪用される恐れがある。', 'スマートフォンが盗難にあうと、保存されている他人の個人情報が悪用される恐れがある。', 'スマートフォンが盗難にあった場合、位置検索機能が役に立つことがある。']),
#                3:np.array(['デバイスのパスワードに関して、正しいものを一つ選んで下さい。', 'パソコンは席を外すときは面倒であるがパスワードロックをした方がよい。', 'スマートフォンのパターンロックで画面に残った指の跡がセキュリティホールになることがある。', '指紋認証や顔認証は積極的に利用したほうがよい。']),
#                4:np.array(['データのバックアップ・暗号化について、正しいものを一つ選んで下さい。', 'スマートフォンのバックアップはネットワーク等を使って外部に取る必要がある。', 'パソコンのバックアップで十分な空き容量がない時は外付けのストレージが必要である。', 'パソコンで使うUSBやSDは基本的に暗号化しておくのがよい。']),
#                5:np.array(['ソフトウェア更新について、正しいものを一つ選んで下さい。', 'サポートが終了した Windows マシンはまだ使えそうだけど、廃棄することにした。', 'ソフトウェアの更新は時間がかかることもあるが、優先的に適用すべきだ。', '家族や友人がサポートの終了したソフトウェアを使用していたので注意した。'])}

# quiz2_2_l_dic = {1:np.array(['パスワードの仕組みについて、正しいものを一つ選んで下さい。', 'パスワードは、通常、任意の長さの文字列(平文)が、ハッシュ関数によって一定の長さの文字列(パスワードハッシュ)となって保存される。', 'パスワードの確認は、入力された文字列を直接比較されるのではなく、ハッシュパスワード同士が比較されて行われる。', 'ハッシュパスワードを元のパスワードに直接戻す関数はないが、ハッシュパスワードも外部に漏えいさせてはいけない。']),
#                2:np.array(['多要素認証・多段階認証について、正しいものを一つ選んで下さい。', '多段階認証では、例えば、通常のパスワードに加え、OTP (One Time Password) などが使用される。', '多段階認証とは、認証に2つ以上の情報を使用する認証方式である。', '多要素認証では、指紋や静脈などの生体情報が用いられる。']),
#                3:np.array(['パスワードが他人に知られる理由，知られにくくできる理由について，正しいものを一つ選んで下さい。', 'ハッシュパスワードを元のパスワードに直接戻す関数はないが、ハッシュパスワードが一致する文字列が見つけられれば、元パスワードがわかってしまう。', '元のパスワードが簡単なものであると、ハッシュパスワードも容易に特定される。', '元のパスワードが複雑であれば、ハッシュパスワードの特定に時間がかかる。']),
#                4:np.array(['無線LAN あるいはその使用方法ついて、正しいものを一つ選んで下さい。', 'WPA 以下の方式の無線LAN を利用しなければならない時はセキュリティの問題に注意する。', 'パスワードが設定されていない無線LANは盗聴される危険性がある。', '自宅の無線LANの利用者は限定的であるが、パスワードの設定は必要である。']),
#                5:np.array(['使用すべきではない無線LANサービスへの対応として、正しいものを一つ選んで下さい。', 'スマートフォンを街中で使用する時、時々、どのネットワークに接続されているのか、確認する。', 'スマートフォンを大学や自宅以外で使用する時は無線LANを無効にしている。', 'お店などが正式に提供している無線LAN でも、他人に漏えいすると困るような通信はしないようにしている。'])}

# quiz3_l_dic = {1:np.array(['研究における、中立、客観性について、正しいものを一つ選んで下さい。', '検証可能なデータがなければ、どんなに素晴らしい研究でもその成果の信ぴょう性がないことになる。', '研究成果の公表で重要な立場とは、中立性・客観性のある立場である。', '研究成果の公表はその成果・結果だけではなく、データ・資料の提示も重要である。']),
#              2:np.array(['研究不正となる 行為を一つ選んで下さい。', '他の研究者等のアイディア、研究結果などを、当該研究者の了解又は適切な表示なく流用する。', '存在しないデータ、研究成果等を作成する。', '研究活動によって得られた結果等を真正でないものに加工する。']),
#              3:np.array(['共同研究について、正しいものを一つ選んで下さい。', '立場の異なる研究体制であるので、コミュニケーションが重要である。', '立場の異なる研究体制であるので、役割分担の意識が重要である。', '立場の異なる研究体制であるので、その成果や研究後の資材の扱いなどの取り決めを事前にしておくことが重要である。']),
#              4:np.array(['利益相反について、正しいものを一つ選んで下さい。', '利益相反とは、個人的な利害によって、専門家として行う判断に妥協が生じる状況のことである。', '利益相反とは、金銭の利害によって、専門家として行う判断に偏向が生じる状況のことである。', '利益相反とは、個人的な利害によって、専門家として客観性が失われる状況のことである。']),
#              5:np.array(['九州大学情報倫理規定・セキュリティポリシーについて、正しいものを一つ選んで下さい。', '九州大学には、情報やICTの扱いの規範をまとめた、情報倫理規定・セキュリティポリシーが定められている。', '九州大学には、情報の扱いの規範をまとめた、情報倫理規定が定められている。', '九州大学には、ICTの扱いの規範をまとめた、セキュリティポリシーが定められている。'])}

# quiz4_l_dic = {1:np.array(['機密性について、正しいものを一つ選んで下さい。', '機密性とは、重要な機材の設置してある部屋へ入室可能な者を制限することである。', '機密性とは、重要な情報へアクセスすることのできる者を制限することである。', '機密性とは、重要なデータの更新することのできる者を制限することである。']),
#              2:np.array(['完全性について、正しいものを一つ選んで下さい。', '完全性とは、重要な情報が改竄されることを防ぐことである。', '完全性とは、重要なデータが消去されることを防ぐことである。', '完全性とは、重要な機材の設置してある部屋で、機材の破壊を防ぐことである。']),
#              3:np.array(['可用性について、正しいものを一つ選んで下さい。', '可用性とは、重要な情報に認められた者が何時でもアクセスできるようにすることである。', '可用性とは、重要なデータに認められた者が何時でも更新できるようにすることである。', '可用性とは、重要な機材の設置してある部屋へ、認められた者が何時でも入室できるようにすることである。']),
#              4:np.array(['共通鍵方式について、正しいものを一つ選んで下さい。', '共通鍵方式では、機密性の高い一つの鍵が用いられる。', '共通鍵方式では、構成員に変更が生じた時、鍵を変更する必要がある。', '共通鍵方式では、鍵を構成員間で安全に共有する手段の確保が重要である。']),
#              5:np.array(['公開鍵方式について、正しいものを一つ選んで下さい。', '公開鍵方式では、秘密通信のための鍵を構成員間で共有することが容易である。', '公開鍵方式では、構成員に変更が生じた場合でも、鍵を変更する必要がない。', '公開鍵方式では、機密性の高い秘密鍵と、完全性・可用性の高い公開鍵が用いられる。'])}

# quiz5_l_dic = {1:np.array(['他人にDoS攻撃を行った場合について、正しいものを一つ選んで下さい。', '他人にDoS攻撃を行った場合、刑法に従って処罰を受ける。', '他人にDoS攻撃を行った場合、電子計算機損壊等業務妨害罪に問われる。', '他人にDoS攻撃を行った場合、業務妨害の罪に問われる。']),
#              2:np.array(['サイバーセキュリティ基本法について、正しいものを一つ選んで下さい。', 'サイバーセキュリティ基本法は、サイバーセキュリティに関する施策を総合的に推進するため、国の責務等を明らかにするなどして、サイバーセキュリティ戦略の策定などの基本となる事項等を規定している。', 'サイバーセキュリティ基本法は、サイバーセキュリティ対策に関する国の責務や基本方針を定め、国の主導的な役割を明確化にしている。', 'サイバーセキュリティ基本法は、サイバーセキュリティ対策に関する国の責務や基本方針を定めた初めての法律である。']),
#              3:np.array(['マルウェアを配布した場合について、正しいものを一つ選んで下さい。', 'マルウェアを配布すると、不正指令電磁的記録の犯罪に問われる。', 'マルウェアを配布すると、不正指令電磁的記録を提供したとみなされる。', 'マルウェアを配布すると、刑法に従って処罰を受ける。']),
#              4:np.array(['不正アクセス行為の禁止等に関する法律について、正しいものを一つ選んで下さい。', '不正アクセス行為の禁止等に関する法律は、他人のログインのための情報を不正に保管する行為の禁止、処罰等が規定されている。', '不正アクセス行為の禁止等に関する法律は、他人のログインのための情報を不正に取得する行為の禁止、処罰等が規定されている。', '不正アクセス行為の禁止等に関する法律は、不正アクセス行為を助長する行為の禁止、処罰等が規定されている。']),
#              5:np.array(['個人情報の保護に関する法律について、正しいものを一つ選んで下さい。', '個人情報とは、例えば、住所や顔写真である。', '個人情報とは、例えば、パスポートに記載されている情報である。', '個人情報とは、例えば、氏名や住所である。'])}

# quiz6_l_dic = {1:np.array(['知的財産権と著作権の関係について、正しいものを一つ選んで下さい。', '知的財産権は、知的な創作活動によって何かを創り出した人に対して付与される権利である。', '著作権は知的財産権の一つである。', '知的財産権は、知的な創作活動が他人に無断で利用されない権利である。']),
#              2:np.array(['著作者人格権と著作権について、正しいものを一つ選んで下さい。', '著作権は、その一部又は全部を譲渡したり相続したりすることができる。', '著作権は、財産的利益を守るための権利である。', '著作者人格権は、著作者が精神的に傷つけられないようにするための権利である。']),
#              3:np.array(['著作物について、当てはまるものを一つ選んで下さい。', '音楽', '小説', 'コンピュータ・プログラム']),
#              4:np.array(['著作権（財産権）について、当てはまるものを一つ選んで下さい。', '複製権', '二次的著作物の利用権', '貸与権']),
#              5:np.array(['著作物等の例外的な無断利用について、正しいものを一つ選んで下さい。', '大学での授業での利用', '聴覚障害者等向けの字幕の作成での利用', '私的使用のための自宅での複製利用'])}

# quiz7_l_dic = {1:np.array(['ICTを使わない脅威について、正しいものを一つ選んで下さい。', '非常時に電話でパスワードを聞き出される。', '背後から盗みとられたパスワードで不正アクセスされる。', 'モニターに貼っていた付箋に書かれていたパスワードを盗み見られる。']),
#              2:np.array(['ICTを使わない方法で、ICTに与える脅威について、正しいものを一つ選んで下さい。', '画面の指の跡からパターンを類推され不正アクセスされる。', '紙に書いていたパスワードが盗み見られ、スパムメールを送られる。', '複数の組織で分担されるシステム・サービスの企画・設計・製造・運用・保守・利用・提供等のプロセスのことである。']),
#              3:np.array(['個人を特定される可能性が最も高いものを一つ選んで下さい。', '匿名のブログに掲載された写真', 'SNS の秘密の ID で公開している自撮りの写真', 'SNS の秘密の ID の友人の写真']),
#              4:np.array(['ICT環境上で匿名性の高いものを一つ選んで下さい。', '本人確認が不要の無料のブログ', '本人確認が不要のフリーメール', '本人確認が不要の無料のSNS']),
#              5:np.array(['ICT環境が乗っ取られた後、対応が困難なものを一つ選んで下さい。', '漏えいしたパスワードで無料のブログで勝手な発信をされた。', '漏えいしたパスワードでフリーメールが不正に使用された。', '漏えいしたパスワードで無料の SNS で勝手な発信をされた。'])}


# %%
# 小テストの内容をpickleにしておく
# import pickle
# # %%
# with open('./pickle/quiz1_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz1_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz2_1_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz2_1_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz2_2_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz2_2_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz3_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz3_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz4_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz4_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz5_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz5_dic, f)                   # オブジェクトをシリアライズ 
# # %%
# with open('./pickle/quiz6_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz6_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz7_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz7_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz1_l_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz1_l_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz2_1_l_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz2_1_l_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz2_2_l_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz2_2_l_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz3_l_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz3_l_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz4_l_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz4_l_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz5_l_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz5_l_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz6_l_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz6_l_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/quiz7_l_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(quiz7_l_dic, f)                   # オブジェクトをシリアライズ

# %%
