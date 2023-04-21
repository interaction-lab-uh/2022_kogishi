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
import pickle
from datetime import datetime
# %%
# データの読み込み
df = pd.read_csv('./reading_v2.csv')
df['cid_page'] = df['contents_id'] + '_' + df['page_no'].astype(str).str.zfill(3)
df
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
import sys
import pyocr.builders
from PIL import Image, ImageEnhance
import pyocr
# %%
path_tesseract = "C:\\Program Files (x86)\\Tesseract-OCR"
if path_tesseract not in os.environ["PATH"].split(os.pathsep):
    os.environ["PATH"] += os.pathsep + path_tesseract

# %%
# 画像をテキストに変換するためのコード
pyocr.tesseract.TESSERACT_CMD = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("OCRツールが見つかりませんでした")
    sys.exit(1)
tool = tools[0]
print("use tool :", tool.get_name())

langs = tool.get_available_languages()
print(langs)

# txt = tool.image_to_string(Image.open(png_dir + '/' + "7eacb532570ff6858afd2723755ff790_066.png"), lang="jpn")
# txt = txt.replace(' ', '')
# print(txt)
# %%
# img = Image.open('./png/3c7781a36bcd6cf08c11a970fbe0e2a6_002.png')
# # %%
# builder = pyocr.builders.TextBuilder(tesseract_layout=6)
# # tesseract_layout：精度の強さデフォルト３、あげれば時間かかる
# img_g = img.convert('L') #Gray変換
# enhancer= ImageEnhance.Contrast(img_g) #コントラストを上げる
# img_con = enhancer.enhance(2.0) #コントラストを上げる
# result = tool.image_to_string(img_con, lang="jpn", builder=builder).replace(' ', '')

# print(result)

# img_con.show()
# %%
img_dir = './img_text'
png_dir = './png'
contents1_dic = {}
for i in range(90):
    if os.path.isfile(img_dir + '/' + contents_1 + '_' + str(i).zfill(3) + '.npz'):
        npz = np.load(img_dir + '/' + contents_1 + '_' + str(i).zfill(3) + '.npz')
        contents1_dic[i] = npz['text']
    elif os.path.isfile(png_dir + '/' + contents_1 + '_' + str(i).zfill(3) + '.png'):
        txt = tool.image_to_string(Image.open(png_dir + '/' + contents_1 + '_' + str(i).zfill(3) + '.png'), lang="jpn")
        txt = txt.replace(' ', '')
        contents1_dic[i] = np.array(txt.split())
contents1_dic
# %%
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import UnicodeNormalizeCharFilter
from janome.tokenfilter import POSKeepFilter, POSStopFilter, CompoundNounFilter, LowerCaseFilter
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from math import log
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import termextract.janome
import termextract.mecab
import termextract.core
import termextract.japanese_plaintext
import collections
import MeCab
import regex
# %%
# TF-IDFやってみる
# まずは形態素解析
con1_txt_list = list(contents1_dic.values())
con1_txt_list
# %%
tokenizer = Tokenizer()
token_filters = [POSStopFilter(['記号','助詞','助動詞','動詞'])]
# a = Analyzer(tokenizer=tokenizer, token_filters=token_filters)
# %%
txt_list = []
for n in con1_txt_list:
    for txt in n:
        txt_list.append(txt)
txt_list
# %%
for n in con1_txt_list:
    for sentence in n:
        print("=============================================")
        print(sentence)

        for token in tokenizer.tokenize(sentence):
            print("    " + str(token))
# %%
# 形態素解析後のリスト
# ただこれそのままやと繋がってる部分が繋がっていない判定になってるところあるからどうしようかなあ
# とりあえず記号だけ抜いてそれぞれの文章をページごとにlistにしてそれをlistに突っ込む
# sentence_list = []
# for n in con1_txt_list:
#     tango_list = []
#     for sentence in n:
#         for token in tokenizer.tokenize(sentence, wakati=True):
#             if token not in ['、', '。', '(', ')', '＝', ':', '[', ']', '.', '…', '......', '―', '【', '】', '-', '/', '*', '！', '？•', '・', '•',  '？', '「', '」', ' ', '→', '（', '）', '=']:
#                 tango_list.append(token)
#     sentence_list.append(tango_list)
# sentence_list
# %%
# 設定
vectorizer = TfidfVectorizer(max_df=0.9) # tf-idfの計算
#                            ^ 文書全体の90%以上で出現する単語は無視する
# %%
# TF-IDFやってみた
# skitlearnを使用したヴァージョンかな
# for n in con1_txt_list:
#     X = vectorizer.fit_transform(n)
#     words = vectorizer.get_feature_names()
#     for doc_id, vec in zip(range(len(n)), X.toarray()):
#         print('doc_id:', doc_id)
#         for w_id, tfidf in sorted(enumerate(vec), key=lambda x: x[1], reverse=True):
#             lemma = words[w_id]
#             print('\t{0:s}: {1:f}'.format(lemma, tfidf))
# %%
# 試しに色々やってみる
# やりたいこと ->　キーワードリストをページ単位ではなく、講義単位でやりたい
# ただ講義単位でやるとおそらくテーマごととはちょっと違うくなる気がする
con_tfidf_list = []
for n in con1_txt_list:
    X = vectorizer.fit_transform(n)
    words = vectorizer.get_feature_names()
    con_tfidf_list.append(" ".join(words))
con_tfidf_list