# %%
# !pip install janome
# !pip install wordcloud
# !pip install pyocr
# %%
from cmath import nan
from pydoc import describe
from re import A
from cv2 import merge
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import matplotlib as mpl
import pickle
from sympy import content
import sys
import pyocr.builders
from PIL import Image, ImageEnhance
import pyocr
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
# データの読み込み
df = pd.read_csv('./reading_v2.csv')
df
# %%
tokenizer = Tokenizer()
token_filters = [POSStopFilter(['記号','助詞','助動詞','動詞'])]
# a = Analyzer(tokenizer=tokenizer, token_filters=token_filters)
# %%
# 設定
vectorizer = TfidfVectorizer(max_df=0.9) # tf-idfの計算
#                            ^ 文書全体の90%以上で出現する単語は無視する
# %%
# TF,IDF,TF-IDFの関数、関数なんていらんかったんや…
def tf(t, d):
    return d.count(t) / len(d)


def df_fun(t, docs):
    df = 0
    for doc in docs:
        df += 1 if t in doc else 0
    return df


def idf(t, docs):
    N = len(docs)
    return log(N/df(t, docs)) + 1

def vectorizer_transform(text):
    
    # 単語を生成
    words = []
    for s in text:
        words += s.split(' ')
    words = list(set(words))
    words.sort()

    tf_idf = []
    for txt in text:
        line_tf_idf = []
        for w in words:
            # tfを計算
            tf_v = tf(w, txt)
 
            # idfを計算
            idf_v = idf(w, text)
 
            # tfとidfを乗算
            line_tf_idf.append(tf_v * idf_v)
        tf_idf.append(line_tf_idf)
    return tf_idf

# 単純にTF-IDFだすならこれでよくね関数
def tfidf(t,d):
    return tf(t,d) * idf(t)

# %%
def TFIDF5(con_tfidf_list):
    pd.set_option('display.max_columns', None)
    pd.options.display.float_format = '{:0.2f}'.format

    count_vectorizer = CountVectorizer()
    bow = count_vectorizer.fit_transform(con_tfidf_list).toarray()

    # TF を計算してるところ (行方向の処理)
    # 文書に含まれる単語の数をカウントする
    number_of_words = np.sum(bow, axis=1, keepdims=True)
    # 文書の中での単語の頻度を計算する
    tf = bow / number_of_words
    
    # IDF を計算してるところ (列方向の処理)
    # 文書の数をカウントする
    number_of_docs = len(con_tfidf_list)
    # その単語が一つでも含まれる文書の数をカウントする
    number_of_docs_contain_word = np.count_nonzero(bow, axis=0)
    # 単語の珍しさを計算する
    idf = np.log(number_of_docs / number_of_docs_contain_word) + 1

    # TF-IDF を計算してるところ
    print('--- TF-IDF ---')
    # TF と IDF をかける
    tfidf = tf * idf
    df_tfidf = pd.DataFrame(tfidf,
                      columns=count_vectorizer.get_feature_names())
    return df_tfidf
# %%
# 辞書型のvalueをそのままlist化する
def make_txt_list(contents_dic):
    con_txt_list = list(contents_dic.values())
    txt_list = []
    for n in con_txt_list:
        for txt in n:
            txt_list.append(txt)
    return txt_list
# %%
# 形態素解析のための関数（janome）
def made_tfidf_list_old(contents_dic):
    con_txt_list1 = list(contents_dic.values())
    # vectorizer = TfidfVectorizer(max_df=0.9)
    con_tfidf_list = []
    for n in con_txt_list1:
        X = vectorizer.fit_transform(n)
        words = vectorizer.get_feature_names()
        con_tfidf_list.append(" ".join(words))
    return con_tfidf_list
# %%
# 形態素解析のための関数（janome）(改良版)
def made_tfidf_list(contents_dic):
    con_txt_list1 = list(contents_dic.values())
    con_tfidf_list = []
    for n in con_txt_list1:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(n)
        words = vectorizer.get_feature_names()
        con_tfidf_list.append(" ".join(words))
    return con_tfidf_list
# %%
# # 形態素解析のための関数（janome）（上みたいなlistじゃなくmecabと同じようにしたらどうなる？）
# # あんま変わんなかったので特に気にしなくていいです
# def make_janome_list(txt_list):
#     con_janome_list = []
#     vectorizer = TfidfVectorizer(max_df=0.9)
#     X = vectorizer.fit_transform(txt_list)
#     words = vectorizer.get_feature_names()
#     con_janome_list.append(" ".join(words))
#     return con_janome_list
# %%
# 形態素解析のための関数（MeCab ver.）
def make_mecab_list(txt_list):
    tagger = MeCab.Tagger("-Owakati")
    wakachi_list = tagger.parse(' '.join(txt_list)).split()
    return wakachi_list
# %%
def TFIDF_main(tfidf_list):
    pd.set_option('display.max_columns', None)
    pd.options.display.float_format = '{:0.2f}'.format

    count_vectorizer = CountVectorizer()
    bow = count_vectorizer.fit_transform(tfidf_list).toarray()

    # TF を計算してるところ (行方向の処理)
    # 文書に含まれる単語の数をカウントする
    number_of_words = np.sum(bow, axis=1, keepdims=True)
    # 文書の中での単語の頻度を計算する
    tf = bow / number_of_words
    
    # IDF を計算してるところ (列方向の処理)
    # 文書の数をカウントする
    number_of_docs = len(tfidf_list)
    # その単語が一つでも含まれる文書の数をカウントする
    number_of_docs_contain_word = np.count_nonzero(bow, axis=0)
    # 単語の珍しさを計算する
    idf = np.log(number_of_docs / number_of_docs_contain_word) + 1

    # TF-IDF を計算してるところ
    print('--- TF-IDF ---')
    # TF と IDF をかける
    tfidf = tf * idf
    df_tfidf = pd.DataFrame(tfidf,
                      columns=count_vectorizer.get_feature_names())
    return df_tfidf




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
# そのままリストにしている
txt1_list = make_txt_list(contents1_dic)
# 区切ったやつをスペースでくっつけてる
con1_tfidf_list = made_tfidf_list(contents1_dic)
# やたらめったら細かい
con1_mecab_list = make_mecab_list(txt1_list)
# %%
# 全部のTF-IDF値をだすよ
df_tfidf_1 = TFIDF_main(txt1_list)
df_tfidf1 = TFIDF_main(con1_tfidf_list)
df_tfidf1_1 = TFIDF_main(con1_mecab_list)
# %%
# そのままlistにしたよ
key_list_1 = list(df_tfidf_1)
key_list1 = list(df_tfidf1)
key_list1_1 = list(df_tfidf1_1)
# %%
print('janomeの方にしか存在しないもの')
print(set(key_list1) - set(key_list1_1))
print('\n')
print('mecabの方にしか存在しないもの')
print(set(key_list1_1) - set(key_list1))
print('\n')
print('共通して存在しているもの')
print(set(key_list1) & set(key_list1_1))
# %%
txt2_list = make_txt_list(contents2_dic)
con2_tfidf_list = made_tfidf_list(contents2_dic)
con2_mecab_list = make_mecab_list(txt2_list)
# %%
df_tfidf_2 = TFIDF_main(txt2_list)
df_tfidf2 = TFIDF_main(con2_tfidf_list)
df_tfidf2_2 = TFIDF_main(con2_mecab_list)
# %%
key_list_2 = list(df_tfidf_2)
key_list2 = list(df_tfidf2)
key_list2_2 = list(df_tfidf2_2)
# %%
print('janomeの方にしか存在しないもの')
print(set(key_list2) - set(key_list2_2))
print('\n')
print('mecabの方にしか存在しないもの')
print(set(key_list2_2) - set(key_list2))
print('\n')
print('共通して存在しているもの')
print(set(key_list2) & set(key_list2_2))
# %%
txt3_list = make_txt_list(contents3_dic)
con3_tfidf_list = made_tfidf_list(contents3_dic)
con3_mecab_list = make_mecab_list(txt3_list)
# %%
df_tfidf_3 = TFIDF_main(txt3_list)
df_tfidf3 = TFIDF_main(con3_tfidf_list)
df_tfidf3_3 = TFIDF_main(con3_mecab_list)
# %%
key_list_3 = list(df_tfidf_3)
key_list3 = list(df_tfidf3)
key_list3_3 = list(df_tfidf3_3)
# %%
print('janomeの方にしか存在しないもの')
print(set(key_list3) - set(key_list3_3))
print('\n')
print('mecabの方にしか存在しないもの')
print(set(key_list3_3) - set(key_list3))
print('\n')
print('共通して存在しているもの')
print(set(key_list3) & set(key_list3_3))
# %%
txt4_list = make_txt_list(contents4_dic)
con4_tfidf_list = made_tfidf_list(contents4_dic)
con4_mecab_list = make_mecab_list(txt4_list)
# %%
df_tfidf_4 = TFIDF_main(txt4_list)
df_tfidf4 = TFIDF_main(con4_tfidf_list)
df_tfidf4_4 = TFIDF_main(con4_mecab_list)
# %%
key_list_4 = list(df_tfidf_4)
key_list4 = list(df_tfidf4)
key_list4_4 = list(df_tfidf4_4)
# %%
print('janomeの方にしか存在しないもの')
print(set(key_list4) - set(key_list4_4))
print('\n')
print('mecabの方にしか存在しないもの')
print(set(key_list4_4) - set(key_list4))
print('\n')
print('共通して存在しているもの')
print(set(key_list4) & set(key_list4_4))
# %%
txt5_list = make_txt_list(contents5_dic)
con5_tfidf_list = made_tfidf_list(contents5_dic)
con5_mecab_list = make_mecab_list(txt5_list)
# %%
df_tfidf_5 = TFIDF_main(txt5_list)
df_tfidf5 = TFIDF_main(con5_tfidf_list)
df_tfidf5_5 = TFIDF_main(con5_mecab_list)
# %%
key_list_5 = list(df_tfidf_5)
key_list5 = list(df_tfidf5)
key_list5_5 = list(df_tfidf5_5)
# %%
print('janomeの方にしか存在しないもの')
print(set(key_list5) - set(key_list5_5))
print('\n')
print('mecabの方にしか存在しないもの')
print(set(key_list5_5) - set(key_list5))
print('\n')
print('共通して存在しているもの')
print(set(key_list5) & set(key_list5_5))
# %%
txt6_list = make_txt_list(contents6_dic)
con6_tfidf_list = made_tfidf_list(contents6_dic)
con6_mecab_list = make_mecab_list(txt6_list)
# %%
df_tfidf_6 = TFIDF_main(txt6_list)
df_tfidf6 = TFIDF_main(con6_tfidf_list)
df_tfidf6_6 = TFIDF_main(con6_mecab_list)
# %%
key_list_6 = list(df_tfidf_6)
key_list6 = list(df_tfidf6)
key_list6_6 = list(df_tfidf6_6)
# %%
print('janomeの方にしか存在しないもの')
print(set(key_list6) - set(key_list6_6))
print('\n')
print('mecabの方にしか存在しないもの')
print(set(key_list6_6) - set(key_list6))
print('\n')
print('共通して存在しているもの')
print(set(key_list6) & set(key_list6_6))
# %%
txt7_list = make_txt_list(contents7_dic)
con7_tfidf_list = made_tfidf_list(contents7_dic)
con7_mecab_list = make_mecab_list(txt7_list)
# %%
df_tfidf_7 = TFIDF_main(txt7_list)
df_tfidf7 = TFIDF_main(con7_tfidf_list)
df_tfidf7_7 = TFIDF_main(con7_mecab_list)
# %%
key_list_7 = list(df_tfidf_7)
key_list7 = list(df_tfidf7)
key_list7_7 = list(df_tfidf7_7)
# %%
print('janomeの方にしか存在しないもの')
print(set(key_list7) - set(key_list7_7))
print('\n')
print('mecabの方にしか存在しないもの')
print(set(key_list7_7) - set(key_list7))
print('\n')
print('共通して存在しているもの')
print(set(key_list7) & set(key_list7_7))




# %%
txt1_s = ' '.join(txt1_list)
# with open("txt1_list.txt", 'wt') as f:
#     f.write(txt1_s)
# %%
# MeCabでの形態素解析
tagger = MeCab.Tagger()
parse1 = tagger.parse(txt1_s)
print(parse1)

# %%
# 形態素解析結果方式
# MeCabのかたちを突っ込む方法
frequency = termextract.mecab.cmp_noun_dict(parse1)
LR = termextract.core.score_lr(frequency,
         ignore_words=termextract.mecab.IGNORE_WORDS,
         lr_mode=1, average_rate=1
     )
term_imp = termextract.core.term_importance(frequency, LR)

# 重要度が高い順に並べ替えて出力
data_collection = collections.Counter(term_imp)
for cmp_noun, value in data_collection.most_common():
    print(termextract.core.modify_agglutinative_lang(cmp_noun), value, sep="\t")

# %%
# ストップワード方式
# 複合語を抽出し、重要度を算出
frequency = termextract.japanese_plaintext.cmp_noun_dict(txt1_s)
LR = termextract.core.score_lr(frequency,
         ignore_words=termextract.japanese_plaintext.IGNORE_WORDS,
         lr_mode=1, average_rate=1
     )
term_imp = termextract.core.term_importance(frequency, LR)

# 重要度が高い順に並べ替えて出力
data_collection = collections.Counter(term_imp)
for cmp_noun, value in data_collection.most_common():
    print(termextract.core.modify_agglutinative_lang(cmp_noun), value, sep="\t")
# %%
# 形態素解析器で日本語処理
t = Tokenizer()
tokenize_text1 = t.tokenize(txt1_s)
 
# Frequency生成＝複合語抽出処理（ディクショナリとリストの両方可)
frequency = termextract.janome.cmp_noun_dict(tokenize_text1)
 
# FrequencyからLRを生成する
lr = termextract.core.score_lr(
    frequency,
    ignore_words=termextract.mecab.IGNORE_WORDS,
    lr_mode=1, average_rate=1)
 
# FrequencyとLRを組み合わせFLRの重要度を出す
term_imp = termextract.core.term_importance(frequency, lr)
 
# collectionsを使って重要度が高い順に表示
data_collection = collections.Counter(term_imp)
for cmp_noun, value in data_collection.most_common():
    print(termextract.core.modify_agglutinative_lang(cmp_noun), value, sep="\t")
# %%
for i in range(len(txt1_list)):
    if '•' in txt1_list[i]:
        txt1_list[i] = txt1_list[i].replace('•', '')
txt1_list
# %%
tagger = MeCab.Tagger()
txt1_parse = tagger.parse(' '.join(txt1_list))
print(txt1_parse)
# %%
frequency = termextract.mecab.cmp_noun_dict(txt1_parse)
LR = termextract.core.score_lr(frequency,
         ignore_words=termextract.mecab.IGNORE_WORDS,
         lr_mode=1, average_rate=1
     )
term_imp = termextract.core.term_importance(frequency, LR)

# 重要度が高い順に並べ替えて出力
data_collection = collections.Counter(term_imp)
for cmp_noun, value in data_collection.most_common():
    print(termextract.core.modify_agglutinative_lang(cmp_noun), value, sep="\t")
# %%
# MeCabを使って形態素解析を行った結果をTF-IDFにいれてみる
# うまいこといく気しない
tagger = MeCab.Tagger("-Owakati")
wakachi1_list = tagger.parse(' '.join(txt1_list)).split()
df_tfidf_w1 = TFIDF_main(wakachi1_list)
key_list_w1 = list(df_tfidf_w1)
key_list_w1
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
from torch.utils.data import DataLoader
from keras_bert import load_trained_model_from_checkpoint
import sentencepiece as spm
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
text= '今日の東京は雪になるでしょう'
text2vector(text)
# %%
text2vector(contents1_dic[1][0])
# %%
for i in range(len(contents1_dic)):
    print(contents1_dic[i+1])
    print(text2vector(contents1_dic[i+1][0]))
# %%
for i in range(len(contents2_dic)):
    print(contents2_dic[i+1])
    print(text2vector(contents2_dic[i+1][0]))
# %%
for i in range(len(contents3_dic)):
    print(contents3_dic[i+1])
    print(text2vector(contents3_dic[i+1][0]))
# %%
for i in range(len(contents4_dic)):
    print(contents4_dic[i+1])
    print(text2vector(contents4_dic[i+1][0]))
# %%
for i in range(len(contents5_dic)):
    print(contents5_dic[i+1])
    print(text2vector(contents5_dic[i+1][0]))
# %%
for i in range(len(contents6_dic)):
    print(contents6_dic[i+1])
    print(text2vector(contents6_dic[i+1][0]))
# %%
for i in range(len(contents7_dic)):
    print(contents7_dic[i+1])
    print(text2vector(contents7_dic[i+1][0]))
# %%
for i in range(len(contentsplus_dic)):
    print(contentsplus_dic[i+1])
    print(text2vector(contentsplus_dic[i+1][0]))
# %%
# transformer = models.BERT('cl-tohoku/bert-base-japanese-whole-word-masking')

# %%
transformer = models.Transformer('cl-tohoku/bert-base-japanese-whole-word-masking')

pooling = models.Pooling(
    transformer.get_word_embedding_dimension(), 
    pooling_mode_mean_tokens=True, 
    pooling_mode_cls_token=False, 
    pooling_mode_max_tokens=False
)

model = SentenceTransformer(modules=[transformer, pooling])
# %%
sentences = ['吾輩は猫である',  '本日は晴天なり']
embeddings = model.encode(sentences)

for i, embedding in enumerate(embeddings):
  print("[%d] : %s" % (i, embedding.shape, ))
  
# %%
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

sentences = ["暴走したAI", "暴走した人工知能"]
sentence_embeddings = model.encode(sentences, batch_size=8)

print("Sentence embeddings:", sentence_embeddings)



# %%
df = pd.read_excel(report_path, dtype={'text':str,'tokens':str,'label':int,"cause":int,"effect":int,})

train_exapmle = []
labels = []
for index1, row1 in df.iterrows():    
    if row1['cause'] == 1:
        label = 1
    elif row1["effect"] == 1:
        label = 2
    else:
        label = 0
        
    train_exapmle.append(InputExample(texts=[row1.tokens], label=float(label)))
    labels.append(label)
# %%
# !git clone https://github.com/STAIR-Lab-CIT/STAIR-captions
# !tar zxvf STAIR-captions/stair_captions_v1.2.tar.gz
# !ls -lh *.json
# %%
import json
with open("stair_captions_v1.2_val.json", "r", encoding="utf-8") as f:
  json_data_val = json.load(f)
with open("stair_captions_v1.2_train.json", "r", encoding="utf-8") as f:
  json_data_train = json.load(f)
# %%
dataset = {}
ids = []
captions = []

def build_dataset(dataset, json_data):
  num_samples = len(json_data['annotations'])
  for i in range(num_samples):
    anno = json_data['annotations'][i]
    image_id = anno["image_id"]
    image_captions = dataset.get(image_id, [])
    image_captions.append((anno["id"], anno["caption"]))
    ids.append(anno["id"])
    captions.append(anno["caption"])
    dataset[image_id] = image_captions

build_dataset(dataset, json_data_val)
build_dataset(dataset, json_data_train)
# %%
id2idx = {id:idx for idx, id in enumerate(ids)}
# %%
import spacy
import pkg_resources, imp
imp.reload(pkg_resources)

nlp = spacy.load("ja_ginza")

vectors = []
for caption in captions:
  doc = nlp(caption, disable=['ner'])
  vectors.append(doc.vector)

del nlp


