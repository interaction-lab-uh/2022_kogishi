# %%
# npzファイル,画像から文章を抜き出し、辞書型にする関数
def make_contents_dic(contentsnum):
    contents_dic = {}
    for i in range(90):
        if os.path.isfile(img_dir + '/' + contentsnum + '_' + str(i).zfill(3) + '.npz'):
            npz = np.load(img_dir + '/' + contentsnum + '_' + str(i).zfill(3) + '.npz')
            contents_dic[i] = npz['text']
        elif os.path.isfile(png_dir + '/' + contentsnum + '_' + str(i).zfill(3) + '.png'):
            txt = tool.image_to_string(Image.open(png_dir + '/' + contentsnum + '_' + str(i).zfill(3) + '.png'), lang="jpn")
            txt = txt.replace(' ', '')
            contents_dic[i] = np.array(txt.split())
    return contents_dic
# %%
# npzファイル,画像から文章を抜き出し、辞書型にする関数(高性能版)
def make_contents_dic_hi(contentsnum):
    pyocr.tesseract.TESSERACT_CMD = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    contents_dic = {}
    tools = pyocr.get_available_tools()
    tool = tools[0]
    langs = tool.get_available_languages()
    builder = pyocr.builders.TextBuilder(tesseract_layout=6)
    for i in range(90):
        if os.path.isfile(img_dir + '/' + contentsnum + '_' + str(i).zfill(3) + '.npz'):
            npz = np.load(img_dir + '/' + contentsnum + '_' + str(i).zfill(3) + '.npz')
            contents_dic[i] = npz['text']
        elif os.path.isfile(png_dir + '/' + contentsnum + '_' + str(i).zfill(3) + '.png'):
            img = Image.open(png_dir + '/' + contentsnum + '_' + str(i).zfill(3) + '.png')
            img_g = img.convert('L') #Gray変換
            enhancer= ImageEnhance.Contrast(img_g) #コントラストを上げる
            img_con = enhancer.enhance(2.0) #コントラストを上げる
            txt = tool.image_to_string(img_con, lang="jpn", builder=builder).replace(' ', '')
            contents_dic[i] = np.array(txt.split())
    return contents_dic
# %%
# npzファイル,画像から文章を抜き出し、辞書型にする関数
def make_contents_dic_npz(contentsnum):
    contents_dic = {}
    for i in range(1, 90):
        if os.path.isfile(img_dir + '/' + contentsnum + '_' + str(i).zfill(3) + '.npz'):
            npz = np.load(img_dir + '/' + contentsnum + '_' + str(i).zfill(3) + '.npz')
            contents_dic[i] = npz['text']
    return contents_dic
# %%
# contents1_dic = make_contents_dic_npz(contents_1)
# contents2_dic = make_contents_dic_npz(contents_2)
# contents3_dic = make_contents_dic_npz(contents_3)
# contents4_dic = make_contents_dic_npz(contents_4)
# contents5_dic = make_contents_dic_npz(contents_5)
# contents6_dic = make_contents_dic_npz(contents_6)
# contents7_dic = make_contents_dic_npz(contents_7)
# contents0_dic = make_contents_dic_npz(contents_0)
# contentsplus_dic = make_contents_dic_npz(contents_plus)
# %%
# with open('./pickle/contents1_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(contents1_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/contents2_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(contents2_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/contents3_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(contents3_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/contents4_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(contents4_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/contents5_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(contents5_dic, f)                   # オブジェクトをシリアライズ 
# # %%
# with open('./pickle/contents6_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(contents6_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/contents7_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(contents7_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/contents0_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(contents0_dic, f)                   # オブジェクトをシリアライズ
# # %%
# with open('./pickle/contentsplus_dic.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
#     pickle.dump(contentsplus_dic, f)                   # オブジェクトをシリアライズ
# %%
# MeCabで得た結果の基本形だけを出力したい
class IndependentWord:
  def __init__(self, dictpath=None):
    if dictpath:
      self.m = MeCab.Tagger("-d "+dictpath)
    else:
      self.m = MeCab.Tagger()   
    self.kanaalpha = regex.compile(r'[\p{Script=Hiragana}\p{Script=Katakana}ーA-Za-z]')
    self.number = regex.compile("[0-9０-９]+")

  class Constant:
    BASIC = "basic_form" #原型
    SURFACE = "surface_form" #表層型
    POS = "pos" #品詞
    POS_DETAIL_1 = "pos_detail_1" #品詞詳細1
    POS_DETAIL_2 = "pos_detail_2" #品詞詳細2
    POS_DETAIL_3 = "pos_detail_3" #品詞詳細3
    PRONUNCIATION = "pronunciation" #発音
    READING = "reading" #読み
    CONJUGATED_TYPE = "conjugated_type" #活用
    CONJUGATED_FORM = "conjugated_form" #活用形

  #mecabの出力行をobjectに変換
  #mecabの出力フォーマットに応じて適宜修正する
  def mecabLineToDict(self, line):
    surface, tmp = line.split("\t")
    others = tmp.split(",")

    Const = self.Constant

    return {
      Const.SURFACE: surface,
      Const.POS: others[0],
      Const.POS_DETAIL_1: others[1],
      Const.POS_DETAIL_2: others[2],
      Const.POS_DETAIL_3: others[3],
      Const.CONJUGATED_TYPE: others[4],
      Const.CONJUGATED_FORM: others[5],
      Const.BASIC: others[6],
      Const.READING: others[7],
      Const.PRONUNCIATION: others[8]
      }
  #自立語かどうかの判定
  def isIndependentWord(self, token):
    pos = token[self.Constant.POS]
    pos_detail_1 = token[self.Constant.POS_DETAIL_1]
    if pos == "名詞" and pos_detail_1 in ['一般','固有名詞','サ変接続','形容動詞語幹']: #用途によっては「副詞可能」を足しても良いかもしれません
      return True
    elif pos == '形容詞' and pos_detail_1 == '自立':
      return True
    elif pos == "副詞" and pos_detail_1 == "一般":
      return True
    elif pos == "動詞" and pos_detail_1 == "自立":
      return True
    else:
      return False
  #カナやアルファベット１文字や数字出ないかの判定
  def isReliableWord(self, token):
    surface = token[self.Constant.SURFACE]
    if self.number.fullmatch(surface):
      return False
    elif self.kanaalpha.fullmatch(surface):
      return False
    else:
      return True

  #自立語の原型を抽出
  def extract(self,text):
    lines = self.m.parse(text).splitlines()[:-1]
    tokens = [self.mecabLineToDict(line) for line in lines]

    independent_words = []
    for token in tokens:
      if self.isIndependentWord(token) and self.isReliableWord(token):
        surface = token[self.Constant.SURFACE]
        basic = token[self.Constant.BASIC]
        if basic == "*":
          independent_words.append(surface)
        else:
          independent_words.append(basic)
    return independent_words    
# %%
# 直す必要のあるコード
# なぜうまくいかん…なぜ…out of range してる。ただこれはとても内部のお話で私にはわからない
if __name__ == "__main__":
    txt_list_1 = list(contents1_dic.values())
    waka1_list = []
    for n in txt_list_1:
        ap_list = []
        for i in range(len(n)):
            idptwd = IndependentWord()
            result = idptwd.extract(n[i])
            ap_list.append(result)
        waka1_list.append(ap_list)
# %%
# # 画像の座標をだすためのテスト
# import cv2
# # %%
# def onMouse(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, y)

# %%
# img = cv2.imread('./png/7eacb532570ff6858afd2723755ff790_015.png')
# cv2.imshow('week3_15', img)
# cv2.setMouseCallback('week3_15', onMouse)
# cv2.waitKey(0)
# # クリックしたら出るようにはなったけど全部やるの嫌なんですけど…

# %%
# ここからファインチューニングの話
# 実行しないで
# transformer = models.Transformer("sonoisa/sentence-bert-base-ja-mean-tokens-v2")
# pooling = models.Pooling(
#     transformer.get_word_embedding_dimension(),
#     pooling_mode_mean_tokens=True,
#     pooling_mode_cls_token=False,
#     pooling_mode_max_tokens=False
# )
# model2 = SentenceTransformer(modules=[transformer, pooling])
# %%
# 3. データの読み込み、パラメータの設定
# triplet_reader = TripletReader(".")
# train_dataset = SentencesDataset(
#         triplet_reader.get_examples("triplet.tsv"),
#         model=model2,
# )
# %%
# BATCH_SIZE = 8
# NUM_EPOCH = 15
# EVAL_STEPS = 100
# WARMUP_STEPS = int(len(train_dataset) // BATCH_SIZE * 0.1)
# BATCH_SIZE = 16
# NUM_EPOCHS = 1
# EVAL_STEPS = 1000
# WARMUP_STEPS = int(len(train_dataset) // BATCH_SIZE * 0.1)
# OUTPUT_PATH = "./sbert_stair"
# train_dataloader = DataLoader(
#         train_dataset,
#         shuffle=False,
#         batch_size=BATCH_SIZE,
# )
# train_loss = TripletLoss(
#         model=model2,
#         distance_metric=TripletDistanceMetric.EUCLIDEAN,
#         triplet_margin=1,
# )
# %%
# model2.fit(
#         train_objectives=[(train_dataloader, train_loss)],
#         epochs=NUM_EPOCHS,
#         evaluation_steps=EVAL_STEPS,
#         warmup_steps=WARMUP_STEPS,
#         output_path="./sbert",
#         )
# %%
# 文字数少ないとこきってみる？
# 文字数少ないってどっからやろ20字くらい？
# あと「出典」からはじまっている文章消してもいいかも、ノイズになりそう
# %%
# transformer = models.Transformer('sonoisa/sentence-bert-base-ja-mean-tokens-v2')
# pooling = models.Pooling(
#     transformer.get_word_embedding_dimension(),
#     pooling_mode_mean_tokens=True,
#     pooling_mode_cls_token=False,
#     pooling_mode_max_tokens=False
# )
# model = SentenceTransformer(modules=[transformer, pooling])
# %%
# sentences = ['吾輩は猫である',  '本日は晴天なり']
# embeddings = model.encode(sentences)
# for i, embedding in enumerate(embeddings):
#   print("[%d] : %s" % (i, embedding.shape, ))
# %%