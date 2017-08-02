
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Qt5Agg")
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

plot_name = "nce-comparison-001"

#タイトル等で日本語を使えるようにフォントを指定
fp = FontProperties(fname=r'/usr/share/fonts/truetype/fonts-japanese-gothic.ttf')

#プロットごとのラベルを指定
blue_patch = mpatches.Patch(color='blue',label=u'正規化定数30 ノイズ数5')
red_patch = mpatches.Patch(color='red',label=u'正規化定数30(固定) ノイズ数5')
yellow_patch = mpatches.Patch(color='yellow',label=u'正規化定数30 ノイズ数50')
#green_patch = mpatches.Patch(color='green',label=u'RNN')
sns.plt.legend(handles=[blue_patch,red_patch,yellow_patch],prop=fp)

data1 = pd.read_csv("nce-train-s10000-h50-b15-noise5-z30.csv")
data2 = pd.read_csv("nce-train-s10000-h50-b15-noise5-fz30.csv")
data3 = pd.read_csv("nce-train-s10000-h50-b15-noise50-z30.csv")
data4 = pd.read_csv("rnn-s10000-h50-b15.csv")
sns.pointplot(
    x='epoch',
    y='PPL',
    data=data1,
    color='blue')
sns.pointplot(
    x='epoch',
    y='PPL',
    data=data2,
    color='red')
sns.pointplot(
    x='epoch',
    y='PPL',
    data=data3,
    color='yellow')
'''
sns.pointplot(
    x='epoch',
    y='PPL',
    data=data4,
    color='green')
'''
sns.plt.title(u"NCEを用いたRNN言語モデルの比較",fontproperties=fp)
sns.plt.ylim(30, 85)
sns.plt.xlabel('epoch')
sns.plt.ylabel('perplexity')
sns.plt.savefig(plot_name + ".png")
sns.plt.show()
