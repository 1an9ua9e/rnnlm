
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Qt5Agg")
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

plot_name = "compare-test3000-voc10000-batch15-hidden50-sort1"

#タイトル等で日本語を使えるようにフォントを指定
fp = FontProperties(fname=r'/usr/share/fonts/truetype/fonts-japanese-gothic.ttf')

#プロットごとのラベルを指定
blue_patch = mpatches.Patch(color='blue',label=u'RNN')
red_patch = mpatches.Patch(color='red',label=u'class-based RNN (クラス数 100)')
green_patch = mpatches.Patch(color='green',label=u'BlackOut (ノイズ数 5)')
yellow_patch = mpatches.Patch(color='yellow',label=u'NCE (ノイズ数 50)')
sns.plt.legend(handles=[blue_patch,red_patch,green_patch,yellow_patch],prop=fp)

data1 = pd.read_csv("testPPL-20170829-rnn.csv")
data2 = pd.read_csv("testPPL-20170829-class.csv")
data3 = pd.read_csv("testPPL-20170829-bo.csv")
data4 = pd.read_csv("testPPL-20170829-nce.csv")
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
    color='green')
sns.pointplot(
    x='epoch',
    y='PPL',
    data=data4,
    color='yellow')

sns.plt.title(u"テストデータに対するRNNLM高速化手法の精度比較",fontproperties=fp)
sns.plt.ylim(10, 35)
sns.plt.xlabel('epoch')
sns.plt.ylabel('test perplexity')
sns.plt.savefig(plot_name + ".png")
sns.plt.show()
