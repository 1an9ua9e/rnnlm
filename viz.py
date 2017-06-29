
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Qt5Agg")
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

plot_name = "rnn-vs-gru-s5000-h50-b15"

#タイトル等で日本語を使えるようにフォントを指定
fp = FontProperties(fname=r'/usr/share/fonts/truetype/fonts-japanese-gothic.ttf')

#プロットごとのラベルを指定
blue_patch = mpatches.Patch(color='blue',label=u'RNN')
red_patch = mpatches.Patch(color='red',label=u'GRU')
sns.plt.legend(handles=[blue_patch,red_patch],prop=fp)

data1 = pd.read_csv("rnn-s5000-h50-b15.csv")
data2 = pd.read_csv("gru-s5000-h50-b15.csv")
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

sns.plt.title(u"RNNとGRUの比較、言語モデル実験",fontproperties=fp)
sns.plt.ylim(0, 140)
sns.plt.xlabel('epoch')
sns.plt.ylabel('perplexity')
sns.plt.savefig(plot_name + ".png")
sns.plt.show()
