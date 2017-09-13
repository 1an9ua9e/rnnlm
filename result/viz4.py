
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Qt5Agg")
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

plot_name = "testPPL-compare-2017-09-13"

#タイトル等で日本語を使えるようにフォントを指定
fp = FontProperties(fname=r'/usr/share/fonts/truetype/fonts-japanese-gothic.ttf')

colors = ['blue','red','green','yellow','cyan','magenta','black','white']
l = 5
labels = [u'RNN 隠れ層50',
          u'NCE ノイズ数200',
          u'BlackOut ノイズ数5',
          u'クラス クラス数30',
          u'提案手法 alpha=1.0, eval=500, class=1000']
patches = []
if l < len(colors):
    for i in range(l):
        patches.append(mpatches.Patch(color=colors[i],label=labels[i]))
        
#プロットごとのラベルを指定
'''
blue_patch = mpatches.Patch(color='blue',label=u'正規化定数30 ノイズ数5')
red_patch = mpatches.Patch(color='red',label=u'正規化定数30(固定) ノイズ数5')
yellow_patch = mpatches.Patch(color='yellow',label=u'正規化定数30 ノイズ数50')
green_patch = mpatches.Patch(color='green',label=u'RNN')
green_patch = mpatches.Patch(color='green',label=u'RNN')
sns.plt.legend(handles=[blue_patch,red_patch,yellow_patch],prop=fp)
'''
sns.plt.legend(handles=patches,prop=fp)
my_data = []
my_data.append(pd.read_csv("RNN2017-09-12.csv"))
my_data.append(pd.read_csv("RNNwithNCE2017-09-12.csv"))
my_data.append(pd.read_csv("BlackOut2017-09-12.csv"))
my_data.append(pd.read_csv("classRNN2017-09-12.csv"))
my_data.append(pd.read_csv("EFRNN5-2017-09-12.csv"))
'''
data2 = pd.read_csv("EFRNN2-2017-09-12.csv")
data3 = pd.read_csv("EFRNN3-2017-09-12.csv")
data4 = pd.read_csv("EFRNN4-2017-09-12.csv")
data5 = pd.read_csv("EFRNN5-2017-09-12.csv")
'''
for i in range(l):
    sns.pointplot(
        x='epoch',
        y='testPPL',
        data=my_data[i],
        color=colors[i])
'''
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
'''
sns.pointplot(
    x='epoch',
    y='PPL',
    data=data4,
    color='green')
'''
sns.plt.title(u"RNN言語モデル高速化手法の比較",fontproperties=fp)
sns.plt.ylim(95, 205)
sns.plt.xlabel('epoch')
sns.plt.ylabel('test perplexity')
sns.plt.savefig(plot_name + ".png")
sns.plt.show()
