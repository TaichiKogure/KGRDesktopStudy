import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import numpy as np

# CSVファイルを読み込む
df = pd.read_csv('LiC6_OCV2.csv')

# XとYのデータ抜き出し
data = df[['x', 'Potential(V)']].values

# Sort data by x
data = data[data[:,0].argsort()]

x = data[:, 0]
y = data[:, 1]

# データをスプライン補間

spline = UnivariateSpline(x, y, s=68)

# 曲線上のデータポイント
x_range = np.linspace(min(x), max(x), num=1000)
y_range = spline(x_range)

# プロット
plt.scatter(x, y, label='Original Data', alpha=0.6)
plt.plot(x_range, y_range, color='red', label='Spline Interpolation')
plt.xlabel('x')
plt.ylabel('Potential(V)')
plt.title('Spline Interpolation of Data')
plt.ylim([0, 1]) # ここでyminとymaxを任意の範囲に変更します
plt.legend()
plt.show()