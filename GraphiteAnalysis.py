import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ファイルを読み込む
df = pd.read_csv('LiC6_OCV2.csv')

# x軸とy軸のデータを取得
x = df['x'].values
y = df['Potential(V)'].values

# データの微分を取得
def my_gradient(y, x, dx=1):
    n = len(y)
    gradient = np.zeros(n)

    for i in range(n):
        if i < dx:
            # Forward difference for the first few elements
            dy = y[i+dx] - y[i]
            dx_real = x[i+dx] - x[i]
            gradient[i] = dy / dx_real
        elif i >= n - dx:
            # Backward difference for the last few elements
            dy = y[i] - y[i-dx]
            dx_real = x[i] - x[i-dx]
            gradient[i] = dy / dx_real
        else:
            # Central difference for the other elements
            dy_forward = y[i+dx] - y[i]
            dx_forward = x[i+dx] - x[i]
            dy_backward = y[i] - y[i-dx]
            dx_backward = x[i] - x[i-dx]
            gradient[i] = (dy_forward / dx_forward + dy_backward / dx_backward) / 2
    return gradient
# numpyのgradient関数は数値的に微分を計算する
# pandasのSeriesへ変換
y_series = pd.Series(y)
# 移動平均でスムージング
# window_sizeを調節することでスムージングの程度を制御できます。
window_size = 50
y_smooth = y_series.rolling(window_size, center=True).mean()

# データの微分を取得
y_diff = my_gradient(y_smooth.values, x, dx=20)
y_diff_series = pd.Series(y_diff)# pandasのSeriesへ変換
# 移動平均でスムージング
window_size =10 # window_sizeを調節することでスムージングの程度を制御できます。
y_diff_smooth = y_diff_series.rolling(window_size, center=True).mean()

# プロットを作成
plt.subplot(2, 1, 1)
plt.plot(x, y_smooth, label='Original')
plt.legend()
plt.ylim([0.01, 0.6])

plt.subplot(2, 1, 2)
plt.plot(x, y_diff_smooth, label='Derivative')
plt.legend()
# 追加：Derivativeグラフのy軸の範囲を設定
plt.ylim([-1, 1]) # ここでyminとymaxを任意の範囲に変更します
# グラフを表示
plt.show()