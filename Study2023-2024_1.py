#%%
# from sklearn import datasets
import datasets
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = pd.DataFrame(iris.target, columns=["target"])
df = pd.concat([data, target], axis=1)



# データの読み込み
iris = datasets.load_iris()

# データを特徴量とラベルに分ける
X = iris.data
y = iris.target

# データを訓練用と評価用に分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルを選択し、訓練する
clf = svm.SVC()
clf.fit(X_train, y_train)

# モデルを評価する
y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# データのロード
iris = datasets.load_iris()

X = iris.data[:, :2]  # 最初の2つの特徴量を使います
y = iris.target

# モデルを作成
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)

# 決定境界を描くためのメッシュを生成
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))

plt.subplot(1, 1, 1)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Machine')
plt.show()