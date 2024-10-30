# 完全雲端運行：使用Google CoLaboratory訓練神經網路。
南華大學跨領域-人工智慧期中報告
11215015 江英姿 11215025 詹文儀
# CoLaboratory 訓練神經網路

本文旨在展示如何使用CoLaboratory 訓練神經網路。我們將展示一個在威斯康辛乳癌資料集上訓練神經網路的範例，資料集可在UCI Machine Learning Repository（http://archive.ics.uci.edu/ml/datasets） 取得。本文的範例相對比較簡單。

本文所使用的CoLaboratory notebook 連結：https://colab.research.google.com/notebook#fileId=1aQGl_sH4TVehK8PDBRspwI4pD16xIR0r

# 深度學習

深度學習是一種機器學習技術，它所使用的計算技術一定程度上模仿了生物神經元的運作。各層中的神經元網路不斷將資訊從輸入傳輸到輸出，直到其權重調整到可以產生反映特徵和目標之間底層關係的演算法。

想更了解神經網絡，推薦閱讀這篇論文《Artificial Neural Networks for Beginners》（https://arxiv.org/pdf/cs/0308031.pdf）。

# 程式碼
問題：研究者取得乳房腫塊的細針穿刺（FNA），然後產生數位影像。此資料集包含描述影像中細胞核特徵的實例。每個實例包括診斷結果：M（惡性）或B（良性）。我們的任務是在該數據上訓練神經網路根據上述特徵診斷乳癌。

開啟CoLaboratory，出現一個新的untitled.ipynb 檔案供你使用。

谷歌允許使用其伺服器上的一台linux 虛擬機，這樣你可以存取終端為專案安裝特定套件。如果你只在程式碼單元中輸入!ls 指令（記得指令前加!），那麼你的虛擬機器中會出現一個simple_data 資料夾。

![image](https://github.com/dtanlley/report1/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202023-11-01%20122739.png)

我們的任務是將資料集放置到該機器上，這樣我們的notebook 就可以存取它。你可以使用以下程式碼：

```
#Uploading the Dataset

from google.colab import files

uploaded = files.upload()

with open("breast_cancer.csv", 'wb') as f:

    f.write(uploaded[list(uploaded.keys())[0]])
```
輸入!ls 指令，檢查機器上是否有該檔案。你將會看到datalab 資料夾和breast_cancer_data.csv 檔案。

![image](https://github.com/dtanlley/report1/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202023-11-01%20124115.png)

資料預處理：

現在資料已經在機器上了，我們使用pandas 將其輸入到專案中。

```

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#Importing dataset

dataset = pd.read_csv('breast_cancer.csv')



#Check the first 5 rows of the dataset. 

    dataset.head(5)

```

![image](https://github.com/dtanlley/report1/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202023-11-01%20124600.png)

CoLaboratory 上的輸出結果圖示。

現在，分割因變數（Dependent Variables）和自變數（Independent Variables）。

```

#Seperating dependent and independent variables. 



X = dataset.iloc[:, 2:32].values  #Note: Exclude Last column with all NaN values.

y = dataset.iloc[:, 1].values

```

Y 包含一列，其中的「M」和「B」分別代表「是」（惡性）和「否」（良性）。我們需要將其編碼成數學形式，即“1”和“0”。可以使用Label Encoder 類別完成此任務。

```
#Encoding Categorical Data

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()



y = labelencoder.fit_transform(y)

```

現在資料已經準備好，我們將其分割成訓練集和測試集。在Scikit-Learn 中使用train_test_split 可以輕鬆完成這項工作。

```

#Splitting into Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```
參數test_size = 0.2 定義測試集比例。這裡，我們將訓練集設定為資料集的80%，測試集佔資料集的20%。

# Keras

Keras 是一種建構人工神經網路的高階API。它使用TensorFlow 或Theano 後端執行內部運作。要安裝Keras，必須先安裝TensorFlow。CoLaboratory 已經在虛擬機器上安裝了TensorFlow。使用以下指令可以檢查是否安裝TensorFlow：

!pip show tensorflow

你也可以使用!pip install tensorflow==1.2，安裝特定版本的TensorFlow。

另外，如果你更喜歡用Theano 後端，可以閱讀該文件：https://keras.io/backend/。

# 安裝Keras：

!pip install -q keras

```

# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

```

使用Sequential 和Dense 類別指定神經網路的節點、連接和規格。如上所示，我們將使用這些自訂網路的參數並進行調整。

為了初始化神經網絡，我們將建立一個Sequential 類別的物件。

```

# Initialising the ANN

classifier = Sequential()

```

# 設計網路。

對於每個隱藏層，我們需要定義三個基本參數：units、kernel_initializer 和activation。units 參數定義每層包含的神經元數量。Kernel_initializer 定義神經元在輸入資料上執行時的初始權重（詳見https://faroit.github.io/keras-docs/1.2.2/initializations/）。activation 定義資料的激活函數。

注意：如果現在這些項非常大也沒事，很快就會變得更加清晰。

第一層：

16 個具備統一初始權重的神經元，活化函數為ReLU。此外，定義參數input_dim = 30 作為輸入層的規格。注意我們的資料集中有30 個特徵列。

Cheat：

我們如何決定這一層的單元數？人們往往會說這需要經驗和專業知識。對於初學者來說，一個簡單方式是：x 和y 的總和除以2。如(30+1)/2 = 15.5 ~ 16，因此，units = 16。

第二層：第二層和第一層一樣，不過第二層沒有input_dim 參數。

輸出層：由於我們的輸出是0 或1，因此我們可以使用具備統一初始權重的單一單元。但是，這裡我們使用sigmoid 來活化函數。

```

# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))



# Adding the second hidden layer

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

```

擬合：

運行人工神經網絡，發生反向傳播。你將在CoLaboratory 上看到所有處理過程，而不是在自己的電腦上。

```

# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

```

這裡batch_size 是你希望同時處理的輸入量。epoch 指數據通過神經網路一次的整個週期。它們在Colaboratory Notebook 中顯示如下：

![image](https://github.com/dtanlley/report1/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202023-11-01%20125551.png)


進行預測，建構混淆矩陣。

```

# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

```

訓練網路後，就可以在X_test set 上進行預測，以檢查模型在新資料上的效能。在程式碼單元中輸入和執行cm 查看結果。

# 混淆矩陣

混淆矩陣是模型做出的正確、錯誤預測的矩陣表徵。此矩陣可供個人調查哪些預測和另一種預測混淆。這是一個2×2 的混淆矩陣。

![image](https://github.com/dtanlley/report1/blob/main/595146.png)

混淆矩陣如下所示。[cm (Shift+Enter)]

![image](https://github.com/dtanlley/report1/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202023-11-01%20130027.png)

上圖表示：68 個真負類、0 個假正類、46 個假負類、0 個真正類。很簡單。此平方矩陣的大小隨著分類類別的增加而增加。

這個範例中的準確率幾乎達到100%，只有2 個錯誤預測。但並不總是這樣。有時你可能需要投入更多時間，研究模型的行為，提出更好、更複雜的解決方案。如果一個網路效能不夠好，你需要調整超參數來改進模型。

希望本文可以幫助你開始使用Colaboratory。教學的Notebook 位址：https://colab.research.google.com/notebook#fileId=1aQGl_sH4TVehK8PDBRspwI4pD16xIR0r

原文連結：https://medium.com/@howal/neural-networks-with-google-colaboratory-artificial-intelligence-getting-started-713b5eb07f14
