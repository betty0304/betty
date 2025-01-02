卷积神经网络(CNN)实现图片分类
南華大學跨領域-人工智慧期末報告
11215015 江英姿 11215025 詹文儀
本次使用的数据集为food-11数据集，共有11类

数据集的划分为：

Training set: 9866张
Validation set: 3430张
Testing set: 3347张
数据格式 下载 zip 档后解压缩会有三个资料夹，分别为training、validation 以及 testing

training 以及 validation 中的照片名称格式为 [类别]_[编号].jpg，例如 3_100.jpg 即为类别 3 的照片（编号不重要）

对应类别如图1所示：

类别	含义（英）	含义（中）
0	Bread	面包
1	Dairy product	乳制品
2	Dessert	甜点
3	Egg	蛋类
4	Fried food	油炸食品
5	Meat	肉类
6	Noodles/Pasta	面条/意大利面
7	Rice	米饭
8	Seafood	海鲜
9	Soup	汤
10	Vegetable/Fruit	蔬菜/水果
表1 food-11各类别对应食物
In [1]
import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  
import cv2
import random
import paddle

# 选择运行设备
device = paddle.set_device('cpu')
# device = paddle.set_device('gpu:0')
1.2、数据集解压并移动至工作文件夹，并观察数据集
由于本次实践的数据集稍微比较大，以防出现不好下载的问题，为了提高效率，可以用下面的代码进行数据集的下载。

!unzip -q data/data94114/food-11.zip # 解压缩food-11数据集

!mv  /home/aistudio/food-11 /home/aistudio/work

In [2]
# !unzip -q data/data94114/food-11.zip # 解压缩food-11数据集
# !mv  /home/aistudio/food-11 /home/aistudio/work
In [3]
# 观察样本
img = cv2.imread('work/food-11/training/0_110.jpg')
plt.imshow(img[:,:,::-1])
plt.show()

<Figure size 640x480 with 1 Axes>
1.3 图像预处理
在图像预处理中，我们定义一个名为preprocess的函数，并做如下预处理：

图像尺寸调整：蒋图像尺寸调整为128*128像素，确保输入图像具有相同尺寸；
随机翻转（仅用于训练）：为了使数据更加丰富，从而让模型效果更好；
数据标准化：将图像转换为numpy数组，并使其归一化；
更改数组形状：以符合CNN的输入要求
In [4]
def preprocess(img,mode='train'):
    img = cv2.resize(img,(128,128))
    # 在训练集中随机对数据进行flip操作
    if mode == 'train':
        if random.randint(0,1): # 随机进行预处理
            img = cv2.flip(img,random.randint(-1,1))  # flip操作模式随机选择
    # 转换为numpy数组
    img = np.array(img).astype('float32')
    # 将数据范围改为0-1
    img = img / 255.
    # 最后更改数组的shape，使其符合CNN输入要求
    return img.transpose((2,0,1))
1.4 数据读取
定义了一个用于处理食物分类任务的自定义数据集类 FoodDataSet，并使用这个类来创建训练和评估（验证）的数据加载器（DataLoader），并读取数据。

注：FoodDataSet类所继承的paddle.io.Dataset类为官方建议的数据集类型，可以直接在官方训练方法fit()中使用，API文档见：Dataset。DataLoader可以返回一个迭代器，其支持单进程和多进程的数据加载方式，在数据量较大的时候比较有利。

In [5]
class FoodDataSet(paddle.io.Dataset):
    def __init__(self,data_dir,mode):
        # 获取文件夹下数据名称列表
        self.filenames = os.listdir(data_dir)
        self.data_dir = data_dir
        self.mode = mode
    def __getitem__(self,index):
        file_name = self.data_dir + self.filenames[index]
        # 读取数据
        img = cv2.imread(file_name)
        # 预处理
        img = preprocess(img,mode = self.mode)
        # 获得标签
        label = int(self.filenames[index].split('_')[0]) 
        return img,label
    def __len__(self):
        return len(self.filenames)

train_dataset = FoodDataSet('work/food-11/training/','train')
train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=64, shuffle=True)
eval_dataset = FoodDataSet('work/food-11/validation/','validation')
eval_loader = paddle.io.DataLoader(eval_dataset, places=paddle.CPUPlace(), batch_size=64, shuffle=True)
2、模型构建（网络配置）
首先定义网络结构，这里选择的LeNet()模型。LeNet 是一个经典的卷积神经网络（CNN）架构，最初由 Yann LeCun 等人在1990年代提出，用于手写数字识别（如 MNIST 数据集）。本示例中的 LeNet 模型基于原始的 LeNet 结构进行了适度的修改，以适应更通用的图像分类任务，特别是针对具有更多类别（如11类）的数据集。该模型使用 PaddlePaddle 深度学习框架实现。

LeNet 模型主要由以下几个部分组成：

卷积层（Convolutional Layers）：
Conv0：第一个卷积层，输入通道数为3（适用于RGB图像），输出通道数为10，卷积核大小为5x5，步长为1，使用SAME填充策略以保持特征图尺寸不变。

Conv1：第二个卷积层，输入通道数为10，输出通道数为20，卷积核大小为5x5，步长为1，同样使用SAME填充。

Conv2：第三个卷积层，输入通道数为20，输出通道数为50，卷积核大小为5x5，步长为1，SAME填充。

池化层（Pooling Layers）：
每个卷积层后都跟有一个最大池化层（MaxPool2D），池化核大小为2x2，步长为2，用于减少特征图的尺寸和参数数量，同时提高模型的泛化能力。

全连接层（Fully Connected Layers）：
FC1：第一个全连接层，将卷积层输出的特征图展平后作为输入，输出特征维度为256。

FC2：第二个全连接层，输入特征维度为256，输出特征维度为64。

FC3：第三个全连接层，即输出层，输入特征维度为64，输出特征维度为11，对应于11个类别的分类任务。

激活函数：
在每个卷积层和前两个全连接层后使用 Leaky ReLU 激活函数，以引入非线性因素，提高模型的表达能力。

在输出层使用 Softmax 激活函数，将输出转换为概率分布，便于进行多分类任务。

In [6]
class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet,self).__init__()

        self.conv0 = paddle.nn.Conv2D(in_channels=3,out_channels=10,kernel_size=5,padding="SAME",stride=1)
        self.pool0 = paddle.nn.MaxPool2D(kernel_size=2,stride=2) # 128 * 128 -> 64 * 64

        self.conv1 = paddle.nn.Conv2D(in_channels=10,out_channels=20,kernel_size=5,padding="SAME",stride=1)
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=2,stride=2) # 64 * 64 -> 32 * 32

        self.conv2 = paddle.nn.Conv2D(in_channels=20,out_channels=50,kernel_size=5,padding="SAME",stride=1)
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=2,stride=2) # 32 * 32 -> 16 * 16

        self.fc1 = paddle.nn.Linear(in_features=12800,out_features=256)
        self.fc2 = paddle.nn.Linear(in_features=256,out_features=64)
        self.fc3 = paddle.nn.Linear(in_features=64,out_features=11)
    
    def forward(self,x):
        x = self.conv0(x)
        x = paddle.nn.functional.leaky_relu(x)
        x = self.pool0(x)

        x = self.conv1(x)
        x = paddle.nn.functional.leaky_relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = paddle.nn.functional.leaky_relu(x)
        x = self.pool2(x)

        x = paddle.reshape(x,[x.shape[0],-1])

        x = self.fc1(x)
        x = paddle.nn.functional.leaky_relu(x)
        x = self.fc2(x)
        x = paddle.nn.functional.leaky_relu(x)
        x = self.fc3(x)
        x = paddle.nn.functional.softmax(x)
        return x
network = LeNet()
查看模型结构
In [7]
paddle.summary(network, (1,3,128,128))
---------------------------------------------------------------------------
 Layer (type)       Input Shape          Output Shape         Param #    
===========================================================================
   Conv2D-1      [[1, 3, 128, 128]]   [1, 10, 128, 128]         760      
  MaxPool2D-1   [[1, 10, 128, 128]]    [1, 10, 64, 64]           0       
   Conv2D-2      [[1, 10, 64, 64]]     [1, 20, 64, 64]         5,020     
  MaxPool2D-2    [[1, 20, 64, 64]]     [1, 20, 32, 32]           0       
   Conv2D-3      [[1, 20, 32, 32]]     [1, 50, 32, 32]        25,050     
  MaxPool2D-3    [[1, 50, 32, 32]]     [1, 50, 16, 16]           0       
   Linear-1         [[1, 12800]]           [1, 256]          3,277,056   
   Linear-2          [[1, 256]]            [1, 64]            16,448     
   Linear-3          [[1, 64]]             [1, 11]              715      
===========================================================================
Total params: 3,325,049
Trainable params: 3,325,049
Non-trainable params: 0
---------------------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 2.83
Params size (MB): 12.68
Estimated Total Size (MB): 15.71
---------------------------------------------------------------------------

{'total_params': 3325049, 'trainable_params': 3325049}
3、 模型训练
如果发现运行时间较长，可以将epoch（训练总轮次）降低，但这样会导致训练结果变差。

In [8]
model = paddle.Model(network)

model.prepare(paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters()), 
              paddle.nn.CrossEntropyLoss(), 
              paddle.metric.Accuracy())

visualdl = paddle.callbacks.VisualDL(log_dir='visualdl_log')

# 启动模型全流程训练
model.fit(train_loader,  # 训练数据集
          eval_loader,   # 评估数据集
          epochs=30,       # 训练的总轮次
          batch_size=64,  # 训练使用的批大小
          verbose=1,      # 日志展示形式
          callbacks=[visualdl])  # 设置可视化
The loss value printed in the log is the current step, and the metric is the average value of previous steps.
Epoch 1/30
step 155/155 [==============================] - loss: 2.4318 - acc: 0.1951 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.3413 - acc: 0.2044 - 621ms/step          
Eval samples: 3430
Epoch 2/30
step 155/155 [==============================] - loss: 2.3207 - acc: 0.2214 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.3530 - acc: 0.2172 - 618ms/step          
Eval samples: 3430
Epoch 3/30
step 155/155 [==============================] - loss: 2.1491 - acc: 0.2251 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.3826 - acc: 0.2300 - 621ms/step          
Eval samples: 3430
Epoch 4/30
step 155/155 [==============================] - loss: 2.2994 - acc: 0.2316 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.2151 - acc: 0.2312 - 595ms/step          
Eval samples: 3430
Epoch 5/30
step 155/155 [==============================] - loss: 2.3592 - acc: 0.2366 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.1528 - acc: 0.2353 - 612ms/step          
Eval samples: 3430
Epoch 6/30
step 155/155 [==============================] - loss: 2.2438 - acc: 0.2419 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.4641 - acc: 0.2350 - 615ms/step          
Eval samples: 3430
Epoch 7/30
step 155/155 [==============================] - loss: 2.1917 - acc: 0.2442 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.2508 - acc: 0.2382 - 614ms/step          
Eval samples: 3430
Epoch 8/30
step 155/155 [==============================] - loss: 2.3497 - acc: 0.2458 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.3744 - acc: 0.2417 - 612ms/step          
Eval samples: 3430
Epoch 9/30
step 155/155 [==============================] - loss: 2.1516 - acc: 0.2507 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.3154 - acc: 0.2388 - 613ms/step          
Eval samples: 3430
Epoch 10/30
step 155/155 [==============================] - loss: 2.3362 - acc: 0.2510 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.3045 - acc: 0.2388 - 592ms/step          
Eval samples: 3430
Epoch 11/30
step 155/155 [==============================] - loss: 2.3292 - acc: 0.2554 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.2595 - acc: 0.2417 - 597ms/step          
Eval samples: 3430
Epoch 12/30
step 155/155 [==============================] - loss: 2.1593 - acc: 0.2648 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.3055 - acc: 0.2627 - 597ms/step          
Eval samples: 3430
Epoch 13/30
step 155/155 [==============================] - loss: 2.2378 - acc: 0.2759 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.3169 - acc: 0.2694 - 607ms/step          
Eval samples: 3430
Epoch 14/30
step 155/155 [==============================] - loss: 2.4284 - acc: 0.2885 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.2588 - acc: 0.2638 - 597ms/step          
Eval samples: 3430
Epoch 15/30
step 155/155 [==============================] - loss: 2.3275 - acc: 0.2961 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.2283 - acc: 0.2776 - 599ms/step          
Eval samples: 3430
Epoch 16/30
step 155/155 [==============================] - loss: 2.1556 - acc: 0.2958 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.2509 - acc: 0.2796 - 606ms/step          
Eval samples: 3430
Epoch 17/30
step 155/155 [==============================] - loss: 2.3291 - acc: 0.3087 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.2250 - acc: 0.2790 - 613ms/step          
Eval samples: 3430
Epoch 18/30
step 155/155 [==============================] - loss: 2.3184 - acc: 0.3135 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.2014 - acc: 0.2793 - 621ms/step          
Eval samples: 3430
Epoch 19/30
step 155/155 [==============================] - loss: 2.0596 - acc: 0.3166 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.2324 - acc: 0.2799 - 591ms/step          
Eval samples: 3430
Epoch 20/30
step 155/155 [==============================] - loss: 2.1381 - acc: 0.3185 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.2198 - acc: 0.2851 - 605ms/step          
Eval samples: 3430
Epoch 21/30
step 155/155 [==============================] - loss: 2.2388 - acc: 0.3208 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.1962 - acc: 0.2784 - 608ms/step          
Eval samples: 3430
Epoch 22/30
step 155/155 [==============================] - loss: 2.4273 - acc: 0.3234 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.2545 - acc: 0.2878 - 592ms/step          
Eval samples: 3430
Epoch 23/30
step 155/155 [==============================] - loss: 1.8778 - acc: 0.3329 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.1583 - acc: 0.2883 - 608ms/step          
Eval samples: 3430
Epoch 24/30
step 155/155 [==============================] - loss: 2.3336 - acc: 0.3360 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.2658 - acc: 0.2927 - 582ms/step          
Eval samples: 3430
Epoch 25/30
step 155/155 [==============================] - loss: 1.8549 - acc: 0.3497 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.3349 - acc: 0.2991 - 609ms/step          
Eval samples: 3430
Epoch 26/30
step 155/155 [==============================] - loss: 1.9650 - acc: 0.3569 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.3495 - acc: 0.3102 - 610ms/step          
Eval samples: 3430
Epoch 27/30
step 155/155 [==============================] - loss: 2.2597 - acc: 0.3616 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.1653 - acc: 0.3041 - 595ms/step          
Eval samples: 3430
Epoch 28/30
step 155/155 [==============================] - loss: 2.1626 - acc: 0.3702 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.3113 - acc: 0.2985 - 597ms/step          
Eval samples: 3430
Epoch 29/30
step 155/155 [==============================] - loss: 1.9490 - acc: 0.3766 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.2044 - acc: 0.3117 - 610ms/step          
Eval samples: 3430
Epoch 30/30
step 155/155 [==============================] - loss: 2.2298 - acc: 0.3851 - 2s/step          
Eval begin...
step 54/54 [==============================] - loss: 2.0697 - acc: 0.3321 - 606ms/step          
Eval samples: 3430
代码解释
In [12]
model.save('model/LeNet') 
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
