卷积神经网络(CNN)实现图片分类
南華大學跨領域-人工智慧期末報告
11215015 江英姿 11215025 詹文儀
這段程式碼會從連結中下載food-11的壓縮檔 
並將food-11.zip檔解壓縮至work檔中
解壓縮完成後再將壓縮檔刪除

![image](https://github.com/user-attachments/assets/0db99e0d-7166-4b5a-96ad-1fc98335036d)

影像預處理
在影像預處理中，我們定義一個名為preprocess的函數，並做如下預處理：

影像尺寸調整：蔣影像尺寸調整為128*128像素，確保輸入影像具有相同尺寸；
隨機翻轉（僅用於訓練）：為了讓資料更加豐富，從而讓模型效果更好；
資料標準化：將影像轉換為numpy數組，並使其歸一化；
更改數組形狀：以符合CNN的輸入要求

![image](https://github.com/user-attachments/assets/76a68856-8ebe-4dad-aace-7bd81621242e)

資料讀取
定義了一個用於處理食物分類任務的自訂資料集類別 FoodDataSet，並使用這個類別來建立訓練和評估（驗證）的資料載入器（DataLoader），並讀取資料。

註：FoodDataSet類別所繼承的paddle.io.Dataset類別為官方建議的資料集類型，可以直接在官方訓練方法fit()中使用，API文件請見：Dataset。 DataLoader可以傳回一個迭代器，其支援單進程和多進程的資料載入方式，在資料量較大的時候比較有利。

![image](https://github.com/user-attachments/assets/3a0c0eaa-a371-4bf2-a1e3-607f50c9527c)

模型構建（網路配置）
先定義網路結構，這裡選擇的LeNet()模型。 LeNet 是一個經典的捲積神經網路（CNN）架構，最初由 Yann LeCun 等人在1990年代提出，用於手寫數字識別（如 MNIST 資料集）。本範例中的 LeNet 模型基於原始的 LeNet 結構進行了適度的修改，以適應更通用的影像分類任務，特別是針對具有更多類別（如11類）的資料集。此模型使用 PaddlePaddle 深度學習框架實作。

LeNet 模型主要由以下幾個部分組成：

卷積層（Convolutional Layers）：
Conv0：第一個卷積層，輸入通道數為3（適用於RGB影像），輸出通道數為10，卷積核大小為5x5，步長為1，使用SAME填滿策略以保持特徵圖尺寸不變。

Conv1：第二個卷積層，輸入通道數為10，輸出通道數為20，卷積核大小為5x5，步長為1，同樣使用SAME填充。

Conv2：第三個卷積層，輸入通道數為20，輸出通道數為50，卷積核大小為5x5，步長為1，SAME填滿。

池化層（Pooling Layers）：
每個卷積層後面跟著最大池化層（MaxPool2D），池化核大小為2x2，步長為2，用於減少特徵圖的尺寸和參數數量，同時提高模型的泛化能力。

全連接層（Fully Connected Layers）：
FC1：第一個全連接層，將捲積層輸出的特徵圖展平後作為輸入，輸出特徵維度為256。

FC2：第二個全連接層，輸入特徵維度為256，輸出特徵維度為64。

FC3：第三個全連接層，即輸出層，輸入特徵維度為64，輸出特徵維度為11，對應11個類別的分類任務。

激活函數：
在每個卷積層和前兩個全連接層後使用 Leaky ReLU 激活函數，以引入非線性因素，並提高模型的表達能力。

在輸出層使用 Softmax 啟動函數，將輸出轉換為機率分佈，以便進行多分類任務。

![image](https://github.com/user-attachments/assets/fcb01c88-ad77-4b9c-9303-e80b95505373)

查看模型结构

![image](https://github.com/user-attachments/assets/cac47bac-fa50-4e04-a10e-3c951854167c)

模型訓練
如果發現運行時間較長，可以將epoch（訓練總輪次）降低，但這會導致訓練結果變差。

![image](https://github.com/user-attachments/assets/c3e6b1ee-8941-4c71-9763-7e4ae0fcf797)

在work/food-11中再建立testing資料夾
並隨機生成圖像進行下一步的預測

![image](https://github.com/user-attachments/assets/c22810cd-c8c7-4f77-a649-bebe03e8760b)

簡單地匯入一張圖，然後進行直覺測試

![image](https://github.com/user-attachments/assets/c23c6c66-2a5c-41d9-95c5-71dc244cb69b)

以下為預測完的結果

![image](https://github.com/user-attachments/assets/8841c69f-fc98-4577-8d4a-318e4564f35a)
