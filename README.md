# SFZcode_Recognization
一、使用freestyle数据进行生成训练的模型
ocr_number.py
trainIDCard.py
genIDCard.py

二、使用tensorflow_mnist_cnn_master的参数直接进行预测
ocr_number_MNIST.py
ocr_number_MNIST_test.py

三、使用tensorflow_mnist_cnn_master的参数进行迁移学习的预测
ocr_number_transfer_data.py
ocr_number_transfer_training.py
ocr_number_transfer_test.py

使用mnist的dropout3层，另外搭建fullyconnected层，输出为11，finetunning。
