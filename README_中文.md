
# Deep person reid
此仓库包含pytorch实现的行人重识别模型

- 此仓库支持：
- 多GPU训练
- 基于图象的和基于视频的行人重识别
- 端对端选练和评估
- ？很多paper用到的standardsplits
- 训练好的模型

# 日常更新
- 2018.5：支持了 两个新的模型（请看对于的英文doc），加入了Inception-v4,inception-ResNet-v2,DPN,ResNext于Se-ResNe(x)t(之后奉上训练好的模型)
- 2018 4：加入了几个新模型(请看对应的英文)
- 2018 4：增加了一个新模型，我们在CUHK03上，达到了42.4%的识别准确率，目前排名第一。你下好此代码库之后，利用

"python train_img_model_xent.py -d cuhk03 -a hacnn --save-dir log/hacnn-xent-cuhk03 --height 160 --width 64 --max-epoch 500 --stepsize -1 --eval-step 50"

就可以直接复现

# loading...
