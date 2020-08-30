
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
- 2018 4:代码已升级岛pytorch4.0
- 2018 4:加入了CHUHK03模型，可用
- 2018 4:添加了iLIDS-VID模型与PRID-2011模型
- 2018 3:为启动代码'''train_img_model_xent_htri.py'''和'''train_vid_model_xent_htri.py'''后加入参数'''--htri-only'''   可以再训练的时候只用上'''htri'''。[在这里查看详细解答](https://github.com/KaiyangZhou/deep-person-reid/blob/master/train_img_model_xent_htri.py#L189)
- 2018 3:做了轻微的修改，加入了[Multi-scale Deep CNN (ICCV'17)](https://arxiv.org/abs/1709.05165)，
a.把输入大小从160x60 改为了256x128
b.在最后一个卷积特征映射层后(last conv feature maps)，本文添加了一个平均池化层
c.本文训练模型有一定的策略，本文利用Market1501得到的预训练模型是可行的
- 2018 3：加入了 [center loss (ECCV'16)](https://github.com/KaiyangZhou/pytorch-center-loss)并训练了模型权重
# 实验环境
- pytorch 0.4.0
- torchvision 0.2.1
- 本仓库推荐用使用python2
# 下载本仓库
git clone https://github.com/KaiyangZhou/deep-person-reid
# 准备数据
通过以下代码创建目的存储数据
'''cd deep-person-reid/
mkdir data/'''
还有其他方式可以帮你存储到其他的目录，译者再次不再赘述
本文数据集
## market1501:
1. 从下面链接下载数据集到'''data/'''http://www.liangzheng.org/Project/project_reid.html.
2.解压数据集并重命名为 market1501 文件的结构如下：
'''
market1501/
    bounding_box_test/
    bounding_box_train/
    ...
    '''
3.跑训练数据的时候 使用 '''-d market1501'''
## CUHK03
啦啦啦啦
## DukeMTMC-reID:
啦啦啦啦
## MSMT17:
啦啦啦啦
## MARS:
啦啦啦啦
## iLIDS-VID:
啦啦啦啦
## PRID:
啦啦啦啦
## DukeMTMC-VideoReID
lalalala
# 数据加载器
在'''dataset_loader.py'''中，本文实现了数据加载其，在该python文件中，本文实现了两个继承'''torch.utils.data.Dataset'''的类
- ImageDataset:基于行人重识别数据集处理数据
- VideoDataset:基于行人重识别数据集处理视频
这两个类配合torch.utils.data.DataLoader 则可以批量的提供数据 ，数据加载器ImageDataset可以批量产生数据:(batch,channel,height,weight)，而数据加载器VideoDataset批量产生数据:((batch, sequence, channel, height, width))
# 模型(见源文件！！！)
在'''model/__init__.py'''中可以查看调用这些模型所需要的关键词

# 损失函数
- xnet:交叉熵+标间平滑正则化(label smoothing regularizer)
- htri:triplet loss with hard positive/negative mining 
- cent:center loss
优化器在optimizer.py中实现，有'''adam、sgd'''.用'''--optim string_name'''来管理优化器
# 训练
- train_img_model_xent.py:用交叉熵损失函数来训练image model
- train_img_model_xent_htri.py:用交叉熵和hard trplei loss的结合来训练image model
- train_img_model_cent.py: center losss -> image model
- train_vid_model_xent.py: 交叉熵 -> video model
- train_vid_model_xent_htri.py: 交叉熵+hard triplet loss ->video model
举个例子，若你想基于交叉熵训练ResNet50的image行人重识别模型，你可以在命令提示符中使用下述代码：
'''python train_img_model_xent.py -d market1501 -a resnet50 --max-epoch 60 --train-batch 32 --test-batch 32 --stepsize 20 --eval-step 20 --save-dir log/resnet50-xent-market1501 --gpu-devices 0
'''
若读者想使用多GPU训练，可以设置 --gpu-devices 0,1,2,3
读者还可以通过 python train_blah_blah.py -h 获取更多的调用参数细节
# 结果和预训练权重
看原文
# 测试
首先下载基于market1501,利用xnet训练的ResNet50.模型放在此 saved-models/resnet50_xent_market1501.pth.tar ，然后运行下面的代码：
'''python train_img_model_xent.py -d market1501 -a resnet50 --evaluate --resume saved-models/resnet50_xent_market1501.pth.tar --save-dir log/resnet50-xent-market1501 --test-batch 32
'''
同样的，若为了训练视频行人重识别模型，你摇吧预训练模型保存在saved-models/，如：saved-models/resnet50_xent_mars.pth.tar，之后运行下面的代码：
'''python train_vid_model_xent.py -d mars -a resnet50 --evaluate --resume saved-models/resnet50_xent_mars.pth.tar --save-dir log/resnet50-xent-mars --test-batch 2
'''
注意：上述代码中的--test--batch，若设置为2，那么每次训练的图片为2* 15 = 30(其中15为每个个tracklet内的images量)。你可以根据你的GPU性能来调整该参数值

# Version 1 β finished
