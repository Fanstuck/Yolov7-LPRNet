# Yolov7-LPRNet
基于Yolov7-LPRNet的动态车牌目标识别算法模型
博客地址：https://blog.csdn.net/master_hunter/article/details/133631461
## 数据集
CCPD：https://github.com/detectRecog/CCPD

CCPD是一个大型的、多样化的、经过仔细标注的中国城市车牌开源数据集。CCPD数据集主要分为CCPD2019数据集和CCPD2020(CCPD-Green)数据集。CCPD2019数据集车牌类型仅有普通车牌(蓝色车牌)，CCPD2020数据集车牌类型仅有新能源车牌(绿色车牌)。

**在CCPD数据集中，每张图片仅包含一张车牌，车牌的车牌省份主要为皖。CCPD中的每幅图像都包含大量的标注信息，但是CCPD数据集没有专门的标注文件，每张图像的文件名就是该图像对应的数据标注**。

标注最困难的部分是注释四个顶点的位置。为了完成这项任务，数据发布者首先在10k图像上手动标记四个顶点的位置。然后设计了一个基于深度学习的检测模型，在对该网络进行良好训练后，对每幅图像的四个顶点位置进行自动标注。
## 前言

。我见过很多初学目标识别的同学基本上只花一周时间就可以参照案例实现一个目标检测的项目，这全靠YOLO强大的解耦性和部署简易性。初学者甚至只需要修改部分超参数接口，调整数据集就可以实现目标检测了。但是我想表达的并不是YOLO的原理有多么难理解，原理有多难推理。一般工作中要求我们能够运行并且能够完成目标检测出来就可以了，更重要的是数据集的标注。我们不需要完成几乎难以单人完成的造目标检测算法轮子的过程，我们需要理解YOLO算法中每个超参数的作用以及影响。就算我们能够训练出一定准确度的目标检测模型，我们还需要根据实际情况对生成结果进行一定的改写：例如对于图片来说一共出现了几种目标;对于一个视频来说，定位到具体时间出现了识别的目标。这都是需要我们反复学习再练习的本领。

完成目标检测后，我们应该输出定位出来的信息，YOLO是提供输出设定的超参数的，我们需要根据输出的信息对目标进行裁剪得到我们想要的目标之后再做上层处理。如果是车牌目标识别的项目，我们裁剪出来的车牌就可以进行OCR技术识别出车牌字符了，如果是安全帽识别项目，那么我们可以统计一张图片或者一帧中出现检测目标的个数做出判断，一切都需要根据实际业务需求为主。本篇文章主要是OCR模型对车牌进行字符识别，结合YOLO算法直接定位目标进行裁剪，裁剪后生成OCR训练数据集即可。
## 训练步骤
### 1.安装环境
利用Yolo训练模型十分简单并没有涉及到很复杂的步骤，如果是新手的话注意下载的torch版本是否符合本身NVDIA GPU的版本，需要根据NVIDIA支持最高的cuda版本去下载兼容的Torch版本，查看cuda版本可以通过终端输入：nvidia-smi

![f6b17a7a98f748ebaf58ee703f7489ef](https://github.com/Fanstuck/Yolov7-LPRNet/assets/62112487/04881d58-fc8e-4b82-97ae-a3348e0c8e6b)
### 2.修改Yolo配置文件
首先增加cfg/training/yolov7-e6e-ccpd.yaml文件，此配置文件可以参数动态调整网络规模，这里也不展开细讲，以后会有Yolov7源码详解系列，敬请期待，我们需要根据我们用到的官方yolo模型选择对于的yaml文件配置，我这里用的的yolov7-e6e模型训练，所以直接拿yolov7-e6e.yaml改写：
````
 # parameters
nc: 1  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple
````
其中nc是检测个数，depth_multiple是模型深度，width_multiple表示卷积通道的缩放因子，就是将配置里面的backbone和head部分有关Conv通道的设置，全部乘以该系数。通过这两个参数就可以实现不同复杂度的模型设计。然后是添加数据索引文件data/license.yaml：
````
train: ./split_dataset/images/train  
val: ./split_dataset/images/val 
test: ./split_dataset/images/test 
 
# number of classes
nc : 1
 
#class names
names : ['license']
````
### 训练模型
前面train，val，test都对应着目录存放的训练数据集。之后修改train.py中的参数或者是直接在终端输入对应的参数自动目录，我一般是习惯直接在defalut下面修改,对应参数修改，一般来说修改这些就足够了：

````
parser.add_argument('--weights', type=str, default='weights/yolo7-e6e.pt', help='initial weights path')
parser.add_argument('--cfg', type=str, default='cfg/yolov7-e6e-ccpd', help='model.yaml path')
parser.add_argument('--data', type=str, default='data/license.yaml', help='data.yaml path')
````
当然也可能出现内存溢出等问题，需要修改：
````
arser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
````
 这两个参数，具体参数根据自己硬件条件修改。
 ### 推理
 这里需要将刚刚训练好的最好的权重传入到推理函数中去。然后就可以对图像视频进行推理了。

主要需要修改的参数是：
````
parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp/weights/best.pt', help='model.pt path(s)')
 parser.add_argument('--source', type=str, default='测试数据集目录或者图片', help='source') 
````
有问题的私信博主或者直接评论就可以了博主会长期维护此开源项目，目前此项目运行需要多部操作比较繁琐，我将不断更新版本优化，下一版本将加入UI以及一键部署环境和添加sh指令一键运行项目代码。下篇文章将详细解读LPRNet模型如何进行OCR识别， 再次希望对大家有帮助不吝点亮star~：

 
