- [EasyLossUtil](#easylossutil)
  - [1 简介](#1-简介)
  - [2 requirement](#2-requirement)
  - [3 Easy Loss Util example](#3-easy-loss-util-example)
    - [3.1 loss数据保存到csv与可视化](#31-loss数据保存到csv与可视化)
    - [3.2 接续训练时的loss处理](#32-接续训练时的loss处理)
  - [4 Quick Average Method Example](#4-quick-average-method-example)
  - [5 torchvision.utils.save\_imge的优化版本](#5-torchvisionutilssave_imge的优化版本)
  - [6 增加了一些通用方法](#6-增加了一些通用方法)
    - [6.1 retainTail(num, n)函数, 将给定的数字保留指定的小数位数，可以给整数也可以是小数](#61-retaintailnum-n函数-将给定的数字保留指定的小数位数可以给整数也可以是小数)
    - [6.2 ParamsParent类, 各个参数类的父类](#62-paramsparent类-各个参数类的父类)
    - [6.3 formatSeconds(seconds, targetStr)函数, 将秒钟输出格式化](#63-formatsecondsseconds-targetstr函数-将秒钟输出格式化)
    - [6.4 getEqNum(pred\_vector, label)函数， 用于计算分类器预测正确率](#64-geteqnumpred_vector-label函数-用于计算分类器预测正确率)
    - [6.5 checkDir(dir\_path)函数，如果目录不存在， 则递归的创建多级目录](#65-checkdirdir_path函数如果目录不存在-则递归的创建多级目录)
    - [6.6 get\_lr(optimizer:torch.optim.Optimizer)，返回优化器当前的学习率](#66-get_lroptimizertorchoptimoptimizer返回优化器当前的学习率)


# EasyLossUtil   
## 1 简介   
在训练神经网路模型的过程中常常有loss数据,一般表现为一个epoch出现好几个<br>
如果要使用tensorbord去可视化训练loss，我又懒得配置，
因此自己写了将loss保存为文件以及图片的程序。<br>
在不同的项目之间往往需要复制粘贴完成，所以不如写一个自己喜欢的通用的库出来算了。

安装包在dist目录中, 安装命令如下:   
`pip install .\EasyLossUtil-0.7-py3-none-any.whl`

## 2 requirement
前三个不用保持版本一致, 但是这几个库一定要有  
torch的版本一定要高于1.11.0   
开发这个库使用的第三方包:   
pandas==1.0.5   
numpy==1.21.4   
matplotlib==3.2.2   
torch>=1.11.0 

## 3 Easy Loss Util example

### 3.1 loss数据保存到csv与可视化
```python
import os
from EasyLossUtil.easyLossUtil import EasyLossUtil

root_path = os.path.dirname(__file__)  # 本文件所在目录的绝对路径   
# 需要处理的loss的名字
name_list = ["loss1", "loss2"]
# 初始化工具
lossUtil = EasyLossUtil(
    # loss的名字
    loss_name_list=name_list,
    # loss保存的根目录
    loss_root_dir=os.path.join(root_path, "test_loss")
)
# 模拟的loss数据, 共5个epoch的数据
loss1 = [1, 2, 3, 4]
loss2 = [5, 2, 3, 4]
# for循环模拟训练流程
total_epochs = len(loss1)
for i in range(total_epochs):
    lossUtil.append(
        loss_name=name_list,
        # 数据的顺序要与名字一致
        # 比如名字是loss1和loss2
        # 那么数据也应该是loss1和loss2
        loss_data=[
            loss1[i],
            loss2[i]
        ]
    )
# 自动保存loss数据为csv文件以及折线图文件
lossUtil.autoSaveFileAndImage()
```
**运行代码后会有以下提示:**   
EasyLossUtil---要处理的loss的名字为:   
loss1   
loss2   

**运行结果如下:**   
loss1.png   
<img src="./test_loss/loss1.png" width=400>  
loss2.png   
<img src="./test_loss/loss2.png" width=400>  

loss1.csv:   
1   
2   
3   
4   

loss1.csv:  
5   
2   
3   
4   

### 3.2 接续训练时的loss处理
这个功能是为了以下情况开发的:   
假设使用本工具保存了200个epoch的数据时, 因为未知原因(停电或者主动中断)导致程序停止了    
而我们想要在下一次训练时, 加载上一次训练的模型权重接续训练, 并且使得loss数据也能够连续, 前200个数据不会丢失(主要是折线图),    
那么可以按照以下例子设置:   
主要是需要在初始化工具时, 使用loadArchive=True进行设置, 工具会自行从设置的loss_root_dir目录中以往的日志文件中读取数据

```python
import os
from EasyLossUtil.easyLossUtil import EasyLossUtil

root_path = os.path.dirname(__file__)  # 本文件所在目录的绝对路径
# 需要处理的loss数据
name_list = ["loss1", "loss2"]
# 初始化工具, 
lossUtil = EasyLossUtil(
    loss_name_list=name_list,
    loss_root_dir=os.path.join(root_path, "test_loss"),
    # loadArchive=True表示从loss_root_dir目录下以往的csv文件中读取数据初始化工具
    loadArchive=True
)
# 输出现有的数据查看是否初始化成功(data是一个字典)
print(lossUtil.data)
# 后续可以按照3.1节继续向工具中添加数据, 保存到文件
```

## 4 Quick Average Method Example
这个工具用于快速计算平均值, <br>
例如一个epoch中有多次迭代, 计算多次迭代的loss平均值
```python
from EasyLossUtil.quickAverageMethod import QuickAverageMethod

# 需要处理的loss的名字
name_list = ["loss1", "loss2"]
# 初始化工具
q = QuickAverageMethod(loss_name_list=name_list)
# 需要计算平均值的工具
loss1 = [1, 2, 3, 4, 0]
loss2 = [5, 2, 3, 4, 0]
# 模拟循环训练过程, 向工具类中添加数据
for i in range(len(loss1)):
    q.append(loss_name=name_list, value=[loss1[i], loss2[i]])
# 获取loss数据的平均值
all_avg_loss = q.getAllAvgLoss()
print(all_avg_loss)
```

输出为:   
[2.0, 2.8]   
分别对应loss1的均值和loss2的均值

## 5 torchvision.utils.save_imge的优化版本
torchvision.utils中的保存图片的api在处理灰度图像数据时, 存储开销较大, <br>
因此我根据博客做了更改, 链接如下<br>
https://blog.csdn.net/nyist_yangguang/article/details/119935122 <br>

使用的例子<br>
使用gray_image=True控制, 在保存前对灰度图像做处理<br>
```python
from EasyLossUtil.saveTensor2Img import save_image
import torch
a = torch.randn((10, 1, 64, 128))
save_image(
    a,
    'efficient_save_tensor.png',
    gray_image=True,
    nrow=2,
    padding=5
)
```

## 6 增加了一些通用方法
### 6.1 retainTail(num, n)函数, 将给定的数字保留指定的小数位数，可以给整数也可以是小数
参数如下:   
num: 给定的数字   
n: 要求保留的位数   
返回值: 结果字符串   

### 6.2 ParamsParent类, 各个参数类的父类
各个参数类的父类, 实现了def __repr__(self)方法,    
使得可以直接print该类的对象, 方便输出日志   
示例代码:   
```python
from EasyLossUtil.global_utils import ParamsParent

class MyParams(ParamsParent):
    my_param1 = 1
    my_param2 = 2
    
my_params = MyParams()
print(my_params)

"""
程序的输出为:   
---MyParams---
my_param1: 1   
my_param2: 2   
------------------
"""
```


### 6.3 formatSeconds(seconds, targetStr)函数, 将秒钟输出格式化
将秒钟输出格式化， 大于60秒的用分钟表示   
参数如下:   
seconds: 要输出的秒钟   
targetStr: 在输出秒钟格式化信息之前的提示字符串   
返回值: 输出的字符串   


### 6.4 getEqNum(pred_vector, label)函数， 用于计算分类器预测正确率
模型预测的概率向量中最大概率的类别作为预测类别, 与标签作比较, 计算预测正确的数量 <br>
> pred_vector: 模型输出的概率向量，形状为[Batch_size, num_categories] <br>
> label: 标签，形状为[B_size],  例如1代表图像属于类别1 <br>

### 6.5 checkDir(dir_path)函数，如果目录不存在， 则递归的创建多级目录
dir_path: 需要创建的目录 <br>
例程：<br>
```python
from EasyLossUtil.global_utils import checkDir
dir_path = "./test1/test2"
checkDir(dir_path)
# 如果当前目录下文件夹test1不存在，则创建test1文件夹， 并在其中创建test2文件夹
# 如果test1下不存在test2, 则创建test2文件夹
# 如果两个文件夹都存在，那么不做任何动作
```

### 6.6 get_lr(optimizer:torch.optim.Optimizer)，返回优化器当前的学习率
