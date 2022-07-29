## EasyLossUtil   
### 简介   
在训练神经网路模型的过程中常常有loss数据,一般表现为一个epoch出现好几个。  
如果要使用tensorbord去可视化训练loss，我又懒得配置，因此自己写了将loss保存为文件以及图片的程序。
在不同的项目之间往往需要复制粘贴完成，所以不如写一个自己喜欢的通用的库出来算了。

### requirement
pandas==1.0.5   
numpy==1.21.4   
matplotlib=3.2.2   


### Easy Loss Util example

```python
import os
from EasyLossUtil.easyLossUtil import EasyLossUtil

root_path = os.path.dirname(__file__)  # 本文件所在目录的绝对路径   
name_list = ["loss1", "loss2"]
lossUtil = EasyLossUtil(loss_name_list=name_list, loss_root_dir=os.path.join(root_path, "test_loss"))
loss1 = [1, 2, 3, 4]
loss2 = [5, 2, 3, 4]
for i in range(len(loss1)):
    lossUtil.append(loss_name=name_list, loss_data=[loss1[i], loss2[i]])
lossUtil.autoSaveFileAndImage()
```

loss1.png   
![alt loss1](./test_loss/loss1.png)   
loss2.png   
![alt loss2](./test_loss/loss2.png)   

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

### Quick Average Method Example
```python
import os
from EasyLossUtil.quickAverageMethod import QuickAverageMethod

name_list = ["loss1", "loss2"]
q = QuickAverageMethod(loss_name_list=name_list)
loss1 = [1, 2, 3, 4, 0]
loss2 = [5, 2, 3, 4, 0]
for i in range(len(loss1)):
    q.append(loss_name=name_list, value=[loss1[i], loss2[i]])
all_avg_loss = q.getAllAvgLoss()
print(all_avg_loss)
```

输出为:   
[2.0, 2.8]