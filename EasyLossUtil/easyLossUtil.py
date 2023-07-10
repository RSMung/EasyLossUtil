import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


@torch.no_grad()
class EasyLossUtil:
    def __init__(self, loss_name_list: list, loss_root_dir: str, loadArchive: bool = False):
        """
        :param loss_name_list: the loss data that we need to visualize, no repetitive.
        For example   d_loss, g_loss in training procedure of GAN
        :param loss_root_dir: the root directory for saving the loss data
        :param loadArchive: whether we want to load the checkpoints
        """
        self.loss_name_list = loss_name_list
        self.loss_root_dir = loss_root_dir
        # 一共要处理多少个loss
        self.loss_num = len(loss_name_list)
        # 用于存储所有loss数据, loss_name, list键值对
        self.data = {}
        # 检查loss_name的同时初始化字典
        print("EasyLossUtil---All the loss names:")
        for i in range(self.loss_num):
            # 检测是否已经存在这个name的loss
            check = self.data.get(loss_name_list[i], None)
            if check is None:
                # 不存在, 就新建一个空的list并且存储到字典中
                print(loss_name_list[i])
                self.data[loss_name_list[i]] = []
            else:
                raise RuntimeError(f"the name of loss are not allowed to be repetitive:{str(loss_name_list)}")
        print()
        # 查看数据存储目录是否存在，不存在则创建
        if not os.path.exists(loss_root_dir):
            os.mkdir(loss_root_dir)
        # 加载存档的loss数据
        if loadArchive:
            for i in range(self.loss_num):
                # 加载数据
                loss_data_i = self.loadSingleLossDataArchive(loss_name=loss_name_list[i])
                # 放进字典, 更新数据
                self.data[loss_name_list[i]] = loss_data_i

    def append(self, loss_name, loss_data):
        """
        :param loss_name: 要存储的loss的名字 可以是单个也可以是list
        :param loss_data: 要存储的数据  可以是单个也可以是list
        example:
            单个数据
                loss_name: loss_g
                loss_data: 1
            多个数据
                loss_name: loss_g   loss_d
                loss_data: 1        2
        """
        loss_name_type = type(loss_name)
        if loss_name_type is str:
            # 单个数据的情况
            self.data[loss_name].append(loss_data)
        elif loss_name_type is list and type(loss_data) is list:
            # 多个数据的情况
            assert len(loss_name) == len(loss_data), f"loss_name和loss_data的数据个数不一致"
            for idx in range(len(loss_name)):
                # 依次存储所有数据
                # 获取loss名字
                name = loss_name[idx]
                # 获取数据
                d = loss_data[idx]
                self.data[name].append(d)
        else:
            raise RuntimeError(f"loss_name 的类型: {loss_name_type} 不正确")

    def saveSingleLoss2File(self, loss_name: str):
        """
        将loss数据保存为csv文件存储
        :param loss_name: 要存储的loss的名字
        """
        pdOperator = pd.DataFrame(data=self.data[loss_name])
        pdOperator.to_csv(os.path.join(self.loss_root_dir, loss_name + ".csv"), index=False, header=False)

    def saveSingleLoss2Image(self, loss_name, image_name: str, line_color="red", start_epoch: int = 0):
        """
        将loss数据保存为折线图
        :param loss_name: 要保存的loss
        :param image_name: 保存到的文件的名字
        :param line_color: 折线的颜色
        :param start_epoch: 横坐标的起始值(开始训练的epoch的数值)   例如100-200, 用于接续训练的情况
        """
        plt.figure()
        data_len = len(self.data[loss_name])
        # 注意此处stop是包含在内的, 是全闭合区间
        x = np.linspace(start=start_epoch, stop=start_epoch + data_len - 1, num=data_len, dtype=np.uint32)
        # print(x)
        plt.plot(x, np.array(self.data[loss_name]), line_color, label=loss_name)
        plt.legend(loc='best')
        plt.tight_layout(pad=0.5)  # 控制图像边缘的空白, 使得图像紧凑一些
        plt.savefig(os.path.join(self.loss_root_dir, image_name))
        plt.close()

    def autoSaveFileAndImage(self):
        """
        全自动保存所有loss数据到csv文件和图像
        """
        # 图像的后缀名
        postfix = "png"
        for name in self.loss_name_list:
            self.saveSingleLoss2File(name)
            self.saveSingleLoss2Image(loss_name=name, image_name=name + "." + postfix)

    def loadSingleLossDataArchive(self, loss_name:str):
        """
        从csv文件加载loss数据
        :param loss_name: 加载的loss的名字
        :return: 加载后的list对象
        """
        # header为None表示没有表头
        loss_data = pd.read_csv(
            os.path.join(self.loss_root_dir, loss_name + ".csv"),
            sep=',',
            header=None
        )
        loss_data = loss_data.values.squeeze()
        loss_data = loss_data.tolist()
        return loss_data