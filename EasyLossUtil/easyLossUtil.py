import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class EasyLossUtil:
    def __init__(self, loss_name_list: list, loss_root_dir: str):
        """
        :param loss_name_list: 需要处理的loss的名字的列表,不允许重复   例如训练GAN时的d_loss, g_loss
        :param loss_root_dir: 存放loss数据的根目录
        """
        self.loss_name_list = loss_name_list
        self.loss_root_dir = loss_root_dir
        # 一共要处理多少个loss
        loss_num = len(loss_name_list)
        # 用于检查loss名字是否有重复的，存储 名字-index 键值对
        self.name_dict = {}
        # 存储所有loss数据
        self.list_array = []
        # 创建loss_num个list存储数据
        print("要处理的loss的名字为:", end='')
        for i in range(loss_num):
            check = self.name_dict.get(loss_name_list[i], None)
            if check is None:
                print(loss_name_list[i], end=' ')
                self.name_dict[loss_name_list[i]] = i
            else:
                raise RuntimeError(f"loss的名字不允许重复:{str(loss_name_list)}")
            loss_list_i = []
            self.list_array.append(loss_list_i)
        print()
        # 查看数据存储目录是否存在，不存在则创建
        if not os.path.exists(loss_root_dir):
            os.mkdir(loss_root_dir)

    def append(self, loss_name: str, loss_data_point: float):
        """
        :param loss_name: 要存储的loss的名字
        :param loss_data_point: 要存储的数据
        """
        self.list_array[self.name_dict[loss_name]].append(loss_data_point)

    def save2File(self, loss_name):
        """
        将loss数据保存为csv文件存储
        :param loss_name: 要存储的loss的名字
        """
        pdOperator = pd.DataFrame(data=self.list_array[self.name_dict[loss_name]])
        pdOperator.to_csv(os.path.join(self.loss_root_dir, loss_name + ".csv"), index=False, header=False)

    def save2Image(self, loss_name, image_name: str, line_color = "red", start_epoch: int = 0):
        """
        将loss数据保存为折线图
        :param loss_name: 要保存的loss
        :param image_name: 保存到的文件的名字
        :param line_color: 折线的颜色
        :param start_epoch: 横坐标的起始值(开始训练的epoch的数值)   例如100-200, 用于接续训练的情况
        """
        plt.figure()
        data_len = len(self.list_array[self.name_dict[loss_name]])
        x = np.linspace(start=start_epoch, stop=start_epoch + data_len, num=data_len, dtype=np.uint32)
        plt.plot(x, np.array(self.list_array[self.name_dict[loss_name]]), line_color, label=loss_name)
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.loss_root_dir, image_name))
        plt.close()
