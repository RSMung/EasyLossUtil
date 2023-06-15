import torch

@torch.no_grad()
class QuickAverageMethod:
    def __init__(self, loss_name_list:list):
        """
        快速计算loss的均值
        :param loss_name_list 需要处理的loss数据的list
        """
        self.loss_num = len(loss_name_list)
        self.loss_name_list = loss_name_list
        # 存储所有的loss数据
        self.data = {}
        for index in range(self.loss_num):
            loss_data = []
            self.data[loss_name_list[index]] = loss_data

    def append(self, loss_name, value):
        """
        添加loss值
        :param loss_name: loss的名字  可以是一个str,也可以是str的list
        :param value: loss的值  可以是一个值也可以是list
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
            self.data[loss_name].append(value)
        elif loss_name_type is list:
            assert len(loss_name) == len(value), f"loss_name和value的数据个数不一致"
            # 依次添加
            for idx, name in enumerate(loss_name):
                self.data[name].append(value[idx])
        else:
            raise RuntimeError(f"the type of loss_name:{loss_name_type} is not valid!")

    def getAverageValue(self, loss_name:str):
        """
        获得loss平均值
        """
        assert loss_name in self.loss_name_list, f"loss_name:{loss_name} is not valid!"
        average = 0
        length = len(self.data[loss_name])
        for j in range(length):
            # 第index个loss的第j个loss值
            average += self.data[loss_name][j] / length
        return average

    def getAllAvgLoss(self):
        """
        获取所有loss的均值并且以list的形式返回
        :return: list形式的loss的均值
        """
        avg_loss_list = []
        for name in self.loss_name_list:
            avg_loss = self.getAverageValue(loss_name=name)
            avg_loss_list.append(avg_loss)
        return avg_loss_list

    def clearSpecificLoss(self, loss_name):
        """
        清除指定loss的数据
        :param loss_name: 要清除的loss的名字
        """
        self.data[loss_name].clear()

    def clearAllData(self):
        """
        清除所有loss数据
        """
        for index in range(self.loss_num):
            self.data[self.loss_name_list[index]].clear()


if __name__ == "__main__":
    loss_name_list = ["loss1", "loss2"]
    q = QuickAverageMethod(loss_name_list)
    for i in range(10):
        q.append("loss1", i)
        q.append("loss2", 10 + i)
    a = q.getAverageValue("loss1")
    print(type(a))
    print(a)
    q.clearAllData()
    print(q.getAverageValue("loss1"))
