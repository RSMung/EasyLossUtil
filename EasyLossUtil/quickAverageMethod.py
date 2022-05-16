

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

    def append(self, loss_name:str, value):
        """
        添加一个loss值
        """
        assert loss_name in self.loss_name_list, f"loss_name:{loss_name} is not valid!"
        self.data[loss_name].append(value)

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

    def clearData(self):
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
    print(q.getAverageValue("loss1"))
    q.clearData()
    print(q.getAverageValue("loss1"))
