import torch
import os

def retainTail(num, n):
    """
    将给定的数字保留指定的小数位数，可以给整数也可以是小数
    :param num: 给定的数字
    :param n: 要求保留的位数
    :return: 结果字符串
    """
    # print(f'要处理的数字:{num}')
    # print(f'要处理的数字的类型:{type(num)}')
    num = float(num)
    # 判断正负数, 把符号拿出来
    if num < 0:
        sign = '-'
    else:
        sign = ''
    # 获取数字的绝对值
    num = abs(num)
    # 将数字转换为字符串, tensor类型直接转str就会变成类似tensor(2.)的形式
    str_num = str(num)
    # print(f'str_num:{str_num}')
    # 找到小数点的位置
    point_pos = str_num.find(".")
    # print("小数点位置",point_pos)
    # 没有小数点，即整数
    if point_pos == -1:
        # 直接在后面加上小数点后加上n个0
        str_num += "."
        while n > 0:
            str_num += "0"
            n -= 1
        return str_num
    """
    小数的处理一共有3种情况
    1 小数位数t < 要求保留的位数n，那么在第n位后面补上n - t个0
    2 t = n则直接返回
    3 小数位数t > 要求保留的尾数n，那么找到第n+1个小数，根据它的值四舍五入
        注意进位带来的连环影响
    """
    # 012345  length=6
    # 1.3245
    # 获取小数位数t
    t = len(str_num) - point_pos - 1
    # print("t是",t)
    if t < n:
        temp = n - t
        while temp > 0:
            str_num += "0"
            temp -= 1
        if sign is not None:
            return sign + str_num
    elif t == n:
        if sign is not None:
            return sign + str_num
    else:
        # print("进入第三种情况")
        # t>n
        # 获取第n+1位数字, index是point_pos + n + 1
        num_n1 = int(str_num[point_pos + n + 1])
        # 进位标志
        flag = 1 if num_n1 >= 5 else 0
        # 将字符串变成list
        list_str_num = list(str_num)
        # print(f'list_str_num:{list_str_num}')
        # 存储结果
        result = []
        # 不包含-1
        for i in range(point_pos+n, -1, -1):
            # 如果不进位或者碰见小数点, 直接加入结果list
            if flag == 0 or i == point_pos:
                result.insert(0, list_str_num[i])
                continue
            # 第i个字符转数字
            # print(f'list_str_num[i]:{list_str_num[i]}')
            int_num_i = int(list_str_num[i])
            # 进位
            int_num_i += flag
            if int_num_i == 10:
                # 下一次还要进位
                flag = 1
                list_str_num[i] = '0'
            else:
                # 下一次不用进位了
                list_str_num[i] = str(int_num_i)
                flag = 0
            result.insert(0, list_str_num[i])
        # list转字符串返回
        if sign is not None:
            return sign + "".join(result)


class ParamsParent:
    """
    各个参数类的父类
    """
    def __repr__(self):
        # 直接打印这个类时会调用这个函数, 打印返回的输出的字符串
        str_result = f"---{self.__class__.__name__}---\n"
        # 剔除带__的属性
        # dir(self.__class__)会返回属性的有序列表
        # self.__dir__()返回属性列表, 与前者的区别是不会排序
        for attr in self.__dir__():
            if not attr.startswith('__'):
                str_result += "{}: {}\n".format(attr, self.__getattribute__(attr))
        str_result += "------------------\n"
        return str_result


def formatSeconds(seconds, targetStr):
    """
    将秒钟输出格式化， 大于60秒的用分钟表示
    :param seconds: 要输出的秒钟
    :param targetStr: 在输出秒钟格式化信息之前的提示字符串
    :return: 输出的字符串
    """
    if seconds > 60:
        seconds = seconds / 60
        postfix = 'min'
    else:
        postfix = 'sec'
    return targetStr + ": %.2f " % seconds + postfix

def getEqNum(pred_vector, label):
    """
    模型预测的概率向量中最大概率的类别作为预测类别, 与标签作比较, 计算预测正确的数量
    :param pred_vector: 模型输出的概率向量  [B, num_categories]
    :param label: 标签 [B]
    :return: 预测正确的数量
    """
    # 模型预测的概率向量中最大概率的类别作为预测类别
    # max函数返回的是[value, index]
    pred_label = torch.max(pred_vector, dim=1)[1]   # [B]
    # 与标签作比较
    # print(f'预测标签:{pred_label}, 真实标签:{label}')
    eqNum = torch.eq(pred_label, label)
    # 计算预测正确的数量
    eqNum = torch.sum(eqNum)
    return eqNum.item()

def checkDir(dir_path):
    """
    递归的创建多级目录
    :param dir_path: 需要创建的目录
    """
    # exist_ok=True自动判断当文件夹已经存在就不创建
    os.makedirs(dir_path, exist_ok=True)

def get_lr(optimizer:torch.optim.Optimizer):
    """
    返回优化器当前的学习率
    :param optimizer:
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

