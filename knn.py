import numpy as np
import operator


class KNN(object):

    def __init__(self, k=3):
        self.k = k


    def fit(self, x, y):    #fit()函数将x和y传进去
        self.x = x
        self.y = y


    def _square_distance(self, v1, v2):         #计算任意两点之间的距离平方
        return np.sum(np.square(v1 - v2))


    def _vote(self, ys):    #投票
        ys_unique = np.unique(ys)   #ys取唯一值
        vote_dict = {}  #用字典进行操作
        for y in ys:
            if y not in vote_dict.keys():   #y不在当前字典的键里
                vote_dict[y] = 1    #建立k=0的键，值为1
            else:
                vote_dict[y] += 1   #y在当前字典的键里，值加1
            #第一个参数为可迭代的参数，reverse为从大到小排序
            sorted_vote_dict = sorted(vote_dict.items(), key=operator.itemgetter(1), reverse=True)
            return sorted_vote_dict[0][0]


    def predict(self, x):       #接收x参数，多行的点数据，每行是一个二维的向量
        y_pred = []
        for i in range(len(x)):
            #得到当前的x[i]和所有的训练样点之间的平方距离，保存于数组当中
            dist_arr = [self._square_distance(x[i], self.x[j]) for j in range(len(self.x))]     #循环内部训练数据方法计算
            sorted_index = np.argsort(dist_arr)     #从小到大排序距离，返回索引
            top_k_index = sorted_index[:self.k]
            y_pred.append(self._vote(ys=self.y[top_k_index]))       #添加当前x和y的预测值
        return np.array(y_pred)


    def score(self, y_true=None, y_pred=None):      #计算推测值的精度
        if y_true is None or y_pred is None:
            y_pred = self.predict(self.x)
            y_true = self.y
        score=0.0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                score += 1
        score /= len(y_true)    #得到正确率
        return score
