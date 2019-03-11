import math
import scipy.stats as ss
import numpy as np
class priorized_experience_buffer():
    def __init__(self, buffer_size = 2000):
        self.buffer = []    #缓冲区元素的个数
        self.prob = []    #将值较为固定,需要探究其作用
        self.err = []     #将值无意识放大,后期看看具体作用
        self.buffer_size = buffer_size  #记忆池的大小
        self.alpha = 0.2
    
    #记忆池添加记忆,传入的数据是list类型
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            #经过测试,这几句是将前面的若干越过记忆池的数据清空,以便于后面添加,可以设计成deque
            self.err[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
            self.prob[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
        self.err.extend([10000]*len(experience))
        self.prob.extend([1]*len(experience))

    #更新误差,后期需要探究其作用      
    def updateErr(self, indx, error):
        for i in range(0, len(indx)):
            self.err[indx[i]] = math.sqrt(error[i])
        #这一块并没有改变数据,只是显示它的排名情况1,2,5,4,7,3,6(排名),也存在分数的可能性
        r_err = ss.rankdata(self.err)  #rank of the error from smallest (1) to largest
        self.prob = [1/(len(r_err)-i+1) for i in r_err]
        # print('1',self.prob)

        
    def priorized_sample(self,size):
        prb = [i**self.alpha for i in self.prob]
        t_s = [prb[0]]
        for i in range(1,len(self.prob)):
            t_s.append(prb[i]+t_s[i-1])
        # print('2',t_s)
        batch = []
        mx_p = t_s[-1]
        
        smp_set = set()
        
        while len(smp_set)<size:
            tmp = np.random.uniform(0,mx_p)
            # print('tmp:',tmp)
            for j in range(0, len(t_s)):
                if t_s[j] > tmp:
                    smp_set.add(j)
                    break
        for i in smp_set:
            batch.append([self.buffer[i], i])
        # print(smp_set)
        return np.array(batch)   #返回对映的索引,组成类似二维数组的结构,用来干嘛
#         return np.reshape(np.array(random.sample(self.buffer,size)),[size,6])
class SumTree():
    def __init__(self, capacity):
        #sumtree能存储的最多优先级个数
        self.capacity = capacity
        #顺序表存储二叉树
        self.tree = [0] * (2*capacity - 1)
        #每个优先级所对应的经验片断
        self.data = [None] * capacity
        self.size = 0   #优先级存放时的一个累加器
        self.curr_point = 0  #优先级存放时的索引
    
    #添加一个节点数据,默认优先级为当前的最大优先级值+1
    def add(self, data):
        self.data[self.curr_point] = data
        # print(self.data)
        self.update(self.curr_point, max(self.tree[self.capacity -1 : self.capacity + self.size]) + 1)
        self.curr_point += 1
        if self.curr_point >= self.capacity:
            self.curr_point = 0
        if self.size < self.capacity:
            self.size += 1
        # print(self.data)

    #更新一个节点的优先级权重,对应的索引位置,从叶子节点第一位开始,0出发
    def update(self, point, weight):
        # print(weight)
        # print(self.tree)
        idx = point + self.capacity - 1
        change = weight - self.tree[idx]

        self.tree[idx] = weight
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2
        # print('tree:',self.tree)
        
    def get_total(self):
        return self.tree[0]
    
    #获取最小的优先级,在计算重要性比率中使用
    def get_min(self):
        return min(self.tree[self.capacity - 1 : self.capacity + self.size -1])
    
    #根据一个权重进行抽样
    def sample(self, v):
        idx = 0
        while idx < self.capacity - 1:
            l_idx = idx * 2 +1
            r_idx = l_idx + 1
            if self.tree[l_idx] >= v:
                idx = l_idx
            else:
                idx = r_idx
                v = v - self.tree[l_idx]
        point = idx - (self.capacity - 1)
        #返回抽样得到的位置,transition信息,该样本的概率
        return point, self.data[point], self.tree[idx] / self.get_total()

class Memory(object):
    def __init__(self, batch_size, max_size, beta):
        self.batch_size = batch_size   #mini_batch大小
        self.max_size = 2 ** math.floor(math.log2(max_size))
        self.beta = beta

        self._sum_tree = SumTree(max_size)
    
    def store_transition(self, s, a, r, s_, done):
        self._sum_tree.add((s, a, r, s_, done))
    
    def get_mini_batch(self):
        n_sample = self.batch_size if self._sum_tree.size >= self.batch_size else self._sum_tree.size
        total = self._sum_tree.get_total()

        #生成n_sample个区间
        step = total // n_sample
        points_transsitions_probs = []
        #在每个区间中均匀随机取一个数,并去sumtree中采样
        for i in range(n_sample):
            v = np.random.uniform(i * step, (i+1) * step -1)
            t = self._sum_tree.sample(v)
            points_transsitions_probs.append(t)
        
        points, transitions, probs = zip(*points_transsitions_probs)

        #计算重要性比率
        max_importance_ratio = (n_sample * self._sum_tree.get_min()) ** -self.beta
        importance_ratio = [(n_sample * probs[i])**-self.beta / max_importance_ratio for i in range(len(probs))]

        return points, tuple(np.array(e) for e in zip(*transitions)), importance_ratio
    
    def update(self, points, td_error):
        for i in range(len(points)):
            self._sum_tree.update(points[i], td_error[i]) 


