#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
'''
   @Author:leedom

   Created on Thu Mar 07 12:33:59 2019
   Description:基于优先级的记忆池
        功能:   
            1.定义时需要明确记忆池的大小
            2.add方法向记忆池中添加新的记忆,将其组成一维数组
            3.updateErr更新记忆池中排名情况
            4.priorized_sample从记忆池中进行取样的操作
   License: (C)Copyright 2019
'''
import math
import scipy.stats as ss
import numpy as np

class priorized_experience_buffer(object):
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
        #return np.reshape(np.array(random.sample(self.buffer,size)),[size,6])