#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
'''
   @Author:leedom

   Created on Thu Mar 07 09:23:45 2019
   Description:最普通的经验池,采用了队列的形式,添加数据的同时自带更新操作
            功能:
                1.定义时弄清楚所需的池大小
                2.添加记录时,要将其封装成一维数组存放
                3.取样本时,要明确取样的大小batch_size
   License: (C)Copyright 2019
'''
#历程重现
#这个类赋予了网络存储\重采样来进行训练的能力
#经验回放池
class experience_buffer():
    #回放单元大小为50000
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.experience_pool = collections.deque(maxlen=self.buffer_size) #个人添加  
    
    #添加回放经验至经验池中
    def add(self,experience):
        # if len(self.buffer) + len(experience) >= self.buffer_size:
        #     self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        # self.buffer.extend(experience)
        # print("ADDED", len(self.buffer))
        # 以上所有代码替换成,前提experience是一个一维数组
        self.experience_pool.extend(experience)
    
    #从经验池中取大小为size的经验样本    
    def sample(self,size):
        # print("BUFFer:", len(self.buffer))
        #从列表元素中返回指定长度的数据
        return np.reshape(np.array(random.sample(self.experience_pool,size)),[size,6]) #这边这个数字6主要取决于你存放到库中的数据长度