#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
'''
   @Author:leedom

   Created on Tue Mar 05 15:48:29 2019
   Description:3DQN+CNN结构
   License: (C)Copyright 2019
'''
#The class of dueling DQN with three convolutional layers
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import matplotlib.pyplot as plt
import scipy.misc
import os
import scipy.stats as ss
import math
import collections
#神经网络
class Qnetwork():
    def __init__(self,h_size,action_num):
        #The network recieves a state from the sumo, flattened into an array.
        #从sumo获取一个状态，展平成一个数组,60行 * 60列
        #It then resizes it and processes it through three convolutional layers.
        #修改输入的结构后，通过三个卷积层进行处理
        self.scalarInput =  tf.placeholder(shape=[None,3600,2],dtype=tf.float32) #输入
        self.legal_actions =  tf.placeholder(shape=[None,action_num],dtype=tf.float32) #行为空间
        
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,60,60,2])  #把输入的结构修改
        #slim.conv2d自带卷积功能+激活函数,其中slim可以使建立\训练和评估神经网络更加简单
        #第一层卷积层c，卷积核个数（即输出）32，大小为8*8，步长为4
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', activation_fn=self.relu, biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', activation_fn=self.relu, biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2,num_outputs=128,kernel_size=[3,3],stride=[2,2],padding='VALID', activation_fn=self.relu, biases_initializer=None)
        '''
        #参数说明:
            inputs 需要做卷积的图像
            num_outputs 指定卷积核的个数(就是filter的个数)
            kernel_size 用于指定卷积核的维度(卷积核的宽度,卷积核的高度)
            stride 为卷积时在图像每一维的步长
            padding 为padding方式选择,分成Valid和same两种
            activation_fn 用于激活函数的指定,默认为relu函数
            biases_initializer 用于指定biases的初始化程序
        '''   
        #It is split into Value and Advantage
        #将输出扁平化但保留batch_size,假设第一维是batch
        self.stream = slim.flatten(self.conv3) #将conv3的输出值从三维变为1维的数组，保留batch_size，假设第一维是batch
        #前两个参数分别为网络输入\输出的神经元数量
        self.stream0 = slim.fully_connected(self.stream, 128, activation_fn=self.relu) #输出单元为128个
    
        self.streamA = self.stream0
        self.streamV = self.stream0
        
        
        self.streamA0 = slim.fully_connected(self.streamA,h_size, activation_fn=self.relu)
        self.streamV0 = slim.fully_connected(self.streamV, h_size, activation_fn=self.relu)
        
        xavier_init = tf.contrib.layers.xavier_initializer() #权重的初始化器,这个初始化器是用来保持每一层的梯度大小都差不多相同
        action_num = np.int32(action_num)
        self.AW = tf.Variable(xavier_init([h_size,action_num])) #设置Advantage的权重
        self.VW = tf.Variable(xavier_init([h_size,1])) #设置分析state的value的权重,h_size表示几个状态,它所对应的值
        self.Advantage = tf.matmul(self.streamA0,self.AW)
        self.Value = tf.matmul(self.streamV0,self.VW)
        
        #Then combine them together to get our final Q-values.采用优势函数的平均值代替上述的最优值,采用这种方法,虽然使得值函数V和优势函数A不再完美的表示值函数和优势函数(在语义上的表示),但是这种操作提高了稳定性
        #在实际中,一般要将动作优势流设置为单独动作优势函数减去某状态下所有动作的优势函数的平均值,这样做可以保证该状态下各个动作的优势函数相对排序不变,而且
        #可以缩小Q值的范围,去除多余的自由度,提高算法的稳定性
        self.Qout0 = self.Value  + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        #The final Q value is the addition of the Q value and penelized value for illegal actions
        self.Qout = tf.add(self.Qout0, self.legal_actions)  #这一块也是创新点设计,加大所谓的惩罚力度
        #The predicted action
        #预测的动作,argmax中的0:按列计算  1:按行计算
        self.predict = tf.argmax(self.Qout,1) #tf.argmax(input, axis=None, name=None, dimension=None)，axis为1则为按行取最大的那个数的序号
        
        #Below we obtain the loss by taking the mean square error between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        #将一个值化为一个概率分布的向量
        self.actions_onehot = tf.one_hot(self.actions,np.int32(action_num),dtype=tf.float32)
        #压缩求和,用于降维,axis=1表示行
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
#         self.Q = tf.reduce_sum(self.Qout, axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        
    def relu(self, x, alpha=0.01, max_value=None):
        '''ReLU.这个函数在此处到底有没有用到,待研究

        alpha: slope of negative section.
        '''
        negative_part = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        if max_value is not None:
            #tf.clip_by_value(A, min, max)输出一个张量,把A中的每一个元素的值都压缩到min和max之间
            x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                                 tf.cast(max_value, dtype=tf.float32))
        x -= tf.constant(alpha, dtype=tf.float32) * negative_part#若为正数,则输出该数,若为负数,输出一个很小的正数,缩小一点
        return x

#--------------------------------------------------------------------------------------------------------
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
