# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 08:30:54 2018

@author: LJH
"""
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

#神经网络
class Qnetwork():
    def __init__(self,h_size,action_num):
        #The network recieves a state from the sumo, flattened into an array.
        #从sumo获取一个状态，展平成一个数组
        #It then resizes it and processes it through three convolutional layers.
        #修改输入的结构后，通过三个卷积层进行处理
        self.scalarInput =  tf.placeholder(shape=[None,3600,2],dtype=tf.float32) #输入
        self.legal_actions =  tf.placeholder(shape=[None,action_num],dtype=tf.float32) #行为空间
        
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,60,60,2])  #把输入的结构修改
        #第一层卷积层c，卷积核个数（即输出）32，大小为8*8，步长为4
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', activation_fn=self.relu, biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', activation_fn=self.relu, biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2,num_outputs=128,kernel_size=[3,3],stride=[2,2],padding='VALID', activation_fn=self.relu, biases_initializer=None)

        #It is split into Value and Advantage
        self.stream = slim.flatten(self.conv3) #将conv3的输出值从三维变为1维的数组，保留batch_size，假设第一维是batch
        self.stream0 = slim.fully_connected(self.stream, 128, activation_fn=self.relu) #输出单元为128个
    
        self.streamA = self.stream0
        self.streamV = self.stream0
        
        
        self.streamA0 = slim.fully_connected(self.streamA,h_size, activation_fn=self.relu)
        self.streamV0 = slim.fully_connected(self.streamV, h_size, activation_fn=self.relu)
        
        xavier_init = tf.contrib.layers.xavier_initializer() #权重的初始化器
        action_num = np.int32(action_num)
        self.AW = tf.Variable(xavier_init([h_size,action_num])) #设置Advantage的权重
        self.VW = tf.Variable(xavier_init([h_size,1])) #设置分析state的value的权重
        self.Advantage = tf.matmul(self.streamA0,self.AW)
        self.Value = tf.matmul(self.streamV0,self.VW)
        
        #Then combine them together to get our final Q-values.
        self.Qout0 = self.Value  + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        #The final Q value is the addition of the Q value and penelized value for illegal actions
        self.Qout = tf.add(self.Qout0, self.legal_actions)
        #The predicted action
        #预测的动作
        self.predict = tf.argmax(self.Qout,1) #tf.argmax(input, axis=None, name=None, dimension=None)，axis为1则为按行取最大的那个数的序号
        
        #Below we obtain the loss by taking the mean square error between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,np.int32(action_num),dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
#         self.Q = tf.reduce_sum(self.Qout, axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        
    def relu(self, x, alpha=0.01, max_value=None):
        '''ReLU.

        alpha: slope of negative section.
        '''
        negative_part = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        if max_value is not None:
            x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                                 tf.cast(max_value, dtype=tf.float32))
        x -= tf.constant(alpha, dtype=tf.float32) * negative_part
        return x

#经验回放池
class experience_buffer():
    #回放单元大小为50000
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    #添加回放经验至经验池中
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
        print("ADDED", len(self.buffer))
    
    #从经验池中取大小为size的经验样本    
    def sample(self,size):
        print("BUFFer:", len(self.buffer))
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,6])
    
#目标网络的更新
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
        
#参数
batch_size = 128 #How many experiences to use for each training step.
update_freq = 1 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.01 #Final chance of random action
anneling_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 1500#000 #How many episodes of game environment to train network with.
pre_train_steps = 2000#0000 #How many steps of random actions before training begins.
max_epLength = 500 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
action_num = 9 #total number of actions
path = "./dqn" #The path to save our model to.
h_size = 64 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network