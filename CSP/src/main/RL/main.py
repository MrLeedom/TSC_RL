#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
'''
   @Author:leedom

   Created on Thu Mar 07 14:19:24 2019
   Description:
   License: (C)Copyright 2019
'''
from DDDQN import Qnetwork
from priority_experience_replay import priorized_experience_buffer
import car_env as sumo
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

#使用主网络参数更新目标网络
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    #enumerate函数用于将一个可遍历的数据对象(如列表,元组或字符串)组合成一个索引序列,同时列出数据和数据下标
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
#相关参数配置
path = "C:/Users/Administrator/Desktop/trafficSignalControl/CSP/src/main/RL/models" #储存模型的路径.
batch_size = 64            #每个训练步需要的经验数量,也就是每次训练使用多少训练记录
update_freq = 1             #多久执行一次训练操作
y = .99                     #折扣系数,Q值的折算因子
startE = 1                  #初始的epsilon
endE = 0.01                 #最终的epsilon
anneling_steps = 2000.      #epsilon衰减需要的步数.从startE到endE衰减所需的步骤数
num_episodes = 201           #整个过程训练多少次
pre_train_steps = 500      #预先训练步数
max_epLength = 50         #一次训练最多经历多少步
load_model = False          #是否要载入一个已有的模型.
action_num =  9             #动作空间大小
h_size = 64                 #最后一层卷积层的大小,这个需要研究一下
tau = 0.001                 #更新目标网络的速率
total_steps = 0             #记录一个episode训练经过了多少步,也就是多少次动作的选择
init_phases = [40,25,25,20]         #初始相位时间

#用于绘图的值
loss_plot=[]
r_plot=[]
q_plot=[]
tf.reset_default_graph()  #此函数用于清楚默认图形堆栈并重置全局默认图形,默认图形是当前线程的一个属性

mainQN = Qnetwork(h_size,np.int32(action_num))
targetQN = Qnetwork(h_size,np.int32(action_num))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

trainables = tf.trainable_variables()             #返回的是需要训练的变量列表
targetOps = updateTargetGraph(trainables,tau)     #更新网络

#define the memory,此处可以改成experience_buffer()则为无优先级排序的记忆池
myBuffer0 = priorized_experience_buffer()

#此处只是为了得到衰减率
epsilon = startE
stepDrop = (startE - endE)/anneling_steps

#一系列存放训练过程的列表集合
jList = []                      #一个episode需要的步数,我们周期进行选择,也就是多少周期
rList = []                      #一个episode的奖励值
wList = []                      #一个episode的总等待时间
awList = []                     #一个episode的平均等待时间
tList = []                      #一个episode的吞吐量
nList = []                      #停车率,换算成停车次数/总产生的车辆数


#创建用于保存模型的目录
if not os.path.exists(path):
    os.makedirs(path)


#它能让你运行图的时候,插入一些计算图,这些计算是由某些操作构成的,这对于工作在交互环境中的人们来说非常便利
#tf.Session需要在启动session之前构建整个计算图,然后启动该计算图
sess = tf.InteractiveSession()

#record the loss ,tf.summary的各类方法,能够保存训练过程以及参数分布图并在tensorboard显示
tf.summary.scalar('Loss', mainQN.loss)

rfile = open(path+'/reward-rl.txt', 'w')
wfile = open(path+'/wait-rl.txt', 'w')
awfile = open(path+'/acc-wait-rl.txt', 'w')
tfile = open(path+'/throput-rl.txt', 'w')
r_output = open('C:/Users/Administrator/Desktop/trafficSignalControl/CSP/src/main/RL/output/reward.csv','w+')
q_output = open('C:/Users/Administrator/Desktop/trafficSignalControl/CSP/src/main/RL/output/queue.csv','w+')

#将所有的summary整理在一起
merged = tf.summary.merge_all()
s_writer = tf.summary.FileWriter(path+'/train', sess.graph)
s_writer.add_graph(sess.graph)

sess.run(init)
tf.global_variables_initializer().run()


if load_model == True:
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess,ckpt.model_checkpoint_path)
updateTarget(targetOps,sess)        #Set the target network to be equal to the primary network.

tf.summary.FileWriter("logs/", self.sess.graph)
#所有episode的运行过程
for i in range(1, num_episodes):
    episodeBuffer0 = priorized_experience_buffer()        #首先明确好经验池的大小问题
    environment = sumo.reset()                                    #重置环境
    current_state = sumo.getState()                                      #拿到环境的状态信息
    current_phases = list(init_phases)                    #将初始相位设置成list
    wait_time_map = {}
    wait_time = 0                                         #我添加的，因为wait_time没有定义
    done = False
    rAll = 0                                              #奖励值进行不断的累加
    j = 0                                                 #一个episode下步数累加器
    flag = False
    r_plot_i = []
    q_plot_i = []
    #这三个参数可以根据实际的来调整,先训练一会,然后累计前相位的相关参数
    pre_queue = 12 
    pre_delay = 1000
    pre_throughput = 20
    print('step:' + str(i) + ',precent:%.3f' % (i / num_episodes * 100) + '%')
    while j < max_epLength:    #一个eposide跑50个周期
        j+=1
        #确保当前的信号配时方案在最大绿灯时间和最小绿灯时间之间
        current_phases = sumo.getCorrectCycle(current_phases)
        #根据当前的状态得到有效的动作
        legal_action = sumo.getLegalAction(current_phases)
        #选取动作,注意这个epsilon是一直在变化的
        action = mainQN.choose_action(epsilon, current_state, legal_action, sess, flag)
        #根据动作来修改当前相位时间并返回修改后的相位
        cycle = sumo.getPhaseFromAction(current_phases,action)
        #获得下一个状态、奖励、是否结束、以及等待时间
        next_state, reward, done, wait_time_map, q_m, current_queue, current_delay, current_throughput = sumo.step(environment, cycle, wait_time_map, pre_queue, pre_delay, pre_throughput)

        total_steps += 1

        #往记忆池中添加信息
        legal_a_one = [0 if x!=-1 else -99999 for x in legal_action] #the penalized Q value for illegal actions
        legal_act_s1 = sumo.getLegalAction(cycle)#下一个交通相位中的可选行为
        legal_a_one_s1 = [0 if x!=-1 else -99999 for x in legal_act_s1] #legal_a_one和legal_a_one_s1分别是前一轮相位配置和当前相位配置对应动作空间的值
        episodeBuffer0.add(np.reshape(np.array([current_state, action, reward, next_state,done,legal_a_one, legal_a_one_s1]),[1,7])) #Save the experience to our episode buffer.

        if total_steps > pre_train_steps:
            flag = True                                                   #达到此要求开始进行训练
            if epsilon > endE:
                epsilon -= stepDrop                                                          #选择动作的eplison
            if total_steps % (update_freq) == 0:                                             #多少步来更新一次网络
                trainBatch = myBuffer0.priorized_sample(batch_size)                          #Get a random batch of experiences.
                indx = np.reshape(np.vstack(trainBatch[:,1]), [batch_size])                  #取训练集中所有的执行动作a（即第二个值）。
                indx = indx.astype(int)                                                      #修改indx的数据类型
                trainBatch = np.vstack(trainBatch[:,0])                                      #取训练集中的状态（指当前状态，第一个值）

                #Below we perform the Double-DQN update to the target Q-values
                #action from the main QN
                Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3]),mainQN.legal_actions:np.vstack(trainBatch[:,5])})
                #Q value from the target QN
                Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3]), targetQN.legal_actions:np.vstack(trainBatch[:,6])})
                # get targetQ at s'

                end_multiplier = -(trainBatch[:,4] - 1)  #if end, 0; otherwise 1
                
                doubleQ = Q2[range(batch_size),Q1]
                targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)                     #y是折扣系数，这个值就是TD目标值。

                #Update the network with our target values.
                summry, err, ls, _ = sess.run([merged, mainQN.td_error, mainQN.loss, mainQN.updateModel],  \
                    feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1],mainQN.legal_actions:np.vstack(trainBatch[:,5])})

                s_writer.add_summary(summry, total_steps)
                #update the target QN and the memory's prioritization
                updateTarget(targetOps,sess)                                                #Set the target network to be equal to the primary network.
                myBuffer0.updateErr(indx, err)                                              #更新记忆池
                r_plot_i.append(reward)
                loss_plot.append(ls)
                q_plot_i.append(q_m)
        rAll += reward
        current_state = next_state                                                          #更新状态
        current_phases = cycle                                                              #更新当前的相位时间配置
        pre_queue = current_queue
        pre_delay = current_delay
        pre_throughput = current_throughput
        if done == True:
            break
    sumo.end()
    if len(r_plot_i)>0:
        r_plot.append(sum(r_plot_i)/len(r_plot_i))
        q_plot.append(sum(q_plot_i)/len(q_plot_i))
    #save the data into the myBuffer
    myBuffer0.add(episodeBuffer0.buffer)

    jList.append(j)                                             #训练步数
    rList.append(rAll)                                          #最终奖励值
    rfile.write(str(rAll)+'\n')
    wt = sum(wait_time_map[x] for x in wait_time_map)           #车辆总等待时间
    wtAve = wt/len(wait_time_map)                               #路网中车辆的平均等待时间
    wList.append(wt)
    wfile.write(str(wt)+'\n')
    awList.append(wtAve)
    awfile.write(str(wtAve)+'\n')
    tList.append(len(wait_time_map))                            #有多少辆车进入了路网,这个跟结束标志有点关系,确保结束
    tfile.write(str(len(wait_time_map))+'\n')
    tmp = [x for x in wait_time_map if wait_time_map[x] > 1]
    nList.append(len(tmp)/len(wait_time_map))
#     print("Total Reward---------------",  rAll)
#     #Periodically save the model.
#     if i % 100 == 0:
#         saver.save(sess,path+'/model-'+str(i)+'.cptk')
#         print("Saved Model")
# #         if len(rList) % 10 == 0:
# #             print(total_steps,np.mean(rList[-10:]), e)
# saver.save(sess,path+'/model-'+str(i)+'.cptk')
# print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
saver.save(sess,path+'/model-'+str(i)+'.cptk')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
print(str(r_plot).strip('[').strip(']'),file=r_output)
print(str(q_plot).strip('[').strip(']'),file=q_output)
plt.plot(np.arange(len(q_plot)), q_plot)
plt.ylabel('平均排队长度')
plt.xlabel('训练回合数')
plt.show()