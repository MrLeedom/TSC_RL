#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
'''
   @Author:leedom

   Created on Wed Mar 06 13:54:31 2019
   Description:sumo中环境的实验文件
            调参问题:根据个人电脑的sumo工具各个文件的安装目录
   方法介绍:
        subscribe(objectID, varIDs, begin, end):订阅对象的值在一定时间间隔内
        traci.trafficlights.getIDList() 返回场景中所有交通灯的id列表
        traci.vehicle.getIDList():      返回网络中的所有对象
        traci.vehicle.getSubscriptionResults() 获取订阅值（即获取所有车辆的位置和速度值）
        traci.lanearea.getJamLengthVehicle(det) 车道上面的停车数
   License: (C)Copyright 2019
'''
tools_path = 'D:/SUMO/tools'
sumoBinary = "D:/SUMO/bin/sumo"
sumoConfig = "C:/Users/Administrator/Desktop/code/shixin_shanyin_7/shixin_shanyin.sumocfg"
#SUMO环境代码
import os, sys
sys.path.append(tools_path)   
import traci
import traci.constants as tc
import numpy as np
import datetime
outputfile = open('C:/Users/Administrator/Desktop/trafficSignalControl/CSP/src/main/RL/output/queue_length.csv','w+')

print("phase_1,max_1,sum_1,phase_2,max_2,sum_2,phase_3,max_3,sum_3,phase_4,max_4,sum_4",file=outputfile)

#Environment Model
sumoCmd = [sumoBinary, "-c", sumoConfig]  #The path to the sumo.cfg file


Gmin = 15
Gmax = 60
w1 = 4 / 10
w2 = 3 / 10
w3 = 3 / 10
#重置环境,并返回场景中所有交通灯的ID列表
def reset():
    traci.start(sumoCmd)
    tls = traci.trafficlights.getIDList()
    return tls

#仿真软件给我们提供的状态环境,在这边实际就是我们希望得到的输入三维矩阵数据(代表状态)
def getState():   #实际的速度未测试,需要进行测试一下
    for veh_id in traci.vehicle.getIDList():
        traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ACCUMULATED_WAITING_TIME))
    p_state_2 = np.zeros((20,30,2))
    for veh_id in traci.vehicle.getIDList():
        p = traci.vehicle.getSubscriptionResults(veh_id) #获取订阅值（即获取对应车辆的位置和速度值）
        ps = p[tc.VAR_POSITION]
        spd = p[tc.VAR_SPEED]
    #方法二:拼接
        if (ps[0] > 431 and ps[0] < 581 and ps[1] > 586 and ps[1] < 599): #西进口
            p_state_2[int((ps[1]-586)/3.3), int((ps[0]-431)/5)] = [1, round(spd/13.89)]
        elif (ps[0] > 625 and ps[0] < 775 and ps[1] > 598 and ps[1] < 611): #东进口
            p_state_2[int((ps[1]-598)/3.3)+4, int((ps[0]-625)/5)] = [1, round(spd/13.89)]
        elif (ps[0] > 584 and ps[0] < 603 and ps[1] > 614 and ps[1] < 764): #北进口
            p_state_2[int((ps[0]-584)/3.3)+8, int((ps[1]-614)/5)] = [1, round(spd/13.89)]
        elif (ps[0] > 603 and ps[0] < 622 and ps[1] > 433 and ps[1] < 583): #南进口
            p_state_2[int((ps[0]-603)/3.3)+14, int((ps[1]-433)/5)] = [1, round(spd/13.89)]
    p_state_2 = np.reshape(p_state_2, [-1, 600, 2])
    return p_state_2

#保证信号配时方案在合理范围内
def getCorrectCycle(phases):
    for i in range(len(phases)):
        if phases[i] < Gmin:
            phases[i] = Gmin
        if phases[i] > Gmax:
            phases[i] = Gmax
    return phases

#交通灯在当前的周期信号配时状态下,可以选择的合法动作集合,将相位绿灯时间转换成0~8这样的数,另外-1也要特殊处理
def getLegalAction(phases):
    legal_action = np.zeros(9)-1
    i = 0
    for x in phases: #phases是一个长度为4的list,初始化值为[20,20,20,20]
        if x - 5 > Gmin:
            legal_action[i] = i
        if x + 5 <= Gmax:
            legal_action[i+5] = i+5
        i += 1
    legal_action[4] = 4
    return legal_action#这个循环的意思是如果四个相位的时间都在5-60间，legal_action就是[0,1,2...7,8]这样一个序列，如果有一个不在区间内，那么对应的i和i+5的值就是-1(既不能执行该动作)

#根据当前的相位和选择的动作调整得到新的相位配时方案
def getPhaseFromAction(phases, act):
    if act<4: #如果选择的动作序号小于4，那对应的动作相位时间-5s
        phases[int(act)] -= 5
    elif act>4: #如果选择的动作序号大于4，那么对应的动作相位时间+5s
        phases[int(act)-5] += 5
    return phases

#the process of the action
#input: traffic light; new phases; waiting time in the beginning of this cycle
#output: new state; reward; End or not(Bool); new waiting time at the end of the next cycle
def step(tls, ph, wait_time_map, pre_queue, pre_delay, pre_throughput):  #parameters: the phase duration in the green signals,执行动作的函数
    tls_id = tls[0]                    #获取当前交通灯的id，因为单交叉口只有一个所以取0,多路口的话可能要遍历
    init_p = traci.trafficlights.getPhase(tls_id) #获取交通灯的当前处于第几相位,返回索引
    prev = -1
    changed = False
    queue_max = []   #存放一周期内的所有相位的排队长度
    queue_sum = []
    indx = []
    dets = traci.lanearea.getIDList()

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:              #如果交通网络中的车辆数量大于0,也就是说路网有车的情况下
        c_p = traci.trafficlights.getPhase(tls_id)                  #获取交通灯的当前相位情况,四相位的话返回0,1,2,3,4,5,6,7
        if c_p != prev and c_p%2 == 0:                              #如果当前相位序号不是前一个相位且为序号偶数
            if step > ph[0]:                                        #当第一个相位运行结束之后
                queue_length=[]
                for det in dets:
                    queue=traci.lanearea.getJamLengthVehicle(det)
                    queue_length.append(queue)
                queue_max.append(max(queue_length))
                queue_sum.append(sum(queue_length))  # /len(queue_length)
                indx.append(np.argmax(queue_length))
            traci.trafficlights.setPhaseDuration(tls_id, ph[int(c_p/2)]-0.5)
            #设置当前相位的持续时间为相位序号int(c_p/2)对应时长再-0.5后的时间
            prev = c_p                   #prev表示前一个相位？
        if init_p != c_p:                #如果初始相位与当前相位不同，表示相位已经改变
            changed = True
        if changed:
            if c_p == init_p:            #如果当前相位等于初始相位，表示一个周期结束？
                break
        traci.simulationStep()           #模拟1s
        step += 1
        #车辆的累计等待时间.在这个库包中还有很多的方法调用,后期可以调用
        if step % 5 == 0:
            for veh_id in traci.vehicle.getIDList():
                wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
    p_state = getState()
    print(str(indx[0]) + "," + str(queue_max[0]) + "," + str(queue_sum[0]) + "," + str(indx[1]) + "," + str(
        queue_max[1]) + "," + str(queue_sum[1]) +
          "," + str(indx[2]) + "," + str(queue_max[2]) + "," + str(queue_sum[2]) + "," + str(indx[3]) + "," + str(
        queue_max[3]) + "," + str(queue_sum[3]), file=outputfile)

    wait_t = dict(wait_time_map)                           #计算所有车辆的累计等待时间的和

    done = False
    if traci.simulation.getMinExpectedNumber() == 0:    #判断路网中的车辆是否为空,结束的标准可以再加
        done = True                                     #结束的标准可以修改,这个在高峰期不太合适,可以改成指定时段内
    print('四相位最长车道序号和排队车辆数分别为', indx, queue_max)
    #奖励函数的设计部分
    #1.四个相位结束之后,排队长度最长的记录下来
    max_queue = max(queue_max)
    reward1 = pre_queue**2 - max_queue**2 
    #2.四相位结束之后,累计等待时延的差值
    current_delay = sum(wait_t[x] for x in wait_t)
    reward2 = current_delay - pre_delay
    #3.四相位结束之后,累积的吞吐量
    current_throughput = len(wait_t) 
    reward3 = current_throughput - pre_throughput

    reward = w1 * reward1 + w2 * reward2 + w3 * reward3
    #原设计-----师兄的版本
    # reward = min(queue_max)-max(queue_max)     #也就是目标想要周期内各个相位的排队车辆数较为平均,内部比较得到奖励
    q_m = sum(queue_sum) / sum(ph)             #平均排队长度,这个除以了相位最长的时间,自然会让整个结果缩小
    return p_state, reward, done, wait_t, q_m, max_queue, current_delay, current_throughput       #返回状态矩阵，奖励值，是否最终状态，本次的累计等待总时间

#close the environment after every episode
def end():
    traci.close()
