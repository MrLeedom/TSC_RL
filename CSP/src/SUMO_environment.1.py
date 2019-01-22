# -*- coding: utf-8 -*-
'''
   @author:leedom

   Created on Tue Jan 15 19:29:59 2019
   description:
'''
#SUMO环境代码
import os, sys
sys.path.append('C:/tools/SUMO/sumo-0.32.0/tools')
import traci
import traci.constants as tc
import numpy as np
import datetime

#Environment Model
sumoBinary = "C:/tools/SUMO/sumo-0.32.0/bin/sumo"
sumoConfig = "C:/Users/LJH/Desktop/paper/code/shixin_shanyin_7/shixin_shanyin.sumocfg"
sumoCmd = [sumoBinary, "-c", sumoConfig]  #The path to the sumo.cfg file

#reset the environment
#重置环境
def reset():
    traci.start(sumoCmd)
    tls = traci.trafficlights.getIDList() #返回场景中所有交通灯的id列表
    return tls
#仿真软件给我们提供的状态环境 
def state():
    for veh_id in traci.vehicle.getIDList():
        traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED))
    p = traci.vehicle.getSubscriptionResults() #获取订阅值（即获取所有车辆的位置和速度值）
    p_state_2 = np.zeros((20,30,2))
    for x in p:
        ps = p[x][tc.VAR_POSITION]
        spd = p[x][tc.VAR_SPEED]
    #方法二:拼接
        if (ps[0] > 431 and ps[0] < 581 and ps[1] > 586 and ps[1] < 599): #西进口
            p_state_2[int((ps[1]-586)/3.3), int((ps[0]-431)/5)] = [1, int(round(spd))]
        elif (ps[0] > 625 and ps[0] < 775 and ps[1] > 598 and ps[1] < 611): #东进口
            p_state_2[int((ps[1]-598)/3.3)+4, int((ps[0]-625)/5)] = [1, int(round(spd))]
        elif (ps[0] > 584 and ps[0] < 603 and ps[1] > 614 and ps[1] < 764): #北进口
            p_state_2[int((ps[0]-584)/3.3)+8, int((ps[1]-614)/5)] = [1, int(round(spd))]
        elif (ps[0] > 603 and ps[0] < 622 and ps[1] > 433 and ps[1] < 583): #南进口
            p_state_2[int((ps[0]-603)/3.3)+14, int((ps[1]-433)/5)] = [1, int(round(spd))]
    p_state_2 = np.reshape(p_state_2, [-1, 600, 2]) #这里面的-1是指第一维度不知道,看后面的操作
    return p_state_2

#get the legal actions at the current phases of the traffic light
def getLegalAction(phases):
    legal_action = np.zeros(9)-1 #长度为9的-1数组，动作空间大小为9？
    i = 0
    for x in phases: #phases是一个长度为4的list,初始化值为[20,20,20,20]
        #if x>5: #如果这个相位时间大于5，那么对应的行为i的值为自身的序号。（因为小于5s就不能继续执行-5s的动作了。但是这里需要修改，因为要考虑最小绿灯时间。）
        if x>15:    
            legal_action[i] = i
        if x<60: #如果小于60，对应行为i+5的值也等于序号。（设60s为最大绿灯时间，大于60s就不能继续执行+5s的动作）
            legal_action[i+5] = i+5
        i +=1
    legal_action[4] = 4 #这个循环的意思是如果四个相位的时间都在5-60间，legal_action就是[0,1,2...7,8]这样一个序列，如果有一个不在区间内，那么对应的i和i+5的值就是-1(既不能执行该动作)
    return legal_action
    
#get the new phases after taking action from the current phases
def getPhaseFromAction(phases, act):
    if act<4: #如果选择的动作序号小于4，那对应的动作相位时间-5s
        phases[int(act)] -= 5
    elif act>4: #如果选择的动作序号大于4，那么对应的动作相位时间+5s
        phases[int(act)-5] += 5
    return phases

#the process of the action
#input: traffic light; new phases; waiting time in the beginning of this cycle
#output: new state; reward; End or not(Bool); new waiting time at the end of the next cycle
def action(tls, ph, wait_time):  #parameters: the phase duration in the green signals,执行动作的函数
    tls_id = tls[0] #获取当前交通灯的id，因为单交叉口只有一个所以取0
    init_p = traci.trafficlights.getPhase(tls_id) #获取交通灯的当前处于第几相位？
    prev = -1
    changed = False
    queue_max=[]
    dets=traci.lanearea.getIDList()

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0: #如果交通网络中的车辆数量大于0
        c_p = traci.trafficlights.getPhase(tls_id) #获取交通灯的当前相位情况
        if c_p != prev and c_p%2==0: #如果当前相位序号不是前一个相位且为序号偶数
            if step>ph[0]:
                queue_length=[]
                for det in dets:
                    queue=traci.lanearea.getJamLengthVehicle(det)
                    queue_length.append(queue)
                queue_max.append(max(queue_length))
            traci.trafficlights.setPhaseDuration(tls_id, ph[int(c_p/2)]-0.5) #设置当前相位的持续时间为相位序号int(c_p/2)对应时长再-0.5后的时间
            prev = c_p #prev表示前一个相位？
        if init_p != c_p: #如果初始相位与当前相位不同，表示相位已经改变
            changed = True
        if changed:
            if c_p == init_p: #如果当前相位等于初始相位，表示一个周期结束？
                break
        traci.simulationStep() #模拟1s
        step += 1
    p_state = state()

    wait_t = 0#sum(wait_temp[x] for x in wait_temp) #计算所有车辆的累计等待时间的和
    
    d = False
    if traci.simulation.getMinExpectedNumber() == 0: #判断路网中的车辆是否为空
        d = True
    r = min(queue_max)-max(queue_max)
    return p_state,r,d,wait_t #返回状态矩阵，奖励值，是否最终状态，本次的累计等待总时间

#close the environment after every episode
def end():
    traci.close()