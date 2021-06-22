# -*- coding: utf-8 -*-

import simpy
import random
from A1_BasicFunc import Ordergenerator, RiderGenerator
from A1_Class import Store, Platform_pool
from A2_Func import Platform_process, ResultPrint
import operator

#Parameter define
order_interval = 5
interval = 5
p2 = 20
thres_p = 1
run_time = 50
rider_working_time = 120
#env = simpy.Environment()
store_num = 10
rider_num = 3
rider_gen_interval = 10
rider_speed = 2.5
rider_capacity = 4
ITE_NUM = 2


class scenario(object):
    def __init__(self, name, p1, p2):
        self.name = name
        self.platform_work = p1
        self.unserved_order_break = p2
        self.res = []

scenarios = []

infos = [['A',False, False],['B',True, True],['C',True, False]]
for info in infos:
    sc = scenario(info[0], info[1], info[2])
    scenarios.append(sc)

for ite in range(ITE_NUM):
    #instance generate
    #가게 생성
    index = 0
    rider_history = None
    order_history = None
    store_history = []
    for _ in range(store_num):
        store_history.append(list(random.sample(range(0, 50), 2)))
    for sc in scenarios:
        Rider_dict = {}
        Orders = {}
        Platform2 = Platform_pool()
        Store_dict = {}
        #run
        env = simpy.Environment()
        for store_name in range(store_num):
            loc = store_history[store_name]
            store = Store(env, Platform2, store_name, loc=loc, capacity=10, print_para=False)
            Store_dict[store_name] = store
        env.process(RiderGenerator(env, Rider_dict, Platform2, Store_dict, Orders, speed=rider_speed,
                                   interval=rider_gen_interval, runtime=run_time, gen_num=rider_num,
                                   capacity=rider_capacity, history= rider_history))
        env.process(Ordergenerator(env, Orders, Store_dict, interval=order_interval, history = order_history))
        if sc.platform_work == True:
            env.process(Platform_process(env, Platform2, Orders, Rider_dict, p2, thres_p, interval, speed=rider_speed,
                                         end_t=1000, unserved_order_break=sc.unserved_order_break))
        env.run(run_time)
        res = ResultPrint(sc.name + str(ite), Orders, speed=rider_speed)
        sc.res.append(res)
        if index == 0:
        #필요한 정보 저장
            rider_history = []
            order_history = []
            for rider_name in Rider_dict:
                rider = Rider_dict[rider_name]
                rider_history.append(rider.start_time)
            rider_history.append(1000)
            rider_history.sort()
            for order_name in Orders:
                order = Orders[order_name]
                order_history.append([order.time_info[0], order.store, order.location])
            order_history.append([1000, 1, [25,25]])
            order_history.sort(key = operator.itemgetter(0))
        input('주문 이력 수{}/ 생성 주문 수 {} '.format(len(order_history),len(Orders)))
        index += 1

for sc in scenarios:
    count = 1
    for res_info in sc.res:
        try:
            print('시나리오로 {} ITE {} /전체 고객 {} 중 서비스 고객 {}/ 서비스율 {}/ 평균 LT :{}/ 평균 FLT : {}/직선거리 대비 증가분 : {}'.format(sc.name , count,res_info[0],res_info[1],res_info[2],res_info[3],res_info[4],res_info[5]))
        except:
            print('시나리오로 {} ITE {} 결과 없음'.format(sc.name , count))
        count += 1