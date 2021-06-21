# -*- coding: utf-8 -*-

import simpy
import random
import A1_BasicFunc as Basic
import A1_Class as Class

#Parameter define
order_interval = 1
rider_working_time = 120
interval = 5
p2 = 20
thres_p = 1
run_time = 120
#Run part
env = simpy.Environment()
Orders = {}
Platform = []
store_num = 10
rider_num = 3
Store_dict = {}
Rider_dict = {}
rider_gen_interval = 10
rider_speed = 2.5

#Before simulation, generate the stores.
for store_name in range(store_num):
    loc = list(random.sample(range(0,50),2))
    store = Class.Store(env, Platform, store_name, loc = loc, capacity = 10, print_para= False)
    #env.process(store.StoreRunner(env, Platform, capacity=store.capacity))
    Store_dict[store_name] = store

env.process(Basic.RiderGenerator(env, Rider_dict, Platform, Store_dict, Orders, speed = rider_speed,  interval = rider_gen_interval, runtime = run_time, gen_num = rider_num))
env.process(Basic.ordergenerator(env, Orders, Store_dict, interval = order_interval))
#env.process(Basic.Platform_process(env, Platform, Orders, Rider_dict, p2, thres_p, interval, speed = rider_speed, end_t = 1000))
env.run(run_time)