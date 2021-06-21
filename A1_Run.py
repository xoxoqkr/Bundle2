# -*- coding: utf-8 -*-

import simpy
import random
from A1_BasicFunc import Ordergenerator, RiderGenerator
from A1_Class import Store, Platform_pool
from A2_Func import Platform_process

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
#Platform = {}
Platform2 = Platform_pool()
#input(type(Platform2))
#input(type(Platform2.platform))
store_num = 10
rider_num = 3
Store_dict = {}
Rider_dict = {}
rider_gen_interval = 10
rider_speed = 2.5
unserved_order_break = False
rider_capacity = 5

#Before simulation, generate the stores.
for store_name in range(store_num):
    loc = list(random.sample(range(0,50),2))
    #store = Store(env, Platform, store_name, loc = loc, capacity = 10, print_para= False)
    store = Store(env, Platform2, store_name, loc=loc, capacity=10, print_para=False)
    #env.process(store.StoreRunner(env, Platform, capacity=store.capacity))
    Store_dict[store_name] = store

#env.process(RiderGenerator(env, Rider_dict, Platform, Store_dict, Orders, speed = rider_speed,  interval = rider_gen_interval, runtime = run_time, gen_num = rider_num, capacity = rider_capacity))
env.process(RiderGenerator(env, Rider_dict, Platform2, Store_dict, Orders, speed = rider_speed,  interval = rider_gen_interval, runtime = run_time, gen_num = rider_num, capacity = rider_capacity))
env.process(Ordergenerator(env, Orders, Store_dict, interval = order_interval))
#env.process(Platform_process(env, Platform, Orders, Rider_dict, p2, thres_p, interval, speed = rider_speed, end_t = 1000, unserved_order_break= unserved_order_break))
env.process(Platform_process(env, Platform2, Orders, Rider_dict, p2, thres_p, interval, speed = rider_speed, end_t = 1000, unserved_order_break= unserved_order_break))
env.run(run_time)