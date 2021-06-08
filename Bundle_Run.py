# -*- coding: utf-8 -*-
import simpy
import store_class
import Basic_Class2 as Basic


order_interval = 1
rider_working_time = 120
interval = 5
p2 = 5
thres_p = 1
#실행부
env = simpy.Environment()
#Platform = simpy.Store(env)
Orders = {}
Platform = []
store_num = 2
rider_num = 3
Store_dict = {}
Rider_dict = {}

for store_name in range(store_num):
    store = Basic.Store(env, Platform, store_name, capacity = 1)
    Store_dict[store_name] = store

for rider_name in range(rider_num):
    rider = Basic.rider(env,rider_name,Platform, Store_dict, end_time = rider_working_time)
    Rider_dict[rider_name] = rider

env.process(Basic.ordergenerator(env, Orders, Platform, Store_dict, interval = order_interval))
env.run(120)
#env.process(Basic.Platform_process(env, Platform, Orders, Rider_dict, p2, thres_p, interval, speed = 1, end_t = 1000))
"""
#rider1 = rider(env,0,Platform, Store_list)
#rider2 = rider(env,1,Platform, Store_list)
env.run(100)
#print("queue check",Platform.put_queue)


#실행부
# str for test
env = simpy.Environment()
Platform = simpy.Store(env)
env.process(store_class.ordergenerator(env, Platform))
rider1 = store_class.rider(env,0,Platform)
rider2 = store_class.rider(env,1,Platform)
env.run(200)
print("queue check",Platform.put_queue)

"""
