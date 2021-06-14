# -*- coding: utf-8 -*-
import random
import simpy
#import store_class
import Basic_Class2 as Basic


order_interval = 1
rider_working_time = 120
interval = 5
p2 = 20
thres_p = 1
run_time = 120
#실행부
env = simpy.Environment()
#Platform = simpy.Store(env)
Orders = {}
Platform = []
store_num = 2
rider_num = 3
Store_dict = {}
Rider_dict = {}
rider_gen_interval = 10
rider_speed = 2.5

#Before simulation, generate the stores.
for store_name in range(store_num):
    loc = list(random.sample(range(0,50),2))
    store = Basic.Store(env, Platform, store_name, loc = loc, capacity = 3)
    #env.process(store.StoreRunner(env, Platform, capacity=store.capacity))
    Store_dict[store_name] = store

env.process(Basic.RiderGenerator(env, Rider_dict, Platform, Store_dict, speed = rider_speed, end_time = 120, interval = rider_gen_interval, runtime = run_time, gen_num = rider_num))
env.process(Basic.ordergenerator(env, Orders, Platform, Store_dict, interval = order_interval))
env.process(Basic.Platform_process(env, Platform, Orders, Rider_dict, p2, thres_p, interval, speed = rider_speed, end_t = 1000))
env.run(run_time)
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
