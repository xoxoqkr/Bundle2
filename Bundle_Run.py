# -*- coding: utf-8 -*-
import simpy
import store_class

#실행부
# str for test
env = simpy.Environment()
Platform = simpy.Store(env)
env.process(store_class.ordergenerator(env, Platform))
rider1 = store_class.rider(env,0,Platform)
rider2 = store_class.rider(env,1,Platform)
env.run(200)
print("queue check",Platform.put_queue)
