# -*- coding: utf-8 -*-
import operator
import random
import math
import simpy
import copy

class rider(object):
    def __init__(self, env, name, queue, capacity = 3):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.resource = simpy.Resource(env, capacity = capacity)
        self.done_orders = []
        self.capacity = capacity
        env.process(self.Runner(env, queue))


    def Runner(self, env, queue, ready_time = 2, end_time = 100):
        while env.now < end_time:
            if len(queue.put_queue) > 0 and len(self.resource.put_queue) < self.capacity:
                order = queue.put_queue[0]
                queue.get(order)
                print("라이더",self.name,"시간",int(env.now),"주문 접수", order.name, "대기 주문들", len(queue.put_queue), "라이더 주문",len(self.resource.put_queue))
                with self.resource.request() as req:
                    yield req
                    yield env.timeout(order.process_time)
                    print("라이더",self.name,"시간", int(env.now), "주문 완료", order.name)
                    queue.get(order)
                #주문 수행
                yield env.timeout(ready_time)
                print("라이더",self.name, "시간", int(env.now), "주문 완료", order.name)
            else:
                yield env.timeout(1)


def ordergenerator(env, queue, interval = 5, end_time = 100):
    count = 0
    while env.now < end_time:
        #queue.put('order'+str(count))
        #order = 'order'+str(count)
        #order = env.timeout(10)
        order = simpy.events.Timeout(env,10)
        print('주문확인',order, type(order))
        queue.put(order)
        """
        #with queue.put(order) as request:
        #    yield request
        #    yield env.timeout(interval)
        #    print('T:',int(env.now),'/큐',queue.put_queue)
        """
        print('T:', int(env.now), '/큐', queue.put_queue)
        yield env.timeout(interval)
        count += 1


#실행부
# str for test
env = simpy.Environment()
Platform = simpy.Store(env)
env.process(ordergenerator(env, Platform))
#rider1 = rider(env,0,Platform)
#rider2 = rider(env,1,Platform)
env.run(200)
print("queue check",Platform.put_queue)