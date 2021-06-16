# -*- coding: utf-8 -*-

import simpy
import numpy as np
import random
import operator
import itertools
import Basic_Class2 as Basic

## 현재 env에 있는 모든 process를 쭉 접근할 수 있는 방법이 없음.
## 따라서, 매번 프로세스를 따로 리스트의 형태로 저장해주는 것이 필요함.
current_ps = []

class Rider(object):
    def __init__(self, env, i, platform, capacity = 3, end_t = 120):
        self.name = i
        self.env = env
        self.visited_route = []
        self.route = []
        self.run_process = None
        self.capacity = capacity
        self.onhand = []
        self.end_t = env.now + end_t
        self.last_departure_loc = None
        self.container = []
        env.process(self.OrderSelect(platform))


    def updater(self, point_info):
        pass

    def RunProcess(self, platform):
        while int(self.env.now) < self.end_t:
            point_info = self.route[0]
            #order = sorted(platform, key=operator.attrgetter('fee'))[0]
            move_t = point_info[2]
            exp_arrive_time = env.now + move_t
            try:
                yield env.timeout(move_t)
                point_info[3] = env.now
                self.visited_route.append(point_info)
                self.last_departure_loc = point_info[2]
                if point_info[1] == 0:
                    self.container.append(point_info[0])
                else:
                    self.container.remove(point_info[0])
                del self.route[0]
            except simpy.Interrupt:
                time_diff = exp_arrive_time - env.now
                yield env.timeout(time_diff)
                point_info[3] = env.now
                self.visited_route.append(point_info)
                self.last_departure_loc = point_info[2]
                if point_info[1] == 0:
                    self.container.append(point_info[0])
                else:
                    self.container.remove(point_info[0])
                del self.route[0]
                return None

            """
            positive_value_order = []
            for order in platform:
                if order.fee > 0:
                    positive_value_order.append(order.name)
            if len(self.onhand) < self.capacity and len(positive_value_order) > 0:
                env.process(self.OrderSelect(platform))            
            """
        #주문이 종료된 이후에 탐색을 수행할 것인가?
        env.process(self.OrderSelect(platform))


    def OrderSelect(self, platform):
        order = sorted(platform, key=operator.attrgetter('fee'))[0]
        if len(self.onhand) == 0:
            print('T :{} 라이더: {} 주문 {} 선택(기존 주문X)'.format(int(env.now), self.name, order.name))
            self.onhand.append(order)
            self.route += [[order.name, 0, order.store_loc,0], [order.name, 1, order.loc,0]]
            self.run_process = env.process(self.RunProcess(platform))
        else:
            print('T :{} 라이더: {} 주문 {} 선택(기존 주문:{})'.format(int(env.now), self.name, order.name, self.onhand))
            self.onhand.append(order)
            c = order + self.onhand
            r = self.ShortestRoute(c, speed = self.speed) #r[0] = route
            if r[0] != [] and r[0] != self.route:
                self.run_process.interrupt()
                self.route = r[0]
                self.run_process = env.process(self.RunProcess(platform))


    def ShortestRoute(self, orders, p2 = 0, speed = 1, M = 1000):
        prior_route_infos = []
        prior_route = []
        add_time = {}
        for food in self.container:
            for info in self.visited_route:
                if food == info[0] and info[1] == 0:
                    prior_route_infos.append(info)
                    prior_route.append(info[0] + M)
                    add_time[info[0]] = env.now - info[3]
                    break
        order_names = []  # 가게 이름?
        for order in orders:
            order_names.append(order.name)
        store_names = []
        for name in order_names:
            rev_name = name + M
            if rev_name not in prior_route:
                add_time[name] = 0
                store_names.append(rev_name)
        candi = order_names + store_names
        subset = itertools.permutations(candi, len(candi))
        feasible_subset = []
        for route in prior_route + subset:
            # print('고객이름',order_names,'가게이름',store_names,'경로',route)
            sequence_feasiblity = True  # 모든 가게가 고객 보다 앞에 오는 경우.
            feasible_routes = []
            for order_name in order_names:  # order_name + M : store name ;
                if route.index(order_name + M) < route.index(order_name):
                    pass
                else:
                    sequence_feasiblity = False
                    break
            if sequence_feasiblity == True:
                """
                ftd_feasiblity, ftds = Basic.FLT_Calculate(orders, route, p2, M=M, speed=speed, add_time= add_time)
                if ftd_feasiblity == True:
                    # print('ftds',ftds)
                    # input('멈춤5')
                    route_time = Basic.RouteTime(orders, route, speed=speed, M=M)
                    feasible_routes.append(
                        [route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time])                
                """
                route_time = Basic.RouteTime(orders, route, speed=speed, M=M)
                rev_route = []
                for node in route:
                    if node < M:
                        name = node
                        info = [name, 1, orders[name].location,0]
                    else:
                        name = node - M
                        info = [name, 0, orders[name].store_loc, 0]
                    rev_route.append(info)
                feasible_routes.append([rev_route, None, None, None, order_names, route_time])
            if len(feasible_routes) > 0:
                feasible_routes.sort(key=operator.itemgetter(5)) #가장 짧은 거리의 경로 선택.
                feasible_subset.append(feasible_routes[0])
        if len(feasible_subset) > 0:
            feasible_subset.sort(key=operator.itemgetter(5))
            return feasible_subset[0]
        else:
            return []



class Sample_process:
    def __init__(self, env, i, n, run_t):
        self.env = env
        self.rider = simpy.Resource(env, capacity = 1)
        self.name = i
        self.target = n
        self.end = run_t
        self.pool = list(range(run_t))
        self.run = env.process(self.Running())
    
    def Running(self):
        print('런 타입',type(self.run))
        while True:
            print('잔존 수', self.pool)
            if len(self.pool) > 0:
                time = self.pool[0]
                with self.rider.request() as req:
                    yield req
                    print('리퀘스트 타입',type(req))
                    #yield env.process(self.sinlge_process(time, self.target))
                    try:
                        yield env.process(self.sinlge_process(time, self.target))
                        self.pool.remove(time)
                    except simpy.Interrupt:
                        print('now {} interrupted! {} ~ {} removed'.format( env.now , time, time + self.target - 1))
                        input('확인2')
                        for i in range(time, min(time + self.target, self.end)):
                            self.pool.remove(i)
                        return None
            else:
                print('종료 됨', env.now)
                break
            input('확인1')


    def sinlge_process(self, time,n):
        try:
            print('T {} start {}'.format(int(env.now), time))
            yield env.timeout(time)
            print('T {} done {}'.format(int(env.now), time))
        except simpy.Interrupt:
            print('T {} pass {}'.format(int(env.now), time))
            for i in range(time, time + n):
                self.pool.remove(i)
            yield env.timeout(time + n)
            print('T {} move to {}'.format(int(env.now), time + n))


def stop_any_process(env, process_list):
    ## 2초마다 한번씩 현재 process 중 아무거나 종료시키는 generator
    ## 남아있는 clock이 없을때의 조건도 만들어줌.
    while True:
        try:
            yield env.timeout(3)
            print("취소 시도 ::{}".format(env.now))
            r = np.random.randint(0, len(process_list))
            print('선택된',r)
            print("취소 시도1 ::{}".format(env.now))
            #process_list[r].interrupt()
            process_list[r].run.interrupt()
            print("취소 시도2 ::{}".format(env.now))
            env.process(process_list[r].Running())
            #process_list[r].rider.users[0].interrupt()
            #process_list[r].Running.interrupt()
            #process_list[r].run.sinlge_process.interrupt()
            print("취소 시도3 ::{}".format(env.now))
            #current_ps.remove(current_ps[r])
        except:
            yield env.timeout(3)
            print("#" * 20)
            print("all process was interrupted at {}".format(env.now))


def stop_any_process2(env, process_list):
    ## 2초마다 한번씩 현재 process 중 아무거나 종료시키는 generator
    ## 남아있는 clock이 없을때의 조건도 만들어줌.
    while True:
        yield env.timeout(3)
        rv = 0.2
        if rv > random.random():
            print('취소 현재', env.now)
            print("취소 시도 ::{}".format(env.now))
            r = np.random.randint(0, len(process_list))
            print('선택된',r,'확인', process_list[r])
            input('멈춤')
            print("취소 시도1 ::{}".format(env.now))
            if len(process_list[r].pool) > 0:
                process_list[r].run.interrupt()
                print("취소 시도2 ::{}".format(env.now))
                process_list[r].run = env.process(process_list[r].Running())
                #env.process(process_list[r].Running())
                print("취소 시도3 ::{}".format(env.now))
            else:
                print('종료2')
                break
            #current_ps.remove(current_ps[r])
        else:
            pass

## environment setting
env = simpy.Environment()

## 6 개의 중간에 멈출 수 있는 clock을 만들어서 집어넣음
process_list = []
p = Sample_process(env, 1, 5, 26)
process_list.append(p)

## 2초마다 process를 멈추는 generator도 넘겨줌
#env.process(stop_any_process(env,process_list))
env.process(stop_any_process2(env,process_list))
env.run(until=200)

input('테스트 종료 됨')

"""
class clock2:
    def __init__(self,env, i, tick):
        self.env = env
        self.name = i
        self.tick = tick
        env.process(self.Running(self.env))

    def Running(self):
        while True:
            yield self.env.process(self.unit_process(self.tick))
            print('clock {} end at {}'.format(i, env.now))

    def unit_process(self, tick):
        try:
            yield env.timeout(tick)
            print('clock {} ticks at {}'.format(i, env.now))
        except simpy.Interrupt:
            print('## the clock {} was interrupted at {}'.format(i, env.now))
            return None






def clock(env, i, tick):
    ## generator에 interrupt 가 발생했을 때 종료하는 조건을 넣어주어야 함
    while True:
        print('clock {} start at {}'.format(i, env.now))
        try:
            yield env.timeout(tick)
            print('clock {} ticks at {}'.format(i, env.now))
        except simpy.Interrupt:
            print('## the clock {} was interrupted at {}'.format(i, env.now))
            #return None
            yield env.timeout(tick)
            print('clock {} ticks at {}'.format(i, env.now))
        print('clock {} end at {}'.format(i, env.now))

def stop_any_process(env):
    ## 2초마다 한번씩 현재 process 중 아무거나 종료시키는 generator
    ## 남아있는 clock이 없을때의 조건도 만들어줌.
    while True:
        try:
            yield env.timeout(3)
            r = np.random.randint(0, len(current_ps))
            current_ps[r].interrupt()
            #current_ps.remove(current_ps[r])
        except:
            print("#" * 20)
            print("all process was interrupted at {}".format(env.now))
            return None


## environment setting
env = simpy.Environment()

## 6 개의 중간에 멈출 수 있는 clock을 만들어서 집어넣음
for i in range(0, 3):
    p = env.process(clock(env, i, 2))
    ## 새롭게 만들어진 프로세스에 대해서 외부에서 접근 방법이 없으므로, 따로 저장해두어야 함
    current_ps.append(p)

## 2초마다 process를 멈추는 generator도 넘겨줌
env.process(stop_any_process(env))

env.run(until=20)
"""