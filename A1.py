# -*- coding: utf-8 -*-

import simpy
import operator
import itertools
import Basic_Class2 as Basic
import random
import copy


## 현재 env에 있는 모든 process를 쭉 접근할 수 있는 방법이 없음.
## 따라서, 매번 프로세스를 따로 리스트의 형태로 저장해주는 것이 필요함.
current_ps = []

class Order(object):
    def __init__(self, order_name, customer_names, route, type = '1'):
        self.name = order_name
        self.customers = customer_names
        self.route = route
        self.type = type #1:단주문, 2:B2, 3:B3


class Rider(object):
    def __init__(self, env, i, platform, orders, speed = 1, capacity = 3, end_t = 120):
        self.name = i
        self.env = env
        self.visited_route = []
        self.speed = speed
        self.route = []
        self.run_process = None
        self.capacity = capacity
        self.onhand = []
        self.end_t = env.now + end_t
        self.last_departure_loc = [25,25]
        self.container = []
        self.served = []
        env.process(self.OrderSelect(env, platform, orders))


    def updater(self, point_info):
        pass

    def RunProcess_org(self, env, platform, orders):
        while int(env.now) < self.end_t and len(self.route) > 0:
            if len(self.route) > 0:
            #try:
                point_info = self.route[0]
                #order = sorted(platform, key=operator.attrgetter('fee'))[0]
                move_t = Basic.distance(self.last_departure_loc,point_info[2])/ self.speed
                #move_t = point_info[2]
                exp_arrive_time = env.now + move_t
                print('현재 경로 {}'.format(self.route))
                input('확인2')
                ct_name = orders[point_info[0]].name
                try:
                    print('test')
                    yield env.timeout(move_t)
                    point_info[3] = env.now
                    self.visited_route.append(point_info)
                    self.last_departure_loc = point_info[2]
                    if point_info[1] == 0:
                        self.container.append(ct_name)
                        print('T {}/ 라이더 {}/주문{}의 가게 도착 1'.format(int(env.now),self.name, ct_name))
                    else:
                        print('T {}/ 라이더 {}/ 주문{}의 고객 도착 1'.format(int(env.now), self.name, ct_name))
                        self.container.remove(ct_name)
                        self.served.append(ct_name)
                        onahnd_names = []
                        for i in self.onhand:
                            onahnd_names.append(i.name)
                        print('경로 {} /제거 이름{} : 제거 대상{} -> 현재 보유 중 {}'.format(self.route, ct_name,orders[ct_name],self.onhand))
                        print('경로 {} /제거 이름{} : 제거 대상{} -> 현재 보유 중 {}'.format(self.route, ct_name, orders[ct_name].name,
                                                                              onahnd_names))
                        input('확인11')
                        self.onhand.remove(orders[ct_name])
                    print('경로 ', self.route)
                    del self.route[0]
                    input('확인3')
                except simpy.Interrupt:
                    time_diff = exp_arrive_time - env.now
                    yield env.timeout(time_diff)
                    point_info[3] = env.now
                    self.visited_route.append(point_info)
                    self.last_departure_loc = point_info[2]
                    if point_info[1] == 0:
                        self.container.append(ct_name)
                        print('T {}/ 라이더 {}/주문{}의 가게 도착 2'.format(int(env.now), self.name, ct_name))
                    else:
                        print('T {}/ 라이더 {}/ 주문{}의 고객 도착 2'.format(int(env.now), self.name, ct_name))
                        self.container.remove(ct_name)
                        self.onhand.remove(orders[ct_name])
                        self.served.append(ct_name)
                    del self.route[0]
                    input('확인4')
                    #return None
            else:
            #except simpy.Interrupt:
                input('확인13')
                yield env.timeout(5)
                print('T {}/ onhand주문X -> 주문탐색'.format(int(env.now)))
                input('확인5')
                env.process(self.OrderSelect(env, platform, orders))
                #return None
            """
            positive_value_order = []
            for order in platform:
                if order.fee > 0:
                    positive_value_order.append(order.name)
            if len(self.onhand) < self.capacity and len(positive_value_order) > 0:
                env.process(self.OrderSelect(platform))            
            """
        #주문이 종료된 이후에 탐색을 수행할 것인가?
        env.process(self.OrderSelect(env, platform, orders))

    def RunProcess(self, env, platform, orders):
        while int(env.now) < self.end_t and len(self.route) > 0:
            point_info = self.route[0]
            #order = sorted(platform, key=operator.attrgetter('fee'))[0]
            move_t = Basic.distance(self.last_departure_loc,point_info[2])/ self.speed
            #move_t = point_info[2]
            exp_arrive_time = env.now + move_t
            print('현재 경로 {}'.format(self.route))
            input('확인2')
            ct_name = orders[point_info[0]].name
            try:
                print('test')
                yield env.timeout(move_t)
                point_info[3] = env.now
                self.visited_route.append(point_info)
                self.last_departure_loc = point_info[2]
                if point_info[1] == 0:
                    self.container.append(ct_name)
                    print('T {}/ 라이더 {}/주문{}의 가게 도착 1'.format(int(env.now),self.name, ct_name))
                else:
                    print('T {}/ 라이더 {}/ 주문{}의 고객 도착 1'.format(int(env.now), self.name, ct_name))
                    self.container.remove(ct_name)
                    self.served.append(ct_name)
                    onahnd_names = []
                    for i in self.onhand:
                        onahnd_names.append(i.name)
                    print('경로 {} /제거 이름{} : 제거 대상{} -> 현재 보유 중 {}'.format(self.route, ct_name,orders[ct_name],self.onhand))
                    print('경로 {} /제거 이름{} : 제거 대상{} -> 현재 보유 중 {}'.format(self.route, ct_name, orders[ct_name].name,onahnd_names))
                    input('확인11')
                    self.onhand.remove(orders[ct_name])
                    #self.onhand.remove(platform[point_info[0]])
                print('경로 ', self.route)
                del self.route[0]
                input('확인3')
            except simpy.Interrupt:
                time_diff = exp_arrive_time - env.now
                yield env.timeout(time_diff)
                point_info[3] = env.now
                self.visited_route.append(point_info)
                self.last_departure_loc = point_info[2]
                if point_info[1] == 0:
                    self.container.append(ct_name)
                    print('T {}/ 라이더 {}/주문{}의 가게 도착 2'.format(int(env.now), self.name, ct_name))
                else:
                    print('T {}/ 라이더 {}/ 주문{}의 고객 도착 2'.format(int(env.now), self.name, ct_name))
                    self.container.remove(ct_name)
                    self.onhand.remove(orders[ct_name])
                    self.served.append(ct_name)
                del self.route[0]
                input('확인4')
                return None
            print('T:{}/ 정차 중으로, 추가 탐색 시작'.format(int(env.now)))
            env.process(self.OrderSelect(env, platform, orders))
        print('재시작!!')
        env.process(self.OrderSelect(env, platform, orders))


    def OrderSelect(self, env, platform, customers):
        if len(platform) > 0:
            unserved_orders = []
            unserved_orders_names = []
            for order in platform:
                if order.time_info[1] == None:
                    unserved_orders.append(order)
                    unserved_orders_names.append(order.name)
            order = sorted(unserved_orders, key=operator.attrgetter('fee'))[0]
            print('선택한 주문 {}, 주문 이름 {}'.format(order, order.name))
            order.time_info[1] = env.now
            if len(self.onhand) == 0:
                print('T :{} 라이더: {} 주문 {} 선택(기존 주문X)'.format(int(env.now), self.name, order.name))
                self.onhand.append(order)
                self.route += [[order.name, 0, order.store_loc,0], [order.name, 1, order.location,0]]
                self.run_process = env.process(self.RunProcess(self.env, platform,customers))
            else:
                print('T :{} 라이더: {} 주문 {} 선택(기존 주문:{})'.format(int(env.now), self.name, order.name, self.onhand))
                c = [order] + self.onhand
                rev_c = {}
                for ct_class in c:
                    rev_c[ct_class.name] = ct_class
                r = self.ShortestRoute(rev_c, speed = self.speed, now_t= env.now) #r[0] = route
                self.onhand.append(order)
                print('라우트:{},현재 경로 {})'.format(r,self.route))
                input('확인6')
                if r[0] != [] and r[0] != self.route:
                    if self.route == []:
                        self.route = r[0]
                        self.run_process = env.process(self.RunProcess(self.env, platform, customers))
                    else:
                        print('추가 경로{}'.format(r[0]))
                        self.run_process.interrupt()
                        self.route = r[0]
                        self.run_process = env.process(self.RunProcess(self.env, platform,customers))
        else:
            print('주문이 없으니 대기 합시다.')
            yield env.timeout(3)



    def ShortestRoute(self, orders, now_t = 0, p2 = 0, speed = 1, M = 1000):
        prior_route_infos = []
        prior_route = []
        add_time = {}
        for food in self.container:
            for info in self.visited_route:
                if food == info[0] and info[1] == 0 and info[0] not in self.served:
                    prior_route_infos.append(info)
                    prior_route.append(info[0] + M)
                    add_time[info[0]] = now_t - info[3]
                    break
        print('입력 주문들 {}'.format(orders))
        print('현재 배달 중 {}/ 이전 주문 {}/ 이전 경로{}'.format(self.container, prior_route,self.visited_route))
        input('확인8')
        order_names = []  # 가게 이름?
        for order_name in orders:
            order_names.append(order_name)
        store_names = []
        for name in order_names:
            rev_name = name + M
            if rev_name not in prior_route:
                add_time[name] = 0
                store_names.append(rev_name)
        candi = order_names + store_names
        print('고려 노드들',candi)
        subset = itertools.permutations(candi, len(candi))
        feasible_subset = []
        #print(prior_route, subset, prior_route + list(subset))
        #input('확인7')
        for route_part in subset:
            #route = prior_route + list(route_part)
            route = list(route_part)
            print(route)
            #input('확인8')
            # print('고객이름',order_names,'가게이름',store_names,'경로',route)
            sequence_feasiblity = True  # 모든 가게가 고객 보다 앞에 오는 경우.
            feasible_routes = []
            for order_name in order_names:  # order_name + M : store name ;
                if order_name + M in route:
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
                print(orders, route)
                input('확인10')
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
            print('선택 된 정보 {} / 경로 길이 {}'.format(feasible_subset[0][0], feasible_subset[0][5]))
            input('확인9')
            return feasible_subset[0]
        else:
            return []


def RiderGenerator(env, Rider_dict, Platform, Store_dict, Orders, speed = 1, end_time = 120, interval = 1, runtime = 1000, gen_num = 10):
    """
    Generate the rider until t <= runtime and rider_num<= gen_num
    :param env:
    :param Rider_dict:
    :param rider_name:
    :param Platform:
    :param Store_dict:
    :param end_time:
    :param interval:
    :param runtime:
    :param gen_num:
    """
    rider_num = 0
    while env.now <= runtime and rider_num <= gen_num:
        single_rider = Rider(env,rider_num,Platform, Orders,  speed = speed, end_t = end_time)
        Rider_dict[rider_num] = single_rider
        rider_num += 1
        yield env.timeout(interval)


def ordergenerator(env, orders, platform, stores, interval = 5, end_time = 100):
    """
    Generate customer order
    :param env: Simpy Env
    :param orders: Order
    :param platform: Platform
    :param stores: Stores
    :param interval: order gen interval
    :param end_time: func end time
    """
    name = 0
    while env.now < end_time:
        #process_time = random.randrange(1,5)
        #input_location = [36,36]
        input_location = random.sample(list(range(50)),2)
        store_num = random.randrange(0, len(stores))
        order = Basic.Customer(env, name, input_location, store = store_num, store_loc = stores[store_num].location)
        orders[name] = order
        stores[store_num].received_orders.append(orders[name])
        #print('T:', int(env.now),'/주문:', orders[name].name,'접수')
        #print('가게 큐', stores[store_num].received_orders)
        #platform.append(order)
        yield env.timeout(interval)
        #print('현재 {} 플랫폼 주문 수 {}'.format(int(env.now), len(platform)))
        name += 1

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
rider_num = 1
Store_dict = {}
Rider_dict = {}
rider_gen_interval = 10
rider_speed = 2.5

#Before simulation, generate the stores.
for store_name in range(store_num):
    loc = list(random.sample(range(0,50),2))
    store = Basic.Store(env, Platform, store_name, loc = loc, capacity = 10, print_para= False)
    #env.process(store.StoreRunner(env, Platform, capacity=store.capacity))
    Store_dict[store_name] = store

env.process(RiderGenerator(env, Rider_dict, Platform, Store_dict, Orders, speed = rider_speed, end_time = 120, interval = rider_gen_interval, runtime = run_time, gen_num = rider_num))
env.process(ordergenerator(env, Orders, Platform, Store_dict, interval = order_interval))
#env.process(Basic.Platform_process(env, Platform, Orders, Rider_dict, p2, thres_p, interval, speed = rider_speed, end_t = 1000))
env.run(run_time)