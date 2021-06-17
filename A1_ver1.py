# -*- coding: utf-8 -*-

import simpy
import operator
import itertools
import Basic_Class2 as Basic
import random
import copy

class Order(object):
    def __init__(self, order_name, customer_names, route, order_type):
        self.index = order_name
        self.customers = customer_names
        self.route = route
        self.picked = False
        self.type = order_type #1:단주문, 2:B2, 3:B3

class Rider(object):
    def __init__(self, env, i, platform, customers, stores, speed = 1, capacity = 3, end_t = 120):
        self.name = i
        self.env = env
        self.resource = simpy.Resource(env, capacity=1)
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
        env.process(self.RunProcess(env, platform, customers, stores))


    def RiderMoving(self, env, time):
        """
        라이더가 움직이는 시간의 env.time의 generator를 반환
        :param env: simpy.env
        :param time: 라이더가 움직이는 시간
        """
        yield env.timeout(time)
        print('현재T:',int(env.now),"/라이더 ",self.name,"가게 도착")

    def RunProcess(self, env, platform, customers, stores):
        while int(env.now) < self.end_t:
            if len(self.route) > 0:
                node_info = self.route[0]
                print('T: {} 노드 정보 {}'.format(int(env.now),node_info) )
                input('체크1')
                order = customers[node_info[0]]
                store_name = order.store
                move_t = Basic.distance(self.last_departure_loc, node_info[2]) / self.speed
                with self.resource.request() as req:
                    print('T: {} 노드 {} 시작'.format(int(env.now), node_info))
                    yield req  # users에 들어간 이후에 작동
                    print('T: {} 노드 {} kick1: 이동시간{}'.format(int(env.now), node_info, move_t))
                    #yield env.timeout(move_t)
                    yield env.process(stores[store_name].Cook(env, order)) & env.process(self.RiderMoving(env, move_t))
                    #yield env.process(stores[store_name].Cook(env, order)) & env.timeout(move_t)  # 둘 중 더 오래 걸리는 process가 완료 될 때까지 기다림
                    print('T: {} 노드 {} kick2'.format(int(env.now), node_info))
                    del self.route[0]
                    if node_info[1] == 0: #가게인 경우
                        print('가게 도착')
                        self.container.append(node_info[0])
                    else:#고객인 경우
                        print('고객 도착')
                        input('체크5')
                        self.container.remove(node_info[0])
                        self.onhand.remove(node_info[0])
                        self.served.append(node_info[0])
                    print('T: {} 노드 {} 도착'.format(int(env.now), node_info))
                    if len(self.onhand) < self.capacity:
                        print('T{} 추가 탐색 시작'.format(env.now))
                        order_info = self.OrderSelect(platform, customers, self.route)
                        if order_info != None:
                            print('체크2_기 추가', order_info)
                            input('체크')
                            order = Platform[order_info[0]]
                            self.OrderPcik(order, customers)
                        else:
                            pass
            else:
                order_info = self.OrderSelect(platform, customers, [])
                print('현재 {} 라이더 경로 {} -> 빈 상태 {} 추가'.format(int(env.now), self.route, order_info))
                input('체크')
                if order_info != None:
                    order = Platform[order_info[0]]
                    self.OrderPcik(order, customers)
                    print('라이더 {} -> 경로 {} 할당 '.format(self.name, self.route))
                else:
                    yield env.timeout(5)
                    print('라이더 {} -> 주문탐색 {}~{}'.format(self.name, int(env.now) -5, int(env.now) ))

    def OrderSelect(self, platform, customers, route):
        if len(route) > 0:
            #subsets = itertools.combinations(list(platform.keys()), n)
            score = []
            for order in platform:
                #현재의 경로를 반영한 비용
                if order.picked == False:
                    route_info = self.ShortestRoute(order, customers, route)
                    score.append([order.index] + route_info)
            if len(score) > 0:
                score.sort(key = operator.itemgetter(6))
                return score[0]
            else:
                return None
        else: #주문 중 최고점 주문을 선택
            score = []
            for order in platform:
                if order.picked == False:
                    if order.type == 1:
                        print('order type 1')
                        customer = customers[order.customers[0]]
                        dist = Basic.distance(self.last_departure_loc, customer.store_loc)
                        score.append([order.index,[customer.name], dist])
                    else:
                        print('order type 2')
                        route_info = self.ShortestRoute(order, customers, [])
                        if route_info != None:
                            score_info = [order.index, route_info[4],route_info[5]]
                            score.append(score_info)
            if len(score) > 0:
                print('스코어', score)
                score.sort(key = operator.itemgetter(2))
                return score[0]
            else:
                return None


    def ShortestRoute(self, order, customers, input_route, now_t = 0, p2 = 0, speed = 1, M = 1000):
        prior_route = []
        for node in input_route:
            for visitied_node in self.visited_route:
                if node[0] == visitied_node[0]:
                    prior_route.append(visitied_node[0] + M)
                    break
        prior_route.sort(key = operator.itemgetter(4))
        order_names = []  # 가게 이름?
        store_names = []
        for customer_name in order.customers:
            order_names.append(customer_name)
            store_names.append(customer_name + M)
        for node_info in input_route:
            if node_info[1] == 0:
                order_names.append(node_info[0])
            else:
                store_names.append(node_info[0] + M)
        candi = order_names + store_names
        subset = itertools.permutations(candi, len(candi))
        feasible_subset = []
        for route_part in subset:
            route = prior_route + list(route_part)
            #print('라우트:{}'.format(route))
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
                order_customers = []
                order_customers_names = []
                for customer_name in order.customers:
                    order_customers.append(customers[customer_name])
                    order_customers_names.append(customer_name)
                if len(self.route) > 0:
                    for info in self.route:
                        if info[0] not in order_customers_names:
                            order_customers.append(customers[info[0]])
                            order_customers_names.append(info[0])
                """
                ftd_feasiblity, ftds = Basic.FLT_Calculate(order_customers, route, p2, M=M, speed=speed, add_time= add_time)
                if ftd_feasiblity == True:
                    # print('ftds',ftds)
                    # input('멈춤5')
                    route_time = Basic.RouteTime(order_customers, route, speed=speed, M=M)
                    feasible_routes.append(
                        [route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time])                
                """
                #print('주문 확인', order_customers)
                #input('체크3')
                #route_time = Basic.RouteTime(order_customers, route, speed=speed, M=M) #todo: 이미 수행된 경로(route)에 대해서는 시간을 제외해야함.
                route_time = Basic.RouteTime(order_customers, list(route_part), speed=speed, M=M)
                rev_route = []
                for node in route:
                    if node < M:
                        name = node
                        info = [name, 1, customers[name].location,0]
                    else:
                        name = node - M
                        info = [name, 0, customers[name].store_loc, 0]
                    rev_route.append(info)
                feasible_routes.append([rev_route, None, None, None, order_names, route_time])
            if len(feasible_routes) > 0:
                feasible_routes.sort(key=operator.itemgetter(5)) #가장 짧은 거리의 경로 선택.
                feasible_subset.append(feasible_routes[0])
        if len(feasible_subset) > 0:
            feasible_subset.sort(key=operator.itemgetter(5))
            #print('선택 된 정보 {} / 경로 길이 {}'.format(feasible_subset[0][0], feasible_subset[0][5]))
            #input('확인9')
            return feasible_subset[0]
        else:
            return []

    def OrderPcik(self, order, customers):
        order.picked = True
        if order.type == 1:
            customer = customers[order.customers[0]]
            names = [customer.name]
            route = [[customer.name, 0, customer.store_loc, 0], [customer.name, 1, customer.location, 0]]
        else:
            names = customers[order.customers]
            route = order.route
        print('orderpick 경로확인',route, names)
        self.route += route
        self.onhand += names
        print('orderpick 경로확인2', self.route, self.onhand)

class Store(object):
    """
    Store can received the order.
    Store has capacity. The order exceed the capacity must be wait.
    """
    def __init__(self, env, platform, name, loc = [25,25], order_ready_time = 7, capacity = 6, slack = 2, print_para = True):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.location = loc
        self.order_ready_time = order_ready_time
        self.resource = simpy.Resource(env, capacity = capacity)
        self.slack = slack #자신의 조리 중이 queue가 꽉 차더라도, 추가로 주문을 넣을 수 있는 주문의 수
        self.received_orders = []
        self.wait_orders = []
        self.ready_order = []
        self.loaded_order = []
        self.capacity = capacity
        env.process(self.StoreRunner(env, platform, capacity = capacity, print_para= print_para))


    def StoreRunner(self, env, platform, capacity, open_time = 1, close_time = 900, print_para = True):
        """
        Store order cooking process
        :param env: simpy Env
        :param platform: Platform
        :param capacity: store`s capacity
        :param open_time: store open time
        :param close_time: store close time
        """
        #input('가게 주문 채택')
        #yield env.timeout(open_time)
        now_time = round(env.now, 1)
        #input('가게 주문 채택0')
        while now_time < close_time:
            now_time = round(env.now,1)
            #받은 주문을 플랫폼에 올리기
            #print('값 확인',len(self.resource.users) , len(self.wait_orders), capacity , self.slack,len(self.received_orders))
            if len(self.resource.users) + len(self.resource.put_queue) < capacity + self.slack:  # 플랫폼에 자신이 생각하는 여유 만큼을 게시
            #if len(self.resource.users) + len(self.wait_orders) < capacity + self.slack: #플랫폼에 자신이 생각하는 여유 만큼을 게시
                #self.received_orders.append()
                #print('접수된 고객 수', len(self.received_orders))
                #input('가게 주문 채택1')
                #slack = min(capacity + self.slack - len(self.resource.users), len(self.received_orders))
                slack = capacity + self.slack - len(self.resource.users)
                #print('가게:',self.name,'/ 잔여 용량:', slack,'/대기 중 고객 수:',len(self.received_orders))
                received_orders_num = len(self.received_orders)
                if received_orders_num > 0:
                    for count in range(min(slack,received_orders_num)):
                        order = self.received_orders[0] #앞에서 부터 플랫폼에 주문 올리기
                        route = [order.name, 0, order.store_loc, 0], [order.name, 1, order.location,0]
                        o = Order(len(platform), [order.name],route,1)
                        #print('주문 정보',o.index, o.customers, o.route, o.type)
                        platform.append(o)
                        if print_para == True:
                            print('현재T:', int(env.now), '/가게', self.name, '/주문', order.name, '플랫폼에 접수/조리대 여유:',capacity - len(self.resource.users),'/조리 중',len(self.resource.users))
                        self.wait_orders.append(order)
                        self.received_orders.remove(order)
                        #input('가게 주문 채택2')
            else: #이미 가게의 능력 최대로 조리 중. 잠시 주문을 막는다(block)
                #print("가게", self.name, '/',"여유 X", len(self.resource.users),'/주문대기중',len(self.received_orders))
                pass
            #만약 현재 조리 큐가 꽉차는 경우에는 주문을 더이상 처리하지 X
            yield env.timeout(0.1)
        #print("T",int(env.now),"접수 된 주문", self.received_orders)


    def Cook(self, env, customer, cooking_time_type = 'fixed'):
        """
        Occupy the store capacity and cook the order
        :param env: simpy Env
        :param customer: class customer
        :param cooking_time_type: option
        """
        #print('현재 사용중', len(self.resource.users))
        with self.resource.request() as req:
            yield req #resource를 점유 해야 함.
            print('현재T:',int(env.now),'/가게',self.name,'/주문:',customer.name,"조리시작")
            now_time = round(env.now , 1)
            req.info = [customer.name, now_time]
            if cooking_time_type == 'fixed':
                cooking_time = self.order_ready_time
            elif cooking_time_type == 'random':
                cooking_time = random.randrange(1,self.order_ready_time)
            else:
                cooking_time = 1
            print(cooking_time, '분 후 ', customer.name, "음식준비완료")
            yield env.timeout(cooking_time)
            #print(self.resource.users)
            print('현재T:',int(env.now),'/가게',self.name,'/주문:',customer.name,"음식준비완료")
            customer.food_ready = True
            customer.ready_time = env.now
            self.ready_order.append(customer)
            #print('T',int(env.now),"기다리는 중인 고객들",self.ready_order)


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
        single_rider = Rider(env,rider_num,Platform, Orders,  Store_dict, speed = speed, end_t = end_time)
        Rider_dict[rider_num] = single_rider
        rider_num += 1
        print('라이더 {} 생성'.format(rider_num))
        yield env.timeout(interval)


def ordergenerator(env, orders, stores, interval = 5, end_time = 100):
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
rider_num = 0
Store_dict = {}
Rider_dict = {}
rider_gen_interval = 10
rider_speed = 2.5

#Before simulation, generate the stores.
for store_name in range(store_num):
    loc = list(random.sample(range(0,50),2))
    store = Store(env, Platform, store_name, loc = loc, capacity = 10, print_para= False)
    #env.process(store.StoreRunner(env, Platform, capacity=store.capacity))
    Store_dict[store_name] = store

env.process(RiderGenerator(env, Rider_dict, Platform, Store_dict, Orders, speed = rider_speed, end_time = 120, interval = rider_gen_interval, runtime = run_time, gen_num = rider_num))
env.process(ordergenerator(env, Orders, Store_dict, interval = order_interval))
#env.process(Basic.Platform_process(env, Platform, Orders, Rider_dict, p2, thres_p, interval, speed = rider_speed, end_t = 1000))
env.run(run_time)