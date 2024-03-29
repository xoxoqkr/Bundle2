# -*- coding: utf-8 -*-

import simpy
import operator
import itertools
import Basic_Class2 as Basic
import random
import copy


# customer.time_info = [0 :발생시간, 1: 차량에 할당 시간, 2:차량에 실린 시간, 3:목적지 도착 시간,
# 4:고객이 받은 시간, 5: 보장 배송 시간, 6:가게에서 준비시간,7: 고객에게 서비스 하는 시간]
class Order(object):
    def __init__(self, order_name, customer_names, route, order_type):
        self.index = order_name
        self.customers = customer_names
        self.route = route
        self.picked = False
        self.type = order_type  # 1:단주문, 2:B2, 3:B3


class Rider(object):
    def __init__(self, env, i, platform, customers, stores, speed=1, capacity=3, end_t=120, p2=15):
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
        self.last_departure_loc = [25, 25]
        self.container = []
        self.served = []
        self.p2 = p2
        env.process(self.RunProcess(env, platform, customers, stores, self.p2))

    def RiderMoving(self, env, time, info=''):
        """
        라이더가 움직이는 시간의 env.time의 generator를 반환
        :param env: simpy.env
        :param time: 라이더가 움직이는 시간
        """
        yield env.timeout(time)
        # print('현재1 T:{} 라이더{} 가게 {} 도착'.format(int(env.now),self.name, info ))

    def RunProcess(self, env, platform, customers, stores, p2=0, wait_time=5):
        while int(env.now) < self.end_t:
            if len(self.route) > 0:
                node_info = self.route[0]
                print('T: {} 노드 정보 {} 경로 {}'.format(int(env.now), node_info, self.route))
                # ('체크1')
                order = customers[node_info[0]]
                store_name = order.store
                move_t = Basic.distance(self.last_departure_loc, node_info[2]) / self.speed
                with self.resource.request() as req:
                    print('T: {} 노드 {} 시작'.format(int(env.now), node_info))
                    yield req  # users에 들어간 이후에 작동
                    print('T: {} 노드 {} kick1: 이동시간{}'.format(int(env.now), node_info, move_t))
                    self.visited_route.append(self.route[0])
                    if node_info[1] == 0:  # 가게인 경우
                        yield env.process(stores[store_name].Cook(env, order)) & env.process(
                            self.RiderMoving(env, move_t, info=node_info[0]))
                        print('현재2 T:{} 라이더{} 가게 {} 도착'.format(int(env.now), self.name, customers[node_info[0]].store))
                        self.container.append(node_info[0])
                        order.time_info[2] = env.now
                    else:  # 고객인 경우
                        input('T: {} 고객 {} 이동 시작'.format(int(env.now), node_info[0]))
                        yield env.process(self.RiderMoving(env, move_t, info=node_info[0]))
                        print('T: {} 고객 {} 도착'.format(int(env.now), node_info[0]))
                        input('체크5')
                        order.time_info[3] = env.now
                        self.container.remove(node_info[0])
                        self.onhand.remove(node_info[0])
                        self.served.append(node_info[0])
                    print('T: {} 노드 {} 도착 '.format(int(env.now), node_info))
                    del self.route[0]
                    print('남은 경로 {}'.format(self.route))
            if len(self.onhand) < self.capacity:
                print('T{} 추가 탐색 시작'.format(env.now))
                order_info = self.OrderSelect(platform, customers, p2=p2)
                if order_info != None:
                    print('체크2_기 추가', order_info)
                    input('체크')
                    added_order = Platform[order_info[0]]
                    self.OrderPick(added_order, order_info[1], customers, env.now)
                else:
                    yield env.timeout(wait_time)
                    print('라이더 {} -> 주문탐색 {}~{}'.format(self.name, int(env.now) - 5, int(env.now)))

    def RunProcess_org(self, env, platform, customers, stores, p2=0, wait_time=5):
        while int(env.now) < self.end_t:
            if len(self.route) > 0:
                node_info = self.route[0]
                print('T: {} 노드 정보 {} 경로 {}'.format(int(env.now), node_info, self.route))
                input('체크1')
                order = customers[node_info[0]]
                store_name = order.store
                move_t = Basic.distance(self.last_departure_loc, node_info[2]) / self.speed
                with self.resource.request() as req:
                    print('T: {} 노드 {} 시작'.format(int(env.now), node_info))
                    yield req  # users에 들어간 이후에 작동
                    print('T: {} 노드 {} kick1: 이동시간{}'.format(int(env.now), node_info, move_t))
                    print('T: {} 노드 {} kick2'.format(int(env.now), node_info))
                    self.visited_route.append(self.route[0])
                    if node_info[1] == 0:  # 가게인 경우
                        yield env.process(stores[store_name].Cook(env, order)) & env.process \
                            (self.RiderMoving(env, move_t, info=node_info[0]))
                        print('현재2 T:{} 라이더{} 가게 {} 도착'.format(int(env.now), self.name, customers[node_info[0]].store))
                        self.container.append(node_info[0])
                        order.time_info[2] = env.now
                    else:  # 고객인 경우
                        input('T: {} 고객 {} 이동 시작'.format(int(env.now), node_info[0]))
                        yield env.process(self.RiderMoving(env, move_t, info=node_info[0]))
                        print('T: {} 고객 {} 도착'.format(int(env.now), node_info[0]))
                        input('체크5')
                        order.time_info[3] = env.now
                        self.container.remove(node_info[0])
                        self.onhand.remove(node_info[0])
                        self.served.append(node_info[0])
                    print('T: {} 노드 {} 도착 '.format(int(env.now), node_info))
                    del self.route[0]
                    print('남은 경로 {}'.format(self.route))
                    if len(self.onhand) < self.capacity:
                        print('T{} 추가 탐색 시작'.format(env.now))
                        order_info = self.OrderSelect(platform, customers, p2=p2)
                        if order_info != None:
                            print('체크2_기 추가', order_info)
                            input('체크')
                            added_order = Platform[order_info[0]]
                            self.OrderPick(added_order, order_info[1], customers, env.now)
                        else:
                            pass
            else:
                order_info = self.OrderSelect(platform, customers, p2=p2)
                print('현재 {} 라이더 경로 {} -> 빈 상태 {} 추가'.format(int(env.now), self.route, order_info))
                input('체크')
                if order_info != None:
                    added_order = Platform[order_info[0]]
                    self.OrderPick(added_order, order_info[1], customers, env.now)
                    print('라이더 {} -> 경로 {} 할당 '.format(self.name, self.route))
                else:
                    yield env.timeout(wait_time)
                    print('라이더 {} -> 주문탐색 {}~{}'.format(self.name, int(env.now) - 5, int(env.now)))

    def OrderSelect(self, platform, customers, p2=0, sort_standard=6):
        """
        route를 운행하는 라이더의 입장에서 platform의 주문들 중에서 가장 이윤이 높은 주문을 반환함.
        @param platform: {class order, ...}
        @param customers:
        @param route:
        @return: [order index, route(선택한 고객 반영), route 길이]선택한 주문 정보 / None : 선택할 주문이 없는 경우
        """
        score = []
        for order in platform:
            # 현재의 경로를 반영한 비용
            exp_onhand_order = order.customers + self.onhand
            if order.picked == False and len(exp_onhand_order) <= self.capacity:
                route_info = self.ShortestRoute(order, customers, self.route, p2=p2)
                if len(route_info) > 0:
                    score.append([order.index] + route_info)
                    # score = [[order.index, rev_route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time],...]
        if len(score) > 0:
            input('최단경로 실행1/ 대상 경로 수 {}, 내용{}'.format(len(score), score[0]))
            score.sort(key=operator.itemgetter(sort_standard))
            # input('최단경로 실행1/ 대상 경로 수 {}, 내용{}'.format(len(score), score[0]))
            return score[0]
        else:
            return None

    def OrderSelect_org(self, platform, customers, p2=0):
        """
        route를 운행하는 라이더의 입장에서 platform의 주문들 중에서 가장 이윤이 높은 주문을 반환함.
        @param platform: {class order, ...}
        @param customers:
        @param route:
        @return: [order index, route(선택한 고객 반영), route 길이]선택한 주문 정보 / None : 선택할 주문이 없는 경우
        """
        if len(self.route) > 0:
            # subsets = itertools.combinations(list(platform.keys()), n)
            score = []
            for order in platform:
                # 현재의 경로를 반영한 비용
                if order.picked == False:
                    route_info = self.ShortestRoute(order, customers, self.route, p2=p2)
                    if len(route_info) > 0:
                        score.append([order.index] + route_info)
            if len(score) > 0:
                input('최단경로 실행1/ 대상 경로 수 {}, 내용{}'.format(len(score), score[0]))
                score.sort(key=operator.itemgetter(6))
                # input('최단경로 실행1/ 대상 경로 수 {}, 내용{}'.format(len(score), score[0]))
                return score[0]
            else:
                return None
        else:  # 주문 중 최고점 주문을 선택
            score = []
            for order in platform:
                if order.picked == False:
                    print('order type 2')
                    route_info = self.ShortestRoute(order, customers, [], p2=p2)
                    # input('최단경로 결과2 {}'.format(route_info))
                    if len(route_info) > 0:
                        score_info = [order.index, route_info[0], route_info[5]]
                        score.append(score_info)
            if len(score) > 0:
                # print('스코어', score)
                score.sort(key=operator.itemgetter(2))
                input('최단경로 실행2/ 대상 경로 수 {}, 내용{}'.format(len(score), score[0]))
                return score[0]
            else:
                return None

    def ShortestRoute(self, order, customers, input_route, now_t=0, p2=0, speed=1, M=1000):
        prior_route = []
        # input('주문 {} 고객정보 {} /이미 방문 경로 {}/ 남은 경로 {}'.format(order.index, order.customers, self.visited_route, self.route))
        for visitied_node in self.visited_route:
            for node in input_route:
                # for node in input_route:
                #    for visitied_node in self.visited_route:
                if node[0] == visitied_node[0]:
                    prior_route.append(visitied_node[0] + M)
                    break
        # input('기 방문 노드 {}'.format(prior_route))
        order_names = []  # 가게 이름?
        store_names = []
        for customer_name in order.customers:
            order_names.append(customer_name)
            store_names.append(customer_name + M)
        # input('주문 목록1 {} /가게 목록1 {}'.format(order_names, store_names))
        for node_info in input_route:
            if node_info[1] == 1:
                order_names.append(node_info[0])
            else:
                store_names.append(node_info[0] + M)
        candi = order_names + store_names
        # input('주문 목록2 {} /가게 목록2 {}'.format(order_names, store_names))
        # input('이미 방문한 노드 {} /삽입 대상 {}'.format(prior_route, candi))
        subset = itertools.permutations(candi, len(candi))
        feasible_subset = []
        for route_part in subset:
            route = prior_route + list(route_part)
            # print('라우트:{}'.format(route))
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
                            order_customers.append(customers[info[0]])  # 추가된  고객과 기존에 남아 있는 고객들의 customer class
                            order_customers_names.append(info[0])
                # todo: 추가된 경로로 인한 FLT을 측정.
                ftd_feasiblity, ftds = FLT_Calculate(order_customers, customers, route, p2, M=M, speed=speed,
                                                     now_t=now_t)
                if ftd_feasiblity == True:
                    # print('ftds',ftds)
                    # input('멈춤5')
                    # route_time = Basic.RouteTime(order_customers, route, speed=speed, M=M)
                    route_time = Basic.RouteTime(order_customers, list(route_part), speed=speed, M=M)
                    # feasible_routes.append([route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time])
                    # route_time = Basic.RouteTime(order_customers, list(route_part), speed=speed, M=M)
                    rev_route = []
                    for node in route:
                        if node not in prior_route:
                            # print('node', node)
                            if node < M:
                                name = node
                                info = [name, 1, customers[name].location, 0]
                            else:
                                name = node - M
                                info = [name, 0, customers[name].store_loc, 0]
                            rev_route.append(info)
                    feasible_routes.append(
                        [rev_route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time])
                    # input('기존 경로 중 {} 제외 경로 {} -> 추가될 경로 {}'.format(route,prior_route,rev_route))
                """
                #print('주문 확인', order_customers)
                #input('체크3')
                #route_time = Basic.RouteTime(order_customers, route, speed=speed, M=M)
                route_time = Basic.RouteTime(order_customers, list(route_part), speed=speed, M=M)
                rev_route = []
                for node in route:
                    if node not in prior_route:
                        #print('node', node)
                        if node < M:
                            name = node
                            info = [name, 1, customers[name].location,0]
                        else:
                            name = node - M
                            info = [name, 0, customers[name].store_loc, 0]
                        rev_route.append(info)
                #input('기존 경로 중 {} 제외 경로 {} -> 추가될 경로 {}'.format(route,prior_route,rev_route))
                feasible_routes.append([rev_route, None, None, None, order_names, route_time])                
                """
            if len(feasible_routes) > 0:
                feasible_routes.sort(key=operator.itemgetter(5))  # 가장 짧은 거리의 경로 선택.
                feasible_subset.append(feasible_routes[0])
        if len(feasible_subset) > 0:
            feasible_subset.sort(key=operator.itemgetter(5))
            # print('선택 된 정보 {} / 경로 길이 {}'.format(feasible_subset[0][0], feasible_subset[0][5]))
            # input('확인9')
            return feasible_subset[0]
        else:
            return []

    def OrderPick(self, order, route, customers, now_t):
        order.picked = True
        """
        if order.type == 1:
            customer = customers[order.customers[0]]
            names = [customer.name]
            route = [[customer.name, 0, customer.store_loc, 0], [customer.name, 1, customer.location, 0]]
        else:
            names = customers[order.customers]
            route = order.route        
        """
        names = order.customers
        for name in names:
            customers[name].time_info[1] = now_t
        print('선택된 주문의 고객들 {} / 추가 경로{}'.format(names, route))
        self.route = route
        self.onhand += names
        print('수정후 경로 {}/ 보유 고객 {}'.format(self.route, self.onhand))


class Store(object):
    """
    Store can received the order.
    Store has capacity. The order exceed the capacity must be wait.
    """

    def __init__(self, env, platform, name, loc=[25, 25], order_ready_time=7, capacity=6, slack=2, print_para=True):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.location = loc
        self.order_ready_time = order_ready_time
        self.resource = simpy.Resource(env, capacity=capacity)
        self.slack = slack  # 자신의 조리 중이 queue가 꽉 차더라도, 추가로 주문을 넣을 수 있는 주문의 수
        self.received_orders = []
        self.wait_orders = []
        self.ready_order = []
        self.loaded_order = []
        self.capacity = capacity
        env.process(self.StoreRunner(env, platform, capacity=capacity, print_para=print_para))

    def StoreRunner(self, env, platform, capacity, open_time=1, close_time=900, print_para=True):
        """
        Store order cooking process
        :param env: simpy Env
        :param platform: Platform
        :param capacity: store`s capacity
        :param open_time: store open time
        :param close_time: store close time
        """
        # input('가게 주문 채택')
        # yield env.timeout(open_time)
        now_time = round(env.now, 1)
        # input('가게 주문 채택0')
        while now_time < close_time:
            now_time = round(env.now, 1)
            # 받은 주문을 플랫폼에 올리기
            # print('값 확인',len(self.resource.users) , len(self.wait_orders), capacity , self.slack,len(self.received_orders))
            if len(self.resource.users) + len(
                    self.resource.put_queue) < capacity + self.slack:  # 플랫폼에 자신이 생각하는 여유 만큼을 게시
                # if len(self.resource.users) + len(self.wait_orders) < capacity + self.slack: #플랫폼에 자신이 생각하는 여유 만큼을 게시
                # self.received_orders.append()
                # print('접수된 고객 수', len(self.received_orders))
                # input('가게 주문 채택1')
                # slack = min(capacity + self.slack - len(self.resource.users), len(self.received_orders))
                slack = capacity + self.slack - len(self.resource.users)
                # print('가게:',self.name,'/ 잔여 용량:', slack,'/대기 중 고객 수:',len(self.received_orders))
                received_orders_num = len(self.received_orders)
                if received_orders_num > 0:
                    for count in range(min(slack, received_orders_num)):
                        order = self.received_orders[0]  # 앞에서 부터 플랫폼에 주문 올리기
                        route = [order.name, 0, order.store_loc, 0], [order.name, 1, order.location, 0]
                        o = Order(len(platform), [order.name], route, 1)
                        # print('주문 정보',o.index, o.customers, o.route, o.type)
                        platform.append(o)
                        if print_para == True:
                            print('현재T:', int(env.now), '/가게', self.name, '/주문', order.name, '플랫폼에 접수/조리대 여유:',
                                  capacity - len(self.resource.users), '/조리 중', len(self.resource.users))
                        self.wait_orders.append(order)
                        self.received_orders.remove(order)
                        # input('가게 주문 채택2')
            else:  # 이미 가게의 능력 최대로 조리 중. 잠시 주문을 막는다(block)
                # print("가게", self.name, '/',"여유 X", len(self.resource.users),'/주문대기중',len(self.received_orders))
                pass
            # 만약 현재 조리 큐가 꽉차는 경우에는 주문을 더이상 처리하지 X
            yield env.timeout(0.1)
        # print("T",int(env.now),"접수 된 주문", self.received_orders)

    def Cook(self, env, customer, cooking_time_type='fixed'):
        """
        Occupy the store capacity and cook the order
        :param env: simpy Env
        :param customer: class customer
        :param cooking_time_type: option
        """
        # print('현재 사용중', len(self.resource.users))
        with self.resource.request() as req:
            yield req  # resource를 점유 해야 함.
            print('현재T:', int(env.now), '/가게', self.name, '/주문:', customer.name, "조리시작")
            now_time = round(env.now, 1)
            req.info = [customer.name, now_time]
            if cooking_time_type == 'fixed':
                cooking_time = self.order_ready_time
            elif cooking_time_type == 'random':
                cooking_time = random.randrange(1, self.order_ready_time)
            else:
                cooking_time = 1
            print(cooking_time, '분 후 ', customer.name, "음식준비완료")
            yield env.timeout(cooking_time)
            # print(self.resource.users)
            print('현재T:', int(env.now), '/가게', self.name, '/주문:', customer.name, "음식준비완료")
            customer.food_ready = True
            customer.ready_time = env.now
            self.ready_order.append(customer)
            # print('T',int(env.now),"기다리는 중인 고객들",self.ready_order)


def FLT_Calculate(customer_in_order, customers, route, p2, M=1000, speed=1, now_t=0):
    """
    Calculate the customer`s Food Delivery Time in route(bundle)

    :param orders: customer order in the route. type: customer class
    :param route: customer route. [int,...,]
    :param p2: allowable FLT increase
    :param speed: rider speed
    :return: Feasiblity : True/False, FLT list : [float,...,]
    """
    names = []
    for order in customer_in_order:
        names.append(order.name)
    ftds = []
    # input(''.format())
    for order_name in names:
        rev_p2 = p2
        if customers[order_name].time_info[2] != None:
            print('FLT 고려 대상 {} 시간 정보 {}'.format(order_name, customers[order_name].time_info))
            last_time = now_t - customers[order_name].time_info[2]  # 이미 음식이 실린 후 지난 시간
            rev_p2 = p2 - last_time
        s = route.index(order_name + M)
        e = route.index(order_name)
        ftd = Basic.RouteTime(customer_in_order, route[s: e + 1], speed=speed, M=M)
        if ftd > rev_p2:
            return False, []
        else:
            ftds.append(ftd)
    return True, ftds


def RiderGenerator(env, Rider_dict, Platform, Store_dict, Orders, speed=1, end_time=120, interval=1, runtime=1000,
                   gen_num=10):
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
        single_rider = Rider(env, rider_num, Platform, Orders, Store_dict, speed=speed, end_t=end_time)
        Rider_dict[rider_num] = single_rider
        rider_num += 1
        print('라이더 {} 생성'.format(rider_num))
        yield env.timeout(interval)


def ordergenerator(env, orders, stores, interval=5, end_time=100):
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
        # process_time = random.randrange(1,5)
        # input_location = [36,36]
        input_location = random.sample(list(range(50)), 2)
        store_num = random.randrange(0, len(stores))
        order = Basic.Customer(env, name, input_location, store=store_num, store_loc=stores[store_num].location)
        orders[name] = order
        stores[store_num].received_orders.append(orders[name])
        yield env.timeout(interval)
        # print('현재 {} 플랫폼 주문 수 {}'.format(int(env.now), len(platform)))
        name += 1


order_interval = 1
rider_working_time = 120
interval = 5
p2 = 20
thres_p = 1
run_time = 120
# 실행부
env = simpy.Environment()
# Platform = simpy.Store(env)
Orders = {}
Platform = []
store_num = 2
rider_num = 0
Store_dict = {}
Rider_dict = {}
rider_gen_interval = 10
rider_speed = 2.5

# Before simulation, generate the stores.
for store_name in range(store_num):
    loc = list(random.sample(range(0, 50), 2))
    store = Store(env, Platform, store_name, loc=loc, capacity=10, print_para=False)
    # env.process(store.StoreRunner(env, Platform, capacity=store.capacity))
    Store_dict[store_name] = store

env.process(RiderGenerator(env, Rider_dict, Platform, Store_dict, Orders, speed=rider_speed, end_time=120,
                           interval=rider_gen_interval, runtime=run_time, gen_num=rider_num))
env.process(ordergenerator(env, Orders, Store_dict, interval=order_interval))
# env.process(Basic.Platform_process(env, Platform, Orders, Rider_dict, p2, thres_p, interval, speed = rider_speed, end_t = 1000))
env.run(run_time)