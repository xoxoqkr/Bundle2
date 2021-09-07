# -*- coding: utf-8 -*-

from scipy.stats import poisson
import simpy
import operator
import itertools
import random
import A1_BasicFunc as Basic
import time
import numpy


# customer.time_info = [0 :발생시간, 1: 차량에 할당 시간, 2:차량에 실린 시간, 3:목적지 도착 시간,
# 4:고객이 받은 시간, 5: 보장 배송 시간, 6:가게에서 준비시간,7: 고객에게 서비스 하는 시간]
class Order(object):
    def __init__(self, order_name, customer_names, route, order_type, fee = 0):
        self.index = order_name
        self.customers = customer_names
        self.route = route
        self.picked = False
        self.type = order_type #1:단주문, 2:B2, 3:B3
        self.average_ftd = None
        self.fee = fee



class Rider(object):
    def __init__(self, env, i, platform, customers, stores, start_time = 0, speed = 1, capacity = 3, end_t = 120, p2= 15, bound = 5, freedom = True, max_order_num = 5, order_select_type = 'simple', wait_para = False, uncertainty = False, exp_error = 1):
        self.name = i
        self.env = env
        self.gen_time = int(env.now)
        self.resource = simpy.Resource(env, capacity=1)
        self.visited_route = [[-1, -1, [25, 25], int(env.now)]]
        self.speed = speed
        self.route = []
        self.run_process = None
        self.capacity = capacity
        self.onhand = []
        self.picked_orders = []
        self.end_t = env.now + end_t
        self.last_departure_loc = [25, 25]
        self.container = []
        self.served = []
        self.p2 = p2
        self.start_time = start_time
        self.max_order_num = max_order_num
        self.bound = bound
        self.idle_time = 0
        self.candidates = []
        self.b_select = 0
        self.income = 0
        self.wait_para = wait_para
        self.store_wait = 0
        self.num_bundle_customer = 0
        self.bundle_store_wait = [] # 번들에 속한 주문들에 의해 발생한 대기 시간
        self.single_store_wait = [] # 일반 주문에 의해 발생한 대기 시간
        self.onhand_order_indexs = []
        self.decision_moment = []
        self.exp_error = exp_error
        self.search_lamda = random.randint(2,5)
        self.exp_wage = 0
        env.process(self.RunProcess(env, platform, customers, stores, self.p2, freedom= freedom, order_select_type = order_select_type, uncertainty = uncertainty))


    def RiderMoving(self, env, time):
        """
        라이더가 움직이는 시간의 env.time의 generator를 반환
        :param env: simpy.env
        :param time: 라이더가 움직이는 시간
        """
        yield env.timeout(time)
        #print('현재1 T:{} 라이더{} 가게 {} 도착'.format(int(env.now),self.name, info ))


    def RunProcess(self, env, platform, customers, stores,p2 = 0, wait_time = 2, freedom = True, order_select_type = 'simple', uncertainty = False):
        """
        라이더의 행동 과정을 정의.
        1)주문 선택
        2)선택할만 한 주문이 없는 경우 대기(wait time)
        @param env:
        @param platform:
        @param customers:
        @param stores:
        @param p2:
        @param wait_time:
        """
        while int(env.now) < self.end_t:
            if len(self.route) > 0:
                node_info = self.route[0]
                #print('T: {} 라이더 :{} 노드 정보 {} 경로 {}'.format(int(env.now),self.name, node_info,self.route))
                #('체크1')
                order = customers[node_info[0]] #customer
                store_name = order.store
                move_t = Basic.distance(self.last_departure_loc, node_info[2]) / self.speed
                #self.heading_to = node_info[2]
                with self.resource.request() as req:
                    print('T: {} 노드 {} 시작'.format(int(env.now), node_info))
                    yield req  # users에 들어간 이후에 작동
                    req.loc = node_info[2]
                    print('T: {} 라이더 : {} 노드 {} 이동 시작 예상 시간{}'.format(int(env.now), self.name, node_info, move_t))
                    if node_info[1] == 0: #가게인 경우
                        exp_store_arrive = env.now + move_t
                        if order.type == 'single_order':
                            pool = numpy.random.normal(order.cook_info[1][0], order.cook_info[1][1] * self.exp_error, 1000)
                            order.rider_exp_cook_time = random.choice(pool)
                            exp_cook_time = order.rider_exp_cook_time
                        else:  # 'bundle'
                            exp_cook_time = order.platform_exp_cook_time
                        wait_at_store, food_wait, manual_cook_time = WaitTimeCal1(exp_store_arrive, order.time_info[1], exp_cook_time, order.cook_time,move_t = move_t)
                        ##결과 저장 부##
                        self.store_wait += wait_at_store
                        order.rider_wait = wait_at_store
                        order.food_wait = food_wait
                        if order.inbundle == True:
                            self.bundle_store_wait.append(wait_at_store)
                        else:
                            self.single_store_wait.append(wait_at_store)
                        yield env.process(stores[store_name].Cook(env, order, order.cook_info[0], manual_cook_time = manual_cook_time)) & env.process(self.RiderMoving(env, move_t))
                        """
                        ##필요 없는 부분 시작 ###
                        exp_store_arrive = env.now + move_t
                        yield env.process(stores[store_name].Cook(env, order, order.cook_info[0], manual_cook_time = manual_cook_time)) & env.process(self.RiderMoving(env, move_t))
                        if self.wait_para == True:
                            if order.type == 'single_order':
                                pool = numpy.random.normal(order.cook_info[1][0], order.cook_info[1][1] * self.exp_error, 1000)
                                order.rider_exp_cook_time = random.choice(pool)
                                exp_cook_time = order.rider_exp_cook_time
                                if exp_store_arrive > order.time_info[1] + exp_cook_time:
                                    wait_at_store = 0
                                    food_wait = exp_store_arrive - (order.time_info[1] + exp_cook_time)
                                else:
                                    wait_at_store = (order.time_info[1] + exp_cook_time) - exp_store_arrive
                                    food_wait = 0
                            else: #'bundle'
                                exp_cook_time = order.platform_exp_cook_time
                                if exp_store_arrive > order.time_info[1] + exp_cook_time:
                                    wait_at_store = 0
                                    food_wait = exp_store_arrive - (order.time_info[1] + exp_cook_time)
                                else:
                                    wait_at_store = (order.time_info[1] + exp_cook_time) - exp_store_arrive
                                    food_wait = 0
                            if order.cook_time > wait_at_store:
                                yield env.timeout(wait_at_store)
                                self.store_wait += wait_at_store
                                order.rider_wait = wait_at_store
                            else:
                                wait_at_store = 0
                            if order.inbundle == True:
                                self.bundle_store_wait.append(wait_at_store)
                            else:
                                self.single_store_wait.append(wait_at_store)                        
                        ##필요 없는 부분 종료###
                        """
                        print('T:{} 라이더{} 고객{}을 위해 가게 {} 도착'.format(int(env.now), self.name, customers[node_info[0]].name,customers[node_info[0]].store))
                        self.container.append(node_info[0])
                        order.time_info[2] = env.now
                        order.time_info[8] = exp_store_arrive
                        #input('가게 도착')
                    else:#고객인 경우
                        #input('T: {} 고객 {} 이동 시작'.format(int(env.now),node_info[0]))
                        yield env.process(self.RiderMoving(env, move_t))
                        print('T: {} 라이더 {} 고객 {} 도착 서비스 종료'.format(int(env.now),self.name, node_info[0]))
                        #input('고객 도착')
                        order.time_info[3] = env.now
                        order.who_serve.append([self.name, int(env.now)])
                        try:
                            self.container.remove(node_info[0])
                            self.onhand.remove(node_info[0])
                            self.served.append(node_info[0])
                            self.income += order.fee
                        except:
                            input('현재 컨테이너::{}/들고 있는 주문::{}/대상 주문::{}'.format(self.container,self.onhand,node_info[0]))
                        #todo: order를 완료한 경우 order를 self.picked_orders에서 제거해야함.
                        for order_info in self.picked_orders:
                            done = True
                            for customer_name in order_info[1]:
                                if customer_name not in self.served:
                                    done = False
                                    break
                            if done == True:
                                self.picked_orders.remove(order_info)
                    #print('T: {} 노드 {} 도착 '.format(int(env.now), node_info))
                    self.last_departure_loc = self.route[0][2]
                    self.visited_route.append(self.route[0])
                    self.visited_route[-1][3] = int(env.now)
                    #input('현재 경로 {} 제거 대상 {}'.format(self.route, self.route[0]))
                    del self.route[0]
                    print('남은 경로 {}'.format(self.route))
            freedom_para = True #todo : 라이더의 임의 주문 선택을 방지.
            if freedom == False:
                if len(self.route) > 0:
                    freedom_para == False
            if len(self.onhand) < self.capacity and freedom_para == True:
                print('T{} 라이더 {} 추가 탐색 시작'.format(env.now, self.name))
                test = []
                print('플랫폼 ID 확인 {}'.format(id(platform)))
                for index in platform.platform:
                    test += [platform.platform[index].customers]
                #t1 = time.time()
                uppicked_order_num = 0
                for order_index in platform.platform:
                    if platform.platform[order_index].picked == False:
                        uppicked_order_num += 1
                print('플랫폼의 주문 수 {}/ 대기 중인 주문 수 {}'.format(len(platform.platform),uppicked_order_num))
                self.candidates.append(len(platform.platform))
                order_info = self.OrderSelect(platform, customers, p2 = p2, score_type= order_select_type, uncertainty = uncertainty) #todo : 라이더의 선택 과정
                #t2 = time.time()
                #elapsed_time = t2 - t1
                #print(f"처리시간：{elapsed_time}")
                #print('계산 종료 {} '.format(env.now))
                if order_info != None:
                    #input('체크')
                    print('T {} 라이더 {} 주문 {} 선택 : 주문 고객 {}'.format(int(env.now),self.name, order_info[0], order_info[1]))
                    self.decision_moment.append(round(env.now,2))
                    #print('라이더 {} :: 간격 {} 현재 {} 최근 시점 {}'.format(self.name, self.decision_moment[-1], env.now, self.decision_moment[-2]))
                    added_order = platform.platform[order_info[0]]
                    if len(added_order.customers) > 1:
                        #input('라이더 {} 번들 {} 선택'.format(self.name, added_order.route))
                        pass
                    else:
                        #input('라이더 {} 주문 {} 선택'.format(self.name, added_order.route))
                        pass
                    #print('대상 주문들 선택 여부:: 인덱스 {} 고객들{} 선택여부{}'.format(added_order.index, added_order.customers,added_order.picked))
                    #print('T: {}/ 라이더 {}/ 주문 {} 선택 / 고객들 {}'.format(int(env.now), self.name, added_order.index, added_order.customers))
                    #print('라이더 {} 플랫폼 ID{}'.format(self.name, id(platform)))
                    #input('정보00 {} 확인 {} :: {}'.format(order_info, platform.platform[order_info[0]].picked,platform.platform[order_info[0]]))
                    self.OrderPick(added_order, order_info[1], customers, env.now)
                    #input('정보11 {} 확인 {} :: {}'.format(order_info, platform.platform[order_info[0]].picked, platform.platform[order_info[0]]))
                    #input('확인2',platform.platform[order_info[1][0][0]].picked)
                    if len(added_order.route) > 2:
                        self.b_select += 1
                        self.num_bundle_customer += len(added_order.customers)
                    Basic.UpdatePlatformByOrderSelection(platform,order_info[0])  # 만약 개별 주문 선택이 있다면, 해당 주문이 선택된 번들을 제거.
                else:
                    if len(self.route) > 0:
                        pass
                    else:
                        yield env.timeout(wait_time)
                        self.idle_time += wait_time
                        print('라이더 {} -> 주문탐색 {}~{}'.format(self.name, int(env.now) - wait_time, int(env.now)))


    def OrderSelect(self, platform, customers, p2 = 0, score_type = 'simple',sort_standard = 7, uncertainty = False, current_loc = None, add = 'X' ):
        """
        라이더의 입장에서 platform의 주문들 중에서 가장 이윤이 높은 주문을 반환함.
        1)현재 수행 중인 경로에 플랫폼의 주문을 포함하는 최단 경로 계산
        2)주문들 중 최단 경로가 가장 짧은 주문 선택
        *Note : 선택하는 주문에 추가적인 조건이 걸리는 경우 ShortestRoute 추가적인 조건을 삽입할 수 있음.
        @param platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
        @param customers: 발생한 고객들 {[KY]customer name : [Value]class customer, ...}
        @param p2: 허용 Food Lead Time의 최대 값
        @param sort_standard: 정렬 기준 [2:최대 FLT,3:평균 FLT,4:최소FLT,6:경로 운행 시간]
        @return: [order index, route(선택한 고객 반영), route 길이]선택한 주문 정보 / None : 선택할 주문이 없는 경우
        """
        score = []
        bound_order_names = []
        for index in platform.platform:
            # 현재의 경로를 반영한 비용
            order = platform.platform[index]
            exp_onhand_order = order.customers + self.onhand
            #print('주문 고객 확인 {}/ 자신의 경로 길이 {} / 상태 {}/ ID {}'.format(order.customers, len(self.route), order.picked, id(order)))
            if order.picked == False:
                #if ((len(exp_onhand_order) <= self.capacity and len(self.picked_orders) <= self.max_order_num)) or (len(order.route) > 2 and len(self.onhand) < 3):
                if Basic.ActiveRiderCalculator(self) == True:
                #if len(self.picked_orders) <= self.max_order_num:
                    if type(order.route[0]) != list:
                        input('에러 확인 {} : {}'.format(self.last_departure_loc,order.route))
                    if current_loc != None:
                        dist = Basic.distance(current_loc, order.route[0][2]) / self.speed
                    else:
                        dist = Basic.distance(self.last_departure_loc, order.route[0][2])/self.speed #자신의 현재 위치와 order의 시작점(가게) 사이의 거리.
                    info = [order.index,dist]
                    bound_order_names.append(info)
                #elif len(order.route) > 2: #번들이라는 소리
                #    dist = Basic.distance(self.last_departure_loc, order.route[0][2]) / self.speed
                #    info = [order.index, dist]
                #    bound_order_names.append(info)
            bound_order_names.sort(key = operator.itemgetter(1))
        if len(bound_order_names) > 0:
            for info in bound_order_names[:self.bound]: #todo : route의 시작 위치와 자신의 위치사이의 거리가 가까운 bound개의 주문 중 선택.
                order = platform.platform[info[0]]
                if score_type == 'oracle':
                    route_info = self.ShortestRoute(order, customers, p2=p2, uncertainty = uncertainty)
                    #route_info = [rev_route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time]
                    if len(route_info) > 0:
                        benefit = order.fee / route_info[5]  # 이익 / 운행 시간
                        score.append([order.index] + route_info + [benefit])
                elif score_type == 'simple':
                    mv_time = 0
                    times = []
                    rev_route = [self.last_departure_loc]
                    for route_info in order.route:
                        rev_route.append(route_info[2])
                    for node_index in range(1,len(rev_route)):
                        mv_time += Basic.distance(rev_route[node_index - 1],rev_route[node_index])/self.speed
                    for customer_name in order.customers:
                        mv_time += customers[customer_name].time_info[6] #예상 가게 준비시간
                        mv_time += customers[customer_name].time_info[7] #예상 고객 준비시간
                        times.append(self.env.now - customers[customer_name].time_info[0])
                        #self.income += customers[customer_name].fee
                    WagePerMin = round(order.fee/mv_time,2) #분당 이익
                    if len(order.route) > 2:
                        #WagePerMin = 1000 + (100 - Basic.distance(rev_route[0],rev_route[1])) #현재 위치에서 가까운
                        #WagePerMin = 1000 + max(0,(100 - sum(times)/len(times))) #최신의 고객들 부터 선택하도록
                        pass
                    #input('추가 경로 {}'.format(order.route))
                    if type(order.route) == tuple:
                        order.route = list(order.route)
                    score.append([order.index] + [order.route ,None,None,None,order.customers,None] + [WagePerMin])
        if len(score) > 0:
            score.sort(key=operator.itemgetter(sort_standard), reverse = True)
            for info in score:
                #print('오더 index {} picked 유/무 {}'.format(info[0], platform.platform[info[0]].picked))
                pass
            #input('score 체크{}'.format(score))
            return score[0]
        else:
            print('가능한 주문 X/ 대상 주문{}'.format(len(bound_order_names)))
            return None


    def ShortestRoute(self, order, customers, now_t = 0, p2 = 0, M = 1000, uncertainty = False):
        """
        order를 수행할 수 있는 가장 짧은 경로를 계산 후, 해당 경로의 feasible 여/부를 계산
        반환 값 [경로, 최대 FLT, 평균 FLT, 최소FLT, 경로 내 고객 이름, 경로 운행 시간]
        *Note : 선택하는 주문에 추가적인 조건이 걸리는 경우 feasiblity에 추가적인 조건을 삽입할 수 있음.
        @param order: 주문 -> class order
        @param customers: 발생한 고객들 {[KY]customer name : [Value]class customer, ...}
        @param now_t: 현재 시간
        @param p2: 허용 Food Lead Time의 최대 값
        @param M: 가게와 고객을 구분하는 임의의 큰 수
        @return: 최단 경로 정보 -> [경로, 최대 FLT, 평균 FLT, 최소FLT, 경로 내 고객 이름, 경로 운행 시간]
        """
        prior_route = []
        #input('주문 {} 고객정보 {} /이미 방문 경로 {}/ 남은 경로 {}'.format(order.index, order.customers, self.visited_route, self.route))
        """
        for visitied_node in self.visited_route:
            for node in input_route:
                if node[0] == visitied_node[0]:
                    prior_route.append(visitied_node[0] + M)
                    break        
        """
        index_list = []
        except_names = []
        for visitied_node in self.visited_route:
            for node in self.route:
                if node[0] == visitied_node[0]:
                    index_list.append(self.visited_route.index(visitied_node))
                    break
        already_served_customer_names = []
        if len(index_list) > 0:
            index_list.sort()
            for visitied_node in self.visited_route[index_list[0]:]:
                if visitied_node[1] == 0:
                    prior_route.append(visitied_node[0] + M)
                else:
                    prior_route.append(visitied_node[0])
            #print('index_list {} 내용 {} 현재 경로 {} 과거 경로 {} 대상 경로 {}'.format(index_list, prior_route, self.route, self.visited_route, self.visited_route[index_list[0]:]))
            for prior in prior_route:
                if prior < M and customers[prior].time_info[3] != None:
                    already_served_customer_names.append(prior)
        #self.route에 있는 가장 이른 노드의 정보를 포함하는 prior route가 구성됨.
        #input('기 방문 노드 {}'.format(prior_route))
        order_names = []  # 가게 이름?
        store_names = []
        for customer_name in order.customers:
            order_names.append(customer_name)
            store_names.append(customer_name + M)
        #input('주문 목록1 {} /가게 목록1 {}'.format(order_names, store_names))
        for node_info in self.route:
            if node_info[1] == 1:
                order_names.append(node_info[0])
            else:
                store_names.append(node_info[0] + M)
        candi = order_names + store_names
        #input('주문 목록2 {} /가게 목록2 {}'.format(order_names, store_names))
        #input('이미 방문한 노드 {} /삽입 대상 {}'.format(prior_route, candi))
        t1 = time.time()
        #print('itertools.permutations 시작')
        #check = itertools.permutations(candi, len(candi))
        subset = itertools.permutations(candi, len(candi)) # todo: permutations 사용으로 연산량 부하 지점
        t2 = time.time()
        #print(f"itertools.permutations 처리시간：{t2 - t1}")
        #print('라이더 {} 탐색 대상 subset 수 :  / 입력 수 {}'.format(self.name,len(candi)))
        feasible_subset = []
        for route_part in subset:
            route = prior_route + list(route_part)
            #print('고려 되는 라우트:{}'.format(route))
            sequence_feasiblity = True
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
                order_customers_names = [] #원래 order에 속한 주문 넣기
                for customer_name in order.customers:
                    order_customers.append(customers[customer_name])
                    order_customers_names.append(customer_name)
                if len(self.route) > 0:
                    for info in self.route: #현재 수행 중인 주문들 넣기
                        if info[0] not in order_customers_names:
                            order_customers.append(customers[info[0]]) #추가된  고객과 기존에 남아 있는 고객들의 customer class
                            order_customers_names.append(info[0])
                for past_name in prior_route:
                    #if past_name not in order_customers_names + already_served_customer_names:
                    if past_name not in order_customers_names:
                        if past_name < M:
                            order_customers.append(customers[past_name])  # 사이에 있는 주문 중 고객이 있다는 것은 이미 서비스 받은 고객을 의미.
                            order_customers_names.append(past_name)
                        else:
                            order_customers.append(customers[past_name - M])  # 사이에 있는 주문들 넣기
                            order_customers_names.append(past_name - M)
                if len(already_served_customer_names) > 0:
                    #print('이미 서비스 받아서 고려 필요 X 고객 {}'.format(already_served_customer_names))
                    pass
                # todo: FLT_Calculate 가 모든 형태의 경로에 대한 고려가 가능한기 볼 것.
                #print('FTL계산시작')
                t1 = time.time()
                ftd_feasiblity, ftds = Basic.FLT_Calculate(order_customers, customers, route,  p2, except_names = already_served_customer_names, M=M, speed=self.speed, now_t=now_t, uncertainty = uncertainty, exp_error = self.exp_error)
                #print('FTL계산종료')
                t2 = time.time()
                #print(f"FTL계산종료 itertools.permutations 처리시간：{t2 - t1}")
                if ftd_feasiblity == True:
                    # print('ftds',ftds)
                    # input('멈춤5')
                    #route_time = Basic.RouteTime(order_customers, route, speed=speed, M=M)
                    route_time = Basic.RouteTime(order_customers, list(route_part), speed=self.speed, M=M, uncertainty=uncertainty, error = self.exp_error)
                    #feasible_routes.append([route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time])
                    #route_time = Basic.RouteTime(order_customers, list(route_part), speed=speed, M=M)
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
                    try:
                        feasible_routes.append([rev_route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time])
                    except:
                        #input('대상 경로 {} 고객들 {} '.format(rev_route, order_names))
                        print('대상 경로 {} 고객들 {} '.format(rev_route, order_names))

                    #input('기존 경로 중 {} 제외 경로 {} -> 추가될 경로 {}'.format(route,prior_route,rev_route))
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

    def OrderPick(self, order, route, customers, now_t):
        """
        수행한 order에 대한 경로를 차량 경로self.route에 반영하고, onhand에 해당 주문을 추가.
        @param order: class order
        @param route: 수정될 경로
        @param customers: 발생한 고객들 {[KY]customer name : [Value]class customer, ...}
        @param now_t: 현재 시간
        """
        order.picked = True
        names = order.customers
        for name in names:
            customers[name].time_info[1] = now_t
            if len(names) > 1:
                customers[name].inbundle = True
                customers[name].type = 'bundle'
            #print('주문 {}의 고객 {} 가게 위치{} 고객 위치{}'.format(order.index, name, customers[name].store_loc, customers[name].location))
        #print('선택된 주문의 고객들 {} / 추가 경로{}'.format(names, route))
        if route[0][1] != 0:
            #input('삽입 경로에 문제 발생:: 삽입경로 {}'.format(route))
            pass
        self.route = route
        self.onhand += names
        self.picked_orders.append([order.index, names])
        print('라이더 {} 수정후 경로 {}/ 보유 고객 {}/ 추가된 고객 {}'.format(self.name, self.route, self.onhand, names))


    def select_pr(self, t):
        # t분 이내에 주문을 선택을 할 확률을 반환
        mu = (1.0/self.search_lamda)*t
        x = 0
        return round(1 - poisson.pmf(x, mu),4) # 주문을 1번이상 수행할 확률


    def CurrentLoc(self, t_now):
        """
        현재의 위치를 물어 보는 함수.
        @return:
        """
        nodeA = self.last_departure_loc
        try:
            nodeB = self.resource.users[0].loc
        except:
            #print(' T {} 출발 위치 에러 ; 마지막 노드 {}'.format(t_now, self.last_departure_loc))
            nodeB = self.last_departure_loc
        if nodeA == nodeB:
            return nodeA
        else:
            t = t_now - self.visited_route[-1][3] # nodeA출발 후 경과한 시간.
            ratio = t / Basic.distance(nodeA, nodeB)
            x_inc = (nodeB[0] - nodeA[0])*ratio
            y_inc = (nodeB[1] - nodeA[1])*ratio
            return [nodeA[0] + x_inc, nodeA[1] + y_inc]


class Store(object):
    """
    Store can received the order.
    Store has capacity. The order exceed the capacity must be wait.
    """
    def __init__(self, env, platform, name, loc = (25,25), order_ready_time = 7, capacity = 6, slack = 2, print_para = True):
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
        :param platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
        :param capacity: 발생한 고객들 {[KY]customer name : [Value]class customer, ...}
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
                platform_exist_order = []
                for index in platform.platform:
                    try:
                        platform_exist_order += platform.platform[index].customers
                    except:
                        print(' 에러 확인 용', platform.platform, index,platform.platform[index].customers)
                #print('플랫폼에 있는 주문 {}'.format(platform_exist_order))
                if received_orders_num > 0:
                    for count in range(min(slack,received_orders_num)):
                        order = self.received_orders[0] #앞에서 부터 플랫폼에 주문 올리기
                        route = [order.name, 0, order.store_loc, 0], [order.name, 1, order.location,0]
                        if len(list(platform.platform.keys())) > 0:
                            order_index = max(list(platform.platform.keys())) + 1
                        else:
                            order_index = 1
                        o = Order(order_index, [order.name],route,'single', fee = order.fee)
                        #print('주문 정보',o.index, o.customers, o.route, o.type)
                        if o.customers[0] not in platform_exist_order:
                            #platform[order_index] = o
                            platform.platform[order_index] = o
                            print('T : {} 가게 {} 고객 {} 주문 인덱스 {}에 추가'.format(env.now, self.name, o.customers, o.index))
                            print('플랫폼 ID{}'.format(id(platform)))
                        #platform.append(o)
                        #print('T : {} 가게 {} 고객 {} 주문 인덱스 {}에 추가'.format(env.now, self.name, o.customers, o.index))
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


    def Cook(self, env, customer, cooking_time_type = 'fixed', manual_cook_time = None):
        """
        Occupy the store capacity and cook the order
        :param env: simpy Env
        :param customer: class customer
        :param cooking_time_type: option
        """
        #print('현재 사용중', len(self.resource.users))
        with self.resource.request() as req:
            yield req #resource를 점유 해야 함.
            now_time = round(env.now , 1)
            req.info = [customer.name, now_time]
            if cooking_time_type == 'fixed':
                cooking_time = self.order_ready_time
            elif cooking_time_type == 'random':
                cooking_time = random.randrange(1,self.order_ready_time)
            elif cooking_time_type == 'uncertainty':
                cooking_time = customer.cook_time
            else:
                cooking_time = 0.001
            print('T :{} 가게 {}, {} 분 후 주문 {} 조리 완료'.format(int(env.now),self.name,cooking_time,customer.name))
            if manual_cook_time == None:
                yield env.timeout(cooking_time)
            else:
                yield env.timeout(manual_cook_time)
            #print(self.resource.users)
            print('T :{} 가게 {} 주문 {} 완료'.format(int(env.now),self.name,customer.name))
            customer.food_ready = True
            customer.ready_time = env.now
            self.ready_order.append(customer)
            #print('T',int(env.now),"기다리는 중인 고객들",self.ready_order)

class Customer(object):
    def __init__(self, env, name, input_location, store = 0, store_loc = (25, 25),end_time = 60, ready_time=3, service_time=3, fee = 2500, p2 = 15, cooking_time = (2,5), cook_info = (None, None)):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.time_info = [round(env.now, 2), None, None, None, None, end_time, ready_time, service_time, None]
        # [0 :발생시간, 1: 차량에 할당 시간, 2:차량에 실린 시간, 3:목적지 도착 시간,
        # 4:고객이 받은 시간, 5: 보장 배송 시간, 6:가게 출발 시간),7: 고객에게 서비스 하는 시간, 8: 가게 도착 시간]
        self.location = input_location
        self.store_loc = store_loc
        self.store = store
        self.type = 'single_order'
        self.min_FLT = p2 #Basic.distance(input_location, store_loc) #todo: 고객이 기대하는 FLT 시간.
        self.fee = fee + 150*Basic.distance(input_location, store_loc)
        self.ready_time = None #가게에서 음식이 조리 완료된 시점
        self.who_serve = []
        self.distance = Basic.distance(input_location, store_loc)
        self.p2 = p2
        self.cook_time = cooking_time
        self.inbundle = False
        self.rider_wait = 0
        self.in_bundle_time = None
        self.cook_info = cook_info
        self.exp_info = [None,None,None]
        self.rider_exp_cook_time = None
        self.platform_exp_cook_time = None
        self.food_wait = None
        #self.sensitiveness = random.randrange()

class Platform_pool(object):
    def __init__(self):
        self.platform = {}
        self.info = []


def WaitTimeCal1(exp_store_arrive_t, assign_t, exp_cook_time, cook_time, move_t = 0):
    exp_food_ready_t = assign_t + exp_cook_time
    actual_food_ready_time = assign_t + cook_time
    if exp_store_arrive_t > exp_food_ready_t: #조리 시간에 여유가 있다면, 미이 주문을 실시
        #wait_at_store = 0
        food_wait = exp_store_arrive_t - exp_food_ready_t
        manual_cook_time = 0.001  # 거의 존재하지 않는다는 의미.
        if actual_food_ready_time > exp_store_arrive_t:
            wait_at_store = actual_food_ready_time - exp_store_arrive_t
            food_wait = 0
            manual_cook_time = wait_at_store + move_t
        elif exp_store_arrive_t >= actual_food_ready_time >= exp_food_ready_t:
            wait_at_store =0
            food_wait = exp_store_arrive_t - actual_food_ready_time
            manual_cook_time = 0.001
        elif exp_food_ready_t >= actual_food_ready_time:
            wait_at_store = 0
            food_wait = exp_store_arrive_t - actual_food_ready_time
            manual_cook_time = 0.001
        else:
            input('시간 관계 문제 발생1 ::actual_food_ready_time {}:: exp_store_arrive_t {} :: exp_food_ready_t{} '.format(actual_food_ready_time, exp_store_arrive_t, exp_food_ready_t))
    else:#조리 시간에 여유가 없다면, 가게에서 대기
        wait_at_store = exp_food_ready_t - exp_store_arrive_t
        food_wait = 0
        manual_cook_time = move_t + wait_at_store
        if actual_food_ready_time > exp_food_ready_t:
            wait_at_store = actual_food_ready_time - exp_store_arrive_t
            food_wait = 0
            manual_cook_time = wait_at_store + move_t
        elif exp_food_ready_t >= actual_food_ready_time >= exp_store_arrive_t:
            wait_at_store = actual_food_ready_time - exp_store_arrive_t
            food_wait = 0
            manual_cook_time = wait_at_store + move_t
        elif exp_store_arrive_t >= actual_food_ready_time:
            wait_at_store = 0
            food_wait = exp_store_arrive_t - actual_food_ready_time
            manual_cook_time = 0.001
        else:
            input('시간 관계 문제 발생2 ::actual_food_ready_time {}:: exp_store_arrive_t {} :: exp_food_ready_t{} '.format(actual_food_ready_time, exp_store_arrive_t, exp_food_ready_t))
    return wait_at_store, food_wait, manual_cook_time


def WaitTimeCal0(exp_store_arrive_t, assign_t, exp_cook_time, cook_time, move_t = 0):
    exp_food_ready_t = assign_t + exp_cook_time
    actual_food_ready_time = assign_t + cook_time
    if exp_store_arrive_t > exp_food_ready_t: #조리 시간에 여유가 있다면, 미이 주문을 실시
        wait_at_store = 0
        food_wait = exp_store_arrive_t - exp_food_ready_t
        manual_cook_time = 0.001  # 거의 존재하지 않는다는 의미.
    else:#조리 시간에 여유가 없다면, 가게에서 대기
        wait_at_store = exp_food_ready_t - exp_store_arrive_t
        food_wait = 0
        manual_cook_time = move_t + wait_at_store
    return wait_at_store, food_wait, manual_cook_time



def WaitTimeCal2(exp_store_arrive_t, exp_food_ready_t, move_t = 0):
    if exp_store_arrive_t > exp_food_ready_t:
        wait_at_store = 0
        food_wait = exp_store_arrive_t - exp_food_ready_t
        if food_wait < 0:
            wait_at_store += -food_wait
            food_wait = 0
            manual_cook_time = move_t - food_wait
        else:
            manual_cook_time = 0.001  # 거의 존재하지 않는다는 의미.
    else:
        wait_at_store = exp_food_ready_t - exp_store_arrive_t
        food_wait = 0
        if wait_at_store < 0:
            wait_at_store = 0
            food_wait += - wait_at_store
            manual_cook_time = move_t - wait_at_store
        else:
            manual_cook_time = move_t + wait_at_store
    if wait_at_store < 0 or food_wait < 0:
        pass
    return wait_at_store, food_wait, manual_cook_time
