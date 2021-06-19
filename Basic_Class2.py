# -*- coding: utf-8 -*-
import operator
import random
import math
import simpy
import copy
import itertools


class Order(object):
    def __init__(self, order_name, customer_names, route, type = '1'):
        self.name = order_name
        self.customers = customer_names
        self.route = route
        self.type = type #1:단주문, 2:B2, 3:B3

class Customer(object):
    def __init__(self, env, name, input_location, store = 0, store_loc = [25,25],end_time = 60, ready_time=3, service_time=3, fee = 1000):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.time_info = [round(env.now, 2), None, None, None, None, end_time, ready_time, service_time]
        # [0 :발생시간, 1: 차량에 할당 시간, 2:차량에 실린 시간, 3:목적지 도착 시간,
        # 4:고객이 받은 시간, 5: 보장 배송 시간, 6:가게에서 준비시간,7: 고객에게 서비스 하는 시간]
        self.location = input_location
        self.store_loc = store_loc
        self.store = store
        self.type = 'single_order'
        self.fee = fee
        self.ready_time = None #가게에서 음식이 조리 완료된 시점



class rider(object):
    def __init__(self, env, name, platform, stores, speed = 1, capacity = 3, end_time = 1000):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.start_time = int(env.now)
        self.resource = simpy.Resource(env, capacity = capacity)
        self.done_orders = []
        self.capacity = capacity
        self.end = False
        self.end_time = env.now + end_time
        self.speed = speed
        self.last_loc = [25,25]
        env.process(self.Runner(env, platform,stores))


    def RiderMoving(self, env, time, loc):
        """
        라이더가 움직이는 시간의 env.time의 generator를 반환
        :param env: simpy.env
        :param time: 라이더가 움직이는 시간
        """
        yield env.timeout(time)
        self.last_loc = loc
        print('현재T:',int(env.now),"/라이더 ",self.name,"가게 도착")

    def Runner(self, env, platform, stores, ready_time = 2):
        """
        Rider`s behavior during working time.
        :param env: simpy Env
        :param platform: Platform
        :param stores: Store. resource. limited capacity for cooking
        :param ready_time: ready time for pick-up
        :param end_time: rider exit time. rider will be accept the order until end time.
        """
        while env.now < self.end_time:
            order = RiderChoiceCustomer(self, platform)
            if len(platform) > 0 and len(self.resource.put_queue) < self.capacity and order != None:
                #order = RiderChoiceCustomer(self, platform)  #todo: 라이더가 platform의 주문 중 가장 이윤이 높은 것을 선택.
                #order = platform[0] #큐의 가장 앞에 주문을 잡음.
                order.time_info[1] = env.now
                print('현재T:',int(env.now),'/라이더',self.name,"/주문",order.name,'선택')
                #input('정지')
                store_name = order.store
                #print('가게정보',stores[store_name].wait_orders)
                platform.remove(order)
                #stores[store_name].wait_orders.remove(order)
                #move_time = random.randrange(5,12)
                move_time = distance(self.last_loc, order.store_loc) / self.speed
                exp_arrival_time = env.now + move_time
                print('현재T:', int(env.now), '/라이더', self.name, "/주문", order.name,'/가게까지 이동시간', move_time, '조리시간', stores[store_name].order_ready_time)
                #if len(stores[store_name].resource.users) + len(stores[store_name].wait_orders) > stores[store_name].capacity:
                #    print('음식점이 조리 불가','라이더',self.name,"주문 ", order.name,'/현재시간',int(env.now))
                yield env.process(stores[store_name].Cook(env, order)) & env.process(self.RiderMoving(env, move_time, order.store_loc)) #둘 중 더 오래 걸리는 process가 완료 될 때까지 기다림
                rider_wait_time = max(0, env.now - exp_arrival_time)
                food_wait_time = max(0, env.now - order.ready_time)
                #print(env.now - exp_arrival_time, env.now - order.ready_time)
                print('현재T:', int(env.now), '/라이더', self.name, "/주문", order.name, '픽업 완료/라이더 대기시간:',round(rider_wait_time,2),'/음식대기시간',round(food_wait_time,2))
                with self.resource.request() as req:
                    stores[store_name].ready_order.remove(order)
                    req.info = [order.name, round(env.now, 2)]
                    order.time_info[3] = env.now
                    yield req  # users에 들어간 이후에 작동
                    #move_time = random.randrange(1,5)
                    move_time = distance(order.store_loc, order.location)/self.speed
                    yield env.timeout(move_time)
                    order.time_info[4] = env.now
                    self.last_loc = order.location
                    #stores[store_name].ready_order.remove(order)
                    print('현재T:', int(env.now),'/라이더',self.name,"/배달 완료/ 고객:", order.name,'/이동시간',move_time)
            else: #현재 플랫폼에 주문이 없다면, 아래 시간 만큼 대기 후 다시 탐색 수행
                #print('현재T:', int(env.now),'/라이더',self.name,"/주문X")
                yield env.timeout(1)

class Bundle(object):
    """
    Bundle consists of multiple orders
    """
    def __init__(self, bundle_name, names, route, ftd_info):
        self.name = bundle_name
        self.size = len(names)
        self.customer_names = names
        self.routebyname = route
        self.min_ftd = ftd_info[0]
        self.average_ftd = ftd_info[1]
        self.max_ftds = ftd_info[2]
        self.routebycor = None
        self.gen_t = None
        self.visit_t = []
        self.fee = 0
        self.type = 'bundle'


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
                        platform.append(order)
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
            #print(cooking_time, '분 후 ', customer.name, "음식준비완료")
            yield env.timeout(cooking_time)
            #print(self.resource.users)
            print('현재T:',int(env.now),'/가게',self.name,'/주문:',customer.name,"음식준비완료")
            customer.food_ready = True
            customer.ready_time = env.now
            self.ready_order.append(customer)
            #print('T',int(env.now),"기다리는 중인 고객들",self.ready_order)


def RiderGenerator(env, Rider_dict, Platform, Store_dict, speed = 1, end_time = 120, interval = 1, runtime = 1000, gen_num = 10):
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
    while env.now <= runtime or rider_num <= gen_num:
        single_rider = rider(env,rider_num,Platform, Store_dict, speed = speed, end_time = end_time)
        Rider_dict[rider_num] = single_rider
        rider_num += 1
        yield env.timeout(interval)

def CustomerValueForRiderCalculator(rider, customer):
    """
    rider가 customer에 대해 가지는 가치 계산
    :param rider:
    :param customer:
    """
    value = 10
    return value

def RiderChoiceCustomer(rider, customers):
    """
    Rider pick the highest score orders
    rider의 시각에서 customers 중 가장 높은 가치를 가지는 customer 계산
    :param rider: class rider
    :param customers: customer list [customer, customer,...,]
    :return: highest score customer class
    """
    customer_values = []
    for customer in customers:
        if customer.time_info[1] == None:
            value = CustomerValueForRiderCalculator(rider, customer)
            customer_values.append([value, customer])
    customer_values.sort(key = operator.itemgetter(0), reverse = True)
    if len(customer_values) > 0:
        return customer_values[0][1]
    else:
        return None


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
        order = Customer(env, name, input_location, store = store_num, store_loc = stores[store_num].location)
        orders[name] = order
        stores[store_num].received_orders.append(orders[name])
        print('T:', int(env.now),'/주문:', orders[name].name,'접수')
        #print('가게 큐', stores[store_num].received_orders)
        platform.append(order)
        yield env.timeout(interval)
        print('현재 {} 플랫폼 주문 수 {}'.format(int(env.now), len(platform)))
        name += 1


def CalculateRho(lamda1, lamda2, mu1, mu2, add_lamda = 0, add_mu = 0):
    """
    Calculate rho
    :param lamda1: current lamda
    :param lamda2: expected lamda of the near future time slot
    :param mu1: current mu
    :param mu2: expected mu of the near future time slot
    :param add_lamda: additional lamda
    :param add_mu: additional mu
    :return: rho
    """
    rho = (lamda1 + lamda2 + add_lamda) / (mu1 + mu2 + add_mu)
    return round(rho, 4)


def RequiredBundleNumber(lamda1, lamda2, mu1, mu2, thres = 1):
    """
    Cacluate required b2 and b3 number
    condition : rho <= thres
    :param lamda1: current un-selected order
    :param lamda2: future generated order
    :param mu1: current rider
    :param mu2: future rider
    :param thres: rho thres: system over-load
    :return: b2, b3
    """
    b2 = 0
    b3 = 0
    for index in range(lamda1+lamda2):
        b2 += 1
        rho = CalculateRho(lamda1, lamda2, mu1, mu2, add_lamda = -b2)
        #rho = (lamda1 + lamda2 - b2)/(mu1 + mu2)
        if rho <= thres:
            return b2, b3
    for index in range(lamda1+lamda2):
        b2 -= 1
        b3 += 1
        rho = CalculateRho(lamda1, lamda2, mu1, mu2, add_lamda=-(b2+b3))
        #rho = (lamda1 + lamda2 - b2 - b3)/(mu1 + mu2)
        if rho <= thres:
            return b2, b3
    return b2, b3


def RequiredBreakBundleNum(platform_set, lamda2, mu1, mu2, thres = 1):
    """
    Caclculate availiable break-down bundle number
    :param platform_set: orders set : [order,...]
    :param lamda2: expected lamda of the near future time slot
    :param mu1: current mu
    :param mu2: expected mu of the near future time slot
    :param thres: system level.
    :return:
    """
    org_b2_num = 0
    org_b3_num = 0
    b2_num = 0
    b3_num = 0
    customer_num = 0
    for order in platform_set:
        if order.type == 'bundle':
            if order.size == 2:
                b2_num += 1
                org_b2_num += 1
            else:
                b3_num += 1
                org_b3_num += 1
        else:
            customer_num += 1
    end_para = False
    for count in range(org_b3_num): #break b3 first
        if b3_num > 0:
            b3_num -= 1
            customer_num += 3
        else:
            pass
        p = CalculateRho(b2_num + b3_num + customer_num, lamda2, mu1, mu2)
        if p >= thres:
            end_para = True
            break
    if end_para == False: #if p < thres, than break b2
        for count in range(org_b2_num):
            if b2_num > 0:
                b2_num -= 1
                customer_num += 2
            else:
                pass
            p = CalculateRho(b2_num + b3_num + customer_num, lamda2, mu1, mu2)
            if p >= thres:
                break
    return [org_b2_num,org_b3_num],[b2_num, b3_num]


def BreakBundle(break_info, platform_set, customer_set):
    """
    Break bundle by break_info
    And return the revised platform_set
    :param break_info: bundle breaking info [b2 decrcase num, b2 decrcase num]
    :param platform_set: orders set : [order,...]
    :param customer_set: customer set : [customer class,...]
    :return: breaked platform set
    """
    b2 = []
    b3 = []
    single_orders = []
    breaked_customer_names = []
    for order in platform_set:
        if order.type == 'bundle':
            if order.size == 2:
                b2.append(order)
            else:
                b3.append(order)
        else:
            single_orders.append(order)
    b2.sort(key=operator.attrgetter('average_ftd'), reverse=True)
    b3.sort(key=operator.attrgetter('average_ftd'), reverse=True)
    for break_b2 in range(min(break_info[0],len(b2))):
        breaked_customer_names.append(b2[0].customer_names)
        del b2[0]
    for break_b3 in range(min(break_info[1],len(b3))):
        breaked_customer_names.append(b3[0].customer_names)
        del b3[0]
    breaked_customers = []
    for customer_name in breaked_customer_names:
        breaked_customers.append(customer_set[customer_name])
    res = single_orders + b2 + b3 + breaked_customers
    return res


def distance(p1, p2):
    """
    Calculate 4 digit rounded euclidean distance between p1 and p2
    :param p1:
    :param p2:
    :return: 4 digit rounded euclidean distance between p1 and p2
    """
    euc_dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return round(euc_dist,4)


def RouteTime(orders, route, M = 1000, speed = 1):
    """
    Time to move the route with speed
    :param orders: order in route
    :param route: seq
    :param speed: rider speed
    :return: time : float
    """
    time = 0
    locs = {}
    names = []
    if type(orders) == dict:
        for order_name in orders:
            locs[order_name + M] = [orders[order_name].store_loc, 'store', orders[order_name].time_info[6]]
            locs[order_name] = [orders[order_name].location, 'customer', orders[order_name].time_info[7]]
            names += [order_name + M, order_name]
    elif type(orders) == list:
        for order in orders:
            locs[order.name + M] = [order.store_loc, 'store', order.time_info[6]]
            locs[order.name] = [order.location, 'customer', order.time_info[7]]
            names += [order.name + M, order.name]
    else:
        input('Error')
    #print('고려 대상들{} 경로{}'.format(list(locs.keys()), route))
    for index in range(1,len(route)):
        bf = route[index-1]
        bf_loc = locs[bf][0]
        af = route[index]
        #print(bf,af)
        af_loc = locs[af][0]
        time += distance(bf_loc,af_loc)/speed + locs[af][2]
        """
        print('정보', bf, af, af - M)
        print('고려 고객들', orders)
        input('멈춤4')
        if af < M : #customer process time #도착지가 고객인 경우
            time += orders[af].time_info[7]
        else: # store process time #도착지가 가게인 경우
            input('멈춤5')
            time += orders[af - M].time_info[6]        
        """
    return round(time,4)


def FLT_Calculate(orders, route, p2, M = 1000, speed = 1, add_time = None):
    """
    Calculate the customer`s Food Delivery Time in route(bundle)

    :param orders: customer order in the route. type: customer class
    :param route: customer route. [int,...,]
    :param p2: allowable FLT increase
    :param speed: rider speed
    :return: Feasiblity : True/False, FLT list : [float,...,]
    """
    names = []
    for order in orders:
        names.append(order.name)
    ftds = []
    for order_name in names:
        rev_p2 = p2
        if add_time != None:
            rev_p2 = p2 - add_time[order_name]
        s = route.index(order_name + M)
        e = route.index(order_name)
        ftd = RouteTime(orders, route[s: e + 1], speed = speed, M = M)
        if ftd > rev_p2:
            return False, []
        else:
            ftds.append(ftd)
    return True, ftds


def BundleConsist(orders, p2, speed = 1,M = 1000):
    """
    Construct bundle consists of orders
    :param orders: customer order in the route. type: customer class
    :param p2: allowable FLT increase
    :param M: big number for distinguish order name and store name
    :param speed: rider speed
    :return: feasible route
    """
    order_names = [] #가게 이름?
    for order in orders:
        order_names.append(order.name)
    store_names = []
    for name in order_names:
        store_names.append(name + M)
    candi = order_names + store_names
    subset = itertools.permutations(candi, len(candi))
    feasible_subset = []
    for route in subset:
        #print('고객이름',order_names,'가게이름',store_names,'경로',route)
        sequence_feasiblity = True #모든 가게가 고객 보다 앞에 오는 경우.
        feasible_routes = []
        for order_name in order_names: # order_name + M : store name ;
            if route.index(order_name + M) < route.index(order_name):
                pass
            else:
                sequence_feasiblity = False
                break
        if sequence_feasiblity == True:
            #print('순서는 만족', route)
            #input('멈춤3')
            ftd_feasiblity, ftds = FLT_Calculate(orders, route, p2, M = M ,speed = speed)
            if ftd_feasiblity == True:
                #print('ftds',ftds)
                #input('멈춤5')
                route_time = RouteTime(orders, route, speed=speed, M=M)
                feasible_routes.append([route, max(ftds), sum(ftds)/len(ftds), min(ftds), order_names,route_time])
                #[경로, 최대FTD, 평균FTD, 최소FTD]
        if len(feasible_routes) > 0:
            feasible_routes.sort(key = operator.itemgetter(2))
            feasible_subset.append(feasible_routes[0])
            #print('가능 경로', feasible_routes)
            #input('멈춤6')
    if len(feasible_subset) > 0:
        feasible_subset.sort(key = operator.itemgetter(2))
        return feasible_subset[0]
    else:
        return []



def ConstructBundle(orders, s, n, p2, speed = 1):
    """
    Construct s-size bundle pool based on the customer in orders.
    And select n bundle from the pool
    Required condition : customer`s FLT <= p2
    :param orders: userved customers : [customer class, ...,]
    :param s: bundle size: 2 or 3
    :param n: needed bundle number
    :param p2: max FLT
    :param speed:rider speed
    :return: constructed bundle set
    """
    B = []
    for order_name in orders:
        order = orders[order_name]
        d = []
        dist_thres = p2 - distance(order.store_loc, order.location)/speed
        for order2_name in orders:
            order2 = orders[order2_name]
            dist = distance(order.store_loc, order2.store_loc)/speed
            if order2 != order and dist <= dist_thres:
                d.append(order2.name)
        M = itertools.combinations(d,s-1)
        #M = list(M)
        b = []
        for m in M:
            q = list(m) + [order.name]
            subset_orders = []
            for name in q:
                subset_orders.append(orders[name])
            tem_route_info = BundleConsist(subset_orders, p2, speed = speed)
            if len(tem_route_info) > 0:
                b.append(tem_route_info)
        if len(b) > 0:
            b.sort(key = operator.itemgetter(2))
            B.append(b[0])
    #n개의 번들 선택
    selected_bundles = []
    selected_orders = []
    for bundle_info in B:
        # bundle_info = [[route,max(ftds),average(ftds), min(ftds), names],...,]
        unique = True
        for name in bundle_info[4]:
            if name in selected_orders:
                unique = False
                break
        if unique == True:
            selected_orders.append(bundle_info[4])
            selected_bundles.append(bundle_info)
            if len(selected_bundles) >= n:
                break
    if len(selected_bundles) > 0:
        #print("selected bundle#", len(selected_bundles))
        #print("selected bundle#", selected_bundles)
        #input('멈춤7')
        pass
    #todo: 1)겹치는 고객을 가지는 번들 중 1개를 선택해야함. 2)어떤 번들이 더 좋은 번들인가?
    return selected_bundles


def CountUnpickedOrders(orders, now_t , interval = 10, return_type = 'class'):
    """
    return un-picked order
    :param orders: order list : [order class,...]
    :param now_t : now time
    :param interval : platform`s bundle construct interval # 플랫폼에서 번들을 생성하는 시간 간격.
    :param return_type: 'class'/'name'
    :return: unpicked_orders, lamda2(future generated order)
    """
    unpicked_orders = []
    interval_orders = []
    for order_name in orders:
        order = orders[order_name]
        if order.time_info[1] == None:
            if return_type == 'class':
                unpicked_orders.append(order)
            elif return_type == 'name':
                unpicked_orders.append(order.name)
            else:
                pass
        if now_t- interval <= order.time_info[0] < now_t:
            interval_orders.append(order.name)
    return unpicked_orders, len(interval_orders)


def CountIdleRiders(riders, now_t , interval = 10, return_type = 'class'):
    """
    return idle rider
    :param riders: rider list : [rider class,...]
    :param now_t : now time
    :param interval : platform`s bundle construct interval # 플랫폼에서 번들을 생성하는 시간 간격.
    :param return_type: 'class'/'name'
    :return: idle_riders, mu2(future generated rider)
    """
    idle_riders = []
    interval_riders = []
    for rider_name in riders:
        #Count current idle rider
        rider = riders[rider_name]
        if len(rider.resource.users) == 0:
            if return_type == 'class':
                idle_riders.append(rider)
            elif return_type == 'name':
                idle_riders.append(rider.name)
            else:
                pass
        #count rider occurred from (now_t - interval, now)
        if now_t- interval <= rider.start_time < now_t:
            interval_riders.append(rider.name)
    return idle_riders, len(interval_riders)


def PlatformOrderRevise(bundles, customer_set):
    """
    Construct unpicked_orders with bundled customer
    :param bundles: constructed bundles
    :param customer_set: customer list : [customer class,...,]
    :return: unserved customer set
    """
    unpicked_orders, num = CountUnpickedOrders(customer_set, 0 , interval = 0, return_type = 'name')
    bundle_names = []
    names = []
    for bundle in bundles:
        bundle_names.append(bundle.customer_names)
    for customer_name in unpicked_orders:
        if customer_name not in bundle_names:
            names.append(customer_name)
    res = []
    for customer_name in names:
        res.append(customer_set[customer_name])
    res += bundles
    return res




def Platform_process(env, platform_set, orders, riders, p2,thres_p,interval, speed = 1, end_t = 1000):
    B2 = []
    B3 = []
    while env.now <= end_t:
        now_t = env.now
        unpicked_orders, lamda2 = CountUnpickedOrders(orders, now_t, interval = interval ,return_type = 'class') #lamda1
        lamda1 = len(unpicked_orders)
        idle_riders, mu2 = CountIdleRiders(riders, now_t, interval = interval, return_type = 'class')
        mu1 = len(idle_riders)
        #print(lamda1, lamda2, mu1, mu2)
        #input('확인2')
        p = CalculateRho(lamda1, lamda2, mu1, mu2)
        if p >= thres_p:
            if lamda1/3 < mu1 + mu2:
                b2,b3 = RequiredBundleNumber(lamda1, lamda2, mu1, mu2, thres=thres_p)
            else:
                b2 = 0
                b3 = int(lamda1/3)
            B = []
            if b2 > 0:
                b2_bundle = ConstructBundle(orders, 2, b2, p2, speed = speed)
                #b2_bundle = [[route, max(ftds), average(ftds), min(ftds), names], ..., ]
                B2 = b2_bundle
            if b3 > 0:
                b3_bundle = ConstructBundle(orders, 3, b3, p2, speed = speed)
                # b3_bundle = [[route, max(ftds), average(ftds), min(ftds), names], ..., ]
                B3 = b3_bundle
            count = 1
            print('B2:', B2)
            print('B3:', B3)
            input('B2,B3확인')
            for info in B2+B3:
                B.append(Bundle(count,info[4], info[0],info[1:4]))
                count += 1
            #offer bundle to the rider:
            offered_order = PlatformOrderRevise(B, orders)
            platform_set = offered_order
        else: #Break the offered bundle
            org_bundle_num, rev_bundle_num = RequiredBreakBundleNum(platform_set, lamda2, mu1, mu2, thres=thres_p)
            if sum(rev_bundle_num) < sum(org_bundle_num):
                break_info = [org_bundle_num[0] - rev_bundle_num[0],org_bundle_num[1] - rev_bundle_num[1]]
                #번들의 해체가 필요
                platform_set = BreakBundle(break_info, platform_set, orders)
        yield env.timeout(interval)

"""
# bracnh test
#파라메터 부
order_interval = 1
rider_working_time = 120
#실행부
env = simpy.Environment()
#Platform = simpy.Store(env)
Orders = {}
Platform = []
store_num = 1
rider_num = 3
Store_list = {}
for store_name in range(store_num):
    store = Store(env, Platform, store_name, capacity = 1)
    Store_list[store_name] = store

for rider_name in range(rider_num):
    rider(env,rider_name,Platform, Store_list, end_time = rider_working_time)


env.process(ordergenerator(env, Orders, Platform, Store_list, interval = order_interval))
#rider1 = rider(env,0,Platform, Store_list)
#rider2 = rider(env,1,Platform, Store_list)
env.run(100)
#print("queue check",Platform.put_queue)
"""