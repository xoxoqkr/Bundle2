# -*- coding: utf-8 -*-
import math
import random
import A1_Class as Class



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


def FLT_Calculate(customer_in_order, customers, route, p2, except_names , M = 1000, speed = 1, now_t = 0):
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
        if order.name not in names:
            names.append(order.name)
    ftds = []
    #input(''.format())
    #print('경로 고객들 {} 경로 {}'.format(names, route))
    for order_name in names:
        if order_name not in except_names:
            rev_p2 = p2
            if customers[order_name].time_info[2] != None:
                print('FLT 고려 대상 {} 시간 정보 {}'.format(order_name,customers[order_name].time_info))
                last_time = now_t - customers[order_name].time_info[2] #이미 음식이 실린 후 지난 시간
                rev_p2 = p2 - last_time
            try:
                s = route.index(order_name + M)
                e = route.index(order_name)
                try:
                    ftd = RouteTime(customer_in_order, route[s: e + 1], speed=speed, M=M)
                except:
                    print('경로 {}'.format(route))
                    print('경로 시간 계산 에러/ 현재고객 {}/ 경로 고객들 {}'.format(order_name,names))
                    input('중지')
            except:
                ftd = 0
                print('경로 {}'.format(route))
                print('인덱스 에러 발생 현재 고객 이름 {} 경로 고객들 {} 경로 {}'.format(order_name, names, route))
                #input('인덱스 에러 발생')
            #s = route.index(order_name + M)
            #e = route.index(order_name)
            if ftd > rev_p2:
                return False, []
            else:
                ftds.append(ftd)
    return True, ftds


def RiderGenerator(env, Rider_dict, Platform, Store_dict, Customer_dict, speed = 1, working_duration = 120, interval = 1, runtime = 1000, gen_num = 10):
    """
    Generate the rider until t <= runtime and rider_num<= gen_num
    :param env: simpy environment
    :param Rider_dict: 플랫폼에 있는 라이더들 {[KY]rider name : [Value]class rider, ...}
    :param rider_name: 라이더 이름 int+
    :param Platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
    :param Store_dict: 플랫폼에 올라온 가게들 {[KY]store name : [Value]class store, ...}
    :param Customer_dict:발생한 고객들 {[KY]customer name : [Value]class customer, ...}
    :param working_duration: 운행 시작 후 운행을 하는 시간
    :param interval: 라이더 생성 간격
    :param runtime: 시뮬레이션 동작 시간
    :param gen_num: 생성 라이더 수
    """
    rider_num = 0
    while env.now <= runtime and rider_num <= gen_num:
        single_rider = Class.Rider(env,rider_num,Platform, Customer_dict,  Store_dict, speed = speed, end_t = working_duration)
        Rider_dict[rider_num] = single_rider
        rider_num += 1
        print('T {} 라이더 {} 생성'.format(int(env.now), rider_num))
        yield env.timeout(interval)


def ordergenerator(env, orders, stores, interval = 5, runtime = 100):
    """
    Generate customer order
    :param env: Simpy Env
    :param orders: Order
    :param platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
    :param stores: 플랫폼에 올라온 가게들 {[KY]store name : [Value]class store, ...}
    :param interval: 주문 생성 간격
    :param runtime: 시뮬레이션 동작 시간
    """
    name = 0
    while env.now < runtime:
        #process_time = random.randrange(1,5)
        #input_location = [36,36]
        input_location = random.sample(list(range(50)),2)
        store_num = random.randrange(0, len(stores))
        order = Class.Customer(env, name, input_location, store = store_num, store_loc = stores[store_num].location)
        orders[name] = order
        stores[store_num].received_orders.append(orders[name])
        yield env.timeout(interval)
        #print('현재 {} 플랫폼 주문 수 {}'.format(int(env.now), len(platform)))
        name += 1