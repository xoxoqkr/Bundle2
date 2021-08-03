# -*- coding: utf-8 -*-
import operator
import itertools
from A1_BasicFunc import RouteTime, distance, FLT_Calculate
from A1_Class import Order
import time
import matplotlib.pyplot as plt
import numpy as np

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
    if mu1 + mu2 + add_mu > 0:
        rho = (lamda1 + lamda2 + add_lamda) / (mu1 + mu2 + add_mu)
    else:
        rho = 2
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
    for order_name in platform_set.platform:
        order = platform_set.platform[order_name]
        if order.type == 'bundle':
            if len(order.customers) == 2:
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
    for order_name in platform_set.platform:
        order = platform_set.platform[order_name]
        if order.type == 'bundle':
            if len(order.customers) == 2:
                b2.append(order)
            else:
                b3.append(order)
        else:
            single_orders.append(order)
    b2.sort(key=operator.attrgetter('average_ftd'), reverse=True)
    b3.sort(key=operator.attrgetter('average_ftd'), reverse=True)
    for break_b2 in range(min(break_info[0],len(b2))):
        #breaked_customer_names.append(b2[0].customers)
        breaked_customer_names += b2[0].customers
        del b2[0]
    for break_b3 in range(min(break_info[1],len(b3))):
        #breaked_customer_names.append(b3[0].customers)
        breaked_customer_names += b3[0].customers
        del b3[0]
    breaked_customers = []
    order_nums = []
    for order_name in platform_set.platform:
        order = platform_set.platform[order_name]
        order_nums += order.customers
    order_num = max(order_nums) + 1
    for customer_name in breaked_customer_names:
        route = [[customer_name, 0, customer_set[customer_name].store_loc, 0],[customer_name, 1, customer_set[customer_name].location, 0 ]]
        order = Order(order_num,[customer_name], route, 'single', fee = customer_set[customer_name].fee)
        breaked_customers.append(order)
    res = {}
    for order in single_orders + b2 + b3 + breaked_customers:
        res[order.index] = order
    return res


def BundleConsist(orders, customers, p2, time_thres = 0, speed = 1,M = 1000, option = False, uncertainty = False, platform_exp_error =  1):
    """
    Construct bundle consists of orders
    :param orders: customer order in the route. type: customer class
    :param customers: customer dict  {[KY]customer name: [Value]class customer,...}
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
    if option == False:
        subset = itertools.permutations(candi, len(candi))
    else:
        store_subset = itertools.permutations(store_names, len(store_names))
        store_subset = list(store_subset)
        order_subset = itertools.permutations(order_names, len(order_names))
        order_subset = list(order_subset)
        test = []
        test_names = itertools.permutations(order_names, 2)
        for names in test_names:
            dist = distance(customers[names[0]].location, customers[names[1]].location)
            if dist > 15:
                return []
        subset = []
        for store in store_subset:
            for order in order_subset:
                tem = store + order
                subset.append(tem)
        pass
    #print('번들 처리 시작. 대상 subset{}'.format(subset))
    #print('번들 고려시 탐색 수 {}'.format(len(list(subset))))
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
            ftd_feasiblity, ftds = FLT_Calculate(orders, customers, route, p2, [],M = M ,speed = speed, uncertainty =uncertainty, exp_error=platform_exp_error)
            #customer_in_order, customers, route, p2, except_names, M = 1000, speed = 1, now_t = 0
            if ftd_feasiblity == True:
                route_time = RouteTime(orders, route, speed=speed, M=M, uncertainty = uncertainty, error = platform_exp_error)
                feasible_routes.append([route, round(max(ftds), 2), round(sum(ftds) / len(ftds), 2), round(min(ftds), 2), order_names,round(route_time, 2)])
                #print('시간 정보 번들 경로 시간 {} : 가능한 짧은 시간 {}'.format(route_time, time_thres))
                #if route_time < time_thres :
                #    feasible_routes.append([route, round(max(ftds),2), round(sum(ftds)/len(ftds),2), round(min(ftds),2), order_names,round(route_time,2)])
                #    input('번들 생성 절약 시간 {}'.format(time_thres - route_time))
                #[경로, 최대FTD, 평균FTD, 최소FTD]
        if len(feasible_routes) > 0:
            feasible_routes.sort(key = operator.itemgetter(2))
            feasible_subset.append(feasible_routes[0])
    if len(feasible_subset) > 0:
        feasible_subset.sort(key = operator.itemgetter(2))
        #GraphDraw(feasible_subset[0], customers)
        return feasible_subset[0]
    else:
        return []


def GraphDraw(infos, customers, M = 1000):
    # 그래프 그리기
    x = []
    y = []
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    store_label = []
    loc_label = []
    locs = []
    for node in infos[0]:
        if node > M:
            name = node - M
            x.append(customers[name].store_loc[0])
            y.append(customers[name].store_loc[1])
            x1.append(customers[name].store_loc[0])
            y1.append(customers[name].store_loc[1])
            store_label.append('S' + str(name))
            locs.append(customers[name].store_loc)
        else:
            x.append(customers[node].location[0])
            y.append(customers[node].location[1])
            x2.append(customers[node].location[0])
            y2.append(customers[node].location[1])
            loc_label.append('C' + str(node))
            locs.append(customers[name].location)
    # plt.plot(x, y, linestyle='solid', color='blue', marker = 6)
    x3 = np.array(x)
    y3 = np.array(y)
    plt.quiver(x3[:-1], y3[:-1], x3[1:] - x3[:-1], y3[1:] - y3[:-1], scale_units='xy', angles='xy', scale=1)
    plt.scatter(x1, y1, marker="X", color='g')
    plt.scatter(x2, y2, marker="o", color='r')
    plt.title("Bundle coordinate")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.show()
    # name = str(random.random)
    tm = time.localtime(time.time())
    # print("hour:", tm.tm_hour)
    # print("minute:", tm.tm_min)
    # print("second:", tm.tm_sec)
    plt.savefig('B{}Hr{}Min{}Sec{}.png'.format(len(infos[4]), tm.tm_hour, tm.tm_min, tm.tm_sec))
    plt.close()
    print('경로 {}'.format(locs))
    #input('번들 경로 {} 시간 {}'.format(infos[0],infos[5]))



def ConstructBundle(orders, s, n, p2, speed = 1, option = False, uncertainty = False, platform_exp_error = 1):
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
        #dist_thres = p2 - distance(order.store_loc, order.location)/speed
        dist_thres = order.p2
        for order2_name in orders:
            order2 = orders[order2_name]
            dist = distance(order.store_loc, order2.store_loc)/speed
            if order2 != order and dist <= dist_thres:
                d.append(order2.name)
        #M = itertools.combinations(d,s-1)
        M = itertools.permutations(d, s - 1)
        #print('번들 구성 고려 subset 수 {}'.format(len(list(M))))
        #M = list(M)
        b = []
        for m in M:
            q = list(m) + [order.name]
            subset_orders = []
            time_thres = 0 #3개의 경로를 연속으로 가는 것 보다는
            for name in q:
                subset_orders.append(orders[name])
                time_thres += orders[name].distance/speed
            tem_route_info = BundleConsist(subset_orders, orders, p2, speed = speed, option= option, time_thres= time_thres, uncertainty = uncertainty, platform_exp_error = platform_exp_error)
            if len(tem_route_info) > 0:
                b.append(tem_route_info)
        if len(b) > 0:
            b.sort(key = operator.itemgetter(2))
            B.append(b[0])
            #input('삽입되는 {}'.format(b[0]))
    #n개의 번들 선택
    B.sort(key = operator.itemgetter(5))
    selected_bundles = []
    selected_orders = []
    print('번들들 {}'.format(B))
    for bundle_info in B:
        # bundle_info = [[route,max(ftds),average(ftds), min(ftds), names],...,]
        unique = True
        for name in bundle_info[4]:
            if name in selected_orders:
                unique = False
                break
        if unique == True:
            selected_orders += bundle_info[4]
            selected_bundles.append(bundle_info)
            if len(selected_bundles) >= n:
                break
    if len(selected_bundles) > 0:
        #print("selected bundle#", len(selected_bundles))
        print("selected bundle#", selected_bundles)
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


def PlatformOrderRevise(bundle_infos, customer_set, order_index, platform_set, M = 1000, divide_option = False, now_t = 0):
    """
    Construct unpicked_orders with bundled customer
    :param bundles: constructed bundles
    :param customer_set: customer list : [customer class,...,]
    :return: unserved customer set
    """
    unpicked_orders, num = CountUnpickedOrders(customer_set, 0 , interval = 0, return_type = 'name')
    bundle_names = []
    names = []
    res = {}
    #info = [[route, max(ftds), average(ftds), min(ftds), names],...,]
    for info in bundle_infos:
        bundle_names += info[4]
        if len(info[4]) == 1:
            customer = customer_set[info[4][0]]
            route = [[customer.name, 0, customer.store_loc, 0],[customer.name, 1, customer.location, 0]]
            o = Order(order_index, info[4][0], route, 'single', fee = customer.fee)
        else:
            route = []
            for node in info[0]:
                if node >= M:
                    customer_name = node - M
                    customer = customer_set[customer_name]
                    route.append([customer_name, 0, customer.store_loc, 0])
                else:
                    customer_name = node
                    customer = customer_set[customer_name]
                    route.append([customer_name, 1, customer.location, 0])
            fee = 0
            for customer_name in info[4]:
                fee += customer_set[customer_name].fee #주문의 금액 더하기.
                customer_set[customer_name].in_bundle_time = now_t
            o = Order(order_index, info[4], route, 'bundle', fee = fee)
        o.average_ftd = info[2]
        res[order_index] = o
        #res.append(o)
        order_index += 1
    for index in platform_set.platform:
        order = platform_set.platform[index]
        if order.type == 'single':
            if order.customers[0] not in bundle_names and order.picked == False and customer_set[order.customers[0]].time_info[1] == None:
                res[order.index] = order
            else:
                pass
        else:
            if order.picked == False:
                #res.append(order)
                pass
    already_ordered_customer_names = []
    for index in res:
        already_ordered_customer_names += res[index].customers
    for index in platform_set.platform:
        already_ordered_customer_names += platform_set.platform[index].customers
    for customer_name in unpicked_orders:
        if divide_option == True:
            condition = customer_name not in already_ordered_customer_names
        else:
            condition = customer_name not in bundle_names + already_ordered_customer_names
        #if customer_name not in bundle_names + already_ordered_customer_names:
        if condition == True:
            names.append(customer_name)
            customer = customer_set[customer_name]
            if customer.time_info[1] == None:
                singleroute = [[customer.name , 0 , customer.store_loc,0],[customer.name, 1, customer.location, 0]]
                o = Order(order_index, [customer_name], singleroute, 'single', fee = customer.fee)
                #res.append(o)
                res[order_index] = o
                order_index += 1
                #print('추가 정보22 {}'.format(customer_name))
    return res


def ConsideredCustomer(platform_set, orders, unserved_order_break = False):
    """
    번들 구성에 고려될 수 있는 고객들을 선별함.
    @param platform_set: platform set list [class order, ...,]
    @param orders: customer set {[KY] customer name : [Value] class customer}
    @param unserved_order_break: T: 서비스 되지 않은 번들도 번들 구성에 고려/ F : 이미 구성된 번들은 고려하지 않음.
    @return: 번들 구성에 사용될 수 있는 고객 {[KY] customer name : [Value] class customer}
    """
    rev_order = {}  # 아직 서비스 되지 않은 고객 + 플렛폼에 있으나, 아직 번들로 구성되지 않은 주문 [KY] 고객 이름
    except_names = []
    #input('확인1 {}'.format(platform_set.platform))
    for index in platform_set.platform:
        #input('확인2 {}'.format(index))
        order = platform_set.platform[index]
        if order.type == 'single':
            if order.picked == False and orders[order.customers[0]].time_info[1] == None:
                rev_order[order.customers[0]] = orders[order.customers[0]]
            else: #already picked customer
                pass
        else: #번들인 경우
            if order.picked == False:
                if unserved_order_break == False:
                    except_names += order.customers  # todo: 기존에 구성된 번들은 해체되지 않음.
                else:
                    pass
            else: #이미 선택된 주문의 경우
                pass
    for customer_name in orders:
        customer = orders[customer_name]
        if customer.time_info[1] == None and customer_name not in list(rev_order.keys()) + except_names:
            rev_order[customer_name] = customer
    print1 = []
    for order_name in rev_order:
        order = rev_order[order_name]
        #print1 += order.customers
        print1.append(order.name)
    print2 = []
    for customer_name in orders:
        customer = orders[customer_name]
        if customer.time_info[1] != None:
            print2.append(customer.name)
    print('번들 대상 고객 {}'.format(print1))
    print('실려있는 고객 {}'.format(print2))
    return rev_order

def Platform_process(env, platform_set, orders, riders, p2,thres_p,interval, speed = 1, end_t = 1000, unserved_order_break = True,option = False, divide_option = False, uncertainty = False, platform_exp_error = 1):
    B2 = []
    B3 = []
    while env.now <= end_t:
        now_t = env.now
        unpicked_orders, lamda2 = CountUnpickedOrders(orders, now_t, interval = interval ,return_type = 'class') #lamda1
        lamda1 = len(unpicked_orders)
        idle_riders, mu2 = CountIdleRiders(riders, now_t, interval = interval, return_type = 'class')
        mu1 = len(idle_riders)
        p = CalculateRho(lamda1, lamda2, mu1, mu2)
        #p = 2
        rev_order = ConsideredCustomer(platform_set, orders, unserved_order_break = unserved_order_break)
        print('번들 생성에 고려되는 고객들 {}'.format(sorted(list(rev_order.keys()))))
        if p >= thres_p:
            if lamda1/3 < mu1 + mu2:
                print("번들 계산 시작")
                t1 = time.time()
                b2,b3 = RequiredBundleNumber(lamda1, lamda2, mu1, mu2, thres=thres_p)
                t2 = time.time()
                print(f"번들 계산 처리시간：{t2 - t1}")
            else:
                b2 = 0
                b3 = int(lamda1/3)
            if b3 > 0:
                print("B3 처리 시작")
                t1 = time.time()
                b3_bundle = ConstructBundle(rev_order, 3, b3, p2, speed = speed, option = option , uncertainty = uncertainty, platform_exp_error = platform_exp_error)
                t2 = time.time()
                print(f"B3 처리시간：{t2 - t1}")
                # b3_bundle = [[route, max(ftds), average(ftds), min(ftds), names], ..., ]
                B3 = b3_bundle
            if b2 > 0 or len(B3) < b3:
                b2 = int((b3 - len(B3))*1.5)
                print("B2 처리 시작")
                t1 = time.time()
                b2_bundle = ConstructBundle(rev_order, 2, b2, p2, speed = speed, option = option, uncertainty = uncertainty, platform_exp_error = platform_exp_error)
                t2 = time.time()
                print(f"B2 처리시간：{t2 - t1}")
                #b2_bundle = [[route, max(ftds), average(ftds), min(ftds), names], ..., ]
                B2 = b2_bundle
            print('B2:', B2)
            print('B3:', B3)
            B = B2 + B3
            for index in platform_set.platform:
                bundle_names += platform_set.platform[index].customers
                #print('1 order index : {} added : {}'.format(order.index,order.customers))
            order_indexs = []
            for index in platform_set.platform:
                order_indexs.append(index)
            if len(order_indexs) > 0:
                order_index = max(order_indexs) + 1
            else:
                order_index = 1
            print('인덱스 수 확인', platform_set.platform.keys(), '키', order_index)
            bundle_names = []
            for index in platform_set.platform:
                bundle_names += platform_set.platform[index].customers
                #print('2 order index : {} added : {}'.format(order.index,order.customers))
            print('고객 이름들 2 :: {}'.format(list(bundle_names)))
            new_orders = PlatformOrderRevise(B, orders, order_index,platform_set, divide_option = divide_option, now_t= round(env.now,2)) #todo: 이번에 구성되지 않은 단번 주문은 바로 플랫폼에 계시.
            bundle_names = []
            for index in new_orders:
                bundle_names += new_orders[index].customers
            still_names = []
            for index in platform_set.platform:
                still_names += platform_set.platform[index].customers
            print('고객 이름들 3 :: 기존 {} 추가 {}'.format(list(sorted(still_names)),list(sorted(bundle_names))))
            print('전체함수 플랫폼1 ID{}'.format(id(platform_set)))
            #platform_set.platform = new_orders
            print('원래 index {} :: 추가 index {}'.format(platform_set.platform.keys(),new_orders.keys()))
            #platform_set.platform.update(new_orders)
            platform_set.platform = new_orders
            count = [[],[]]
            for index in platform_set.platform:
                if platform_set.platform[index].type == 'single':
                    count[0].append(platform_set.platform[index].customers)
                else:
                    #print('종류??',platform_set.platform[index].type)
                    count[1].append(platform_set.platform[index].customers)
            print('고객 이름들 4 :: 단건 주문 {} 번들 주문 {}'.format(count[0], count[1]))
            print('전체함수 플랫폼2 ID{}'.format(id(platform_set)))
        else: #Break the offered bundle
            org_bundle_num, rev_bundle_num = RequiredBreakBundleNum(platform_set, lamda2, mu1, mu2, thres=thres_p)
            if sum(rev_bundle_num) < sum(org_bundle_num):
                break_info = [org_bundle_num[0] - rev_bundle_num[0],org_bundle_num[1] - rev_bundle_num[1]] #[B2 해체 수, B3 해체 수]
                #번들의 해체가 필요
                platform_set.platform = BreakBundle(break_info, platform_set, orders)
                #input('확인 3 {}'.format(platform_set.platform))
        print('T: {} B2,B3확인'.format(int(env.now)))
        #input('T: {} B2,B3확인'.format(int(env.now)))
        yield env.timeout(interval)

def ResultPrint(name, customers, speed = 1):
    served_customer = []
    TLT = []
    FLT = []
    MFLT = []
    for customer_name in customers:
        customer = customers[customer_name]
        if customer.time_info[3] != None:
            lt = customer.time_info[3] - customer.time_info[0]
            flt = customer.time_info[3] - customer.time_info[2]
            mflt = distance(customer.store_loc, customer.location)/speed
            TLT.append(lt)
            FLT.append(flt)
            MFLT.append(mflt)
    try:
        served_ratio = round(len(TLT)/len(customers),2)
        av_TLT = round(sum(TLT)/len(TLT),2)
        av_FLT = round(sum(FLT)/len(FLT),2)
        av_MFLT = av_FLT - round(sum(MFLT)/len(MFLT),2)
        print('시나리오 명 {} 전체 고객 {} 중 서비스 고객 {}/ 서비스율 {}/ 평균 LT :{}/ 평균 FLT : {}/직선거리 대비 증가분 : {}'.format(name, len(customers), len(TLT),served_ratio,av_TLT,
                                                                             av_FLT, av_MFLT))
        return [len(customers), len(TLT),served_ratio,av_TLT,av_FLT, av_MFLT]
    except:
        print('TLT 수:  {}'.format(len(TLT)))
        return None