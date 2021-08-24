# -*- coding: utf-8 -*-

#from scipy.stats import poisson
import operator
import itertools
from A1_BasicFunc import RouteTime, distance, FLT_Calculate, ActiveRiderCalculator, WillingtoWork
from A2_Func import BundleConsist
import numpy as np
import random


def CountActiveRider(riders, t, min_pr = 0):
    names = []
    for rider_name in riders:
        rider = riders[rider_name]
        if ActiveRiderCalculator(rider) == True and rider.select_pr(t) >= min_pr:
            names.append(rider_name)
    return names


def WeightCalculator(riders, rider_names, sample_size = 1000):
    """
    시뮬레이션을 활용해서, t_i의 발생 확률을 계산함.
    @param riders: RIDER CLASS DICT
    @param rider_names: active rider names list
    @param sample_size: 포아송 분포 샘플 크기
    @return:
    """
    w = {}
    ava_combinations = list(itertools.permutations(rider_names, len(rider_names))) # 가능한 조합
    count = {}
    for info in ava_combinations:
        count[info] = 0
    poisson_dist = {}
    for rider_name in rider_names:
        rider = riders[rider_name]
        mu = rider.search_lamda
        if mu not in poisson_dist:
            poisson_dist[mu] = np.random.poisson(mu, sample_size)
            #poisson_ = poisson(mu)# 분포 추정
            #poisson_dist[mu] = poisson_.rvs(sample_size)  # 추정한 분포 기반 Sample 1000개 생성
    for _ in sample_size:
        tem = []
        for rider_name in rider_names:
            rider = riders[rider_name]
            val = random.choice(poisson_dist[rider.search_lamda])
            tem.append([rider_name, val])
        tem.sort(key = operator.itemgetter(1))
        seq = []
        for info in tem:
            seq.append(info[0])
        count[seq] += 1
    for key in count:
        if count[key] > 0:
            w[key] = round(count[key]/sample_size,6)
    return w


def Calculate_e_b(stores, riders, infos):
    e_b = []
    for store_name in stores:
        store = stores[store_name]
        for info in infos:
            rider = riders[info[0]]
            val = store.wait*distance(rider.CurrentLoc(), store.loc)
            e_b.append(val)
    return round(sum(e_b),4)


def Calculate_d_b(riders, infos, t_now):
    d_b = []
    for info in infos:
        rider = riders[info[0]]
        val = WillingtoWork(rider, t_now)
        d_b.append(val)
    # state가 주어졌을 때의 d_b를 계산
    return round(sum(d_b),4)


def Calculate_s_b(orders, infos):
    OD_pair_dist = 0
    pass


def BundleScoreSimulator(riders, platform, orders, stores, w, t_now):
    # 매 번들 마다 수행해야함.
    e = []
    d = []
    # 라이더가 t_i의 순서대로 주문을 선택할 때, 선택할 주문을 선택
    for seq in w:
        tem = []
        for rider_name in seq:
            rider = riders[rider_name]
            current_loc = rider.CurrentLoc()
            select_order_name = rider.OrderSelect(platform, orders, current_loc = current_loc)
            select_order_name = None
            tem.append([rider_name, select_order_name])
        e_b_bar = Calculate_e_b(stores, tem)
        d_b_bar = Calculate_d_b(riders, tem, t_now)
        e.append(w[seq]*e_b_bar)
        d.append(w[seq]*d_b_bar)
        #라이더의 주문 선택 시뮬레이터
    return round(sum(e),4), round(sum(d),4)


def BundleSimulator(riders, orders, platform, bundle_infos, stores, t_now, sample_size = 1000):
    # bundle_infos = [route, round(max(ftds),2), round(sum(ftds)/len(ftds),2), round(min(ftds),2), order_names, round(route_time,2), s_b]
    # s_b 계산
    s = []
    for bundle in bundles:
        pass

    # s_b와 차이가 10% 이내인 b에 대해서 e,d 계산 수행
    compare = []
    s.sort(key = operator.itemgetter(1))
    max_s = s[0][1]
    for info in s:
        s_value = info[1]
        if max_s/s_value <= 1.1 :
            compare.append(info)
    if len(compare) > 0 :
        res = []
        w = WeightCalculator(riders, orders, sample_size=sample_size)
        for info in compare:
            e,d = BundleScoreSimulator(riders, platform, orders, stores, w, t_now)
            res.append([info[0],info[1], e,d])
        #pareto count로 결정하기. 대체품
    return None

def ConstructBundleTwoSided(target_order, orders, s, p2, speed = 1, option = False, uncertainty = False, platform_exp_error = 1):
    """
    Construct s-size bundle pool based on the customer in orders.
    And select n bundle from the pool
    Required condition : customer`s FLT <= p2
    :param new_orders: new order genrated during t_bar
    :param orders: userved customers : [customer class, ...,]
    :param s: bundle size: 2 or 3
    :param p2: max FLT
    :param speed:rider speed
    :parm option:
    :parm uncertainty:
    :parm platform_exp_error:
    :return: constructed bundle set
    """
    for order_name in orders:
        order = orders[order_name]
        d = []
        dist_thres = order.p2
        dist = distance(target_order.store_loc , order.store_loc) / speed
        if target_order.name != order.name and dist <= dist_thres:
            d.append(order.name)
    M = itertools.permutations(d, s - 1)
    b = []
    for m in M:
        q = list(m) + [target_order.name]
        subset_orders = []
        time_thres = 0 #3개의 경로를 연속으로 가는 것 보다는
        for name in q:
            subset_orders.append(orders[name])
            time_thres += orders[name].distance/speed
        tem_route_info = BundleConsist(subset_orders, orders, p2, speed = speed, option= option, time_thres= time_thres, uncertainty = uncertainty, platform_exp_error = platform_exp_error, feasible_subset = True)
        if len(tem_route_info) > 0:
            b.append(tem_route_info)
    #s_b를 계산하는 부분 추가.
    #1 OD-pair 계산
    Q = itertools.permutations(q, s)
    OD_pair_dist = []
    for seq in Q:
        route_dist = 0
        tem_route = []
        for name in seq:
            tem_route += [orders[name].store_loc, orders[name].location]
        for index in range(1, len(tem_route)):
            before = tem_route[index-1]
            after = tem_route[index]
            route_dist += distance(before, after)
        OD_pair_dist.append(route_dist)
    OD_pair_dist = min(OD_pair_dist)
    for info in b:
        info.append((OD_pair_dist - info[5]/s))
    b.sort(key = operator.itemgetter(6)) #s_b 순으로 정렬
    #새로 작성한 함수 사용 가장 좋은 번들을 파악
    if len(b) > 0:
        pass

    return selected_bundle


def ConstructBundleTwoSided_ORG(orders, s, n, p2, speed = 1, option = False, uncertainty = False, platform_exp_error = 1):
    """
    Construct s-size bundle pool based on the customer in orders.
    And select n bundle from the pool
    Required condition : customer`s FLT <= p2
    :param orders: userved customers : [customer class, ...,]
    :param s: bundle size: 2 or 3
    :param n: needed bundle number
    :param p2: max FLT
    :param speed:rider speed
    :parm option:
    :parm uncertainty:
    :parm platform_exp_error:
    :return: constructed bundle set
    """
    B = []
    for order_name in orders:
        order = orders[order_name]
        d = []
        dist_thres = order.p2
        for order2_name in orders:
            order2 = orders[order2_name]
            dist = distance(order.store_loc, order2.store_loc)/speed
            if order2 != order and dist <= dist_thres:
                d.append(order2.name)
        #M = itertools.combinations(d,s-1)
        M = itertools.permutations(d, s - 1)
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