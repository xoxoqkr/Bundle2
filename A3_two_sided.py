# -*- coding: utf-8 -*-

#from scipy.stats import poisson
import operator
import itertools
from A1_Class import Order
from A1_BasicFunc import RouteTime, distance, FLT_Calculate, ActiveRiderCalculator, WillingtoWork
from A2_Func import BundleConsist
import numpy as np
import random
import copy

def CountActiveRider(riders, t, min_pr = 0):
    """
    현대 시점에서 t 시점내에 주문을 선택할 확률이 min_pr보다 더 높은 라이더를 계산
    @param riders: RIDER CLASS DICT
    @param t: t시점
    @param min_pr: 최소 확률(주문 선택확률이 min_pr 보다는 높아야 함.)
    @return: 만족하는 라이더의 이름. LIST
    """
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
    @return: w = {KY : ^_t_i ELE : ^_t_i가 발생할 확률}
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
    # todo : t시점이 흐르고 난 다음의 e_b가 계산되어야 함.
    e_b = []
    for store_name in stores:
        store = stores[store_name]
        for info in infos:
            rider = riders[info[0]]
            val = store.wait*distance(rider.CurrentLoc(), store.loc)
            e_b.append(val)
    return round(sum(e_b),4)


def Calculate_d_b(riders, states , t_now):
    # todo : t시점이 흐르고 난 다음의 d_b가 계산되어야 함.
    d_b = []
    for info in states:
        rider = riders[info[0]]
        val = WillingtoWork(rider, t_now)
        d_b.append(val)
    # state가 주어졌을 때의 d_b를 계산
    return round(sum(d_b),4)


def BundleScoreSimulator(riders, platform, orders, stores, w, t, t_now):
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
            tem.append([rider_name, select_order_name])
        e_b_bar = Calculate_e_b(stores, riders, tem)
        d_b_bar = Calculate_d_b(riders, tem, t_now)
        e.append(w[seq]*e_b_bar)
        d.append(w[seq]*d_b_bar)
        #라이더의 주문 선택 시뮬레이터
    return round(sum(e),4), round(sum(d),4)


def MIN_OD_pair(orders, q,s,):
    # 1 OD-pair 계산
    Q = itertools.permutations(q, s)  # 기존 OD pair의 가장 짧은 순서를 결정 하기 위함.
    OD_pair_dist = []
    for seq in Q:
        route_dist = 0
        tem_route = []
        for name in seq:
            tem_route += [orders[name].store_loc, orders[name].location]
        for index in range(1, len(tem_route)):
            before = tem_route[index - 1]
            after = tem_route[index]
            route_dist += distance(before, after)
        OD_pair_dist.append(route_dist)
    return min(OD_pair_dist)


def SelectByTwo_sided_way(target_order, riders, orders, stores, platform, s, p2, t, t_now, min_pr, speed = 1, bundle_search_variant = 1):
    feasible_bundles = ConstructFeasibleBundle_TwoSided(target_order, orders, s, p2, speed=speed, bundle_search_variant = bundle_search_variant)
    count = 0
    scores = []
    for feasible_bundle in feasible_bundles:
        s = feasible_bundle[6]
        e,d = Two_sidedScore(feasible_bundle, riders, orders, stores, platform, t, t_now, min_pr, M=1000, sample_size=1000)
        scores.append([count, s,e,d,0])
    for base_score in scores:
        dominance_count = 0
        for score in scores:
            if base_score[0] != score[0]:
                if base_score[2] <= score[2] and base_score[3] <= score[3]:
                    dominance_count += 1
    scores.sort(key = operator.itemgetter(4))
    return scores[0]


def ConstructFeasibleBundle_TwoSided(target_order, orders, s, p2, thres = 0.1, speed = 1, option = False, uncertainty = False, platform_exp_error = 1, bundle_search_variant = 1):
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
    d = []
    for order_name in orders:
        order = orders[order_name]
        if bundle_search_variant == 1:
            if order.type == 'bundle':
                continue
            else:
                pass
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
            OD_pair_dist = MIN_OD_pair(orders, q, s)
            for info in tem_route_info:
                info.append((OD_pair_dist - info[5] / s))
        b += tem_route_info
    b.sort(key=operator.itemgetter(6))  # s_b 순으로 정렬  #target order를 포함하는 모든 번들에 대해서 s_b를 계산.
    comparable_b = []
    b_star = b[0][6]
    for ele in b:
        if (ele[6] - b_star)/b_star <= thres: #percent loss 가 thres 보다 작아야 함.
            comparable_b.append(ele)
    return comparable_b

def Two_sidedScore(bundle, riders, orders, stores, platform, t, t_now, min_pr , M = 1000, sample_size=1000, platform_exp_error = 1):
    active_rider_names = CountActiveRider(riders, t, min_pr=min_pr)
    p_s_t = WeightCalculator(riders, active_rider_names, sample_size=sample_size)
    mock_platform = copy.deepcopy(platform)
    mock_index = max(mock_platform.keys()) + 1
    route = []
    for node in bundle[0]:
        if node >= M:
            customer_name = node - M
            customer = orders[customer_name]
            route.append([customer_name, 0, customer.store_loc, 0])
        else:
            customer_name = node
            customer = orders[customer_name]
            route.append([customer_name, 1, customer.location, 0])
    fee = 0
    for customer_name in bundle[4]:
        fee += orders[customer_name].fee  # 주문의 금액 더하기.
        orders[customer_name].in_bundle_time = t_now
        pool = np.random.normal(customer.cook_info[1][0], customer.cook_info[1][1] * platform_exp_error, 1000)
        orders[customer_name].platform_exp_cook_time = random.choice(pool)
    o = Order(mock_index, bundle[4], route, 'bundle', fee=fee)
    o.average_ftd = bundle[2]
    mock_platform[mock_index] = o #가상의 번들을 추가.
    e,d = BundleScoreSimulator(riders, mock_platform, orders, stores, p_s_t, t, t_now)
    return e,d




def BundleSimulator(riders, orders, platform, bundle_infos, stores, t_now, sample_size = 1000):
    # bundle_infos = [route, round(max(ftds),2), round(sum(ftds)/len(ftds),2), round(min(ftds),2), order_names, round(route_time,2), s_b]
    # s_b 계산
    s = []
    for bundle in bundle_infos:
        pass
    compare_bundles = []
    # s_b와 차이가 10% 이내인 b에 대해서 e,d 계산 수행
    compare = []
    s.sort(key = operator.itemgetter(1))
    max_s = s[0][1]
    for info in s:
        s_value = info[1]
        if max_s/s_value <= 1.1 :
            compare.append(info)
    res = []
    if len(compare) > 0 :
        w = WeightCalculator(riders, orders, sample_size=sample_size)
        for info in compare:
            e,d = BundleScoreSimulator(riders, platform, orders, stores, w, t_now)
            res.append([info[0],info[1], e,d])
        #pareto count로 결정하기. 대체품
    return res