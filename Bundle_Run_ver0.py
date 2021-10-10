# -*- coding: utf-8 -*-

#from scipy.stats import poisson
import time
from A2_Func import CountUnpickedOrders, CountIdleRiders, CalculateRho, ConsideredCustomer, RequiredBundleNumber, ConstructBundle, PlatformOrderRevise ,\
    RequiredBreakBundleNum, BreakBundle, PlatformOrderRevise2,PlatformOrderRevise3, PlatformOrderRevise4
from A3_two_sided import SelectByTwo_sided_way, ParetoDominanceCount, BundleConsideredCustomers, SelectByTwo_sided_way2, Two_sidedScore, CountActiveRider, WeightCalculator, WeightCalculator2
import copy
import operator
from Bundle_selection_problem import Bundle_selection_problem, Bundle_selection_problem2
import math
import numpy as np


def Platform_process(env, platform_set, orders, riders, p2,thres_p,interval, speed = 1, end_t = 1000, unserved_order_break = True,option = False, divide_option = False, uncertainty = False, platform_exp_error = 1, bundle_select_type = 'normal'):
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
            #todo : 제안된 번들 중에서 특정 번들을 선택하는 과정
            if bundle_select_type == 'normal':
                new_orders = PlatformOrderRevise(B, orders, order_index,platform_set, divide_option = divide_option, now_t= round(env.now,2), platform_exp_error = platform_exp_error) #todo: 이번에 구성되지 않은 단번 주문은 바로 플랫폼에 계시.
            else:
                #new_orders = SelectByTwo_sided_way(target_order, riders, orders, stores, platform, s, p2, t, t_now, min_pr, speed=speed, bundle_search_variant=1, input_data=B)
                pass
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

def Platform_process2(env, platform_set, orders, riders, p2,thres_p,interval, speed = 1, end_t = 1000, unserved_order_break = True,option = False, divide_option = False, uncertainty = False, platform_exp_error = 1, bundle_select_type = 'normal'):
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
        if p >= thres_p and len(orders) > past_customer_num: #새롭게 발생한 고객이 존재하며, 번들의 구성이 필요함.
            if lamda1/3 < mu1 + mu2:
                t1 = time.time()
                b2,b3 = RequiredBundleNumber(lamda1, lamda2, mu1, mu2, thres=thres_p)
                t2 = time.time()
            else:
                b2 = 0
                b3 = int(lamda1/3)
            b_info = [[3, b3], [2, b2]]
            for b in b_info:
                if b[1] > 0:
                    if b[0] == 2 and len(B3) < b3:
                        b[1] = int((b_info[0][1] - len(B3)) * 1.5)
                    print('B {} 처리시작'.format(b[0]))
                    t1 = time.time()
                    searched_bundle = ConstructBundle(rev_order, b[0], b[1], p2, speed=speed, option=option,
                                                uncertainty=uncertainty, platform_exp_error=platform_exp_error)
                    t2 = time.time()
                    print(f"처리시간：{t2 - t1}")
                    if b[0] == 3:
                        B3 = searched_bundle
                    else:
                        B2 = searched_bundle
            print('B2: {} / B3: {}'.format(len(B2), len(B3)))
            B = B2 + B3
            for index in platform_set.platform:
                bundle_names += platform_set.platform[index].customers
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
            #todo : 제안된 번들 중에서 특정 번들을 선택하는 과정
            if bundle_select_type == 'normal':
                new_orders = PlatformOrderRevise(B, orders, order_index,platform_set, divide_option = divide_option, now_t= round(env.now,2), platform_exp_error = platform_exp_error) #todo: 이번에 구성되지 않은 단번 주문은 바로 플랫폼에 계시.
            else:
                #for b in B:
                #new_orders = SelectByTwo_sided_way(target_order, riders, orders, stores, platform, s, p2, t, t_now, min_pr, speed=speed, bundle_search_variant=1, input_data=B)
                pass
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
        past_customer_num = copy.deepcopy(len(orders))
        print('T: {} B2,B3확인'.format(int(env.now)))
        #input('T: {} B2,B3확인'.format(int(env.now)))
        yield env.timeout(interval)


def Platform_process3(env, platform_set, orders, riders, stores, p2,thres_p,interval, bundle_permutation_option = False,
                      speed = 1, end_t = 1000, unserved_bundle_order_break = True, divide_option = False,
                      platform_exp_error = 1, min_pr = 0.05, scoring_type = 'myopic'):
    while env.now <= end_t:
        now_t = env.now
        unpicked_orders, lamda2 = CountUnpickedOrders(orders, now_t, interval = interval ,return_type = 'class') #lamda1
        lamda1 = len(unpicked_orders)
        idle_riders, mu2 = CountIdleRiders(riders, now_t, interval = interval, return_type = 'class')
        mu1 = len(idle_riders)
        p = CalculateRho(lamda1, lamda2, mu1, mu2)
        #p = 2
        rev_order = ConsideredCustomer(platform_set, orders, unserved_order_break = unserved_bundle_order_break)
        print('번들 생성에 고려되는 고객들 {}'.format(sorted(list(rev_order.keys()))))
        if p >= thres_p:
            B = []
            new_customer_names = []
            for customer_name in orders:
                customer = orders[customer_name]
                if env.now - interval <= customer.time_info[0] and customer.time_info[1] == None:
                    new_customer_names.append(customer.name)
            print('새로 생긴 고객들 {}'.format(new_customer_names))
            for customer_name in new_customer_names:
                start = time.time()
                target_order = orders[customer_name]
                selected_bundle = SelectByTwo_sided_way(target_order, riders, orders, stores, platform_set, p2, interval, env.now, min_pr,
                                                   speed=speed, scoring_type = scoring_type,bundle_permutation_option= bundle_permutation_option,
                                                        unserved_bundle_order_break=unserved_bundle_order_break)
                end = time.time()
                print('고객 당 계산 시간 {}'.format(end - start))
                print('선택 번들1',selected_bundle)
                #input('T {} 계산 완료'.format(env.now))
                if selected_bundle != None:
                    B.append(selected_bundle)
            B = ParetoDominanceCount(B, 0, 8, 9, 10, strict_option = False)
            selected_customer_name_check = []
            #기존에 제시되어 있던 번들 중 새롭게 구성된 번들과 겹치는 부분이 있으면 삭제해야 함.
            print('T {} '.format(env.now))
            unique_bundles = []
            for bundle_info in B:
                duplicate = False
                for ct_name in bundle_info[4]:
                    if ct_name in selected_customer_name_check:
                        duplicate = True
                        break
                if duplicate == True:
                    continue
                else:
                    unique_bundles.append(bundle_info[:7])
                    selected_customer_name_check += bundle_info[4]
            for rider_name in riders:
                rider = riders[rider_name]
                duplicate_customers = list(set(rider.onhand).intersection(set(selected_customer_name_check)))
                if len(duplicate_customers) > 0:
                    input('T {} / 겸침 발생 :: 라이더 {} :: {}, {}'.format(int(env.now), rider_name, rider.onhand,selected_customer_name_check))
            print('선택 번들2{}'.format(unique_bundles))
            #B = unique_bundles
            for index in platform_set.platform:
                bundle_names += platform_set.platform[index].customers
            order_indexs = []
            for index in platform_set.platform:
                order_indexs.append(index)
            order_index = 1
            if len(order_indexs) > 0:
                order_index = max(order_indexs) + 1
            bundle_names = []
            order_subset_names = []
            for index in platform_set.platform:
                bundle_names += platform_set.platform[index].customers
                order_subset_names.append(platform_set.platform[index].customers)
            #input('제안되고 있는 고객 들 {} '.format(order_subset_names))
            #todo : 제안된 번들 중에서 특정 번들을 선택하는 과정
            new_orders = PlatformOrderRevise(unique_bundles, orders, order_index,platform_set, divide_option = divide_option, now_t= round(env.now,2), platform_exp_error = platform_exp_error)
            #new_orders = PlatformOrderRevise2(unique_bundles, orders, order_index,platform_set, divide_option = divide_option, now_t= round(env.now,2), platform_exp_error = platform_exp_error, unserved_bundle_order_break=unserved_bundle_order_break)
            #new_orders = PlatformOrderRevise3(unique_bundles, orders, order_index,platform_set, divide_option = divide_option, now_t= round(env.now,2), platform_exp_error = platform_exp_error, unserved_bundle_order_break=unserved_bundle_order_break)
            bundle_names = []
            bundle_check = []
            for index in new_orders:
                bundle_names += new_orders[index].customers
                bundle_check.append([new_orders[index].customers])
            still_names = []
            for index in platform_set.platform:
                still_names += platform_set.platform[index].customers
            print('고객 이름들 3 :: 기존 {} 추가 {} 추가 2 {}'.format(list(sorted(still_names)),list(sorted(bundle_names)),bundle_check))
            print('전체함수 플랫폼1 ID{}'.format(id(platform_set)))
            print('원래 index {} :: 추가 index {}'.format(platform_set.platform.keys(),new_orders.keys()))
            platform_set.platform = new_orders
            count = [[],[]]
            for index in platform_set.platform:
                if platform_set.platform[index].type == 'single':
                    count[0].append(platform_set.platform[index].customers)
                else:
                    count[1].append(platform_set.platform[index].customers)
            print('고객 이름들 4 :: 단건 주문 {} 번들 주문 {}'.format(count[0], count[1]))
            print('T {} 계산 완료'.format(env.now))
        else: #Break the offered bundle
            print('ELSE 문 실행')
            org_bundle_num, rev_bundle_num = RequiredBreakBundleNum(platform_set, lamda2, mu1, mu2, thres=thres_p)
            if sum(rev_bundle_num) < sum(org_bundle_num):
                break_info = [org_bundle_num[0] - rev_bundle_num[0],org_bundle_num[1] - rev_bundle_num[1]] #[B2 해체 수, B3 해체 수]
                #번들의 해체가 필요
                platform_set.platform = BreakBundle(break_info, platform_set, orders)
                #input('확인 3 {}'.format(platform_set.platform))
        past_customer_num = copy.deepcopy(len(orders))
        print('T: {} B2,B3확인'.format(int(env.now)))
        #input('T: {} B2,B3확인'.format(int(env.now)))
        yield env.timeout(interval)

def LamdaMuCalculate(orders, riders, now_t, interval = 5, return_type = 'class'):
    unpicked_orders, lamda2 = CountUnpickedOrders(orders, now_t, interval=interval, return_type=return_type)  # lamda1
    lamda1 = len(unpicked_orders)
    idle_riders, mu2 = CountIdleRiders(riders, now_t, interval=interval, return_type=return_type)
    mu1 = len(idle_riders)
    return lamda1, lamda2, mu1, mu2

def NewCustomer(cusotmers, now_t, interval = 5):
    new_customer_names = []
    for customer_name in cusotmers:
        customer = cusotmers[customer_name]
        if now_t - interval <= customer.time_info[0] and customer.time_info[1] == None:
            new_customer_names.append(customer.name)
    return new_customer_names

def Platform_process4(env, platform_set, orders, riders, stores, p2,thres_p,interval, bundle_permutation_option = False, speed = 1, end_t = 1000, min_pr = 0.05, divide_option = False,\
                      unserved_bundle_order_break = True,  scoring_type = 'myopic',bundle_selection_type = 'greedy', considered_customer_type = 'new'):
    yield env.timeout(10)
    while env.now <= end_t:
        now_t = env.now
        print('T {} 과정 시작'.format(int(now_t)))
        past_select_t  = []
        for rider_name in riders:
            rider = riders[rider_name]
            if now_t - interval <= rider.last_pick_time < now_t:
                past_select_t.append([rider_name, rider.last_pick_time])
        past_select_t.sort( key = operator.itemgetter(1))
        #sorted_dict = sorted(weight2.items(), key = lambda item: item[1])
        #print('과거 예상 라이더 선택 순서{}'.format(sorted_dict))
        input('C!@ T {} // 과거 라이더 선택 순서 {}'.format(now_t - interval, past_select_t))
        lamda1, lamda2, mu1, mu2 = LamdaMuCalculate(orders, riders, now_t, interval=interval, return_type='class')
        p = CalculateRho(lamda1, lamda2, mu1, mu2)
        if p >= thres_p:
            B = []
            if considered_customer_type == 'new':
                considered_customers_names = NewCustomer(orders, now_t, interval = interval)
            else:
                considered_customers_names, interval_orders = CountUnpickedOrders(orders, now_t, interval = interval,  return_type='name')
            print('탐색 대상 고객들 {}'.format(considered_customers_names))
            active_rider_names = CountActiveRider(riders, interval, min_pr=min_pr, t_now=now_t, option = 'w')
            input('돌아오는 시기에 주문 선택 예쌍 라이더 {}'.format(active_rider_names))
            #weight2 = WeightCalculator(riders, active_rider_names)
            weight2 = WeightCalculator2(riders, active_rider_names, now_t, interval= interval)
            w_list = list(weight2.values())
            sorted_dict = sorted(weight2.items(), key=lambda item: item[1])
            print('C!@ T {} // 과거 예상 라이더 선택 순서{}'.format(now_t, sorted_dict))
            try:
                input('T {} / 대상 라이더 수 {}/시나리오 수 {} 중 {} / w평균 {} /w표준편차 {}'.format(now_t, len(active_rider_names),math.factorial(len(active_rider_names)),len(weight2), np.average(w_list),np.std(w_list)))
            except:
                input('T {} 출력 에러'.format(now_t))
            for customer_name in considered_customers_names:
                start = time.time()
                target_order = orders[customer_name]
                considered_customers = BundleConsideredCustomers(target_order, platform_set, riders, orders,
                                                                 bundle_search_variant=unserved_bundle_order_break,
                                                                 d_thres_option=True, speed=speed)
                selected_bundle = SelectByTwo_sided_way2(target_order, riders, considered_customers, stores, platform_set, p2, interval, env.now, min_pr, speed=speed, \
                                                         scoring_type = scoring_type,bundle_permutation_option= bundle_permutation_option,\
                                                         unserved_bundle_order_break=unserved_bundle_order_break, input_weight= weight2)
                end = time.time()
                print('고객 당 계산 시간 {} : 선택 번들1 {}'.format(end - start, selected_bundle))
                # selected_bundle 구조 : [(1151, 1103, 103, 151), 16.36, 10.69, 5.03, [103, 151], 16.36, 23.1417(s), 23.1417(s), 1000000(e), 1000000(d), 0]
                if selected_bundle != None:
                    B.append(selected_bundle)
            #Part2 기존에 제시되어 있던 번들 중 새롭게 구성된 번들과 겹치는 부분이 있으면 삭제해야 함.
            if unserved_bundle_order_break == False:
                pass
                #기존 번들을 추가하는 부분.
            else:
                #active_rider_names = CountActiveRider(riders, interval, min_pr=min_pr, t_now=now_t)
                #weight2 = WeightCalculator(riders, active_rider_names)
                for order_index in platform_set.platform:
                    order = platform_set.platform[order_index]
                    if order.type == 'bundle':
                        print('확인 {}'.format(order.old_info))
                        order.old_info += [order.old_info[6]]
                        e,d = Two_sidedScore(order.old_info, riders, orders, stores, platform_set , interval, now_t, min_pr, M=1000,
                                       sample_size=1000, platform_exp_error=1, weight = weight2)
                        order.old_info += [e,d,0]
                        B.append(order.old_info)
            unique_bundles = [] #P의 역할
            if len(B) > 0:
                if scoring_type == 'myopic':
                    print('정렬 정보{}'.format(B))
                    B.sort(key = operator.itemgetter(6)) #todo : search index 확인
                else:
                    B = ParetoDominanceCount(B, 0, 8, 9, 10, strict_option = False)
                #Part 2 -1 Greedy한 방식으로 선택
                selected_customer_name_check = [] #P의 확인용
                #unique_bundles = [] #P의 역할
                if bundle_selection_type == 'greedy':
                    for bundle_info in B:
                        duplicate = False
                        for ct_name in bundle_info[4]:
                            if ct_name in selected_customer_name_check:
                                duplicate = True
                                break
                        if duplicate == True:
                            continue
                        else:
                            unique_bundles.append(bundle_info[:7])
                            selected_customer_name_check += bundle_info[4]
                else: # set cover problem 풀이
                    #input('문제 풀이 시작:: B {}'.format(B))
                    #feasiblity, unique_bundles = Bundle_selection_problem(B)
                    feasiblity, unique_bundles = Bundle_selection_problem2(B)
                    print('결과확인 {} : {}'.format(feasiblity, unique_bundles))
            #part 3 Upload P
            #todo : PlatformOrderRevise 를 손볼 것.
            new_orders = PlatformOrderRevise4(unique_bundles, orders, platform_set, now_t = now_t, unserved_bundle_order_break = unserved_bundle_order_break, divide_option = divide_option)
            platform_set.platform = new_orders
        else: #Break the offered bundle
            print('ELSE 문 실행')
            org_bundle_num, rev_bundle_num = RequiredBreakBundleNum(platform_set, lamda2, mu1, mu2, thres=thres_p)
            if sum(rev_bundle_num) < sum(org_bundle_num):
                break_info = [org_bundle_num[0] - rev_bundle_num[0],org_bundle_num[1] - rev_bundle_num[1]] #[B2 해체 수, B3 해체 수]
                #번들의 해체가 필요
                platform_set.platform = BreakBundle(break_info, platform_set, orders)
                #input('확인 3 {}'.format(platform_set.platform))
        print('T: {} B2,B3확인'.format(int(env.now)))
        #input('T: {} B2,B3확인'.format(int(env.now)))
        yield env.timeout(interval)

def TaskSelect(rider, platform, customers, p2 = 0, score_type ='simple', sort_standard = 7, uncertainty = False, current_loc = None, add ='X'):
    """
    라이더에게 가장 적합한 번들을 탐색 후 제안
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
        exp_onhand_order = order.customers + rider.onhand
        #print('주문 고객 확인 {}/ 자신의 경로 길이 {} / 상태 {}/ ID {}'.format(order.customers, len(rider.route), order.picked, id(order)))
        if order.picked == False:
            #if ((len(exp_onhand_order) <= rider.capacity and len(rider.picked_orders) <= rider.max_order_num)) or (len(order.route) > 2 and len(rider.onhand) < 3):
            if Basic.ActiveRiderCalculator(rider) == True:
            #if len(rider.picked_orders) <= rider.max_order_num:
                if type(order.route[0]) != list:
                    input('에러 확인 {} : {}'.format(rider.last_departure_loc,order.route))
                if current_loc != None:
                    dist = Basic.distance(current_loc, order.route[0][2]) / rider.speed
                else:
                    dist = Basic.distance(rider.last_departure_loc, order.route[0][2])/rider.speed #자신의 현재 위치와 order의 시작점(가게) 사이의 거리.
                info = [order.index,dist]
                bound_order_names.append(info)
            #elif len(order.route) > 2: #번들이라는 소리
            #    dist = Basic.distance(rider.last_departure_loc, order.route[0][2]) / rider.speed
            #    info = [order.index, dist]
            #    bound_order_names.append(info)
        bound_order_names.sort(key = operator.itemgetter(1))
    if len(bound_order_names) > 0:
        for info in bound_order_names[:rider.bound]: #todo : route의 시작 위치와 자신의 위치사이의 거리가 가까운 bound개의 주문 중 선택.
            order = platform.platform[info[0]]
            if score_type == 'oracle':
                route_info = rider.ShortestRoute(order, customers, p2=p2, uncertainty = uncertainty)
                #route_info = [rev_route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time]
                print('선택 주문 정보 {}'.format(route_info))
                if len(route_info) > 0:
                    benefit = order.fee / route_info[5]  # 이익 / 운행 시간
                    score.append([order.index] + route_info + [benefit])
            elif score_type == 'simple':
                mv_time = 0
                times = []
                rev_route = [rider.last_departure_loc]
                for route_info in order.route:
                    rev_route.append(route_info[2])
                for node_index in range(1,len(rev_route)):
                    mv_time += Basic.distance(rev_route[node_index - 1],rev_route[node_index])/rider.speed
                for customer_name in order.customers:
                    mv_time += customers[customer_name].time_info[6] #예상 가게 준비시간
                    mv_time += customers[customer_name].time_info[7] #예상 고객 준비시간
                    times.append(rider.env.now - customers[customer_name].time_info[0])
                    #rider.income += customers[customer_name].fee
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



def Platform_process5(env, platform_set, orders, riders, stores, p2,thres_p,interval, bundle_permutation_option = False, speed = 1, end_t = 1000, min_pr = 0.05, divide_option = False,\
                      unserved_bundle_order_break = True,  scoring_type = 'myopic',bundle_selection_type = 'greedy', considered_customer_type = 'new'):
    yield env.timeout(5) #warm-up time
    while env.now <= end_t:
        now_t = env.now
        print('T {} 과정 시작'.format(int(now_t)))
        past_select_t  = []
        for rider_name in riders:
            rider = riders[rider_name]
            if now_t - interval <= rider.last_pick_time < now_t:
                past_select_t.append([rider_name, rider.last_pick_time])
        past_select_t.sort( key = operator.itemgetter(1))
        print('C!@ T {} // 과거 라이더 선택 순서 {}'.format(now_t - interval, past_select_t))
        lamda1, lamda2, mu1, mu2 = LamdaMuCalculate(orders, riders, now_t, interval=interval, return_type='class')
        p = CalculateRho(lamda1, lamda2, mu1, mu2)
        platform_set.p = p
        for customer in orders:
            if customer.customer.time_info[1]  == None and now_t - customer.time_info[0] >= 30:
                customer.priority_weight = customer.priority_weight*1.2
        if p >= thres_p:
            #번들이 필요한 라이더에게 번들 계산.
            B = []
            if considered_customer_type == 'new':
                considered_customers_names = NewCustomer(orders, now_t, interval = interval)
            else:
                considered_customers_names, interval_orders = CountUnpickedOrders(orders, now_t, interval = interval,  return_type='name')
            print('탐색 대상 고객들 {}'.format(considered_customers_names))
            active_rider_names = CountActiveRider(riders, interval, min_pr=min_pr, t_now=now_t, option = 'w')
            input('돌아오는 시기에 주문 선택 예쌍 라이더 {}'.format(active_rider_names))
            #weight2 = WeightCalculator(riders, active_rider_names)
            weight2 = WeightCalculator2(riders, active_rider_names, now_t, interval= interval)
            w_list = list(weight2.values())
            sorted_dict = sorted(weight2.items(), key=lambda item: item[1])
            print('C!@ T {} // 과거 예상 라이더 선택 순서{}'.format(now_t, sorted_dict))
            try:
                input('T {} / 대상 라이더 수 {}/시나리오 수 {} 중 {} / w평균 {} /w표준편차 {}'.format(now_t, len(active_rider_names),math.factorial(len(active_rider_names)),len(weight2), np.average(w_list),np.std(w_list)))
            except:
                input('T {} 출력 에러'.format(now_t))
            for customer_name in considered_customers_names:
                start = time.time()
                target_order = orders[customer_name]
                considered_customers = BundleConsideredCustomers(target_order, platform_set, riders, orders,
                                                                 bundle_search_variant=unserved_bundle_order_break,
                                                                 d_thres_option=True, speed=speed)
                selected_bundle = SelectByTwo_sided_way2(target_order, riders, considered_customers, stores, platform_set, p2, interval, env.now, min_pr, speed=speed, \
                                                         scoring_type = scoring_type,bundle_permutation_option= bundle_permutation_option,\
                                                         unserved_bundle_order_break=unserved_bundle_order_break, input_weight= weight2)
                end = time.time()
                print('고객 당 계산 시간 {} : 선택 번들1 {}'.format(end - start, selected_bundle))
                # selected_bundle 구조 : [(1151, 1103, 103, 151), 16.36, 10.69, 5.03, [103, 151], 16.36, 23.1417(s), 23.1417(s), 1000000(e), 1000000(d), 0]
                if selected_bundle != None:
                    B.append(selected_bundle)
            #Part2 기존에 제시되어 있던 번들 중 새롭게 구성된 번들과 겹치는 부분이 있으면 삭제해야 함.
            if unserved_bundle_order_break == False:
                pass
                #기존 번들을 추가하는 부분.
            else:
                #active_rider_names = CountActiveRider(riders, interval, min_pr=min_pr, t_now=now_t)
                #weight2 = WeightCalculator(riders, active_rider_names)
                for order_index in platform_set.platform:
                    order = platform_set.platform[order_index]
                    if order.type == 'bundle':
                        print('확인 {}'.format(order.old_info))
                        order.old_info += [order.old_info[6]]
                        e,d = Two_sidedScore(order.old_info, riders, orders, stores, platform_set , interval, now_t, min_pr, M=1000,
                                       sample_size=1000, platform_exp_error=1, weight = weight2)
                        order.old_info += [e,d,0]
                        B.append(order.old_info)
            unique_bundles = [] #P의 역할
            if len(B) > 0:
                if scoring_type == 'myopic':
                    print('정렬 정보{}'.format(B))
                    B.sort(key = operator.itemgetter(6)) #todo : search index 확인
                else:
                    B = ParetoDominanceCount(B, 0, 8, 9, 10, strict_option = False)
                #Part 2 -1 Greedy한 방식으로 선택
                selected_customer_name_check = [] #P의 확인용
                #unique_bundles = [] #P의 역할
                if bundle_selection_type == 'greedy':
                    for bundle_info in B:
                        duplicate = False
                        for ct_name in bundle_info[4]:
                            if ct_name in selected_customer_name_check:
                                duplicate = True
                                break
                        if duplicate == True:
                            continue
                        else:
                            unique_bundles.append(bundle_info[:7])
                            selected_customer_name_check += bundle_info[4]
                else: # set cover problem 풀이
                    #input('문제 풀이 시작:: B {}'.format(B))
                    #feasiblity, unique_bundles = Bundle_selection_problem(B)
                    feasiblity, unique_bundles = Bundle_selection_problem2(B)
                    print('결과확인 {} : {}'.format(feasiblity, unique_bundles))
            #part 3 Upload P
            #todo : PlatformOrderRevise 를 손볼 것.
            new_orders = PlatformOrderRevise4(unique_bundles, orders, platform_set, now_t = now_t, unserved_bundle_order_break = unserved_bundle_order_break, divide_option = divide_option)
            platform_set.platform = new_orders
        else: #Break the offered bundle
            print('ELSE 문 실행')
            org_bundle_num, rev_bundle_num = RequiredBreakBundleNum(platform_set, lamda2, mu1, mu2, thres=thres_p)
            if sum(rev_bundle_num) < sum(org_bundle_num):
                break_info = [org_bundle_num[0] - rev_bundle_num[0],org_bundle_num[1] - rev_bundle_num[1]] #[B2 해체 수, B3 해체 수]
                #번들의 해체가 필요
                platform_set.platform = BreakBundle(break_info, platform_set, orders)
                #input('확인 3 {}'.format(platform_set.platform))
        print('T: {} B2,B3확인'.format(int(env.now)))
        #input('T: {} B2,B3확인'.format(int(env.now)))
        yield env.timeout(interval)