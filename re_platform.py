# -*- coding: utf-8 -*-
import time
import math
from A2_Func import CountUnpickedOrders, CalculateRho, RequiredBreakBundleNum, BreakBundle, PlatformOrderRevise4, GenBundleOrder
from A3_two_sided import BundleConsideredCustomers, CountActiveRider,  ConstructFeasibleBundle_TwoSided
import operator
from Bundle_selection_problem import Bundle_selection_problem3, Bundle_selection_problem4
from Bundle_Run_ver0 import LamdaMuCalculate, NewCustomer
import numpy
import matplotlib.pyplot as plt

def distance(p1, p2):
    """
    Calculate 4 digit rounded euclidean distance between p1 and p2
    :param p1:
    :param p2:
    :return: 4 digit rounded euclidean distance between p1 and p2
    """
    euc_dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return round(euc_dist,4)


def Platform_process5(env, platform, orders, riders, p2,thres_p,interval, end_t = 1000,
                      divide_option = False,unserved_bundle_order_break = True, bundle_para = False,
                      delete_para = True, obj_type = 'simple_max_s'):
    yield env.timeout(5) #warm-up time
    while env.now <= end_t:
        if bundle_para == True:
            lamda1, lamda2, mu1, mu2 = LamdaMuCalculate(orders, riders, env.now, interval=interval, return_type='class')
            p = CalculateRho(lamda1, lamda2, mu1, mu2)
            if p > thres_p:
                feasible_bundle_set, phi_b, d_matrix, s_b, D, lt_matrix = Bundle_Ready_Processs(env.now, platform, orders, riders, p2, interval, speed = riders[0].speed, bundle_permutation_option= True)
                print('phi_b {}:{} d_matrix {}:{} s_b {}:{}'.format(len(phi_b), numpy.average(phi_b),
                                                                    d_matrix.shape, numpy.average(d_matrix),len(s_b),numpy.average(s_b),))
                print('d_matrix : {}'.format(d_matrix))
                #문제 풀이
                #unique_bundle_indexs = Bundle_selection_problem3(phi_b, d_matrix, s_b, min_pr = 0.05)
                unique_bundle_indexs = Bundle_selection_problem4(phi_b, D, s_b, lt_matrix, min_pr = 0.05, obj_type= obj_type)
                #input('결과 확인')
                unique_bundles = []
                for index in unique_bundle_indexs:
                    unique_bundles.append(feasible_bundle_set[index])
                print('문제 풀이 결과 {} '.format(unique_bundles))
                # 번들을 업로드
                task_index = max(list(platform.platform.keys())) + 1
                if len(unique_bundles) > 0:
                    #플랫폼에 새로운 주문을 추가하는 작업이 필요.
                    print('주문 수 {} :: 추가 주문수 {}'.format(len(platform.platform),len(unique_bundles)))
                    for info in unique_bundles:
                        o = GenBundleOrder(task_index, info, orders, env.now)
                        o.old_info = info
                        platform.platform[task_index] = o
                        task_index += 1
                    """
                    new_orders = PlatformOrderRevise4(unique_bundles, orders, platform, now_t=env.now,
                                                      unserved_bundle_order_break=unserved_bundle_order_break,
                                                      divide_option=divide_option)                    
                    print(new_orders)
                    input('주문 수 {} -> {} : 추가 번들 수 {}'.format(len(platform.platform), len(new_orders), len(unique_bundles)))
                    platform.platform = new_orders                    
                    """
                    print('주문 수2 {}'.format(len(platform.platform)))
            else:
                print('번들 삭제 수행')
                org_bundle_num, rev_bundle_num = RequiredBreakBundleNum(platform, lamda2, mu1, mu2, thres=thres_p)
                Break_the_bundle(platform, orders, org_bundle_num, rev_bundle_num)
        yield env.timeout(interval)
        if bundle_para == True and delete_para == True:
            delete_task_names = []
            for task_name in platform.platform:
                if len(platform.platform[task_name].customers) > 1 and platform.platform[task_name].picked == False:
                    delete_task_names.append(task_name)
            for task_name in delete_task_names:
                del platform.platform[task_name]
        print('Simulation Time : {}'.format(env.now))


def Calculate_Phi(rider, customers, bundle_infos, l=4):
    #라이더와 번들들에 대해서, t시점에서 phi_b...r을 계산 할 것.
    dp_br = []
    dist_list = []
    displayed_values = []
    gamma = 1 - numpy.exp(-rider.search_lamda)
    for customer_name in customers:
        customer = customers[customer_name]
        dist = distance(rider.last_departure_loc,customer.location)
        try:
            displayed_value = customer.fee / ((dist + distance(customer.store_loc, customer.location))/rider.speed)
        except:
            input('dist {} dist2{} speed{} '.format(dist, distance(customer.store_loc, customer.location), rider.speed))
        dist_list.append([customer.name, dist, displayed_value])
    dist_list.sort(key = operator.itemgetter(1))
    displayed_dist = []
    for info in dist_list:
        displayed_values.append(info[2])
        displayed_dist.append(info[1])
    dist_list = dist_list[:(len(rider.p_j)+1)*l] #todo : 계산 확인
    for bundle_info in bundle_infos:
        #input(bundle_info)
        bundle_dist = distance(rider.last_departure_loc, customers[bundle_info[4][0]].store_loc)
        bundle_fee = 0
        for customer_name in bundle_info[4]:
            bundle_fee += customers[customer_name].fee
        bundle_value = bundle_fee / bundle_info[6]
        tem_dp_br = []
        for page_index in range(len(rider.p_j)):
            #print('dist_list {} page_index {} displayed_values {} bundle_info {}'.format(dist_list,page_index,displayed_values, bundle_info[6]))
            if bundle_value > max(displayed_values[:(page_index + 1)*l]) and  bundle_dist < max(displayed_dist[:(page_index + 1)*l]):
            #if len(dist_list) <= (page_index + 1)*l or (bundle_dist < dist_list[(page_index + 1)*l][1] and bundle_info[6] > max(displayed_values[:(page_index + 1)*l])):
                tem_dp_br.append(rider.p_j[page_index])
            else:
                tem_dp_br.append(0)
        dp_br.append(1-gamma*sum(tem_dp_br))
        #dp_br.append(gamma * (1 - sum(tem_dp_br)))
    return dp_br


def Bundle_Ready_Processs(now_t, platform_set, orders, riders, p2,interval, bundle_permutation_option = False, speed = 1, min_pr = 0.05,
                      unserved_bundle_order_break = True, considered_customer_type = 'new'):
    # 번들이 필요한 라이더에게 번들 계산.
    if considered_customer_type == 'new':
        considered_customers_names = NewCustomer(orders, now_t, interval=interval)
    else:
        considered_customers_names, interval_orders = CountUnpickedOrders(orders, now_t, interval=interval,return_type='name')
    print('탐색 대상 고객들 {}'.format(considered_customers_names))
    active_rider_names = CountActiveRider(riders, interval, min_pr=min_pr, t_now=now_t, option='w')
    print('돌아오는 시기에 주문 선택 예쌍 라이더 {}'.format(active_rider_names))
    #weight2 = WeightCalculator2(riders, active_rider_names, now_t, interval=interval)
    #sorted_dict = sorted(weight2.items(), key=lambda item: item[1])
    #print('C!@ T {} // 과거 예상 라이더 선택 순서{}'.format(now_t, sorted_dict))
    Feasible_bundle_set = []
    for customer_name in considered_customers_names:
        start = time.time()
        target_order = orders[customer_name]
        considered_customers = BundleConsideredCustomers(target_order, platform_set, riders, orders,
                                                         bundle_search_variant=unserved_bundle_order_break,
                                                         d_thres_option=True, speed=speed)
        print('번들 탐색 대상 고객들 {}'.format(len(considered_customers)))
        thres = 1
        size3bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, 3, p2, speed=speed, bundle_permutation_option = bundle_permutation_option, thres= thres)
        size2bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, 2, p2, speed=speed,bundle_permutation_option=bundle_permutation_option , thres= thres)
        max_index = 2
        tem_infos = []
        try:
            size3bundle.sort(key=operator.itemgetter(6))
            for info in size3bundle[:max_index]:
                tem_infos.append(info)
        except:
            pass
        try:
            size2bundle.sort(key=operator.itemgetter(6))
            for info in size2bundle[:max_index]:
                tem_infos.append(info)
        except:
            pass
        Feasible_bundle_set += tem_infos
        #Feasible_bundle_set += size3bundle + size2bundle
        end = time.time()
        print('고객 당 계산 시간 {} : B2::{} B3::{}'.format(end - start, len(size2bundle),len(size3bundle)))
        if len(size3bundle + size2bundle) > 1:
            print('번들 생성 가능')
    print('T {} 번들 수 {}'.format(now_t, len(Feasible_bundle_set)))
    #문제에 필요한 데이터 계산
    #1 phi 계산
    phi_br = []
    for rider_name in riders:
        rider = riders[rider_name]
        phi_r = Calculate_Phi(rider, orders, Feasible_bundle_set)
        phi_br.append(phi_r)
        #input('라이더 {} / 확인 phi_r {}'.format(rider.name, phi_r))
    phi_b = []
    for bundle_index in range(len(phi_br[0])):
        tem = 1
        for rider_index in range(len(phi_br)):
            value = phi_br[rider_index][bundle_index]
            if value > 0:
                tem = tem * value
        phi_b.append(tem)
    #input('phi_b {}'.format(phi_b))
    #2 d-matrix계산
    d_matrix = numpy.zeros((len(Feasible_bundle_set),len(Feasible_bundle_set)))
    for index1 in range(len(Feasible_bundle_set)):
        b1 = Feasible_bundle_set[index1][4]
        for index2 in range(len(Feasible_bundle_set)):
            b2 = Feasible_bundle_set[index2][4]
            if index1 > index2 and len(list(set(b1 + b2))) < len(b1) + len(b2) : #겹치는 것이 존재
                    d_matrix[index1, index2] = 1
                    d_matrix[index2, index1] = 1
            else:
                d_matrix[index1, index2] = 0
                d_matrix[index2, index1] = 0
    #3 s_b계산
    s_b = []
    for info in Feasible_bundle_set:
        s_b.append(info[6])
    #4 D 계산
    D = []
    #info1 : [route, round(max(ftds), 2), round(sum(ftds) / len(ftds), 2), round(min(ftds), 2), order_names, round(route_time, 2)]
    info1_index = 0
    info2_index = 0
    for info1 in Feasible_bundle_set:
        for info2 in Feasible_bundle_set:
            if info1_index > info2_index and len(info1[4]) + len(info2[4]) > len(list(set(info1[4] + info2[4]))):
                D.append([Feasible_bundle_set.index(info1), Feasible_bundle_set.index(info2)])
            info2_index += 1
        info1_index += 1
    lt_vector = []
    for info in Feasible_bundle_set:
        bundle_names = info[4]
        over_t = []
        for name in bundle_names:
            customer = orders[name]
            over_t.append(now_t - customer.time_info[0])
        lt_vector.append(sum(over_t)/len(over_t))
    return Feasible_bundle_set, phi_b, d_matrix, s_b, D, lt_vector


def Break_the_bundle(platform, orders, org_bundle_num, rev_bundle_num):
    if sum(rev_bundle_num) < sum(org_bundle_num):
        break_info = [org_bundle_num[0] - rev_bundle_num[0],
                      org_bundle_num[1] - rev_bundle_num[1]]  # [B2 해체 수, B3 해체 수]
        # 번들의 해체가 필요
        platform.platform = BreakBundle(break_info, platform, orders)


def Rider_Bundle_plt(rider):
    count = 0
    for bundle_info in rider.bundles_infos:
        save_name = 'R{};C{}'.format(rider.name, count)
        #nodes = []
        #for info in bundle_info:
        #    nodes.append(info[2])
        print('Title:{}save_name/Route:{}'.format(save_name,bundle_info))
        Bundle_plot(bundle_info, save_name)
        count += 1
        #input('확인2')
def Bundle_plot(bundle_points, save_name='nan'):
    ax = plt.axes()
    # arrow
    #ax.title(save_name)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.set_title(save_name)
    for index in range(1, len(bundle_points)):
        bf = bundle_points[index - 1][2]
        af = bundle_points[index][2]
        ax.arrow(bf[0], bf[1], af[0]-bf[0], af[1]-bf[1], head_width=0.5, head_length=0.5, fc='k', ec='k')
    for index in range(0, len(bundle_points)):
        node = bundle_points[index][2]
        if bundle_points[index][1] == 1:
            ax.plot(node[0], node[1], 'r-o', lw=1)
        else:
            ax.plot(node[0], node[1], 'b-*', lw=1)
    plt.savefig('test' + save_name + '.png', dpi=100)
    plt.close()

