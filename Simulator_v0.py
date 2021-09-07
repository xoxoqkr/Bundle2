# -*- coding: utf-8 -*-

import simpy
import random

from astropy import uncertainty

from A1_BasicFunc import Ordergenerator, RiderGenerator, ResultSave
from A1_Class import Store, Platform_pool
from A2_Func import Platform_process, ResultPrint
import operator
from Bundle_Run_ver0 import Platform_process3
import matplotlib.pyplot as plt


#Parameter define
order_interval = 1.1
interval = 5
run_time = 150
cool_time = 30 #run_time - cool_time 시점까지만 고객 생성
uncertainty_para = True #음식 주문 불확실성 고려
rider_exp_error = 1.5 #라이더가 가지는 불확실성
platform_exp_error = 1.2 #플랫폼이 가지는 불확실성
cook_time_type = 'uncertainty'
cooking_time = [7,1] #[평균, 분산]
thres_p = 1

rider_working_time = 120
#env = simpy.Environment()
store_num = 20
rider_num = 5
rider_gen_interval = 2 #라이더 생성 간격.
rider_speed = 3
rider_capacity = 1
ITE_NUM = 1
option_para = True #True : 가게와 고객을 따로 -> 시간 단축 가능 :: False : 가게와 고객을 같이 -> 시간 증가
customer_max_range = 50
store_max_range = 30
divide_option = True # True : 구성된 번들에 속한 고객들을 다시 개별 고객으로 나눔. False: 번들로 구성된 고객들은 번들로만 구성
p2_set = True
p2 = 3 #p2_set이 False인 경우에는 p2만큼의 시간이 p2로 고정됨. #p2_set이 True인 경우에는 p2*dis(가게,고객)/speed 만큼이 p2시간으로 설정됨.
#order_p2 = [[1.5,2,3],[0.3,0.3,0.4]] #음식 별로 민감도가 차이남.
order_p2 = 3
wait_para = False #True: 음식조리로 인한 대기시간 발생 #False : 음식 대기로 인한 대기시간 발생X



class scenario(object):
    def __init__(self, name, p1, p2, scoring_type, search_option):
        self.name = name
        self.platform_work = p1
        self.unserved_order_break = p2
        self.res = []
        self.scoring_type = scoring_type
        self.bundle_search_option = search_option

scenarios = []

f = open("결과저장0706.txt", 'a')
f.write('결과저장 시작' + '\n')
f.close()

#infos = [['A',False, False],['B',True, True],['C',True, False]]
#infos = [['A',False, False],['B',True, True],['C',True, False]]
#infos = [['A',False, False],['B',True, True],['C',True, False]]
#infos = [['B',True, True]]
#infos = [['B',True, True, 'myopic'],['B',True, True, 'two_sided'],['C',True, False, 'myopic'],['C',True, False, 'two_sided']]
#infos = [['B',True, True, 'myopic'],['B',True, True, 'two_sided']]
#infos = [['B',True, True, 'myopic'],['B',True, True, 'two_sided']]
#infos = [['B',True, True, 'two_sided', True],['B',True, True, 'two_sided', False]]
#infos = [['B',True, True, 'myopic', True],['B',True, True, 'two_sided', True]]
#infos = [['B',True, True, 'myopic', True],['B',True, True, 'two_sided', True],['C',True, False, 'myopic', True],['C',True, False, 'two_sided', True]]
infos = [['C',True, False, 'myopic', True],['C',True, False, 'two_sided', True]]
infos = [['C',True, False, 'myopic', True]]
for info in infos:
    sc = scenario(info[0], info[1], info[2], info[3], info[4])
    scenarios.append(sc)

for ite in range(ITE_NUM):
    #instance generate
    #가게 생성
    index = 0
    rider_history = None
    order_history = None
    store_history = []
    for _ in range(store_num):
        store_history.append(list(random.sample(range(20 , store_max_range), 2)))
    for sc in scenarios:
        if sc.name == 'A':
            score_type = 'oracle'
            rider_capacity = 3
        else:
            score_type = 'simple'
            rider_capacity = 1
        Rider_dict = {}
        Orders = {}
        Platform2 = Platform_pool()
        Store_dict = {}
        #run
        env = simpy.Environment()
        for store_name in range(store_num):
            loc = store_history[store_name]
            store = Store(env, Platform2, store_name, loc=loc, order_ready_time= 0.1, capacity=10, print_para=False)
            Store_dict[store_name] = store
        env.process(RiderGenerator(env, Rider_dict, Platform2, Store_dict, Orders, speed=rider_speed,
                                   interval=rider_gen_interval, runtime=run_time, gen_num=rider_num,
                                   capacity=rider_capacity, history= rider_history, score_type= score_type, wait_para= wait_para, uncertainty = uncertainty_para, exp_error = rider_exp_error))
        env.process(Ordergenerator(env, Orders, Store_dict, max_range= customer_max_range, interval=order_interval, history = order_history,runtime= run_time - cool_time, p2 = order_p2, p2_set= p2_set, speed= rider_speed, cooking_time = cooking_time, cook_time_type= cook_time_type))
        if sc.platform_work == True:
            """
            env.process(Platform_process(env, Platform2, Orders, Rider_dict, p2, thres_p, interval, speed=rider_speed,
                                         end_t=1000, unserved_order_break=sc.unserved_order_break, option = option_para, divide_option = divide_option, uncertainty = uncertainty_para, platform_exp_error = platform_exp_error))
            
            """
            env.process(Platform_process3(env, Platform2, Orders, Rider_dict, Store_dict,p2, thres_p, interval, speed=rider_speed,bundle_search_option = option_para,
                                         end_t=1000, unserved_order_break=sc.unserved_order_break, divide_option = divide_option, platform_exp_error = platform_exp_error, scoring_type = sc.scoring_type))

        env.run(run_time)
        res = ResultPrint(sc.name + str(ite), Orders, speed=rider_speed)
        sc.res.append(res)
        if index == 0:
        #필요한 정보 저장
            rider_history = []
            rider_gen_times = []
            order_history = []
            for rider_name in Rider_dict:
                rider = Rider_dict[rider_name]
                rider_gen_times.append(rider.start_time)
                #rider_history.append(rider.start_time)
            for index in range(1,len(rider_gen_times)):
                rider_history.append(rider_gen_times[index] - rider_gen_times[index - 1])
            rider_history.append(1000)
            #rider_history.sort()
            for order_name in Orders:
                order = Orders[order_name]
                order_history.append([order.time_info[0], order.store, order.location])
            order_history.append([1000, 1, [25,25]])
            order_history.sort(key = operator.itemgetter(0))
        #input('주문 이력 수{}/ 생성 주문 수 {} '.format(len(order_history),len(Orders)))
        index += 1
        c_x = []
        c_y = []
        s_x = []
        s_y = []
        lead_times = []
        for customer_name in Orders:
            customer = Orders[customer_name]
            if len(customer.who_serve) > 1:
                print('문제 고객 {} 서비스 수 {}'.format(customer.name, customer.who_serve))
            c_x.append(customer.location[0])
            c_y.append(customer.location[1])
            s_x.append(customer.store_loc[0])
            s_y.append(customer.store_loc[1])
            if customer.time_info[3] != None:
                lead_times.append(customer.time_info[3] - customer.time_info[0])
        """
        plt.scatter(c_x, c_y, color='g', marker="*", label = 'customer')
        plt.scatter(s_x, s_y, color='b', marker="X", label = 'store')
        plt.xlabel(' x', labelpad=10)
        plt.ylabel(' y', labelpad=10)
        plt.xlim([0,50])  # X축의 범위: [xmin, xmax]
        plt.ylim([0,50])  # Y축의 범위: [ymin, ymax]
        plt.title('Sactter plot of customer and store')
        plt.legend()
        plt.show()
        plt.close()
        """
        res = []
        wait_time = 0
        candis = []
        b_select = 0
        store_wait_time = 0
        bundle_store_wait_time = []
        single_store_wait_time = []
        served_num = 0
        for rider_name in Rider_dict:
            rider = Rider_dict[rider_name]
            res += rider.served
            wait_time += rider.idle_time
            candis += rider.candidates
            b_select += rider.b_select
            store_wait_time += rider.store_wait
            bundle_store_wait_time += rider.bundle_store_wait
            single_store_wait_time += rider.single_store_wait
            served_num += len(rider.served)
            print('라이더 {} 경로 :: {}'.format(rider.name, rider.visited_route))
        wait_time_per_customer = bundle_store_wait_time + single_store_wait_time
        try:
            wait_time_per_customer= round(sum(wait_time_per_customer)/len(wait_time_per_customer),2)
        except:
            wait_time_per_customer = None
        if len(bundle_store_wait_time) > 0:
            bundle_store_wait_time = round(sum(bundle_store_wait_time)/len(bundle_store_wait_time),2)
        else:
            bundle_store_wait_time = None
        if len(single_store_wait_time) > 0:
            single_store_wait_time = round(sum(single_store_wait_time)/len(single_store_wait_time),2)
        else:
            single_store_wait_time = None
        ave_wait_time = round(wait_time/len(Rider_dict),2)
        #print('candis 수 {}'.format(candis))
        print('고객 서비스 율 {} 전체 {} 중 {} 서비스 됨/ 평균 리드타임 {}'.format(round(served_num/len(Orders),2),len(Orders), served_num, round(sum(lead_times)/len(lead_times),2)))
        print('라이더 수 {} 평균 수행 주문 수 {} 평균 유휴 분 {} 평균 후보 수 {} 평균 선택 번들 수 {} 가게 대기 시간 {} 번들가게대기시간 {} 단건가게대기시간 {} 고객 평균 대기 시간 {}'.format(len(Rider_dict), round(len(res)/len(Rider_dict),2),round(wait_time/len(Rider_dict),2),round(sum(candis)/len(candis),2), b_select/len(Rider_dict), round(store_wait_time/len(Rider_dict),2),bundle_store_wait_time,single_store_wait_time,wait_time_per_customer))
        res_info = sc.res[-1]
        info = str(sc.name) + ';' + str(ite) + ';' + str(res_info[0]) + ';' + str(res_info[1]) + ';' + str(res_info[2]) + ';' +str(res_info[3]) + ';' + str(res_info[4]) + ';' + str(round(res_info[5],4)) + ';' + str(ave_wait_time) +';' +str(b_select) +'\n'
        #'시나리오로 {} ITE {} /전체 고객 {} 중 서비스 고객 {}/ 서비스율 {}/ 평균 LT :{}/ 평균 FLT : {}/직선거리 대비 증가분 : {}'
        f = open("결과저장0706.txt", 'a')
        f.write(info)
        f.close()
        #input('파일 확인')
        sub_info = 'divide_option : {}, p2: {}, divide_option: {}, unserved_order_break : {}'.format(divide_option, p2,sc.platform_work, sc.unserved_order_break)
        ResultSave(Rider_dict, Orders, title='Test', sub_info= sub_info, type_name= sc.name)
        #input('저장 확인')

for sc in scenarios:
    count = 1
    for res_info in sc.res:
        try:
            print('시나리오 {} 타입 {} ITE {} /전체 고객 {} 중 서비스 고객 {}/ 서비스율 {}/ 평균 LT :{}/ 평균 FLT : {}/직선거리 대비 증가분 : {}'.format(sc.name ,sc.scoring_type, count,res_info[0],res_info[1],res_info[2],res_info[3],res_info[4],res_info[5]))
        except:
            print('시나리오 {} ITE {} 결과 없음'.format(sc.name , count))
        count += 1