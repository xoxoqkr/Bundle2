# -*- coding: utf-8 -*-
import csv
import time
import simpy
from A1_Class import Platform_pool
from re_A1_class import scenario
from A1_BasicFunc import ResultSave, GenerateStoreByCSV, RiderGeneratorByCSV, OrdergeneratorByCSV
from A2_Func import ResultPrint
from re_platform import Platform_process5

# Parameter define
order_interval = 1
interval = 5
run_time = 200
cool_time = 30  # run_time - cool_time 시점까지만 고객 생성
uncertainty_para = True  # 음식 주문 불확실성 고려
rider_exp_error = 1.5  # 라이더가 가지는 불확실성
platform_exp_error = 1.2  # 플랫폼이 가지는 불확실성
cook_time_type = 'uncertainty'
cooking_time = [7, 1]  # [평균, 분산]
thres_p = 0

rider_working_time = 120
# env = simpy.Environment()
store_num = 20
rider_num = 8
rider_gen_interval = 2  # 라이더 생성 간격.
rider_speed = 3
rider_capacity = 2
start_ite = 0
ITE_NUM = 1
option_para = True  # True : 가게와 고객을 따로 -> 시간 단축 가능 :: False : 가게와 고객을 같이 -> 시간 증가
customer_max_range = 50
store_max_range = 30
divide_option = True  # True : 구성된 번들에 속한 고객들을 다시 개별 고객으로 나눔. False: 번들로 구성된 고객들은 번들로만 구성
p2_set = True
p2 = 3  # p2_set이 False인 경우에는 p2만큼의 시간이 p2로 고정됨. #p2_set이 True인 경우에는 p2*dis(가게,고객)/speed 만큼이 p2시간으로 설정됨.
# order_p2 = [[1.5,2,3],[0.3,0.3,0.4]] #음식 별로 민감도가 차이남.
order_p2 = 3
wait_para = False  # True: 음식조리로 인한 대기시간 발생 #False : 음식 대기로 인한 대기시간 발생X
scenarios = []
run_para = True  # True : 시뮬레이션 작동 #False 데이터 저장용

f = open("결과저장0706.txt", 'a')
f.write('결과저장 시작' + '\n')
f.close()


infos = [['C',True, False, 'myopic', True]]
#v1 = ['new', 'all']
#v1 = ['all']
v1 = ['new']
#v2 = [True, False]
v2 = [False]
v3 = ['myopic', 'two_sided']
#v4 = ['setcover','greedy']
v4 = ['greedy']
#platform_recommend = True  #True ; False
#rider_bundle_construct = True
order_select_type = 'simple' #oracle ; simple
info = ['C', False, False, 'myopic', True]

sc_index = 0
for i in [True, False]:
    for j in [True, False]:
        sc = scenario('{}:P:{}/R:{}'.format(str(sc_index),i,j))
        sc.platform_recommend = i
        sc.rider_bundle_construct = j
        scenarios.append(sc)
        sc_index += 1


#scenarios = [scenarios[2]]

#input('확인 {}'.format(len(scenarios)))
for ite in range(0, 5):
    # instance generate
    for sc in scenarios:
        print('시나리오 정보 {} : {} : {} : {}'.format(sc.platform_recommend,sc.rider_bundle_construct,sc.scoring_type,sc.bundle_selection_type))
        sc.store_dir = 'Instance_random_store/Instancestore_infos'+str(ite) #Instance_random_store/Instancestore_infos
        sc.customer_dir = 'Instance_random_store/Instancecustomer_infos'+str(ite) #Instance_random_store/Instancecustomer_infos
        sc.rider_dir = 'Instance_random_store/Instancerider_infos0' #+str(ite) #Instance_random_store/Instancerider_infos
        Rider_dict = {}
        Orders = {}
        Platform2 = Platform_pool()
        Store_dict = {}
        # run
        env = simpy.Environment()
        GenerateStoreByCSV(env, sc.store_dir, Platform2, Store_dict)
        env.process(RiderGeneratorByCSV(env, sc.rider_dir,  Rider_dict, Platform2, Store_dict, Orders, input_speed = rider_speed, input_capacity= rider_capacity,
                                        platform_recommend = sc.platform_recommend, input_order_select_type = order_select_type, bundle_construct= sc.rider_bundle_construct,
                                        rider_num = rider_num))
        env.process(OrdergeneratorByCSV(env, sc.customer_dir, Orders, Store_dict, Platform2))
        env.process(Platform_process5(env, Platform2, Orders, Rider_dict, p2,thres_p,interval, bundle_para= sc.platform_recommend))
        env.run(run_time)
        res = ResultPrint(sc.name + str(ite), Orders, speed=rider_speed, riders = Rider_dict)
        sc.res.append(res)
        #저장 부
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
            #candis += rider.candidates
            b_select += rider.b_select
            store_wait_time += rider.store_wait
            bundle_store_wait_time += rider.bundle_store_wait
            single_store_wait_time += rider.single_store_wait
            served_num += len(rider.served)
            print('라이더 {} 경로 :: {}'.format(rider.name, rider.visited_route))
        wait_time_per_customer = bundle_store_wait_time + single_store_wait_time
        try:
            wait_time_per_customer = round(sum(wait_time_per_customer) / len(wait_time_per_customer), 2)
        except:
            wait_time_per_customer = None
        if len(bundle_store_wait_time) > 0:
            bundle_store_wait_time = round(sum(bundle_store_wait_time) / len(bundle_store_wait_time), 2)
        else:
            bundle_store_wait_time = None
        if len(single_store_wait_time) > 0:
            single_store_wait_time = round(sum(single_store_wait_time) / len(single_store_wait_time), 2)
        else:
            single_store_wait_time = None
        ave_wait_time = round(wait_time / len(Rider_dict), 2)
        try:
            print(
                '라이더 수 ;{} ;평균 수행 주문 수 ;{} ;평균 유휴 분 ;{} ;평균 후보 수 {} 평균 선택 번들 수 {} 가게 대기 시간 {} 번들가게대기시간 {} 단건가게대기시간 {} 고객 평균 대기 시간 {}'.format(
                    len(Rider_dict), round(len(res) / len(Rider_dict), 2), round(wait_time / len(Rider_dict), 2),
                    round(sum(candis) / len(candis), 2), b_select / len(Rider_dict),
                    round(store_wait_time / len(Rider_dict), 2), bundle_store_wait_time, single_store_wait_time,
                    wait_time_per_customer))
        except:
            print('에러 발생으로 프린트 제거')
        res_info = sc.res[-1]
        info = str(sc.name) + ';' + str(ite) + ';' + str(res_info[0]) + ';' + str(res_info[1]) + ';' + str(
            res_info[2]) + ';' + str(res_info[3]) + ';' + str(res_info[4]) + ';' + str(
            round(res_info[5], 4)) + ';' + str(ave_wait_time) + ';' + str(b_select) + '\n'
        f = open("결과저장0706.txt", 'a')
        f.write(info)
        f.close()
        # input('파일 확인')
        sub_info = 'divide_option : {}, p2: {}, divide_option: {}, unserved_order_break : {}'.format(divide_option, p2,
                                                                                                     sc.platform_work,
                                                                                                     sc.unserved_order_break)
        ResultSave(Rider_dict, Orders, title='Test', sub_info=sub_info, type_name=sc.name)
        # input('저장 확인')
        # 시나리오 저장
        #SaveInstanceAsCSV(Rider_dict, Orders, Store_dict, instance_name='res')
        #결과 저장 부
        tm = time.localtime()
        string = time.strftime('%Y-%m-%d %I:%M:%S %p', tm)
        info = [string, ite, sc.name, sc.considered_customer_type, sc.unserved_order_break, sc.scoring_type, sc.bundle_selection_type, 0, \
        sc.res[-1][0],sc.res[-1][1], sc.res[-1][2], sc.res[-1][3], sc.res[-1][4], sc.res[-1][5], sc.res[-1][6], sc.res[-1][7], sc.res[-1][8]]
        #[len(customers), len(TLT),served_ratio,av_TLT,av_FLT, av_MFLT, round(sum(MFLT)/len(MFLT),2), rider_income_var,customer_lead_time_var]
        f = open("InstanceRES.csv", 'a', newline='')
        wr = csv.writer(f)
        wr.writerow(info)
        f.close()
for sc in scenarios:
    count = 1
    for res_info in sc.res:
        try:
            print(
                'SC:{}/플랫폼번들{}/라이더번들{}/ITE ;{}; /전체 고객 ;{}; 중 서비스 고객 ;{};/ 서비스율 ;{};/ 평균 LT ;{};/ 평균 FLT ;{};/직선거리 대비 증가분 ;{};원래 O-D길이;{};라이더 수익 분산;{};LT분산;{};'.format(
                    sc.name, sc.platform_recommend,sc.rider_bundle_construct, count, res_info[0],
                    res_info[1], res_info[2], res_info[3], res_info[4], res_info[5], res_info[6], res_info[7], res_info[8]))
        except:
            print('시나리오 {} ITE {} 결과 없음'.format(sc.name, count))
        count += 1
