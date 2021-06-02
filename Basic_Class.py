# -*- coding: utf-8 -*-
import operator
import numpy as np
import random
import math
import simpy
import copy

class Platform(object):
    def __init__(self, env):
        self.resource = simpy.Store(env)






class Store(object):
    def __init__(self, env, name, order_ready_time = 15, capacity = 6):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.order_ready_time = order_ready_time
        self.resource = simpy.Resource(env, capacity = capacity)
        self.recieved_orders = []


    def StoreRunner(self, env, capacity, open_time = 1, end_time = 900):
        yield env.timeout(open_time)
        now_time = round(env.now, 1)
        while now_time < end_time:
            now_time = round(env.now,1)
            #받은 주문을 플랫폼에 올리기
            if len(self.resource.put_queue) < capacity:
                #주문을 플랫폼에 올리기

                pass

            yield env.timeout(1)

    def OrderRecive(self, env, customer):
        #주문을 받아서, 주문을 플랫폼에 올린 후


        #라이더가 주문을 선택시 음식 조리
        self.cook(env, customer)



    def Cook(self, env, customer, cooking_time_type = 'fixed'):
        with self.veh.request() as req:
            now_time = round(env.now , 1)
            req.info = [customer.name, now_time]
            if cooking_time_type == 'fixed':
                cooking_time = self.order_ready_time
            elif cooking_time_type == 'random':
                cooking_time = random.randrange(1,self.order_ready_time)
            else:
                pass
            yield env.timeout(cooking_time)
            customer.food_ready = True
            #음식을 준비하고
            #라이더에게 전달






class Customer(object):
    def __init__(self, env, name, input_location, store = 0, end_time = 60, ready_time=3, service_time=3, fee = 1000,wait = True, far = 0, end_type = 4):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.time_info = [round(env.now, 2), None, None, None, None, end_time, ready_time, service_time]
        # [0 :발생시간, 1: 차량에 할당 시간, 2:차량에 실린 시간, 3:목적지 도착 시간,
        # 4:고객이 받은 시간, 5: 보장 배송 시간, 6:가게에서 준비시간,7: 고객에게 서비스 하는 시간]
        self.location = input_location
        self.store = store
        self.assigned = False
        self.loaded = False
        self.done = False
        self.cancelled = False
        self.server_info = None
        self.fee = [fee, 0, None, None] # [기본 요금, 지급된 보조금, 할당된 라이더]
        self.wait = wait
        self.far = far
        self.error = 0
        self.type = random.randrange(1,end_type)
        env.process(self.Decline(env, end_time))

    def Decline(self, env, end_time):
        """
        고객이 생성된 후 endtime 동안 서비스를 받지 못하면 고객을 취소 시킴
        취소된 고객은 더이상 서비스 받을 수 없음.
        :param env:
        """
        yield env.timeout(end_time)
        if self.assigned == False and self.done == False:
            self.cancelled = True
            self.server_info= ["N", round(env.now,2)]
        else:
            print("고객은 종료 시점이나, 차량에 삽입되어 있기 때문에 진행.")
            pass

class Rider(object):
    def __init__(self, env, name, speed, customer_set, wageForHr = 9000, wait = True, toCenter = True, run_time = 900, error = 0, ExpectedCustomerPreference = [1,1,1,1], pref_info = 'None', save_info = False, left_time = 120):
        self.env = env
        self.name = name
        self.veh = simpy.Resource(env, capacity=1)
        self.last_location = [36, 36]  # 송파 집중국
        self.served = []
        self.speed = speed
        self.wageForHr = wageForHr
        self.idle_times = [[],[]]
        self.gen_time = int(env.now)
        self.left_time = None
        self.wait = wait
        self.end_time = 0
        self.left = False
        self.earn_fee = []
        self.fee_analyze = []
        self.subsidy_analyze = []
        self.choice = []
        self.choice_info = []
        self.now_ct = 0
        for slot_num in range(int(math.ceil(run_time / 60))):
            self.fee_analyze.append([])
            self.subsidy_analyze.append([])
        self.exp_last_location = [36,36]
        self.error = int(error)
        pref = list(range(1, len(ExpectedCustomerPreference) + 1))
        random.shuffle(pref)
        self.CustomerPreference = pref
        self.expect = ExpectedCustomerPreference
        cost_coeff = round(random.uniform(0.8,1.2),1)
        type_coeff = round(1000*random.uniform(0.8,1.2),1)
        self.coeff = [cost_coeff,type_coeff,1]
        self.p_coeff = [1,1000,1]
        env.process(self.Runner(env, customer_set, toCenter = toCenter, pref = pref_info, save_info = save_info))
        env.process(self.RiderLeft(left_time))


    def RiderLeft(self, left_time):
        """
        운행 시작 후 left_time이 지나면, 기사는 더 이상 주문 수행X.
        :param env:
        """
        yield self.env.timeout(left_time)
        self.left = True
        self.left_time = int(self.env.now)


    def CustomerSelector(self,customer_set, now_time, toCenter = True, pref = 'None', save_info = False):
        """
        고객 중 가장 높은 가치를 가지는 고객을 선택하는 함수.
        :param customer_set: 고객 set
        :return: 가장 높은 가치를 가지는 고객 이름
        """
        ava_cts = UnloadedCustomer(customer_set, now_time)
        ava_cts_class = []
        #print('test1', ava_cts)
        ava_cts_names = []
        if len(ava_cts) > 0:
            if type(ava_cts[0]) == int:
                for ct_name in ava_cts:
                    ava_cts_class.append(customer_set[ct_name])
                ava_cts_names = ava_cts
            else:
                ava_cts_class = ava_cts
                for info in ava_cts:
                    ava_cts_names.append(info.name)
        if len(ava_cts_class) > 0:
            #print('test2',ava_cts_class)
            priority_orders = PriorityOrdering(self, ava_cts_class, now_time = self.env.now, toCenter = toCenter, who = pref, save_info = save_info)
            #print('rider', self.name,'//Now',round(self.env.now,2),'//un_ct',len(ava_cts),'//candidates', priority_orders[:min(3,len(priority_orders))],'//ava_cts:',ava_cts_names)
            #input('Stop')
            for ct_info in priority_orders:
                ct = customer_set[ct_info[0]]
                print(self.name, 'selects', ct.name, 'at', self.env.now)
                return ct.name, priority_orders
        return None, None


    def Runner(self, env, customer_set, wait_time=1, toCenter = True, pref = 'None', save_info = False):
        """
        라이더가 고객을 선택하면, 고객을 서비스 하도록 수행하는 과정을 표현
        :param env:
        :param customer_set:
        :param end_time:
        :param wait_time:
        """
        while self.left == False:
            #print('rider test', self.name, env.now, len(customer_set),self.veh.put_queue)
            if len(self.veh.put_queue) == 0 and self.wait == False:
                #print('Rider', self.name, 'assign1 at', env.now)
                ct_name, infos = self.CustomerSelector(customer_set, env.now, toCenter = toCenter, pref = pref, save_info = save_info)
                if infos != None: #infos == None인 경우에는 고를 고객이 없다는 의미임.
                    """
                    rev_infos = []
                    for info in infos:
                        rev_infos.append([info[0],info[2]])
                    """
                    if pref == 'test_rider' or pref == 'test_platform':
                        #self.choice_info.append([int(env.now), ct_name, self.last_location , rev_infos])
                        self.choice_info.append([int(env.now), ct_name, self.last_location, infos])
                #print('Rider',self.name,'assign2',ct_name, 'at', env.now)
                if infos == None:
                    pass
                else:
                    print('Now',int(env.now),'Rider ::',self.name ,' /select::', infos)
                select_time = round(env.now,2)
                if type(ct_name) == int and ct_name > 0:
                    self.now_ct = ct_name
                    self.choice.append([ct_name, int(env.now)])
                    ct = customer_set[ct_name]
                    self.earn_fee.append(ct.fee[1])
                    ct.assigned = True
                    ct.time_info[1] = round(env.now, 2)
                    end_time = env.now + (distance(self.last_location, ct.location[0]) / self.speed) + ct.time_info[6]
                    end_time += ((distance(ct.location[0], ct.location[1]) / self.speed) + ct.time_info[7])
                    if int(env.now // 60) >= len(self.fee_analyze):
                        print(env.now, self.fee_analyze)
                    self.fee_analyze[int(env.now // 60)].append(ct.fee[0])
                    self.subsidy_analyze[int(env.now // 60)].append(ct.fee[1])
                    self.end_time = end_time
                    self.exp_last_location = ct.location[1]
                    #print('Rider', self.name, 'select', ct_name, 'at', env.now, 'EXP T', self.end_time)
                    #print('1:', self.last_location, '2:', ct.location)
                    with self.veh.request() as req:
                        #print(self.name, 'select', ct.name, 'Time:', env.now)
                        req.info = [ct.name, round(env.now,2)]
                        yield req  # users에 들어간 이후에 작동
                        time = distance(self.last_location, ct.location[0]) / self.speed
                        #print('With in 1:',self.last_location, '2:', ct.location[0])
                        time += ct.time_info[6]
                        end_time += time
                        ct.loaded = True
                        #ct.time_info[2] = round(env.now, 2)
                        yield env.timeout(time)
                        ct.time_info[2] = round(env.now, 2)
                        time = distance(ct.location[0], ct.location[1]) / self.speed
                        time += ct.time_info[7]
                        end_time += time
                        self.served.append([ct.name, 0])
                        #print('3:', ct.location[1])
                        yield env.timeout(time)
                        ct.time_info[3] = round(env.now, 2) - ct.time_info[7]
                        ct.time_info[4] = round(env.now,2)
                        ct.done = True
                        ct.server_info = [self.name, round(env.now,2)]
                        self.served.append([ct.name,1])
                        self.last_location = ct.location[1]
                        #임금 분석
                        print('Rider', self.name, 'done', ct_name, 'at', int(env.now))
                else:
                    self.end_time = env.now + wait_time
                    self.idle_times[0].append(wait_time)  #수행할 주문이 없는 경우
                    yield self.env.timeout(wait_time)
            else:
                self.end_time = env.now + wait_time
                self.idle_times[1].append(wait_time) #이미 수행하는 주문이 있는 경우
                yield self.env.timeout(wait_time)


def distance(x1, x2):
    return round(math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2), 2)


def UnloadedCustomer(customer_set, now_time):
    """
    아직 라이더에게 할당되지 않은 고객들을 반환
    :param customer_set:
    :param now_time: 현재시간
    :return: [고객 class, ...,]
    """
    res = []
    for ct_name in customer_set:
        customer = customer_set[ct_name]
        cond1 = now_time - customer.time_info[0] < customer.time_info[5]
        cond2 = customer.assigned == False and customer.loaded == False and customer.done == False
        cond3 = customer.wait == False
        #print('CT check',cond1, cond2, cond3, customer.name, customer.time_info[0])
        #input('STOP')
        if cond1 == True and cond2 == True and cond3 == True and ct_name > 0 and customer.server_info == None:
            res.append(customer)
    return res

def PriorityOrdering(veh, ava_customers, now_time = 0, minus_para = False, toCenter = True, who = 'driver', save_info = False, last_location = None):
    """
    veh의 입장에서 ava_customers를 가치가 높은 순서대로 정렬한 값을 반환
    :param veh: class veh
    :param ava_customer_names: 삽입 가능한 고객들의 class의 list
    :return: [[고객 이름, 이윤],...] -> 이윤에 대한 내림차순으로 정렬됨.
    """
    res = []
    add_info = []
    for customer in ava_customers:
        tem = []
        #print('test',customer)
        #time = CalTime(veh.last_location, veh.speed, customer)
        if last_location == None:
            time = CalTime2(veh.last_location, veh.speed, customer, center = [36,36], toCenter = toCenter, customer_set = ava_customers)
        else:
            time = CalTime2(last_location, veh.speed, customer, center = [36,36], toCenter = toCenter, customer_set = ava_customers)
        cost = (time/60)*veh.wageForHr
        org_cost = copy.deepcopy(cost)
        fee = customer.fee[0]
        paid = 0
        t2 = time - distance(customer.location[1],[36,36])/veh.speed
        if customer.fee[2] == veh.name or customer.fee[2] == 'all':
            fee += customer.fee[1]
            paid += customer.fee[1]
        time_para = now_time + t2 < customer.time_info[0] + customer.time_info[5]
        print("입렵 값 확인",customer.name, cost)
        #('R#',veh.name,'//CT#' ,customer.name,'//Fee$',customer.fee[0],paid, int(cost),'//Earn$',int(customer.fee[0] + paid - cost), '//ExpT',now_time + t2,'//EndT',customer.time_info[0] + customer.time_info[5], 'time_para', time_para )
        #print('check2',fee, cost, time_para)
        cost2 = 0
        if who == 'platform':
            cost2 = veh.error
        elif who == 'test_rider':
            cost2 = customer.type * veh.coeff[1]
            cost = cost*veh.coeff[0]
        elif who == 'test_platform':
            #cost2 = veh.expect[customer.type]*1000
            cost2 = customer.type * veh.p_coeff[1]
            cost = cost*veh.p_coeff[0]
        else:
            pass
        #print('cost2', cost2,'::', fee - cost- cost2)
        if minus_para == True:
            """
            if who == 'platform':
                res.append([customer.name, fee + veh.error - cost])
            else:
                res.append([customer.name, fee - cost])
            """
            res.append([customer.name, int(fee - cost - cost2), int(org_cost), int(fee)])

        elif time_para == True:
            """
            if who == 'platform':
                res.append([customer.name, fee + veh.error - cost])
            else:
                res.append([customer.name, fee - cost])
            """
            res.append([customer.name, int(fee - cost- cost2), int(org_cost), int(fee)])
        elif fee > cost + cost2 :
            res.append([customer.name, int(fee - cost - cost2), int(org_cost), int(fee)])
        else:
            print('negative value',int(fee - cost- cost2))
            pass
    if len(res) > 0:
        res.sort(key=operator.itemgetter(1), reverse = True)
        #print(res)
    return res


def CalTime2(veh_location,veh_speed, customer, center=[25,25], toCenter = True, customer_set = []):
    """
    cost(1) : customer를 서비스하는 비용
    cost(2) : 종료 후 다시 중심으로 돌아오는데 걸리는 시간.
    :param veh_location: 차량의 시작 위치
    :param veh_speed: 차량 속도
    :param customer: 고객
    :param center: 중심지의 위치(가게들이 밀접한 지역)
    :return: 필요한 시간
    """
    #print('Cal Time2',veh_location, customer.location, center)
    time = distance(veh_location, customer.location[0]) / veh_speed
    time += distance(customer.location[0], customer.location[1]) / veh_speed
    time += (customer.time_info[6] + customer.time_info[7])
    if toCenter == True:
        time += distance(customer.location[1], center)/veh_speed
    else:
        dist = []
        for ct in customer_set:
            dist.append([ct.name, distance(customer.location[1], ct.location[0])])
        if len(dist) > 0:
            dist.sort(key=operator.itemgetter(1))
            time += dist[0][1]/veh_speed
            #aveage = []
            #for info in dist:
            #    aveage.append(info[1])
            #time += sum(aveage)/len(aveage)
    return time