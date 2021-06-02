# -*- coding: utf-8 -*-
import operator
import random
import math
import simpy
import copy

class Customer(object):
    def __init__(self, env, name, input_location, store = 0, end_time = 60, ready_time=3, service_time=3, fee = 1000):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.time_info = [round(env.now, 2), None, None, None, None, end_time, ready_time, service_time]
        # [0 :발생시간, 1: 차량에 할당 시간, 2:차량에 실린 시간, 3:목적지 도착 시간,
        # 4:고객이 받은 시간, 5: 보장 배송 시간, 6:가게에서 준비시간,7: 고객에게 서비스 하는 시간]
        self.location = input_location
        self.store = store
        self.fee = fee
        self.ready_time = None #가게에서 음식이 조리 완료된 시점



class rider(object):
    def __init__(self, env, name, platform, stores, capacity = 3, end_time = 1000):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.resource = simpy.Resource(env, capacity = capacity)
        self.done_orders = []
        self.capacity = capacity
        self.end = False
        self.end_time = env.now + end_time
        env.process(self.Runner(env, platform,stores))


    def RiderMoving(self, env, time):
        """
        라이더가 움직이는 시간의 env.time의 generator를 반환
        :param env: simpy.env
        :param time: 라이더가 움직이는 시간
        """
        yield env.timeout(time)
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
            if len(platform) > 0 and len(self.resource.put_queue) < self.capacity:
                order = RiderChoiceCustomer(self, platform)  #todo: 라이더가 platform의 주문 중 가장 이윤이 높은 것을 선택.
                #order = platform[0] #큐의 가장 앞에 주문을 잡음.
                print('현재T:',int(env.now),'/라이더',self.name,"/주문",order.name,'선택')
                store_name = order.store
                platform.remove(order)
                stores[store_name].wait_orders.remove(order)
                move_time = random.randrange(5,12)
                #move = env.timeout(move_time)
                #food_ready = env.timeout(stores[store_name].order_ready_time)
                #env.process(stores[store_name].Cook(env, order))
                exp_arrival_time = env.now + move_time
                print('현재T:', int(env.now), '/라이더', self.name, "/주문", order.name,'/가게까지 이동시간', move_time, '조리시간', stores[store_name].order_ready_time)
                #if len(stores[store_name].resource.users) + len(stores[store_name].wait_orders) > stores[store_name].capacity:
                #    print('음식점이 조리 불가','라이더',self.name,"주문 ", order.name,'/현재시간',int(env.now))
                yield env.process(stores[store_name].Cook(env, order)) & env.process(self.RiderMoving(env, move_time)) #둘 중 더 오래 걸리는 process가 완료 될 때까지 기다림
                rider_wait_time = max(0, env.now - exp_arrival_time)
                food_wait_time = max(0, env.now - order.ready_time)
                #print(env.now - exp_arrival_time, env.now - order.ready_time)
                print('현재T:', int(env.now), '/라이더', self.name, "/주문", order.name, '픽업 완료/라이더 대기시간:',rider_wait_time,'/음식대기시간',food_wait_time)
                with self.resource.request() as req:
                    stores[store_name].ready_order.remove(order)
                    req.info = [order.name, round(env.now, 2)]
                    yield req  # users에 들어간 이후에 작동
                    move_time = random.randrange(1,5)
                    yield env.timeout(move_time)
                    #stores[store_name].ready_order.remove(order)
                    print('현재T:', int(env.now),'/라이더',self.name,"/주문", order.name,'/이동시간',move_time)
            else: #현재 플랫폼에 주문이 없다면, 아래 시간 만큼 대기 후 다시 탐색 수행
                yield env.timeout(1)


class Store(object):
    """
    Store can received the order.
    Store has capacity. The order exceed the capacity must be wait.
    """
    def __init__(self, env, platform, name, order_ready_time = 7, capacity = 6, slack = 2):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.order_ready_time = order_ready_time
        self.resource = simpy.Resource(env, capacity = capacity)
        self.slack = slack #자신의 조리 중이 queue가 꽉 차더라도, 추가로 주문을 넣을 수 있는 주문의 수
        self.received_orders = []
        self.wait_orders = []
        self.ready_order = []
        self.loaded_order = []
        self.capacity = capacity
        env.process(self.StoreRunner(env, platform, capacity = capacity))


    def StoreRunner(self, env, platform, capacity, open_time = 1, close_time = 900):
        """
        Store order cooking process
        :param env: simpy Env
        :param platform: Platform
        :param capacity: store`s capacity
        :param open_time: store open time
        :param close_time: store close time
        """
        yield env.timeout(open_time)
        now_time = round(env.now, 1)
        while now_time < close_time:
            now_time = round(env.now,1)
            #받은 주문을 플랫폼에 올리기
            if len(self.resource.users) + len(self.wait_orders) < capacity + self.slack: #플랫폼에 자신이 생각하는 여유 만큼을 게시
                slack = min(capacity + self.slack - len(self.resource.users), len(self.received_orders))
                if len(self.received_orders) > 0:
                    for count in range(slack):
                        order = self.received_orders[0] #앞에서 부터 플랫폼에 주문 올리기
                        platform.append(order)
                        print('현재T:', int(env.now), '/가게', self.name, '/주문', order.name, '플랫폼에 접수/조리대 여유:',
                              capacity - len(self.resource.users),'/조리 중',len(self.resource.users))
                        self.wait_orders.append(order)
                        self.received_orders.remove(order)
            else: #이미 가게의 능력 최대로 조리 중. 잠시 주문을 막는다(block)
                #print("가게", self.name, '/',"여유 X", len(self.resource.users),'/주문대기중',len(self.received_orders))
                pass
            #만약 현재 조리 큐가 꽉차는 경우에는 주문을 더이상 처리하지 X
            yield env.timeout(0.5)
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
        value = CustomerValueForRiderCalculator(rider, customer)
        customer_values.append([value, customer])
    customer_values.sort(key = operator.itemgetter(0), reverse = True)
    return customer_values[0][1]


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
        input_location = [36,36]
        store_num = random.randrange(0, len(stores))
        order = Customer(env, name, input_location, store = store_num)
        orders[name] = order
        stores[store_num].received_orders.append(orders[name])
        #print('주문확인', orders[name], type(orders[name]))
        platform.append(order)
        #print('T:', int(env.now), '/큐', len(queue))
        yield env.timeout(interval)
        name += 1

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