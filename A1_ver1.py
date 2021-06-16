# -*- coding: utf-8 -*-

import simpy
import operator
import itertools
import Basic_Class2 as Basic
import random
import copy

class Order(object):
    def __init__(self, order_name, customer_names, route, type = '1'):
        self.name = order_name
        self.customers = customer_names
        self.route = route
        self.type = type #1:단주문, 2:B2, 3:B3


class Rider(object):
    def __init__(self, env, i, platform, orders, speed = 1, capacity = 3, end_t = 120):
        self.name = i
        self.env = env
        self.visited_route = []
        self.speed = speed
        self.route = []
        self.run_process = None
        self.capacity = capacity
        self.onhand = []
        self.end_t = env.now + end_t
        self.last_departure_loc = [25,25]
        self.container = []
        self.served = []
        env.process(self.OrderSelect(env, platform, orders))


    def RunProcess(self, env, platform, orders):
        while int(env.now) < self.end_t:
            if len(self.route) > 0:
                node_info = self.route[0]
                move_t = Basic.distance(self.last_departure_loc, node_info[2]) / self.speed
                yield env.timeout(move_t)
                if node_info[1] == 0: #가게인 경우
                    self.container.append(node_info[0])
                else:#고객인 경우
                    self.container.remove(node_info[0])
                    self.onhand.remove(node_info[0])
                    self.served.append(node_info[0])
                if len(self.onhand) < self.capacity:
                    order
                    if order != None:
                        #todo:update route
            else:
                order = self.OrderSelect(env, platform, orders)
                if order != None:
                    route = [order.store_loc, order.location]
                    self.route.append(route)
                    print('라이더 {} -> 경로 {} 할당 '.format(self.name, route))
                else:
                    yield env.timeout(5)
                    print('라이더 {} -> 주문탐색 {}~{}'.format(self.name, int(env.now), int(env.now) + 5))

    def OrderSelect(self, env, platform, customers, route = []):
        if len(route) > 0:
            #현재의 경로를 반영한 비용
            new_route, customer = self.ShortestRoute(platform, customers, route)
            return [new_route, customer]
        else: #주문 중 최고점 주문을 선택
            score = []
            for order in platform:
                dist = Basic.distance(self.last_departure_loc, order.store_loc)
                score.append([order.name, dist])
            if len(score) > 0:
                score.sort(key = operator.itemgetter(1))
                return score[0][0]
            else:
                return None
            #단순비용



    def ShortestRoute(self, orders, now_t = 0, p2 = 0, speed = 1, M = 1000):
        prior_route_infos = []
        prior_route = []
        add_time = {}
        for food in self.container:
            for info in self.visited_route:
                if food == info[0] and info[1] == 0 and info[0] not in self.served:
                    prior_route_infos.append(info)
                    prior_route.append(info[0] + M)
                    add_time[info[0]] = now_t - info[3]
                    break
        print('입력 주문들 {}'.format(orders))
        print('현재 배달 중 {}/ 이전 주문 {}/ 이전 경로{}'.format(self.container, prior_route,self.visited_route))
        input('확인8')
        order_names = []  # 가게 이름?
        for order_name in orders:
            order_names.append(order_name)
        store_names = []
        for name in order_names:
            rev_name = name + M
            if rev_name not in prior_route:
                add_time[name] = 0
                store_names.append(rev_name)
        candi = order_names + store_names
        print('고려 노드들',candi)
        subset = itertools.permutations(candi, len(candi))
        feasible_subset = []
        #print(prior_route, subset, prior_route + list(subset))
        #input('확인7')
        for route_part in subset:
            #route = prior_route + list(route_part)
            route = list(route_part)
            print(route)
            #input('확인8')
            # print('고객이름',order_names,'가게이름',store_names,'경로',route)
            sequence_feasiblity = True  # 모든 가게가 고객 보다 앞에 오는 경우.
            feasible_routes = []
            for order_name in order_names:  # order_name + M : store name ;
                if order_name + M in route:
                    if route.index(order_name + M) < route.index(order_name):
                        pass
                    else:
                        sequence_feasiblity = False
                        break
            if sequence_feasiblity == True:
                """
                ftd_feasiblity, ftds = Basic.FLT_Calculate(orders, route, p2, M=M, speed=speed, add_time= add_time)
                if ftd_feasiblity == True:
                    # print('ftds',ftds)
                    # input('멈춤5')
                    route_time = Basic.RouteTime(orders, route, speed=speed, M=M)
                    feasible_routes.append(
                        [route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time])                
                """
                print(orders, route)
                input('확인10')
                route_time = Basic.RouteTime(orders, route, speed=speed, M=M)
                rev_route = []
                for node in route:
                    if node < M:
                        name = node
                        info = [name, 1, orders[name].location,0]
                    else:
                        name = node - M
                        info = [name, 0, orders[name].store_loc, 0]
                    rev_route.append(info)
                feasible_routes.append([rev_route, None, None, None, order_names, route_time])
            if len(feasible_routes) > 0:
                feasible_routes.sort(key=operator.itemgetter(5)) #가장 짧은 거리의 경로 선택.
                feasible_subset.append(feasible_routes[0])
        if len(feasible_subset) > 0:
            feasible_subset.sort(key=operator.itemgetter(5))
            print('선택 된 정보 {} / 경로 길이 {}'.format(feasible_subset[0][0], feasible_subset[0][5]))
            input('확인9')
            return feasible_subset[0]
        else:
            return []