# -*- coding: utf-8 -*-

import operator
import itertools
import random
import numpy
import simpy
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


class Order(object):
    def __init__(self, name, input_value):
        self.name = name
        self.selected = False
        self.data_vector = input_value

class Rider(object):
    def __init__(self, name, input_value):
        self.name = name
        self.coeff_vector = input_value

    def SelectOrder(self, orders):
        scores = []
        for order_name in orders:
            order = orders[order_name]
            if order.selected == False:
                score = (numpy.dot(self.coeff_vector,order.data_vector),2)
                scores.append([order_name, score])
        scores.sort(key=operator.itemgetter(1), reverse = True)
        if len(scores) > 0:
            orders[scores[0][0]].selected = True
            names = []
            for score in scores[1:]:
                names.append(score[0])
            return [scores[0][0] ,names]
        else:
            return [None, []]


class StepwiseSearch(object):
    def __init__(self, name, func, init_nets, r_k, beta = 0.4, T = 50, info = 'Basic'):
        self.name = name
        self.info = info
        self.func = func
        self.nets = init_nets
        self.r_k = r_k
        self.alpha = 0.1
        self.beta = beta
        self.ite = 0
        self.T = T
        self.min_nets = -1
        self.max_nets = 1
        self.ratio = []

    def NetUpdater(self):
        scores = []
        test = []
        for ele in self.nets:
            scores.append([ele, self.nets[ele]])
            test.append(self.nets[ele])
        #input('점수 분포 {}'.format(sorted(test)))
        scores.sort(key=operator.itemgetter(1), reverse = True)
        update_num = min(5,int(len(scores) * self.alpha) )
        upper_scores = scores[:update_num]
        self.r_k = self.r_k * self.beta
        neighbors = []
        for j in [-1,0,1]: #todo : 위 아래로 1칸 씩 변수의 수가 증가하면 이 과정이 증가되어야 함.
            for k in [-1,0,1]:
                neighbors.append(numpy.array([j,k])*self.r_k)
        #todo : 더 촘촘한 net 추가
        added_nets = []
        for info in upper_scores:
            pivot = numpy.array(info[0])
            for ele in neighbors:
                added_nets.append([pivot + ele, self.nets[info[0]]])
        count = 0
        for added_net_info in added_nets:
            #print('자료 확인 {}'.format(added_net_info))
            if self.min_nets < added_net_info[0][0] < self.max_nets and self.min_nets < added_net_info[0][1] < self.max_nets:
                self.nets[tuple(added_net_info[0])] = added_net_info[1]
                count += 1
        #input('전체 {} / 추가 {} 개'.format(len(scores), count))


    def Check(self, ele, data, orders):
        scores = []
        for order_name in [data[0]] + data[1]:
            order = orders[order_name]
            scores.append([order_name, numpy.dot(ele, order.data_vector)]) #todo: 가치함수의 형태가 달라지면, 달라져야 함.
        scores.sort(key=operator.itemgetter(1), reverse = True)
        return scores[0][0]


    def TimeDiscount(self, discount_ratio = 0.9):
        for net in self.nets:
            self.nets[net] = round(self.nets[net]*discount_ratio,4)


    def Updater(self, data, orders, rider = None):
        if self.ite > 1 :
            if self.ite % self.T == 0:
                #self.NetUpdater()#input('그물 추가')
                #GrahDraw(self, rider)
                pass
            elif self.ite % 10 == 0:
                #self.TimeDiscount()
                pass
        #점수 갱신
        res_1s = []
        res_0s = []
        for ele in self.nets:
            if self.Check(list(ele), data, orders) == data[0]:
                res_1s.append([ele, 1])
            else:
                res_0s.append([ele, 0])
        """
        for res_0 in res_0s:#0.5인 ele를 갱신하는 과정
            for res_1 in res_1s:
                dist = math.sqrt((res_0[0][0] - res_1[0][0])**2 + (res_0[0][1] - res_1[0][1])**2)
                if dist <= self.r_k:
                    res_0[1] += 0.5
                    break
        """
        for info in res_1s:
            self.nets[info[0]] += info[1]
        self.ratio.append(len(res_1s)/len(self.nets))

def ReviseCoeff_MJ(init_coeff, now_data, past_data, error = 10, print_para = False):
    coeff = list(range(len(init_coeff)))
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(coeff), vtype=GRB.CONTINUOUS, name="x")
    z = m.addVars(1 + len(past_data), vtype = GRB.CONTINUOUS, name= "z")
    a = m.addVars(len(coeff), vtype=GRB.CONTINUOUS, name="a")
    u = m.addVars(len(coeff), vtype=GRB.CONTINUOUS, name="a")

    #m.setObjective(gp.quicksum(x[i] for i in coeff), GRB.MINIMIZE)
    m.setObjective(gp.quicksum(u[i] for i in coeff), GRB.MINIMIZE)
    m.addConstrs(x[i] <= u[i] for i in coeff)
    m.addConstrs(-x[i] <= u[i] for i in coeff)


    #m.setObjective(gp.quicksum(a[i] for i in coeff), GRB.MINIMIZE)
    #m.addConstrs(a[i] == gp.abs_(x[i]) for i in coeff)
    z_count = 0
    #이번 selected와 other에 대한 문제 풀이
    if print_para == True:
        print('선택 고객 z {} '.format(numpy.dot(init_coeff, now_data[0])))
    m.addConstr(gp.quicksum((x[i] + init_coeff[i])*now_data[0][i] for i in coeff) == z[z_count])
    #m.addConstr(z[z_count] >= 0)
    for other_info in now_data[1:]:
        if print_para == True:
            print('현재 데이터 제약식 {} : {}'.format(init_coeff, other_info))
            print('현재 데이터 고객 z {} '.format(numpy.dot(init_coeff, other_info)))
        m.addConstr(gp.quicksum((x[i] + init_coeff[i])*other_info[i] for i in coeff) <= z[z_count] - error)
    z_count += 1
    #과거 정보를 적층하는 작업
    if len(past_data) > 0:
        for data in past_data:
            p_selected = data[0]
            p_others = data[1:]
            m.addConstr(gp.quicksum((x[i] + init_coeff[i]) * p_selected[i] for i in coeff) == z[z_count])
            for p_other_info in p_others:
                if print_para == True:
                    print('과거 {} 데이터 제약식 {} : {}'.format(z_count, init_coeff, p_other_info))
                    print('과거 {}  데이터 고객 z {} '.format(z_count, numpy.dot(init_coeff, p_other_info)))
                m.addConstr(gp.quicksum((x[i] + init_coeff[i]) * p_other_info[i] for i in coeff) <= z[z_count] - error)
            z_count += 1
    #풀이
    m.setParam(GRB.Param.OutputFlag, 0)
    m.optimize()
    try:
        print('Obj val: %g' % m.objVal)
        res = []
        for val in m.getVars():
            if val.VarName[0] == 'x':
                res.append(float(val.x))
        return True, res
    except:
        print('Infeasible')
        return False, None

class LP_search(object):
    def __init__(self, name, func, init, T = 50):
        self.name = name
        self.func = func
        self.init = init
        self.past_data = []
        self.true_coeff = None

    def LP_Solver(self, org_data, customers):
        data = [customers[org_data[0]].data_vector]
        for name in org_data[1]:
            data.append(customers[name].data_vector)
        #input('초기 값 {} 입력 데이터 {}'.format(self.init, data))
        feasiblity, res = ReviseCoeff_MJ(self.init, data, self.past_data, error = 0, print_para= False)
        if feasiblity == True:
            for index in range(len(res)):
                self.init[index] += res[index]
        else:
            #input('해 없음'.format())
            feasiblity2, res2 = ReviseCoeff_MJ(self.true_coeff, data, self.past_data, error=0)
            #input('진짜 해에 대한 결과 {} : {}'.format(feasiblity2, res2))
            print('진짜 해에 대한 결과 {} : {}'.format(feasiblity2, res2))
        self.past_data.append(data)
        #input('LP_Solver 확인'.format())


def GrahDraw(engine, rider):
    vectors = []
    test = []
    x = []
    y = []
    z = []
    for net in engine.nets:
        vectors.append([net, engine.nets[net]])
        test.append(engine.nets[net])
        x.append(net[0])
        y.append(net[1])
        z.append(engine.nets[net])
    rev_z = []
    max_val = max(z)
    for info in z:
        rev_z.append(info / max_val)
    vectors.sort(key=operator.itemgetter(1), reverse=True)
    rev_z = numpy.array(rev_z)
    alphas = numpy.array(rev_z)
    # 색깔 농도 관련 -> https://stackoverflow.com/questions/24767355/individual-alpha-values-in-scatter-plot
    rgba_colors = numpy.zeros((len(x), 4))
    # for red the first column needs to be one
    rgba_colors[:, 0] = 1.0
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alphas
    plt.scatter(x, y, color=rgba_colors, s=5)
    plt.scatter(rider.coeff_vector[0], rider.coeff_vector[1], color='b', marker="X", s=20)
    plt.xlabel(' c1', labelpad=10)
    plt.ylabel(' c2', labelpad=10)
    plt.title('ITE {} :: Target Value -> c1:{} c2:{}'.format(engine.ite, rider.coeff_vector[0], rider.coeff_vector[1]))
    plt.savefig('ITE {}.png'.format(engine.ite))
    #plt.show()
    #input('Next -> ITE {}'.format(engine.ite))
##실행부
org_value = []
exp_value = []
dis_value = []
for ITE_num in range(1):
    #1라이더 정의
    Riders = {}
    vector = [round(random.random(),2),-round(random.random(),2)]
    for name in range(3):
        r = Rider(name, vector)
        Riders[name] = r

    #2Stepwise 시작 정의
    ITE = 100
    beta = 0.8
    init_nets = {}
    for i in numpy.arange(-1,1,0.4):
        for j in numpy.arange(-1,1,0.4):
            init_nets[i,j] = 0
    engine = StepwiseSearch(1, None, init_nets, 0.4)
    init_vector = [round(random.random(),2),-round(random.random(),2)]
    print('초기 값', init_vector)
    LP_engine = LP_search(1, None, init_vector)
    LP_engine.true_coeff = vector

    Orders = {}
    pool = list(numpy.arange(0, 10, 0.1))
    for t in range(ITE):
        observation = []
        possible_customer_count = 0
        for name in Orders:
            if Orders[name].selected == False:
                possible_customer_count += 1
        if possible_customer_count < len(Riders) + 10:
            name_start = len(Orders)
            name_end = len(Orders) + 10
            for name in range(name_start, name_end):
                vector = random.sample(pool,2)
                o = Order(name, vector)
                Orders[name] = o
        for rider_name in Riders:
            rider = Riders[rider_name]
            ob = rider.SelectOrder(Orders)
            if ob[0] != None:
                observation.append(ob)
        for ob in observation:
            print('대상 데이터 {}'.format(ob))
            engine.Updater(ob, Orders, Riders[0])
            LP_engine.LP_Solver(ob, Orders) # [선택한 주문 이름, [나머지 주문 이름]]
            print('실제값 {} -> 예측 값 {}'.format(Riders[0].coeff_vector, LP_engine.init))
        if engine.ite % engine.T == 0 and engine.ite > 0:
            engine.NetUpdater()
            #GrahDraw(engine, Riders[0])
            #input('LP_search 결과 {} : 실제 {}'.format(LP_engine.init, Riders[0].coeff_vector))
            print('LP_search 결과 {} : 실제 {}: 시작 값 {}'.format(LP_engine.init, Riders[0].coeff_vector, init_vector))
        engine.ite += 1
        print('ITE {} 종료'.format(t))
    try:
        print('라이더의 벡터 : {} 비율 : {}'.format(Riders[0].coeff_vector,Riders[0].coeff_vector[0]/Riders[0].coeff_vector[1]))
    except:
        pass
    vectors = []
    test = []
    x = []
    y = []
    z = []
    for net in engine.nets:
        vectors.append([net, engine.nets[net]])
        test.append(engine.nets[net])
        x.append(net[0])
        y.append(net[1])
        z.append(engine.nets[net])

    rev_z = []
    max_val = max(z)
    for info in z:
        rev_z.append(info/max_val)
    vectors.sort(key= operator.itemgetter(1), reverse= True)
    count = 1
    for info in vectors[:20]:
        #print('순위 : {}, 정보 {} 점수 {} 비율 {} '.format(count, info[0], info[1],info[0][0]/info[0][1]))
        count += 1

    vectors.sort(key= operator.itemgetter(1))
    count = 1
    for info in vectors[:20]:
        #print('순위 : 하위 {}, 정보 {} 점수 {} 비율 {}'.format(count, info[0], info[1],info[0][0]/info[0][1]))
        count += 1
    print('평균 1 획득률 {}/ 평균 점수 {}'.format(sum(engine.ratio)/len(engine.ratio), sum(test)/len(test)))

    #클러스터 탐색
    rev_data = []
    for index in range(len(x)):
        #for _ in range(int(z[index])):
        #    rev_data.append([x[index], y[index]])
        rev_data.append([x[index], y[index]])
    rev_data = numpy.array(rev_data)
    #kmeans = KMeans(n_clusters=1, random_state=0).fit(rev_data)
    kmeans = KMeans(n_clusters=1, random_state=0).fit(rev_data, sample_weight=z)
    print('목표 {}'.format(Riders[0].coeff_vector))
    print('결과 StepWise{}'.format(kmeans.cluster_centers_[0]))
    print('결과 LP {}'.format(LP_engine.init))
    input('결과 확인')
    rev_z = numpy.array(rev_z)
    alphas = numpy.array(rev_z)
    #색깔 농도 관련 -> https://stackoverflow.com/questions/24767355/individual-alpha-values-in-scatter-plot
    rgba_colors = numpy.zeros((len(x),4))
    # for red the first column needs to be one
    rgba_colors[:,0] = 1.0
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alphas
    plt.scatter(x,y, color=rgba_colors, s = 5)
    plt.scatter(Riders[0].coeff_vector[0], Riders[0].coeff_vector[1], color = 'b', marker= "X", s = 20)
    cluster = [round(kmeans.cluster_centers_[0][0],4), round(kmeans.cluster_centers_[0][1],4)]
    plt.scatter(cluster[0], cluster[1], color = 'g', marker= "*", s = 20)
    #plt.scatter(x,y, label='sample', alpha=rev_z)
    plt.xlabel(' c1', labelpad= 10)
    plt.ylabel(' c2', labelpad= 10)
    plt.title('Target Value -> c1:{} c2:{}/Cluster {} {}'.format(Riders[0].coeff_vector[0],Riders[0].coeff_vector[1],cluster[0], cluster[1]))
    #todo : weighted linear regression 삽입.
    z = numpy.array(z)
    p1, p2 = numpy.polynomial.polynomial.polyfit(x, y, 1, w=z)
    def func(p1, p2, x):
        return  p1 * x + p2
    x = numpy.array(x)
    plt.plot(x, func(p1, p2, x))
    """
    plt.scatter(x, y)
    
    w = np.ones(x.shape[0])
    w[1] = 12
    # p1, p2 = np.polyfit(x, y, 1, w=w)
    p1, p2 = np.polynomial.polynomial.polyfit(x, y, 1, w=w)
    print(p1, p2, w)
    
    plt.plot(x, func(p1, p2, x))
    
    
    x = numpy.array(x)
    m, b = numpy.polyfit(x, y, 1)
    plt.plot(x, m*x + b)
    plt.show()
    """
    #plt.show()
    plt.savefig('ITE{}.png'.format(ITE_num))
    plt.clf()  # Clear figure
    org_value.append(Riders[0].coeff_vector)
    exp_value.append(cluster)
    res_dist = round(math.sqrt((Riders[0].coeff_vector[0] - cluster[0])**2 + (Riders[0].coeff_vector[1] - cluster[1])**2),4)
    dis_value.append(res_dist)
    f = open("StepWise.txt", 'a')
    info = '{};{};{};{};{};{} {}'.format(ITE_num,Riders[0].coeff_vector[0],Riders[0].coeff_vector[1],cluster[0],cluster[1],res_dist,'\n')
    f.write(info)
    f.close()