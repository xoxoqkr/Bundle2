# -*- coding: utf-8 -*-

import operator
import itertools
import random
import numpy
import simpy
import math
import matplotlib.pyplot as plt

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

    def Updater(self, data, orders):
        if self.ite > 1 and self.ite % self.T == 0:
            #input('그물 추가')
            self.NetUpdater()
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
##실행부

#1라이더 정의
Riders = {}
vector = [round(random.random(),2),-round(random.random(),2)]
for name in range(3):
    r = Rider(name, vector)
    Riders[name] = r

#2Stepwise 시작 정의
beta = 0.8
init_nets = {}
for i in numpy.arange(-1,1,0.4):
    for j in numpy.arange(-1,1,0.4):
        init_nets[i,j] = 0
engine = StepwiseSearch(1, None, init_nets, 0.4)


Orders = {}
pool = list(numpy.arange(0, 10, 0.1))
for t in range(500):
    observation = []
    possible_customer_count = 0
    for name in Orders:
        if Orders[name].selected == False:
            possible_customer_count += 1
    if possible_customer_count < len(Riders):
        for name in range(len(Orders), len(Orders) + 4):
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
        engine.Updater(ob, Orders)
    engine.ite += 1
    print('ITE {} 종료'.format(t))

print('라이더의 벡터 : {} 비율 : {}'.format(Riders[0].coeff_vector,Riders[0].coeff_vector[0]/Riders[0].coeff_vector[1]))
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
    print('순위 : {}, 정보 {} 점수 {} 비율 {} '.format(count, info[0], info[1],info[0][0]/info[0][1]))
    count += 1

vectors.sort(key= operator.itemgetter(1))
count = 1
for info in vectors[:20]:
    print('순위 : 하위 {}, 정보 {} 점수 {} 비율 {}'.format(count, info[0], info[1],info[0][0]/info[0][1]))
    count += 1
print('평균 1 획득률 {}/ 평균 점수 {}'.format(sum(engine.ratio)/len(engine.ratio), sum(test)/len(test)))

rev_z = numpy.array(rev_z)
alphas = numpy.array(rev_z)
#색깔 농도 관련 -> https://stackoverflow.com/questions/24767355/individual-alpha-values-in-scatter-plot
rgba_colors = numpy.zeros((len(x),4))
# for red the first column needs to be one
rgba_colors[:,0] = 1.0
# the fourth column needs to be your alphas
rgba_colors[:, 3] = alphas

plt.scatter(x,y, color=rgba_colors)
#plt.scatter(x,y, label='sample', alpha=rev_z)
plt.xlabel(' x1', labelpad= 10)
plt.ylabel(' x2', labelpad= 10)
plt.title('Target Value -> x1:{} x2:{}'.format(Riders[0].coeff_vector[0],Riders[0].coeff_vector[1]))
plt.show()