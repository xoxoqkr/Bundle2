# -*- coding: utf-8 -*-

import copy
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
from docplex.mp.model import Model
#from docplex.mp.constr import *
from docplex.mp.constr import (LinearConstraint as DocplexLinearConstraint,
                               QuadraticConstraint as DocplexQuadraticConstraint,
                               NotEqualConstraint, IfThenConstraint)

def ValueCal(coeff, vector, cal_type = 'linear'):
    if cal_type == 'log':
        val = LogScore(coeff, vector)
    elif cal_type == 'linear':
        val = numpy.dot(coeff, vector)
    else:
        val = 0
        print("Error")
    return round(val,2)

def LogScore(coeff, vector):
    score = 0
    #print('coeff {} vector {} '.format(coeff, vector))
    for index in range(len(coeff)):
        try:
            score += (coeff[index] ** vector[index])*((-1)**index)
        except:
            input('스칼라 오류 coeff :{}, vector : {}'.format(coeff, vector))
    #print('지수 점수 {}'.format(score) )
    return round(score,4)

class Order(object):
    def __init__(self, name, input_value, cal_type = 'linear'):
        self.name = name
        self.selected = False
        self.data_vector = input_value
        self.cal_type = cal_type
        self.subsidy = 0

class Rider(object):
    def __init__(self, name, input_value, cal_type = 'linear'):
        self.name = name
        self.coeff_vector = input_value
        self.cal_type = cal_type

    def SelectOrder(self, orders):
        scores = []
        for order_name in orders:
            order = orders[order_name]
            if order.selected == False:
                #print('주문 점수 정보{} {}'.format(self.coeff_vector, order.data_vector))
                score = ValueCal(self.coeff_vector, order.data_vector, cal_type= self.cal_type) + order.subsidy
                scores.append([order_name, score])
        scores.sort(key=operator.itemgetter(1), reverse = True)
        #print('점수들 {} ~ {}'.format(scores[:4],scores[-3:]))
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
        self.xlim = [-1,1]
        self.ylim = [-1,1]

    def NetUpdater(self):
        scores = []
        test = []
        for ele in self.nets:
            scores.append([ele, self.nets[ele]])
            test.append(self.nets[ele])
        #input('점수 분포 {}'.format(sorted(test)))
        rev_test = list(set(test))
        rev_test.sort(reverse = True)
        second_val = rev_test[1]
        test.sort(reverse = True)
        index = test.index(second_val)
        scores.sort(key=operator.itemgetter(1), reverse = True)
        upper_scores = scores[:index]
        #update_num = min(5,int(len(scores) * self.alpha) )
        #upper_scores = scores[:update_num]
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


    def NetUpdaterReverse(self):
        scores = []
        test = []
        for ele in self.nets:
            scores.append([ele, self.nets[ele]])
            test.append(self.nets[ele])
        cut_value = max(test) #ele has cut_value inherited to next nets.
        #delete nets
        delete_keys = []
        for ele in self.nets:
            if self.nets[ele] < cut_value:
                delete_keys.append(ele)
        for delete_key in delete_keys:
            del self.nets[delete_key]

    def Check(self, ele, data, orders):
        scores = []
        for order_name in [data[0]] + data[1]:
            order = orders[order_name]
            score = ValueCal(ele, order.data_vector, cal_type=order.cal_type)
            scores.append([order_name, score])
            #scores.append([order_name, numpy.dot(ele, order.data_vector)]) #todo: 가치함수의 형태가 달라지면, 달라져야 함.
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

def Coeff_Check(coeff, datas, cal_type = 'linear'):
    count1 = 0
    for data in datas:
        val = ValueCal(coeff, data[0], cal_type=cal_type)
        count2 = 0
        for info in data[1:]:
            val2 = ValueCal(coeff, info, cal_type=cal_type)
            if val >= val2:
                #print('{}-{} Z : {} >= {} : 대상'.format(count1, count2, val, val2))
                pass
            else:
                input('에러 발생 : 데이터 {} : 회차 {} : 선택된 주문 가치 {} < 다른 주문 {} '.format(count1, count2,val, val2))
            count2 += 1
        count1 += 1


def ReviseCoeff_MJByCplex(init_coeff, now_data, past_data, error = 10, print_para = False):
    coeff = list(range(len(init_coeff)))
    # D.V. and model set.
    md1 = Model('Coeff_model')
    x = md1.continuous_var_list(len(coeff),name = 'x')
    #z = md1.continuous_var_list(1 + len(past_data),name = 'v')
    #a = md1.continuous_var_list(len(coeff) ,name = 'a')
    u = md1.continuous_var_list(len(coeff), name = 'u')

    #Define Obj
    md1.minimize(md1.sum(u[i] for i in coeff))

    #Add Constrsints
    md1.add_constraints( x[i] <= u[i] for i in coeff)
    md1.add_constraints(-x[i] <= u[i] for i in coeff)


    for other_info in now_data[1:]:
        md1.add_constraint((x[0] + init_coeff[0]) * now_data[0][0] + (x[1] + init_coeff[1]) * now_data[0][1] - error >= (
                    x[0] + init_coeff[0]) * other_info[0] + (x[1] + init_coeff[1]) * other_info[1])
    if len(past_data) > 0:
        for data in past_data:
            p_selected = data[0]
            p_others = data[1:]
            for p_other_info in p_others:
                md1.add_constraint((x[0] + init_coeff[0]) * p_selected[0] + (x[1] + init_coeff[1]) * p_selected[1] - error >= (
                            x[0] + init_coeff[0]) * p_other_info[0] + (x[1] + init_coeff[1]) *
                            p_other_info[1])
    msol = md1.solve()
    print(msol)
    if msol == None:
        #input('CPLEX 확인1')
        return None, None
    else:
        #print(msol.get_infeasibility())
        print(msol.get_all_values())
        res = msol.get_all_values()[:2]
        print(res)
        #input('CPLEX 확인2')
        return True, res

def ReviseCoeff_MJByGurobi(init_coeff, now_data, past_data, error = 10, print_para = False, cal_type = 'linear', M = 100):
    coeff = list(range(len(init_coeff)))
    # D.V. and model set.
    m = gp.Model("problem1")
    x = m.addVars(len(coeff), vtype=GRB.CONTINUOUS, name="x")
    a = m.addVars(len(coeff), vtype=GRB.CONTINUOUS, name="a")
    u = m.addVars(len(coeff), vtype=GRB.CONTINUOUS, name="u")

    m.setObjective(gp.quicksum(x[i] for i in coeff), GRB.MINIMIZE) #obj

    #m.setObjective(u[0] + u[1], GRB.MINIMIZE)
    m.addConstrs(x[i] <= u[i] for i in coeff)
    m.addConstrs(-x[i] <= u[i] for i in coeff)


    #m.setObjective(gp.quicksum(a[i] for i in coeff), GRB.MINIMIZE)
    #m.setObjective(a[0] + a[1], GRB.MINIMIZE)
    #m.addConstrs(a[i] == gp.abs_(x[i]) for i in coeff)
    #m.setObjective(x[0]^2 + x[1]^2, GRB.MINIMIZE)
    z_count = 0
    #이번 selected와 other에 대한 문제 풀이
    if print_para == True:
        score = ValueCal(init_coeff, now_data[0], cal_type=cal_type)
        print('선택 고객 z {} '.format(score))
    #m.addConstr(gp.quicksum((x[i] + init_coeff[i])*now_data[0][i] for i in coeff) == z[z_count])

    #m.addConstr(z[z_count] - error == (x[0] + init_coeff[0]) * now_data[0][0] + (x[1] + init_coeff[1]) * now_data[0][1])
    #m.addConstr(z[z_count] >= 0)
    z_val = ValueCal(init_coeff, now_data[0], cal_type=cal_type)
    index2 = 0
    for other_info in now_data[1:]:
        compare_val = ValueCal(init_coeff, other_info, cal_type=cal_type)
        if print_para == True:
            if z_val < compare_val:
                print('Current {}-{} 비교 결과 Z : {} < {} : Val'.format(z_count, index2,  z_val, compare_val))
            else:
                print('Current {}-{} 비교 결과 Z : {} > {} : Val'.format(z_count, index2, z_val, compare_val))
        #m.addConstr(gp.quicksum((x[i] + init_coeff[i])*other_info[i] for i in coeff) <= z[z_count] - error)
        #m.addConstr( z[z_count] - error >= (x[0] + init_coeff[0]) * other_info[0] + (x[1] + init_coeff[1]) * other_info[1])
        m.addConstr( (x[0] + init_coeff[0]) * now_data[0][0] + (x[1] + init_coeff[1]) * now_data[0][1] - error >= (x[0] + init_coeff[0]) * other_info[0] + (x[1] + init_coeff[1]) * other_info[1])
        index2 += 1
    z_count += 1
    #과거 정보를 적층하는 작업
    if len(past_data) > 0:
        for data in past_data:
            z_val_old = ValueCal(init_coeff, data[0], cal_type=cal_type)
            p_selected = data[0]
            p_others = data[1:]
            #m.addConstr(gp.quicksum((x[i] + init_coeff[i]) * p_selected[i] for i in coeff) == z[z_count])
            #m.addConstr(z[z_count] - error == (x[0] + init_coeff[0]) * p_selected[0] + (x[1] + init_coeff[1]) * p_selected[1])
            index2 = 0
            for p_other_info in p_others:
                compare_val_old = ValueCal(init_coeff, p_other_info, cal_type=cal_type)
                if print_para == True:
                    #print('과거 {} 데이터 제약식 {} : {}'.format(z_count, init_coeff, p_other_info))
                    #print('과거 {}  데이터 고객 z {} '.format(z_count, numpy.dot(init_coeff, p_other_info)))
                    #print('Past 비교 결과 Z : {} < {} : Val'.format(z_val, compare_val))
                    if z_val_old < compare_val_old:
                        print('Past {}-{} 비교 결과 Z : {} < {} : Val'.format(z_count, index2,  z_val_old, compare_val_old))
                    else:
                        print('Past {}-{} 비교 결과 Z : {} > {} : Val'.format(z_count, index2, z_val_old, compare_val_old))
                    pass
                #m.addConstr(gp.quicksum((x[i] + init_coeff[i]) * p_other_info[i] for i in coeff) <= z[z_count] - error)
                #m.addConstr(z[z_count] - error >= (x[0] + init_coeff[0]) * p_other_info[0] + (x[1] + init_coeff[1]) * p_other_info[1])
                m.addConstr((x[0] + init_coeff[0]) * p_selected[0] + (x[1] + init_coeff[1]) * p_selected[1] - error >= (x[0] + init_coeff[0]) * p_other_info[0] + (x[1] + init_coeff[1]) *
                            p_other_info[1])
                index2 += 1
            z_count += 1
    #풀이
    m.setParam(GRB.Param.OutputFlag, 0)
    m.Params.method = -1
    m.optimize()

    try:
        print('Obj val: %g' % m.objVal)
        res = []
        for val in m.getVars():
            if val.VarName[0] == 'x':
                res.append(float(val.x))
        return True, res, m
    except:
        print('Infeasible')
        return False, None, m

def ModelUpdate(m, coeff, data, error = 10, print_para = False, cal_type = 'linear'):
    #1 제약식 추가 부분.
    z_count = 0
    index2 = 0
    z_val = ValueCal(coeff, data[0], cal_type=cal_type)
    for other_info in data[1:]:
        compare_val = ValueCal(coeff, other_info, cal_type=cal_type)
        if print_para == True:
            if z_val < compare_val:
                print('Current {}-{} 비교 결과 Z : {} < {} : Val'.format(z_count, index2,  z_val, compare_val))
            else:

                print('Current {}-{} 비교 결과 Z : {} > {} : Val'.format(z_count, index2, z_val, compare_val))
        m.addConstr((m.x[0] + coeff[0]) * data[0][0] + (m.x[1] + coeff[1]) * data[0][1] - error >= (m.x[0] + coeff[0]) * other_info[0] + (m.x[1] + coeff[1]) * other_info[1])
        index2 += 1
    #2 model 수정
    m.Reset()
    m.setParam(GRB.Param.OutputFlag, 0)
    m.Params.method = -1
    m.optimize()
    try:
        print('Obj val: %g' % m.objVal)
        res = []
        for val in m.getVars():
            if val.VarName[0] == 'x':
                res.append(float(val.x))
        return True, res, m
    except:
        print('Infeasible')
        return False, None, m

def UpdateGurobiModel(coeff, data, past_data = [], print_para = False, cal_type = 'linear', M = 100):
    """
    주어진 coeff를 새로운 data에 대해 재계산하고 갱신함.
    Args:
        coeff: weight vector
        data: new data [[selected order], [other 1],...,[other n]]
        past_data: optional [data0, data1,...,data n]
        print_para: print option
        cal_type: print option 2
        M: penalty for slack variable y

    Returns: weight vector

    """
    #확장형 1.2를 고려한 확장형2
    w_index = list(range(len(coeff)))
    # D.V. and model set.
    model = gp.Model("problem1_3")
    w = model.addVars(len(coeff), vtype=GRB.CONTINUOUS, name="w")
    z = model.addVars(len(coeff), vtype=GRB.CONTINUOUS, name="z")
    y = model.addVars(len(data) + len(past_data), vtype=GRB.CONTINUOUS, name="y")

    model.setObjective(gp.quicksum(z) + M*gp.quicksum(y), GRB.MINIMIZE)

    model.addConstrs(coeff[i] - w[i] <= z[i] for i in w_index) #linearization part
    model.addConstrs(coeff[i] - w[i]  >= -z[i] for i in w_index)

    z_count = 0 #D_new part
    if print_para == True:
        score = ValueCal(coeff, data[0], cal_type=cal_type)
        print('선택 고객 z {} '.format(score))
    #print('확인 {} : {}'.format(coeff, data[0]))
    z_val = ValueCal(coeff, data[0], cal_type=cal_type)
    data_index = 0
    for other_info in data[1:]:
        compare_val = ValueCal(coeff, other_info, cal_type=cal_type)
        if print_para == True:
            if z_val < compare_val:
                print('Current {}-{} 비교 결과 Z : {} < {} : Val'.format(z_count, data_index, z_val, compare_val))
            else:
                print('Current {}-{} 비교 결과 Z : {} > {} : Val'.format(z_count, data_index, z_val, compare_val))
        model.addConstr(gp.quicksum(data[0][i]*(coeff[i] + w[i]) for i in w_index) + y[z_count]
                    >= gp.quicksum(data[data_index][i]*(coeff[i] + w[i]) for i in w_index))
        data_index += 1
    z_count += 1
    #2 model 수정
    if len(past_data) > 0:
        for data in past_data:
            z_val_old = ValueCal(coeff, data[0], cal_type=cal_type)
            p_selected = data[0]
            p_others = data[1:]
            data_index2 = 0
            for p_other_info in p_others:
                compare_val_old = ValueCal(coeff, p_other_info, cal_type=cal_type)
                if print_para == True:
                    if z_val_old < compare_val_old:
                        print('Past {}-{} 비교 결과 Z : {} < {} : Val'.format(z_count, data_index2,  z_val_old, compare_val_old))
                    else:
                        print('Past {}-{} 비교 결과 Z : {} > {} : Val'.format(z_count, data_index2, z_val_old, compare_val_old))
                    pass
                model.addConstr(gp.quicksum(p_selected[i] * (coeff[i] + w[i]) for i in w_index) + y[z_count]
                                >= gp.quicksum(p_other_info[i] * (coeff[i] + w[i]) for i in w_index))
                data_index2 += 1
            z_count += 1
    #3 model 풀이
    model.setParam(GRB.Param.OutputFlag, 0)
    #model.Params.method = -1
    model.optimize()
    try:
        print('Obj val 출력: %g' % model.objVal)
        res = []
        for val in model.getVars():
            if val.VarName[0] == 'w':
                res.append(float(val.x))
        return True, res, model
    except:
        print('Infeasible')
        return False, None, model



class LP_search(object):
    def __init__(self, name, func, init, T = 50, engine_type = 'Gurobi'):
        self.name = name
        self.func = func
        self.init = init
        self.past_data = []
        self.true_coeff = None
        self.path = []
        self.engine_type = engine_type
        self.xlim = [-1,1]
        self.ylim = [-1,1]

    def LP_Solver(self, org_data, customers):
        data = [customers[org_data[0]].data_vector]
        for name in org_data[1]:
            data.append(customers[name].data_vector)
        #input('초기 값 {} 입력 데이터 {}'.format(self.init, data))
        if self.engine_type == 'Gurobi':
            #feasiblity, res, model = ReviseCoeff_MJByGurobi(self.init, data, self.past_data, error = 0, print_para= False)
            feasiblity, res, model = UpdateGurobiModel(self.init, data, self.past_data)
            #feasiblity, res, model = UpdateGurobiModel(self.init, data)
        else:
            feasiblity,res = ReviseCoeff_MJByCplex(self.init, data, self.past_data, error=0, print_para=False)
        if feasiblity == True:
            for index in range(len(res)):
                self.init[index] += res[index]
            if sum(res) != 0:
                self.path.append(res)
        else:
            #input('해 없음'.format())
            #print('확인용 계산: 라이더의 Coeff {} : 오라클의 Coeff {}'.format())
            if self.engine_type == 'Gurobi':
                feasiblity2, res2, model = ReviseCoeff_MJByGurobi(self.true_coeff, data, self.past_data, error= 0, print_para= False)
            else:
                feasiblity2, res2 = ReviseCoeff_MJByCplex(self.true_coeff, data, self.past_data, error=0, print_para=False)
            #input('진짜 해에 대한 결과 {} : {}'.format(feasiblity2, res2))
            print('진짜 해에 대한 결과 {} : {}'.format(feasiblity2, res2))
            if feasiblity2 == False:
                Coeff_Check(self.true_coeff, [data] + self.past_data, cal_type = customers[0].cal_type)
                input('확인용 계산에도 에러 발생')
        self.past_data.append(data)
        #input('LP_Solver 확인'.format())

def func(p1, p2, x):
    return p1 * x + p2

def GrahDraw(engine, rider, centre = False, saved_info = 'forward'):
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
    plt.xlim(engine.xlim)  # X축의 범위: [xmin, xmax]
    plt.ylim(engine.ylim)  # Y축의 범위: [ymin, ymax]
    plt.title('ITE {} :: Target Value -> c1:{} c2:{}'.format(engine.ite, rider.coeff_vector[0], rider.coeff_vector[1]))
    cluster = None
    if centre == True:
        rev_data = []
        for index in range(len(x)):
            # for _ in range(int(z[index])):
            #    rev_data.append([x[index], y[index]])
            rev_data.append([x[index], y[index]])
        rev_data = numpy.array(rev_data)
        # kmeans = KMeans(n_clusters=1, random_state=0).fit(rev_data)
        kmeans = KMeans(n_clusters=1, random_state=0).fit(rev_data, sample_weight=z)
        plt.scatter(rider.coeff_vector[0], rider.coeff_vector[1], color='b', marker="X", s=20)
        cluster = [round(kmeans.cluster_centers_[0][0], 4), round(kmeans.cluster_centers_[0][1], 4)]
        plt.scatter(cluster[0], cluster[1], color='g', marker="*", s=20)
        z = numpy.array(z)
        p1, p2 = numpy.polynomial.polynomial.polyfit(x, y, 1, w=z)
        x = numpy.array(x)
        plt.plot(x, func(p1, p2, x))
        plt.savefig('ITE {} _ {} .png'.format(engine.ite, saved_info))
        plt.close()
    else:
        plt.savefig('ITE {} _ {} .png'.format(engine.ite, saved_info))
        plt.close()
    return cluster


def NetMaker(range_list, interval_list):
    net_candis = []
    nets = []
    for index in range(len(range_list)):
        s = range_list[index][0]
        e = range_list[index][1]
        i = interval_list[index]
        tem = numpy.arange(s,e,i)
        net_candis.append(tem)
    pass

def A2_part1(exp_coeff, orders, cal_type='linear', buffer = 0.01):
    exp_values = []
    for order_name in orders:
        order = orders[order_name]
        if order.selected == False:
            exp_value = ValueCal(exp_coeff, order.data_vector, cal_type=cal_type)
            exp_values.append([order_name, exp_value])
    exp_values.sort(key = operator.itemgetter(1), reverse = True)
    s = exp_values[0][1] - exp_values[1][1]
    #orders[exp_values[1][0]].subsidy = s + buffer
    orders[exp_values[1][0]].subsidy = s*(1+buffer)
    print(exp_values)
    print('필요 s {}: 지급된 s {}'.format(s,s*(1+buffer)))
    return exp_values[1][0]


##실행부
org_value = []
exp_value = []
dis_value = []
for ITE_num in range(1):
    #1라이더 정의
    Riders = {}
    vector = [0.5 + round(random.random(),2), -round(random.random(),2), round(random.random(),2)] #선형인 경우
    #vector = [2 + round(random.random(), 2), 2 + round(random.random(), 2),2 + round(random.random(), 2)] #지수형인 경우
    #vector = []
    cal_type = 'linear' #linear / log
    for name in range(3):
        r = Rider(name, vector, cal_type = cal_type)
        Riders[name] = r
    print('라이더 벡터 {}'.format(Riders[0].coeff_vector))
    #2Stepwise 시작 정의
    ITE = 100
    ITE2 = 150
    beta = 0.8
    init_nets_forward = {}
    for i in numpy.arange(2,3,0.05):
        for j in numpy.arange(2,3,0.05):
            for k in numpy.arange(2,3,0.05):
                init_nets_forward[i, j, k] = 0
    engine_forward = StepwiseSearch(1, None, init_nets_forward, 0.4)
    engine_reverse = StepwiseSearch(1, None, init_nets_forward, 0.4, T=2)
    engine_reverse.xlim = [-1.5,1.5]
    engine_reverse.ylim = [-1.5,1.5]
    #init_vector = [round(random.random(),2),-round(random.random(),2)]
    init_vector = [0.5, -0.5, 0.5]
    print('초기 값', init_vector)
    LP_engineByGurobi = LP_search(1, None, init_vector, engine_type='Gurobi')
    LP_engineByCplex = LP_search(1, None, init_vector, engine_type='Cplex')
    LP_engineByGurobi.true_coeff = vector
    LP_engineByCplex.true_coeff = vector
    print('LPGurobi_engine 벡터 {} :: LPCplex_engine 벡터 {}'.format(LP_engineByGurobi.true_coeff, LP_engineByCplex.true_coeff))
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
                vector = random.sample(pool,3)
                #vector = random.sample(pool, 2) + [random.choice([0,1])]
                o = Order(name, vector, cal_type = cal_type)
                Orders[name] = o
        for rider_name in Riders:
            rider = Riders[rider_name]
            ob = rider.SelectOrder(Orders)
            #print('라이더 선택 주문 {}'.format(ob[0]))
            if ob[0] != None:
                observation.append(ob)
        for ob in observation:
            #print('대상 데이터 {}'.format(ob))
            #engine_forward.Updater(ob, Orders, Riders[0])
            #engine_reverse.Updater(ob, Orders, Riders[0])
            LP_engineByGurobi.LP_Solver(ob, Orders) # [선택한 주문 이름, [나머지 주문 이름]]
            #LP_engineByCplex.LP_Solver(ob, Orders)
            #Coeff_Check(Riders[0].coeff_vector, LP_engine.past_data)
            #print('확인용 계산: 라이더의 Coeff {} : 오라클(Gurobi)의 Coeff {} : 오라클(Cplex)의 Coeff {}'.format(Riders[0].coeff_vector, LP_engineByGurobi.true_coeff, LP_engineByCplex.true_coeff))
            #print('실제값 {} -> Gurobi 예측 값 {} : Cplex 예측 값 {}'.format(Riders[0].coeff_vector, LP_engineByGurobi.init , LP_engineByCplex.init))
        if engine_forward.ite % engine_forward.T == 0 and engine_forward.ite > 0:
            #engine_forward.NetUpdater()
            pass
        if engine_reverse.ite % engine_reverse.T == 0 and engine_reverse.ite > 0:
            #engine_reverse.NetUpdaterReverse()
            #print('현재 ite : {} / 존재 p 수 : {}'.format(t, len(engine_reverse.nets)))
            pass
        #GrahDraw(engine_forward, Riders[0])
        #GrahDraw(engine_reverse, Riders[0], info = 'backward')
        #input('LP_search 결과 {} : 실제 {}'.format(LP_engine.init, Riders[0].coeff_vector))
        #print('LP_Gurobi 결과 {} : LP_Cplex 결과 {} :실제 {}: 시작 값 {}'.format(LP_engineByGurobi.init,LP_engineByCplex.init, Riders[0].coeff_vector, init_vector))
        """
        if engine_reverse.ite % engine_reverse.T == 0 and engine_reverse.ite > 0:
            engine_reverse.NetUpdaterReverse()
            if engine_reverse.ite < 20:
                #GrahDraw(engine_reverse, Riders[0], saved_info='backwardforexp')
                pass
            print('남은 p 수 : {}'.format(len(engine_reverse.nets)))
        """
        engine_forward.ite += 1
        engine_reverse.ite += 1
        print('ITE {} 실제 값 {} :: {} 계산 값'.format(t, Riders[0].coeff_vector, LP_engineByGurobi.init))
    #input('확인')
    """
    try:
        print('True 라이더 계수 {}'.format(Riders[0].coeff_vector))
        print('남은 p 수 : {} 내용 {}'.format(len(engine_reverse.nets), engine_reverse.nets))
        print('라이더의 벡터 : {} 비율 : {}'.format(Riders[0].coeff_vector,Riders[0].coeff_vector[0]/Riders[0].coeff_vector[1]))
    except:
        pass
    #foward_Search = GrahDraw(engine_forward, Riders[0], centre= True)
    #reverse_Search = GrahDraw(engine_reverse, Riders[0], centre= True, saved_info='backward2')
    #print('foward_Search 클러스터 {}'.format(foward_Search))
    
    #A2알고리즘 부분
    engine_coeff = list(engine_reverse.nets.keys())[0]
    engine_coeff = list(engine_coeff)
    init_engine = copy.deepcopy(engine_coeff)
    print('A1 종료. p 확인 {}'.format(engine_coeff))
    Orders = {}
    pool = list(numpy.arange(0, 10, 0.1))
    for t in range(ITE2):
        possible_customer_count = 0
        for name in Orders:
            if Orders[name].selected == False:
                possible_customer_count += 1
        if possible_customer_count < len(Riders) + 10:
            name_start = len(Orders)
            name_end = len(Orders) + 10
            for name in range(name_start, name_end):
                #vector = random.sample(pool, 3)
                vector = random.sample(pool, 2) + [random.choice([0, 1])]
                o = Order(name, vector, cal_type=cal_type)
                Orders[name] = o
        for rider_name in Riders:
            rider = Riders[rider_name]
            exp_order = A2_part1(engine_coeff, Orders, cal_type=cal_type)
            ob = rider.SelectOrder(Orders)
            if ob[0] != None:
                count = 1
                theta = 0.05*(0.99**(len(Riders)*t+count))
                print('theta {} :: 예상 고객 {} 라이더 선택 고객 {}'.format(theta,exp_order , Orders[ob[0]].name))
                if exp_order == Orders[ob[0]].name:
                    print('일치')
                    for index in range(len(engine_coeff)):
                        #engine_coeff[index] -= theta
                        engine_coeff[index] = engine_coeff[index]*(1- theta)
                        pass
                else:
                    print('불일치')
                    for index in range(len(engine_coeff)):
                        #engine_coeff[index] += theta
                        engine_coeff[index] = engine_coeff[index] * (1 + theta)
                        pass
                for order_name in Orders:
                    Orders[order_name].subsidy = 0
                    pass
                count += 1
                print('ITE {} 목표 {} 현재 {}'.format(t, Riders[0].coeff_vector, engine_coeff))
        #input('A2 확인')
    f = open("A1andA2_res.txt", 'a')
    info = '목표 {};A1 종료 후 ;{}; A2 종료 후; {} ;{}'.format(Riders[0].coeff_vector,init_engine, engine_coeff,  '\n')
    f.write(info)
    f.close()
    #input('A2 정지')
    
    #print('reverse_Search 클러스터 {} : 남은 해들 {}'.format(reverse_Search, list(engine_reverse.nets.keys())))
    input('정지')

    vectors = []
    test = []
    x = []
    y = []
    z = []
    for net in engine_forward.nets:
        vectors.append([net, engine_forward.nets[net]])
        test.append(engine_forward.nets[net])
        x.append(net[0])
        y.append(net[1])
        z.append(engine_forward.nets[net])
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
    print('평균 1 획득률 {}/ 평균 점수 {}'.format(sum(engine_forward.ratio) / len(engine_forward.ratio), sum(test) / len(test)))
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
    print('결과 Gurobi LP {}, LP 변화 경로{}'.format(LP_engineByGurobi.init, LP_engineByGurobi.path))
    print('결과 Cplex LP {}, LP 변화 경로{}'.format(LP_engineByCplex.init, LP_engineByCplex.path))
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
    """


