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

    def SelectOrder(self, orders, predict = []):
        scores = []
        for order_name in orders:
            order = orders[order_name]
            if order.selected == False:
                #print('주문 점수 정보{} {}'.format(self.coeff_vector, order.data_vector))
                if len(predict) == 0:
                    score = ValueCal(self.coeff_vector, order.data_vector, cal_type= self.cal_type) + order.subsidy
                else:
                    score = ValueCal(predict, order.data_vector, cal_type=self.cal_type) + order.subsidy
                scores.append([order_name, score])
        scores.sort(key=operator.itemgetter(1), reverse = True)
        #print('점수들 {} ~ {}'.format(scores[:4],scores[-3:]))
        if len(scores) > 0:
            if len(predict) == 0:
                orders[scores[0][0]].selected = True
            else:
                pass
            names = []
            for score in scores[1:]:
                names.append(score[0])
            return [scores[0][0] ,names]
        else:
            return [None, []]

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

def UpdateGurobiModel(coeff, data, past_data = [], print_para = False, cal_type = 'linear', M = 1000):
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
    model.Params.method = -1
    model.optimize()
    try:
        #print('Obj val 출력: %g' % model.objVal)
        res = []
        for val in model.getVars():
            if val.VarName[0] == 'w':
                res.append(float(val.x))
        print('결과 {} '.format(res))
        return True, res, model
    except:
        print('Infeasible')
        return False, None, model


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
            pass
        self.past_data.append(data)
        #input('LP_Solver 확인'.format())


##실행부
org_value = []
exp_value = []
dis_value = []
for ITE_num in range(1):
    #1라이더 정의
    Riders = {}
    vector = [0.5 + round(random.random(),2), -round(random.random(),2), round(random.random(),2)] #선형인 경우
    cal_type = 'linear' #linear / log
    for name in range(1):
        r = Rider(name, vector, cal_type = cal_type)
        Riders[name] = r
    ITE = 100
    ITE2 = 150
    beta = 0.8
    init_vector = [0.5, -0.5, 0.5]
    LP_engineByGurobi = LP_search(1, None, init_vector, engine_type='Gurobi')
    LP_engineByGurobi.true_coeff = vector
    Orders = {}
    pool = list(numpy.arange(0, 10, 0.1))
    #2실험 시작
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
                o = Order(name, vector, cal_type = cal_type)
                Orders[name] = o
        for rider_name in Riders:
            rider = Riders[rider_name]
            ob2 = rider.SelectOrder(Orders, predict=LP_engineByGurobi.init)
            ob = rider.SelectOrder(Orders)
            print('결과 {} :: {} 예상'.format(ob[0], ob2[0]))
            if ob[0] != None:
                observation.append(ob)
        for ob in observation:
            LP_engineByGurobi.LP_Solver(ob, Orders) # [선택한 주문 이름, [나머지 주문 이름]]
        print('ITE {} 실제 값 {} :: {} 계산 값'.format(t, Riders[0].coeff_vector, LP_engineByGurobi.init))