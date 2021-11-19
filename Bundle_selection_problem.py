# -*- coding: utf-8 -*-

import gurobipy as gp
from gurobipy import GRB


def Bundle_selection_problem(F):
    bundle_index = list(range(len(F)))
    s = [] #s value
    C_b = [] # customer set in bundle b
    C_b_len = []
    for info in F:
        C_b.append(info[4])
        C_b_len.append(list(range(len(info[4]))))
        s.append(info[7])
    print(C_b_len)
    customer_index = list(range(len(C_b_len)))
    m = gp.Model("mip1")
    y = m.addVars(len(F), vtype=GRB.BINARY, name="y")
    x = m.addVars(len(F),len(C_b), vtype=GRB.BINARY, name="x")

    m.setObjective(gp.quicksum(s[i]*y[i] for i in bundle_index), GRB.MAXIMIZE)

    for i in bundle_index:
        m.addConstrs(y[i] <= x[i,j] for j in C_b_len[i])

    for i in customer_index:
        m.addConstr(gp.quicksum(x[i,j] for j in C_b_len[i]) <= 1)
    #풀이
    m.optimize()
    try:
        print('Obj val: %g' % m.objVal)
        res = []
        count = 0
        for val in m.getVars():
            if val.VarName[0] == 'y' and float(val.x) == 1.0:
                res.append(F[count])
            count += 1
        return True, res
    except:
        print('Infeasible')
        return False, []


def Bundle_selection_problem2(F):
    bundle_index = list(range(len(F)))
    s = [] #s value
    C_b = [] # customer set in bundle b
    C_b_len = []
    C_reset = []
    for info in F:
        C_b.append(info[4])
        C_reset +=  info[4]
        C_b_len.append(list(range(len(info[4]))))
        s.append(info[7])
    unique_ct_num_list = list(set(C_reset))
    unique_ct_num_list.sort()
    C_b_len2 = []
    for i in C_b:
        tem = []
        for j in i:
            index = unique_ct_num_list.index(j)
            tem.append(index)
        C_b_len2.append(tem)
    print('번들 인덱스 {} C_b {}'.format(bundle_index, C_b))
    print(C_b_len2)
    unique_ct_num = len(unique_ct_num_list)
    customer_index = list(range(unique_ct_num))
    m = gp.Model("mip1")
    y = m.addVars(len(F), vtype=GRB.BINARY, name="y")
    x = m.addVars(len(F),unique_ct_num, vtype=GRB.BINARY, name="x")

    m.setObjective(gp.quicksum(s[i]*y[i] for i in bundle_index), GRB.MAXIMIZE)

    for i in bundle_index:
        m.addConstrs(y[i] <= x[i,j] for j in C_b_len2[i])

    for j in customer_index:
        m.addConstr(gp.quicksum(x[i,j] for i in bundle_index) <= 1)
    #풀이
    m.optimize()
    try:
        print('Obj val: %g' % m.objVal)
        res = []
        count = 0
        for val in m.getVars():
            if val.VarName[0] == 'y' and float(val.x) == 1.0:
                res.append(F[count])
            count += 1
        return True, res
    except:
        print('Infeasible')
        return False, []

def Bundle_selection_problem3(phi_b, d_matrix, s_b, min_pr):
    bundle_indexs = list(range(len(s_b)))
    try:
        w = sum(s_b) / len(s_b)
    except:
        w = 1

    m = gp.Model("mip1")
    x = m.addVars(len(bundle_indexs), vtype=GRB.BINARY, name="x")
    z = m.addVars(len(bundle_indexs), vtype=GRB.BINARY, name="z")
    #Set objective function
    m.setObjective(gp.quicksum(s_b[i]*x[i] - w*z[i] for i in bundle_indexs) , GRB.MAXIMIZE)

    for i in bundle_indexs:
        m.addConstr(x[i] * gp.quicksum(x[j]*d_matrix[i][j]  for j in bundle_indexs) == 0)

    m.addConstrs(x[i] * phi_b[i] - z[i] <= min_pr for i in bundle_indexs)
    #풀이
    m.optimize()
    try:
        print('Obj val: %g' % m.objVal)
        res = []
        count = 0
        for val in m.getVars():
            if val.VarName[0] == 'x' and float(val.x) == 1.0:
                res.append(count)
            count += 1
        return res
    except:
        print('Infeasible')
        return []


def Bundle_selection_problem4(phi_b, D, s_b, lt_matrix,min_pr=0.05, obj_type = 'simple_max_s'):
    bundle_indexs = list(range(len(s_b)))
    try:
        if obj_type == 'max_s+probability':
            w = sum(s_b)/len(s_b)
        elif obj_type == 'over_lt+probability':
            w = sum(lt_matrix) / len(lt_matrix)
        else:
            w = 1
    except:
        w = 1
    m = gp.Model("mip1")
    x = m.addVars(len(bundle_indexs), vtype=GRB.BINARY, name="x")
    z = m.addVars(len(bundle_indexs), vtype=GRB.BINARY, name="z")
    #Set objective function
    if obj_type == 'simple_max_s':
        m.setObjective(gp.quicksum(s_b[i]*x[i] for i in bundle_indexs), GRB.MAXIMIZE)
    elif obj_type == 'max_s+probability':
        m.setObjective(gp.quicksum(s_b[i] * x[i] - w * z[i] for i in bundle_indexs), GRB.MAXIMIZE)
    elif obj_type == 'simple_over_lt':
        m.setObjective(gp.quicksum(lt_matrix[i] * x[i] for i in bundle_indexs), GRB.MAXIMIZE)
    elif obj_type == 'over_lt+probability':
        m.setObjective(gp.quicksum(lt_matrix[i] * x[i] - w * z[i] for i in bundle_indexs), GRB.MAXIMIZE)
    else:
        pass
    for info in D:
        m.addConstr(x[info[0]] + x[info[1]] <= 1)

    m.addConstrs(x[i] * phi_b[i] - z[i] <= min_pr for i in bundle_indexs)
    #풀이
    m.optimize()
    test = []
    count = 0
    for val in m.getVars():
        if val.VarName[0] == 'x' and float(val.x) == 1.0:
            test.append(phi_b[count])
    try:
        #input('선택된 번들 p_b 평균 : {} // p_b {} '.format(sum(test)/len(test), test))
        pass
    except:
        #input('선택된 번들 p_b 없음 : {}'.format(test))
        pass
    try:
        print('Obj val: %g' % m.objVal)
        res = []
        count = 0
        for val in m.getVars():
            if val.VarName[0] == 'x' and float(val.x) == 1.0:
                res.append(count)
            count += 1
        return res
    except:
        print('Infeasible')
        return []