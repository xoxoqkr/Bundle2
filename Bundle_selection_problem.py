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
    customer_index = list(range(len(C_b_len)))
    m = gp.Model("mip1")
    y = m.addVars(len(F), vtype=GRB.BINARY, name="y")
    x = m.addVars(len(F),len(C_b), vtype=GRB.BINARY, name="x")

    m.setObjective(gp.quicksum(s[i]*y[i] for i in bundle_index), GRB.MAXIMIZE)
    for i in bundle_index:
        m.addConstr(y[i] <= x[i,j] for j in C_b_len[i])

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


