# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
import os


def func1():
    # Create a new model
    m = gp.Model("mip1")
    # Create variables
    x = m.addVar(vtype=GRB.BINARY, name="x")
    y = m.addVar(vtype=GRB.BINARY, name="y")
    z = m.addVar(vtype=GRB.BINARY, name="z")
    # Set objective
    m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)
    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
    # Add constraint: x + y >= 1
    m.addConstr(x + y >= 1, "c1")
    # Optimize model
    m.optimize()
    return m

def func2(length):
    # Create a new model
    m = gp.Model("mip1")
    # Create variables
    x = m.addVars(length, vtype=GRB.BINARY, name="x")
    # Set objective
    m.setObjective(gp.quicksum(x), GRB.MAXIMIZE)
    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(x[0] + 2 * x[1] + 3 * x[2] <= 4, "c0")
    # Add constraint: x + y >= 1
    m.addConstr(x[0] + x[1] >= 1, "c1")
    # Optimize model
    m.optimize()
    return m

def func2(length):
    """
    단순한 목적식 함수를 만들고, i번째 여유 변수를 만든다.
    @param length: 목적식 결정변수 수
    @return: None
    """
    # Create a new model
    m = gp.Model("mip1")
    # Create variables
    x = m.addVars(length, vtype=GRB.BINARY, name="x")
    # Set objective
    m.setObjective(gp.quicksum(x), GRB.MAXIMIZE)
    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(x[0] + 2 * x[1] + 3 * x[2] <= 4, "c0")
    # Add constraint: x + y >= 1
    m.addConstr(x[0] + x[1] >= 1, "c1")
    # Optimize model
    m.optimize()
    return m

def func3(length, i):
    """
    단순한 목적식 함수를 만들고, i번째 여유 변수를 만든다.
    @param length: 목적식 결정변수 수
    @param i: 여유 공간
    @return:
    """
    # Create a new model
    m = gp.Model("mip1")
    # Create variables
    x = m.addVars(length, vtype=GRB.CONTINUOUS, name="x")
    y = m.addVars(i, vtype=GRB.CONTINUOUS, name="y")
    # Set objective
    m.setObjective(gp.quicksum(x) + gp.quicksum(y), GRB.MAXIMIZE)
    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(x[0] + 2 * x[1] + 3 * x[2] <= 4, "c0")
    # Add constraint: x + y >= 1
    m.addConstr(x[0] + x[1] >= 1, "c1")
    for index in list(range(i)):
        m.addConstr(y[index] <= 3)
    for index in list(range(i)):
        m.addConstr(y[index] == 0)
    # Optimize model
    m.optimize()
    return m



model = func2(3)
coeff = [2,2,2]
coeff2 = [1,1,1]
range1 = [0,1,2]
model.addConstr(gp.quicksum(model.x[i] + coeff[i] for i in range1) >= gp.quicksum(model.x[i] + coeff2[i] for i in range1))
#model의 변수를 지정해 주고 갱신해야 함.
#-> ojective function에 새로운 변수를 넣는 것은 불가능. 대신 꼼수 가능.
#->dummy y_i를 생성. y_i == 0 제약식 추가 ->새로운 데이터에 대해 모델을 갱신할 때 마다, y_i == 0 제약식을 제거.
# model.remove(model.getConstrs()[len(w) + i])
# model.reset()
# model.optimize()
#-> i in I 가 다 되면, 모델을 새로 짜야함. 혹은 binding constraint만으로 다시 문제를 정의.
model.update()
os.system('pause')