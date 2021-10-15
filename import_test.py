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

os.system('pause')