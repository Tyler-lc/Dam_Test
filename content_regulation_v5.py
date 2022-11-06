# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:22:47 2022

@author: lucas
"""

import pyomo.core as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




def optimization_model():
    
    m = pyo.ConcreteModel()

    
    m.cap_max = 1000000 #m3
    m.cap_min = 0 #m3
    m.cap_0 = 700000 #m3
    m.cap_target = 300000 #m3
    
    m.out_max = 30*3600 #m3/h
    m.out_min = 10*3600 #m3/h
    
    m.inflow = pd.read_csv("inflow2.csv", index_col = [0])["inflow"]
    m.timesteps = pd.read_csv("inflow2.csv")["Time"]
    
    #sets
    
    m.t = pyo.Set(initialize = m.timesteps, ordered = True)
    
    #Variables
    
    m.power_out = pyo.Var(m.t, within = pyo.NonNegativeReals)
    m.capacity = pyo.Var(m.t, within=pyo.NonNegativeReals)
    m.costs = pyo.Var(m.t, within=pyo.NonNegativeReals)
    
    #constraints
    
    def capacity_current_rule(m, t):
        if t == m.timesteps[0]:
            return (m.capacity[t] == float(m.cap_0) + m.inflow[t] - m.power_out[t])
        else:
            return (m.capacity[t] == m.capacity[t-1] + m.inflow[t] - m.power_out[t])
            
    m.capacity_current = pyo.Constraint(m.t, rule = capacity_current_rule)
    
    
    def capacity_max_rule(m, t):
        return (m.capacity[t] <= float(m.cap_max))
    m.capacity_max = pyo.Constraint(m.t, rule = capacity_max_rule)
    
    
    def outflow_max_rule(m, t):
        return (m.power_out[t] <= float(m.out_max))
    
    m.outflow_max = pyo.Constraint(m.t, rule = outflow_max_rule)
    
    
    def outflow_min_rule(m, t):
        return m.power_out[t] >= float(m.out_min)
    
    m.outflow_min = pyo.Constraint(m.t, rule = outflow_min_rule)
    
    #Cost functions
    
    def costs_rule(m, t):
        return m.costs[t] == m.capacity[t] - float(m.cap_target)
    
    m.costs_func = pyo.Constraint(m.t, rule = costs_rule)
    
    
    def objective_rule(m):
        return pyo.summation(m.costs)
    
    m.objectivefunc = pyo.Objective(rule = objective_rule, sense = pyo.minimize)
    
    return m


#optimisation starts
m = optimization_model()    
m.objectivefunc.activate()
optim = SolverFactory('glpk')
optim.solve(m, tee=True)

#plotting
print(m.power_out.extract_values())
print(m.capacity.extract_values())

power_dict = dict(m.power_out.extract_values())
capacity_dict = dict(m.capacity.extract_values())  

capacity_plot = list(capacity_dict.values())    
power_plot = list(power_dict.values())

inflow_plot = pd.read_csv("inflow2.csv", index_col = [0])["inflow"]

sns.set_style("darkgrid")
plt.plot(capacity_plot)
plt.show()

plt.plot(power_plot)
plt.plot(inflow_plot)
plt.legend(labels=["outflow", "inflow"])
plt.show()
    
    