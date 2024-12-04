#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#%%

'''
First example w/ no kra
'''

F = 100

I1b = 10 # Initial 1 Base Cost

Mb = 10 # Malacca Base 
Mc = 30 # Malacca Congestion 

Sb = 20 # Sundra Base
Sc = 20 # Sundra Congestion

Lb = 30 # Lombok Base
Lc = 10 # Lombok Congestion

def phi(f):
    a, b, c = f
    malacca = Mb * b + Mc/2 * b**2

    sundra = Sb * (a-c) + Sc/2 * (a-c)**2
    lombok = Lb * c + Lc/2 * c**2
    return  malacca + sundra + lombok + I1b * a

# Flow conservation constraint
def flow_conservation(f):
    a, b, c = f
    return a + b - F

# Outlet flow condition a > c
def outlet_flow_condition(f):
    a, b, c = f
    return a - c

# Bounds for flows
bounds = [(0, F), (0, F), (0, F)]  # Non-negative flows

# Constraints
constraints = [
    {"type": "eq", "fun": flow_conservation},  # Flow conservation
    {"type": "ineq", "fun": outlet_flow_condition},  # Flow distribution after Initial 1
]

# Initial guess for flows
initial_guess = [F / 3, F / 3, F / 3]  # Ensure constraints are approximately satisfied

#%%

'''
No Toll at Malacca Case
'''

# Minimize phi(f)
result = minimize(phi, initial_guess, bounds=bounds, constraints=constraints)

# Extract results
if result.success:
    a_eq, b_eq, c_eq = result.x
    print(f"Equilibrium flows:")
    print(f"  b (Malacca): {b_eq:.2f}")
    print(f"  a (Initial 1): {a_eq:.2f}")
    print(f"  a-c (Sundra): {a_eq - c_eq:.2f}")
    print(f"  c (Lombok): {c_eq:.2f}")
    print(f"Minimum total cost (phi): {phi([a_eq, b_eq, c_eq]):.2f}")
else:
    print("Optimization failed.")

# %%

'''
Toll at Malacca Case
'''

# change one variable, i.e. reduce tolls at malacca to see how the ships change direction 

Ts = np.linspace(-1000, 100, 1000)
Mc = 30
results = []

x_var = Ts

for toll in Ts:
    def phi(f):
        a, b, c = f
        malacca = Mb * b + Mc/2 * b**2 + toll * b
        sundra = Sb * (a-c) + Sc/2 * (a-c)**2
        lombok = Lb * c + Lc/2 * c**2
        return  malacca + sundra + lombok + I1b * a
    
    result = minimize(phi, initial_guess, bounds=bounds, constraints=constraints)
    results.append(result.x)

#%%

'''
Changing Congestion Constant at Malacca
'''

Mcs = np.linspace(0, 100, 1000)
toll = 0
results = []

x_var = Mcs

for Mc in Mcs:
    def phi(f):
        a, b, c = f
        malacca = Mb * b + Mc/2 * b**2 + toll * b
        sundra = Sb * (a-c) + Sc/2 * (a-c)**2
        lombok = Lb * c + Lc/2 * c**2
        return  malacca + sundra + lombok + I1b * a
    
    result = minimize(phi, initial_guess, bounds=bounds, constraints=constraints)
    results.append(result.x)

# %%

# Plot the results

results = np.array(results)

plt.plot(x_var, results[:, 0], label='Initial 1')
plt.plot(x_var, results[:, 1], label='Malacca')
plt.plot(x_var, results[:, 2], label='Lombok')
plt.plot(x_var, results[:, 0] - results[:, 2], label='Sundra')
plt.xlabel('x_var')
plt.ylabel('Flow')
plt.legend()
plt.show()

# %%

