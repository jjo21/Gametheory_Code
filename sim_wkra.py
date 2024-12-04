#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#%%

'''
First example 
'''

F = 100

I1b = 10 # Initial 1 Base Cost
I2b = 5 # Initial 2 Base Cost

Mb = 10 # Malacca Base 
Mc = 30 # Malacca Congestion 

Kb = 5 # Kra Base
Kc = 20 # Kra Congestion

Sb = 20 # Sundra Base
Sc = 20 # Sundra Congestion

Lb = 30 # Lombok Base
Lc = 10 # Lombok Congestion

T = 100 # Toll at Kra Canal

def phi(f):
    a, b, c, d = f
    malacca = Mb * (b - d) + Mc/2 * (b - d)**2
    kra = Kb * d + Kc/2 * d**2 + T * d

    sundra = Sb * (a-c) + Sc/2 * (a-c)**2 
    lombok = Lb * c + Lc/2 * c**2
    return  kra + malacca + sundra + lombok + I1b * a + I2b * b

# Flow conservation constraint
def flow_conservation(f):
    a, b, c, d = f
    return a + b - F

# Outlet flow condition a > c
def outlet_flow_condition(f):
    a, b, c, d = f
    return a - c

def outlet_flow_condition2(f):
    a, b, c, d = f
    return b - d

# Bounds for flows
bounds = [(0, F), (0, F), (0, F), (0, F)]  # Non-negative flows

# Constraints
constraints = [
    {"type": "eq", "fun": flow_conservation},  # Flow conservation
    {"type": "ineq", "fun": outlet_flow_condition},  # Flow distribution after Initial 1
    {"type": "ineq", "fun": outlet_flow_condition2},  # Flow distribution after Initial 2
]

# Initial guess for flows
initial_guess = [F / 4, F / 4, F / 4, F / 4]  # Ensure constraints are approximately satisfied

#%%

'''
W Toll at Kra Canal
'''

# Minimize phi(f)
result = minimize(phi, initial_guess, bounds=bounds, constraints=constraints)

# Extract results
if result.success:
    a_eq, b_eq, c_eq, d_eq = result.x
    print(f"Equilibrium flows:")
    print(f"  a (Initial 1): {a_eq:.2f}")
    print(f"  b (Initial 2): {b_eq:.2f}")
    print(f"  c (Lombok): {c_eq:.2f}")
    print(f"  d (Kra): {d_eq:.2f}")
    print(f"  a-c (Sundra): {a_eq - c_eq:.2f}")
    print(f"  b-d (Malacca): {b_eq - d_eq:.2f}")
    print(f"Minimum total cost (phi): {phi([a_eq, b_eq, c_eq, d_eq]):.2f}")
else:
    print("Optimization failed.")

# %%

'''
Toll at Kra Canal
'''

# change one variable, i.e. reduce tolls at malacca to see how the ships change direction 

Ts = np.linspace(-1000, 1000, 1000)
results = []

for toll in Ts:
    def phi(f):
        a, b, c, d = f
        malacca = Mb * (b - d) + Mc/2 * (b - d)**2
        kra = Kb * d + Kc/2 * d**2 + toll * d

        sundra = Sb * (a-c) + Sc/2 * (a-c)**2 
        lombok = Lb * c + Lc/2 * c**2
        return  kra + malacca + sundra + lombok + I1b * a + I2b * b
    
    result = minimize(phi, initial_guess, bounds=bounds, constraints=constraints)
    results.append(result.x)

#%%

# Plot the results

results = np.array(results)

plt.plot(Ts, results[:, 0], label="Initial 1")
plt.plot(Ts, results[:, 1], label="Initial 2")
plt.plot(Ts, results[:, 2], label="Lombok")
plt.plot(Ts, results[:, 3], label="Kra")
plt.plot(Ts, results[:, 0] - results[:, 2], label="Sundra")
plt.plot(Ts, results[:, 1] - results[:, 3], label="Malacca")
plt.xlabel("Toll at Kra Canal")
plt.ylabel("Flow")
plt.legend()
plt.show()
# %%
