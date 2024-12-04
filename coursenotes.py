#%%
import numpy as np
from scipy.optimize import minimize

F = 10  # Example total flow

def phi(f):
    x, y, z = f
    return (x**2) / 2 + 2 * x + (y**2) / 2 + (z**2) / 2 + 3 * (y - z)

# Flow conservation constraint
def flow_conservation(f):
    x, y, z = f
    return x + y - F

# Outlet flow condition: y = a + z
def outlet_flow_condition(f):
    x, y, z = f
    return y - z

# Bounds for flows
bounds = [(0, F), (0, F), (0, F)]  # Non-negative flows

# Constraints
constraints = [
    {"type": "eq", "fun": flow_conservation},  # Flow conservation
    {"type": "ineq", "fun": outlet_flow_condition},  # Flow distribution at B
]

# Initial guess for flows
initial_guess = [F / 4, F / 4, F / 4]  # Ensure constraints are approximately satisfied

# Minimize phi(f)
result = minimize(phi, initial_guess, bounds=bounds, constraints=constraints)

# Extract results
if result.success:
    x_eq, y_eq, z_eq = result.x
    print(f"Equilibrium flows:")
    print(f"  x (A->C via library café): {x_eq:.2f}")
    print(f"  y (A->B via JCR): {y_eq:.2f}")
    print(f"  z (B->C via outlet 2): {z_eq:.2f}")
    print(f"Minimum total cost (phi): {phi([x_eq, y_eq, z_eq]):.2f}")
else:
    print("Optimization failed.")
# %%

F = 10  

def ab(y):
    return y  # Cost for A -> B (JCR)

def ac(x):
    return x + 3  # Cost for A -> C (library café)

def bc1(z):
    return z  # Cost for B -> C (outlet 2)

def bc2(a):
    return 3 * np.ones(len(a)) # Constant cost for B -> C (outlet 1)

def phi(f):
    x, y, z, a = f

    y_range = np.linspace(0, y, 1000)
    x_range = np.linspace(0, x, 1000)
    z_range = np.linspace(0, z, 1000)
    a_range = np.linspace(0, a, 1000)

    intab = np.trapz(ab(y_range), y_range)   # Integral for A -> B (JCR)
    intac = np.trapz(ac(x_range), x_range)   # Integral for A -> C (library café)
    intbc1 = np.trapz(bc1(z_range), z_range)  # Integral for B -> C (outlet 2)
    intbc2 = np.trapz(bc2(a_range), a_range)  # Integral for B -> C (outlet 1)

    # Return the total cost
    return intab + intac + intbc1 + intbc2

# Flow conservation constraint
def flow_conservation(f):
    x, y, z, a = f
    return x + y - F

# Outlet flow condition: y = a + z
def outlet_flow_condition(f):
    x, y, z, a = f
    return y - (a + z)

# Bounds for flows
bounds = [(0, F), (0, F), (0, F), (0, F)]  # Non-negative flows

# Constraints
constraints = [
    {"type": "eq", "fun": flow_conservation},  # Flow conservation
    {"type": "eq", "fun": outlet_flow_condition},  # Flow distribution at B
]

# Initial guess for flows
initial_guess = [F / 4, F / 4, F / 4, F / 4]  # Ensure constraints are approximately satisfied

# Minimize phi(f)
result = minimize(phi, initial_guess, bounds=bounds, constraints=constraints)

# Extract results
if result.success:
    x_eq, y_eq, z_eq, a_eq = result.x
    print(f"Equilibrium flows:")
    print(f"  x (A->C via library café): {x_eq:.2f}")
    print(f"  y (A->B via JCR): {y_eq:.2f}")
    print(f"  z (B->C via outlet 2): {z_eq:.2f}")
    print(f"  a (B->C via outlet 1): {a_eq:.2f}")
    print(f"Minimum total cost (phi): {phi([x_eq, y_eq, z_eq, a_eq]):.2f}")
else:
    print("Optimization failed.")

# %%

F = 10  # Example total flow

def phi(f):
    x, y = f
    return x + 1 / 3 * x**3 + 2 * y + 1 / 3 * y **3

# def phi(f):
#     x, y = f

#     y_range = np.linspace(0, y, 1000)
#     x_range = np.linspace(0, x, 1000)

#     int1 = np.trapz(x_range + 1 / 3 * x_range**3, x_range)  # Integral for x
#     int2 = np.trapz(2 * y_range + 1 / 3 * y_range**3, y_range)  # Integral for y

#     return int1 + int2

# Flow conservation constraint
def flow_conservation(f):
    x, y= f
    return x + y - F

bounds = [(0, F), (0, F)]

# Constraints
constraints = [
    {"type": "eq", "fun": flow_conservation},  # Flow conservation
]

# Initial guess for flows
initial_guess = [F / 4, F / 4]

# Minimize phi(f)
result = minimize(phi, initial_guess, bounds=bounds, constraints=constraints)

# Extract results
if result.success:
    x_eq, y_eq = result.x
    print(f"Equilibrium flows:")
    print(f"  x: {x_eq:.2f}")
    print(f"  y: {y_eq:.2f}")
    print(f"Minimum total cost (phi): {phi([x_eq, y_eq]):.2f}")
else:
    print("Optimization failed.")

# %%
