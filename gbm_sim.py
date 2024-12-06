#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#%%
# Parameters
np.random.seed(43)  # For reproducibility
T = 1  # Time horizon in years
N = 252  # Number of time steps (trading days in a year)
dt = T / N  # Time step
mu = 0.1  # Drift coefficient
sigma = 0.2  # Volatility
S0 = 100  # Initial stock price
rho = 0.8  # Correlation coefficient

# Generate correlated random components
mean = [0, 0]
cov = [[1, rho], [rho, 1]]
normal_samples = np.random.multivariate_normal(mean, cov, N)

# First correlated series
W1 = np.cumsum(normal_samples[:, 0]) * np.sqrt(dt)
S1 = S0 * np.exp((mu - 0.5 * sigma**2) * np.linspace(0, T, N) + sigma * W1)

# Second correlated series
W2 = np.cumsum(normal_samples[:, 1]) * np.sqrt(dt)
S2 = S0 * np.exp((mu - 0.5 * sigma**2) * np.linspace(0, T, N) +  sigma * W2)

# Uncorrelated series
uncorrelated_noise = np.random.normal(0, np.sqrt(dt), N).cumsum()
S_uncorrelated = S0 * np.exp((mu - 0.5 * sigma**2) * np.linspace(0, T, N) + sigma * uncorrelated_noise)

# Plot the series
time = np.linspace(0, T, N)

plt.figure(figsize=(12, 8))
plt.plot(range(0, len(time)), S1, label="Price 1 (Correlated)")
plt.plot(range(0, len(time)), S2, label="Price 2 (Correlated)")
plt.plot(range(0, len(time)), S_uncorrelated, label="Price (Uncorrelated)", linestyle="--")
plt.title("Correlated and Uncorrelated Price Series")
plt.xlabel("Time (Years)")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# %%

'''
First example w/ no kra
'''

F = 100

loc1_dist = 100
loc2_dist = 150
loc3_dist = 180

price1 = S1[0]
price2 = S2[1]
price3 = S_uncorrelated[2]

def phi(f):
    a, b, c = f

    loc1 = 0.5* a** 2 + price1 * a
    loc2 = 0.5* b** 2 + price2 * b
    loc3 = 0.5* c** 2 + price3 * c

    return  loc1 + loc2 + loc3

# Flow conservation constraint
def flow_conservation(f):
    a, b, c = f
    return a + b + c - F

# Bounds for flows
bounds = [(0, F), (0, F), (0, F)]  # Non-negative flows

# Constraints
constraints = [
    {"type": "eq", "fun": flow_conservation},  # Flow conservation
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
    print(f"  a (Loc1): {a_eq:.2f}")
    print(f"  b (Loc2): {b_eq:.2f}")
    print(f"  c (Loc3): {c_eq:.2f}")
    print(f"Minimum total cost (phi): {phi([a_eq, b_eq, c_eq]):.2f}")
else:
    print("Optimization failed.")

# %%

'''
Price Motions
'''

# change one variable, i.e. reduce tolls at malacca to see how the ships change direction 

results = []

for i in range(len(S1)):
    def phi(f):
        a, b, c = f

        loc1 = 0.5* a** 2 + S1[i] * a
        loc2 = 0.5* b** 2 + S2[i] * b
        loc3 = 0.5* c** 2 + S_uncorrelated[i] * c
        return  loc1 + loc2 + loc3
    
    result = minimize(phi, initial_guess, bounds=bounds, constraints=constraints)
    results.append(result.x)

# %%

# plot two graphs one above and one below, one for the results and one for the prices 

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax[0].plot(range(0, len(time)), [r[0] for r in results], label="Loc1")
ax[0].plot(range(0, len(time)), [r[1] for r in results], label="Loc2")
ax[0].plot(range(0, len(time)), [r[2] for r in results], label="Loc3")

ax[0].set_title("Optimal Flows")
ax[0].set_ylabel("Flow")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(range(0, len(time)), S1, label="Price 1 (Correlated)")
ax[1].plot(range(0, len(time)), S2, label="Price 2 (Correlated)")
ax[1].plot(range(0, len(time)), S_uncorrelated, label="Price (Uncorrelated)", linestyle="--")
ax[1].set_title("Correlated and Uncorrelated Price Series")
ax[1].set_xlabel("Time (Days)")
ax[1].set_ylabel("Price")
ax[1].legend()
ax[1].grid(True)

plt.show()
# %%
