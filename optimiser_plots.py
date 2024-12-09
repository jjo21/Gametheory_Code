#%%

import numpy as np
import matplotlib.pyplot as plt
import optimiser_funcs as of

#%%

# Example usage
A = np.diag([1, 1, 5, 3, 1.5, 20])
b = np.array([12, 23, 16, 14, 32, 10])
M = 100
x = of.shipping_analysis(A, b, M, kra=False, eq=True, flag=True)
print(of.route_cost(x, A, b))
print(of.average_cost(x, A, b, M))
of.POA(A, b, M)

# %%

'''
Changing toll value at Kra
'''
A = np.diag([1, 1, 5, 3, 1.5, 20])
M = 100

results_equi = []
average_costs_equi = []
results_soci = []
average_costs_soci = []

tolls = np.linspace(-100, 100, 200)

for toll in tolls: 
    b = np.array([12, 23, 16, 14, 32, 10 + toll])
    x_equi = of.shipping_analysis(A, b, M, kra=True, eq=True, flag=False)
    x_soci = of.shipping_analysis(A, b, M, kra=True, eq=False, flag=False)
    
    results_equi.append(x_equi)
    average_costs_equi.append(of.average_cost(x_equi, A, b, M))

    results_soci.append(x_soci)
    average_costs_soci.append(of.average_cost(x_soci, A, b, M))

results_equi = np.array(results_equi)
results_soci = np.array(results_soci)

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
fig.suptitle('Toll Factor Analysis with Kra Canal', fontsize=16)

axs[0].plot(tolls, results_equi[:, 2], label='Malacca', color='blue')
axs[0].plot(tolls, results_equi[:, 3], label='Sundra', color='orange')
axs[0].plot(tolls, results_equi[:, 4], label='Lombok', color='green')
axs[0].plot(tolls, results_equi[:, 5], label='Kra', color='red')

axs[0].plot(tolls, results_soci[:, 2], linestyle='--', color='blue')
axs[0].plot(tolls, results_soci[:, 3], linestyle='--', color='orange')
axs[0].plot(tolls, results_soci[:, 4], linestyle='--', color='green')
axs[0].plot(tolls, results_soci[:, 5], linestyle='--', color='red')
axs[0].legend(fontsize=14) 
axs[0].grid()
axs[0].set_xlabel('Toll Factor ($\omega$)', fontsize=14)
axs[0].set_ylabel('Flow', fontsize=14)

axs[1].plot(tolls, average_costs_equi, label='Equilibrium')
axs[1].plot(tolls, average_costs_soci, label='Social Optimum')
axs[1].set_xlabel('Toll Factor ($\omega$)', fontsize=14)
axs[1].set_ylabel('Average Cost', fontsize=14)
axs[1].legend(fontsize=14)
axs[1].grid()

plt.tight_layout()
plt.show()

plt.plot(tolls, np.array(average_costs_equi) - np.array(average_costs_soci))
plt.xlabel('Toll Factor ($\omega$)', fontsize=14)
plt.ylabel('Difference in Average Cost', fontsize=14)
plt.grid()
plt.show()

# %%

'''
Changing congestion factor at Kra
'''

b = np.array([12, 23, 16, 14, 32, 10])
M = 100

results_equi = []
average_costs_equi = []
results_soci = []
average_costs_soci = []

congestions = np.linspace(1, 30, 100)

for cong in congestions: 
    A = np.diag([1, 1, 5, 3, 1.5, cong])
    x_equi = of.shipping_analysis(A, b, M, kra=True, eq=True, flag=False)
    x_soci = of.shipping_analysis(A, b, M, kra=True, eq=False, flag=False)
    
    results_equi.append(x_equi)
    average_costs_equi.append(of.average_cost(x_equi, A, b, M))

    results_soci.append(x_soci)
    average_costs_soci.append(of.average_cost(x_soci, A, b, M))

results_equi = np.array(results_equi)
results_soci = np.array(results_soci)

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
fig.suptitle('Congestion Factor Analysis with Kra Canal', fontsize=16)

axs[0].plot(congestions, results_equi[:, 2], label='Malacca', color='blue')
axs[0].plot(congestions, results_equi[:, 3], label='Sundra', color='orange')
axs[0].plot(congestions, results_equi[:, 4], label='Lombok', color='green')
axs[0].plot(congestions, results_equi[:, 5], label='Kra', color='red')

axs[0].plot(congestions, results_soci[:, 2], linestyle='--', color='blue')
axs[0].plot(congestions, results_soci[:, 3], linestyle='--', color='orange')
axs[0].plot(congestions, results_soci[:, 4], linestyle='--', color='green')
axs[0].plot(congestions, results_soci[:, 5], linestyle='--', color='red')
axs[0].legend(fontsize=14)
axs[0].grid()
axs[0].set_xlabel('Congestion Factor ($\gamma$)', fontsize=14)
axs[0].set_ylabel('Flow', fontsize=14)

axs[1].plot(congestions, average_costs_equi, label='Equilibrium')
axs[1].plot(congestions, average_costs_soci, label='Social Optimum')
axs[1].set_xlabel('Congestion Factor ($\gamma$)', fontsize=14)
axs[1].set_ylabel('Average Cost', fontsize=14)
axs[1].legend(fontsize=14)
axs[1].grid()

plt.tight_layout()
plt.show()

plt.plot(congestions, np.array(average_costs_equi) - np.array(average_costs_soci))
plt.xlabel('Congestion Factor ($\gamma$)', fontsize=14)
plt.ylabel('Difference in Average Cost', fontsize=14)
plt.grid()
plt.show()

# %%

# plot contour plot for POA

congestions = np.linspace(1, 30, 10)
tolls = np.linspace(-100, 100, 10)
M = 100

X, Y = np.meshgrid(congestions, tolls)
Z = np.zeros((len(congestions), len(tolls)))

for i, width in enumerate(congestions):
    for j, toll in enumerate(tolls):
        A = np.diag([1, 1, 5, 3, 1.5, width])
        b = np.array([12, 23, 16, 14, 32, 10 + toll])
        POA_value = of.POA(A, b, M)
        Z[i, j] = POA_value

fig, ax = plt.subplots(figsize=(8, 8))
c = ax.contourf(X, Y, Z, levels=20)
fig.colorbar(c)
ax.set_xlabel('Congestion Factor ($\gamma$)', fontsize=14)
ax.set_ylabel('Toll Factor ($\omega$)', fontsize=14)
plt.show()

# %%
