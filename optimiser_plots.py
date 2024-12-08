#%%
import scipy as sp
import numpy as np
from numbers import Number
import matplotlib.pyplot as plt
import copy
#%%

def shipping_analysis(A, b, M, kra=True, eq=True, flag=True):
    """
    Compute minimising solution for relevant optimisation problems related to
    the congestion network as defined in the coursework.
    ===
    Inputs:
    A (numpy.ndarray) : Congestion gradient coefficients of cost functions in a
                        2D matrix. Must be a diagonal matrix.
    b (numpy.ndarray) : Constant coefficients of cost functions in a 1D vector.
    M (numbers.Number): Total mass in the network.
    kra (bool)        : Determine whether to include the Kra Canal or not.
    eq (bool)         : Determine optimisation problem to solve: equilibrium
                        solution if True, social optimum if False.
    flag (bool)       : if True, will print information about solution.
    ---
    Output:
    constrained (numpy.ndarray): Flow that minimises the optimisation problem.
    ===
    """
    if not (isinstance(A, np.ndarray) and len(A.shape) == 2
            and isinstance(b, np.ndarray) and len(b.shape) == 1
            and isinstance(M, Number)):
        raise TypeError("A has to be a 2D numpy array, b a 1D numpy array"
                        "and M a number.")
    
    if kra:
        A_w = copy.deepcopy(A)
        b_w = copy.deepcopy(b)
        C = np.array([[1, 1, 0, 0, 0, 0],
                      [-1, 0, 1, 0, 0, 1],
                      [0, -1, 0, 1, 1, 0]])
    else:
        A_w = copy.deepcopy(A[:-1, :-1])
        b_w = copy.deepcopy(b[:-1])
        C = np.array([[1, 1, 0, 0, 0],
                      [-1, 0, 1, 0, 0],
                      [0, -1, 0, 1, 1]])
    d = np.array([M, 0, 0])
    if not eq:
        A_w *= 2

    # Unconstrained solution
    A_1 = np.diag(1/np.diag(A_w))
    un = -A_1 @ b_w
    
    # Constrained optimization with non-negativity and equality constraints
    def objective(x):
        return 0.5 * x @ A_w @ x + b_w @ x
    
    def equality_constraints(x):
        return C @ x - d

    constraints = [{'type': 'eq', 'fun': equality_constraints}]
    bounds = [(0, None) for _ in range(len(b_w))]  # Non-negativity constraints
    result = sp.optimize.minimize(objective, un,
                                  bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError("Optimization failed to converge.")
    constrained = np.round(result.x, 2)
    if flag:
        if eq:
            text = "Equilibrium Analysis"
        else:
            text = "Social Optimum Analysis"
        if kra:
            text += " (With Kra Canal)"
        else:
            text += " (Without Kra Canal)"
        print(text)
        print(f"Optimal Solution: {constrained}")
    
    return constrained

def route_cost(x, A, b):
    """
    Compute the total cost of traversing each of the routes of the
    congestion network from Node A to Node D, given flow x.
    ===
    Inputs:
    x (numpy.ndarray): Flow of system. Must be a 5 or 6 dimensional vector.
    ---
    Outputs:
    c (numpy.ndarray): Cost of each route.
    ===
    """
    if len(x) == 6:
        A_w = copy.deepcopy(A)
        b_w = copy.deepcopy(b)
        r = np.array([[1, 0, 0, 0, 0, 1],
                      [1, 0, 1, 0, 0, 0],
                      [0, 1, 0, 1, 0, 0],
                      [0, 1, 0, 0, 1, 0]])
    elif len(x) == 5:
        A_w = copy.deepcopy(A[:-1, :-1])
        b_w = copy.deepcopy(b[:-1])
        r = np.array([[1, 0, 1, 0, 0],
                      [0, 1, 0, 1, 0],
                      [0, 1, 0, 0, 1]])
    else:
        raise ValueError('x needs to be 5 or 6 dimensional.')
    return np.round(r @ (A_w @ x + b_w), 2)

def average_cost(x, A, b, M):
    """
    Compute the average cost of any given user of the congestion network
    from Node A to Node D, given flow x.
    ===
    Inputs:
    x (numpy.ndarray): Flow of system. Must be a 5 or 6 dimensional vector.
    ---
    Outputs:
    c (numpy.ndarray): Average cost per single user of network.
    ===
    """
    if len(x) == 6:
        A_w = copy.deepcopy(A)
        b_w = copy.deepcopy(b)
    elif len(x) == 5:
        A_w = copy.deepcopy(A[:-1, :-1])
        b_w = copy.deepcopy(b[:-1])
    return np.round((x.T @ A_w @ x + b_w .T @ x) / M, 2)

def POA(A, b, M):
    """
    Calculate Price of Anarchy of the congestion network.
    ===
    Inputs:
    x_bar (numpy.ndarray):   Flow of system in equilibrium state.
    x_tilde (numpy.ndarray): Flow of system in social optimum.
    ---
    Outputs:
    c (numpy.ndarray): Average cost per single user of network.
    ===
    """

    x_bar = shipping_analysis(A, b, M, kra=False, eq=True, flag=False)
    x_tilde = shipping_analysis(A, b, M, kra=False, eq=False, flag=False)
    print(x_bar)
    print(x_tilde)

    # if x_bar.shape != x_tilde.shape:
    #     raise ValueError("x_bar and x_tilde must be vectors of the same size.")
    return average_cost(x_bar, A, b, M)/average_cost(x_tilde, A, b, M)

#%%

# Example usage
A = np.diag([1, 1, 5, 3, 1.5, 20])
b = np.array([12, 23, 16, 14, 32, 10])
M = 100
x = shipping_analysis(A, b, M, kra=False, eq=False, flag=True)
print(x[2] * A[2, 2] + b[2] + x[0] + b[0])
print(x[3] * A[3, 3] + b[3] + x[1] + b[1])
print(x[4] * A[4, 4] + b[4] + x[1] + b[1])
print(route_cost(x, A, b))
print(average_cost(x, A, b, M))
POA(A, b, M)

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

tolls = np.linspace(-50, 50, 100)

for toll in tolls: 
    b = np.array([12, 23, 16, 14, 32, 10 + toll])
    x_equi = shipping_analysis(A, b, M, kra=True, eq=True, flag=False)
    x_soci = shipping_analysis(A, b, M, kra=True, eq=False, flag=False)
    
    results_equi.append(x_equi)
    average_costs_equi.append(average_cost(x_equi, A, b, M))

    results_soci.append(x_soci)
    average_costs_soci.append(average_cost(x_soci, A, b, M))

results_equi = np.array(results_equi)
results_soci = np.array(results_soci)

# generate 2 plots side by side 

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
fig.suptitle('Toll Factor Analysis with Kra Canal', fontsize=16)

# share an x axis

# plot 1
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

# plot 2
axs[1].plot(tolls, average_costs_equi, label='Equilibrium')
axs[1].plot(tolls, average_costs_soci, label='Social Optimum')
axs[1].set_xlabel('Toll Factor ($\omega$)', fontsize=14)
axs[1].set_ylabel('Average Cost', fontsize=14)
axs[1].legend(fontsize=14)
axs[1].grid()

plt.tight_layout()
plt.show()

# %%

'''
Changing congestion value at Kra
'''
b = np.array([12, 23, 16, 14, 32, 10 + toll])
M = 100

results_equi = []
average_costs_equi = []
results_soci = []
average_costs_soci = []

widths = np.linspace(1, 10, 100)

for width in widths: 
    A = np.diag([1, 1, 5, 3, 1.5, width])
    x_equi = shipping_analysis(A, b, M, kra=True, eq=True, flag=False)
    x_soci = shipping_analysis(A, b, M, kra=True, eq=False, flag=False)
    
    results_equi.append(x_equi)
    average_costs_equi.append(average_cost(x_equi, A, b, M))

    results_soci.append(x_soci)
    average_costs_soci.append(average_cost(x_soci, A, b, M))

results_equi = np.array(results_equi)
results_soci = np.array(results_soci)

# generate 2 plots side by side 

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
fig.suptitle('Congestion Factor Analysis with Kra Canal', fontsize=16)

# share an x axis

# plot 1
axs[0].plot(widths, results_equi[:, 2], label='Malacca', color='blue')
axs[0].plot(widths, results_equi[:, 3], label='Sundra', color='orange')
axs[0].plot(widths, results_equi[:, 4], label='Lombok', color='green')
axs[0].plot(widths, results_equi[:, 5], label='Kra', color='red')

axs[0].plot(widths, results_soci[:, 2], linestyle='--', color='blue')
axs[0].plot(widths, results_soci[:, 3], linestyle='--', color='orange')
axs[0].plot(widths, results_soci[:, 4], linestyle='--', color='green')
axs[0].plot(widths, results_soci[:, 5], linestyle='--', color='red')
axs[0].legend(fontsize=14)
axs[0].grid()
axs[0].set_xlabel('Congestion Factor ($\gamma$)', fontsize=14)
axs[0].set_ylabel('Flow', fontsize=14)

# plot 2
axs[1].plot(widths, average_costs_equi, label='Equilibrium')
axs[1].plot(widths, average_costs_soci, label='Social Optimum')
axs[1].set_xlabel('Congestion Factor ($\gamma$)', fontsize=14)
axs[1].set_ylabel('Average Cost', fontsize=14)
axs[1].legend(fontsize=14)
axs[1].grid()

plt.tight_layout()
plt.show()

# %%

# plot contour plot for 

widths = np.linspace(1, 10, 100)
tolls = np.linspace(-50, 50, 100)

# plot the contour plot and find the minimum of this function, using meshgrid

X, Y = np.meshgrid(widths, tolls)
Z = np.zeros((len(widths), len(tolls)))

for i, width in enumerate(widths):
    for j, toll in enumerate(tolls):
        A = np.diag([1, 1, 5, 3, 1.5, width])
        b = np.array([12, 23, 16, 14, 32, 10 + toll])
        x = shipping_analysis(A, b, M, kra=True, eq=True, flag=False)
        Z[i, j] = average_cost(x, A, b, M)

fig, ax = plt.subplots(figsize=(8, 8))
c = ax.contourf(X, Y, Z, levels=20)
fig.colorbar(c)
ax.set_xlabel('Width')
ax.set_ylabel('Toll')
plt.show()

# %%
