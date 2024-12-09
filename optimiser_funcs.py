import scipy as sp
import numpy as np
from numbers import Number
import matplotlib.pyplot as plt
import copy

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
    A (numpy.ndarray) : Congestion gradient coefficients of cost functions in a
                        2D matrix. Must be a diagonal matrix.
    b (numpy.ndarray) : Constant coefficients of cost functions in a 1D vector.
    M (numbers.Number): Total mass in the network.
    ---
    Outputs:
    c (numpy.ndarray): Average cost per single user of network.
    ===
    """

    x_bar = shipping_analysis(A, b, M, kra=True, eq=True, flag=False)
    x_tilde = shipping_analysis(A, b, M, kra=True, eq=False, flag=False)
    return average_cost(x_bar, A, b, M)/average_cost(x_tilde, A, b, M)

