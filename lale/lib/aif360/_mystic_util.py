from mystic.solvers import diffev2
from mystic.coupler import and_
from mystic.penalty import quadratic_equality
import numpy as np


def create_ci_penalties(n_ci):
    C = n_ci.shape[0]+1
    ci_penalties = []

    # specify C-1 class imbalance constraints as heavy penalties
    for k in range(C-1):
        def condition(x):
            reshape_list = []
            for k in range(A):
                reshape_list.append(2)
            reshape_list.append(C)
            ndx = np.array(x).reshape(reshape_list)
            return (np.sum(ndx[(slice(None),) * A + (k,)])/np.sum(ndx[(slice(None),) * A + (k+1,)])) - n_ci[k,0]

        @quadratic_equality(condition,k=1e6,h=10)
        def penalty(x):
            return 0
        ci_penalties.append(penalty)
    return ci_penalties


def create_di_penalties(n_di, F):
    A = n_di.shape[0]
    di_penalties = []

    # specify A disparate imapct ratio constraints as heavy penalties
    for k in range(A):
        def condition(x):
            reshape_list = []
            for k in range(A):
                reshape_list.append(2)
            reshape_list.append(C)
            ndx = np.array(x).reshape(reshape_list)
            di_ratio_top = np.sum(ndx[(slice(None),) * k + (0,) + (slice(None),) * (A-k-1) + (tuple(F),)]) / np.sum(ndx[(slice(None),) * k + (0,) + (slice(None),) * (A-k-1) + (slice(None),)])
            di_ratio_bottom = np.sum(ndx[(slice(None),) * k + (1,) + (slice(None),) * (A-k-1) + (tuple(F),)]) / np.sum(ndx[(slice(None),) * k + (1,) + (slice(None),) * (A-k-1) + (slice(None),)])
            return (di_ratio_top / di_ratio_bottom) - n_di[k,0]

        @quadratic_equality(condition,k=1e6,h=10)
        def penalty(x):
            return 0
        di_penalties.append(penalty)

def calc_oversample_soln(o_flat, F, n_ci, n_di):
    # integer constraint
    ints = np.round

    # minimize sum of new number of examples
    def cost(x):
        return np.sum(x)

    # specify observed example counts as lower bounds
    bounds = list(map(lambda x: (x, float('inf')), o_flat))

    # combine all penalties
    ci_penalties = create_ci_penalties(n_ci)
    di_penalties = create_di_penalties(n_di, F)
    all_penalties = and_(*ci_penalties, *di_penalties)

    # integer constraint
    constraint = ints

    # pass to solver
    result = diffev2(cost, x0=o_flat, bounds=bounds, constraints=constraint, penalty=all_penalties, full_output=False, disp=False, npop=50, gtol=100,)
    return result[0]

def calc_undersample_soln(o_flat, F, n_ci, n_di):
    # integer constraint
    ints = np.round

    # minimize negative sum of new number of examples (equivalent to maximizing positive sum)
    def cost(x):
        return -np.sum(x)

    # specify observed example counts as upper bounds and 0 as lower bounds
    bounds = list(map(lambda x: (0, x), o_flat))

    # combine all penalties
    ci_penalties = create_ci_penalties(n_ci)
    di_penalties = create_di_penalties(n_di, F)
    all_penalties = and_(*ci_penalties, *di_penalties)

    # integer constraint
    constraint = ints

    # pass to solver
    result = diffev2(cost, x0=o_flat, bounds=bounds, constraints=constraint, penalty=all_penalties, full_output=False, disp=False, npop=50, gtol=100,)
    return result[0]