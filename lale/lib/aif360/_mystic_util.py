from typing import Dict, Set

import numpy as np
from mystic.coupler import and_
from mystic.penalty import quadratic_equality
from mystic.solvers import diffev2


def parse_solver_soln(n_flat, group_mapping):
    sorted_osize_keys = sorted(group_mapping.keys())
    mapped_nsize_tups = list(zip(sorted_osize_keys, n_flat))
    mapped_nsize_dict = {k: v for (k, v) in mapped_nsize_tups}
    nsizes = {g2: mapped_nsize_dict[g1] for g1, g2 in group_mapping.items()}
    return nsizes


def obtain_solver_info(
    osizes: Dict[str, int],
    imbalance_repair_level: float,
    bias_repair_level: float,
    favorable_labels: Set[int],
):
    # get class counts
    class_count_dict = {}
    for k, v in osizes.items():
        c = k[-1]
        if c not in class_count_dict:
            class_count_dict[c] = 0
        class_count_dict[c] += v

    # sorting by class count ensures that ci ratios will be <= 1
    sorted_by_count = sorted(class_count_dict.items(), key=lambda x: x[1])
    oci = []
    for i in range(len(sorted_by_count) - 1):
        oci.append(sorted_by_count[i][1] / sorted_by_count[i + 1][1])

    # if any class reordering has happened, update the group mapping and favorable_labels (for di calculations) accordingly
    class_mapping = {old: new for new, (old, _) in enumerate(sorted_by_count)}
    group_mapping = {k: k for k in osizes.keys()}
    for old, new in class_mapping.items():
        if int(old) in favorable_labels:
            favorable_labels.remove(int(old))
            favorable_labels.add(int(new))
        old_groups = list(filter(lambda x: x[-1] == old, group_mapping.keys()))
        for g in old_groups:
            group_mapping[g] = group_mapping[g][:-1] + str(new)

    mapped_osizes = {k1: osizes[k2] for k1, k2 in group_mapping.items()}

    # calculate di ratios and invert if needed
    odi = []
    num_prot_attr = len(list(group_mapping.keys())[0]) - 1
    for pa in range(num_prot_attr):
        disadv_grp = list(filter(lambda x: x[pa] == "0", group_mapping.keys()))
        adv_grp = list(filter(lambda x: x[pa] == "1", group_mapping.keys()))
        disadv_grp_adv_cls = list(
            filter(lambda x: int(x[-1]) in favorable_labels, disadv_grp)
        )
        disadv_grp_adv_cls_ct = sum(
            list(map(lambda x: mapped_osizes[x], disadv_grp_adv_cls))
        )
        disadv_grp_disadv_cls = list(
            filter(lambda x: int(x[-1]) not in favorable_labels, disadv_grp)
        )
        disadv_grp_disadv_cls_ct = sum(
            list(map(lambda x: mapped_osizes[x], disadv_grp_disadv_cls))
        )
        adv_grp_disadv_cls = list(
            filter(lambda x: int(x[-1]) not in favorable_labels, adv_grp)
        )
        adv_grp_disadv_cls_ct = sum(
            list(map(lambda x: mapped_osizes[x], adv_grp_disadv_cls))
        )
        adv_grp_adv_cls = list(
            filter(lambda x: int(x[-1]) in favorable_labels, adv_grp)
        )
        adv_grp_adv_cls_ct = sum(list(map(lambda x: mapped_osizes[x], adv_grp_adv_cls)))
        calc_di = (
            (disadv_grp_adv_cls_ct) / (disadv_grp_adv_cls_ct + disadv_grp_disadv_cls_ct)
        ) / ((adv_grp_adv_cls_ct) / (adv_grp_adv_cls_ct + adv_grp_disadv_cls_ct))

        if calc_di <= 1:
            odi.append(calc_di)
        else:
            odi.append(1 / calc_di)
            for g in disadv_grp:
                group_mapping[g] = (
                    group_mapping[g][0:pa] + "1" + group_mapping[g][pa + 1 :]
                )
            for g in adv_grp:
                group_mapping[g] = (
                    group_mapping[g][0:pa] + "0" + group_mapping[g][pa + 1 :]
                )

    # recompute mapping based on any flipping of protected attribute values
    mapped_osizes = {k1: osizes[k2] for k1, k2 in group_mapping.items()}
    sorted_osizes = list(
        map(lambda x: x[1], sorted(mapped_osizes.items(), key=lambda x: x[0]))
    )
    # construct variables for solver
    o_flat = np.array(sorted_osizes)
    oci_vec = np.array(oci).reshape(-1, 1)
    nci_vec = oci_vec + imbalance_repair_level * (1 - oci_vec)
    odi_vec = np.array(odi).reshape(-1, 1)
    ndi_vec = odi_vec + bias_repair_level * (1 - odi_vec)

    return group_mapping, o_flat, nci_vec, ndi_vec


def create_ci_penalties(n_ci, n_di):
    C = n_ci.shape[0] + 1
    A = n_di.shape[0]
    ci_penalties = []

    # specify C-1 class imbalance constraints as heavy penalties
    for i in range(C - 1):

        def condition(x):
            reshape_list = []
            for j in range(A):
                reshape_list.append(2)
            reshape_list.append(C)
            ndx = np.array(x).reshape(reshape_list)
            return (
                np.sum(ndx[(slice(None),) * A + (i,)])
                / np.sum(ndx[(slice(None),) * A + (i + 1,)])
            ) - n_ci[i, 0]

        @quadratic_equality(condition, k=1e6, h=10)
        def penalty(x):
            return 0

        ci_penalties.append(penalty)
    return ci_penalties


def create_di_penalties(n_ci, n_di, F):
    C = n_ci.shape[0] + 1
    A = n_di.shape[0]
    di_penalties = []

    # specify A disparate imapct ratio constraints as heavy penalties
    for i in range(A):

        def condition(x):
            reshape_list = []
            for j in range(A):
                reshape_list.append(2)
            reshape_list.append(C)
            ndx = np.array(x).reshape(reshape_list)
            di_ratio_top = np.sum(
                ndx[
                    (slice(None),) * i
                    + (0,)
                    + (slice(None),) * (A - i - 1)
                    + (tuple(F),)
                ]
            ) / np.sum(
                ndx[
                    (slice(None),) * i
                    + (0,)
                    + (slice(None),) * (A - i - 1)
                    + (slice(None),)
                ]
            )
            di_ratio_bottom = np.sum(
                ndx[
                    (slice(None),) * i
                    + (1,)
                    + (slice(None),) * (A - i - 1)
                    + (tuple(F),)
                ]
            ) / np.sum(
                ndx[
                    (slice(None),) * i
                    + (1,)
                    + (slice(None),) * (A - i - 1)
                    + (slice(None),)
                ]
            )
            return (di_ratio_top / di_ratio_bottom) - n_di[i, 0]

        @quadratic_equality(condition, k=1e6, h=10)
        def penalty(x):
            return 0

        di_penalties.append(penalty)
    return di_penalties


def calc_oversample_soln(o_flat, F, n_ci, n_di):
    # integer constraint
    ints = np.round

    # minimize sum of new number of examples
    def cost(x):
        return np.sum(x)

    # specify observed example counts as lower bounds
    bounds = list(map(lambda x: (x, float("inf")), o_flat))

    # combine all penalties
    ci_penalties = create_ci_penalties(n_ci, n_di)
    di_penalties = create_di_penalties(n_ci, n_di, F)
    all_penalties = and_(*ci_penalties, *di_penalties)

    # integer constraint
    constraint = ints

    # pass to solver
    result = diffev2(
        cost,
        x0=o_flat,
        bounds=bounds,
        constraints=constraint,
        penalty=all_penalties,
        full_output=False,
        disp=False,
        npop=50,
        gtol=100,
    )
    return result


def calc_undersample_soln(o_flat, F, n_ci, n_di):
    # integer constraint
    ints = np.round

    # minimize negative sum of new number of examples (equivalent to maximizing positive sum)
    def cost(x):
        return -np.sum(x)

    # specify observed example counts as upper bounds and 0 as lower bounds
    bounds = list(map(lambda x: (0, x), o_flat))

    # combine all penalties
    ci_penalties = create_ci_penalties(n_ci, n_di)
    di_penalties = create_di_penalties(n_ci, n_di, F)
    all_penalties = and_(*ci_penalties, *di_penalties)

    # integer constraint
    constraint = ints

    # pass to solver
    result = diffev2(
        cost,
        x0=o_flat,
        bounds=bounds,
        constraints=constraint,
        penalty=all_penalties,
        full_output=False,
        disp=False,
        npop=50,
        gtol=100,
    )
    return result

def calc_mixedsample_soln(o_flat, F, n_ci, n_di):
    # integer constraint
    ints = np.round

    # minimize sum of absolute value of differences from original numbers of examples
    def cost(x):
        return np.sum(np.abs(x - o_flat))

    # specify 0 as lower bounds and infinity as upper bounds (objective function will implicitly enforce bounds)
    bounds = list(map(lambda x: (0, float('inf')), o_flat))

    # combine all penalties
    ci_penalties = create_ci_penalties(n_ci, n_di)
    di_penalties = create_di_penalties(n_ci, n_di, F)
    all_penalties = and_(*ci_penalties, *di_penalties)

    # integer constraint
    constraint = ints

    # pass to solver
    result = diffev2(cost, x0=o_flat, bounds=bounds, constraints=constraint, penalty=all_penalties, full_output=False, disp=False, npop=50, gtol=100,)
    return result