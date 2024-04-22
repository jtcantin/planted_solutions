import random
import numpy as np
import openfermion as of
import CAS.ferm_utils as feru
from CAS_Cropping.sdstate import *


def add_balance(two_body_tensor, k, ne_each):
    """
    Return the two body tensor with a random balance added.
    Args:
        two_body_tensor: Two body tensor to be modified
        k: the partitioning of the indices
        ne_each: The number of electrons in each block

    Returns: Two body tensor with a balance added.

    """
    tbt = two_body_tensor
    balance_t = 2
    E_total = 0
    for i in range(len(k)):
        orbs = k[i]

        s = orbs[0]
        t = orbs[-1] + 1
        norbs = len(orbs)

        balance_tbt = np.zeros([norbs, norbs, norbs, norbs, ])
        for p, q in product(range(norbs), repeat=2):
            balance_tbt[p, p, q, q] += 1
        for p in range(len(orbs)):
            balance_tbt[p, p, p, p] -= 2 * ne_each[i]
        strength = balance_t * (1 + random.random())
        balance_tbt *= strength
        tbt[s:t, s:t, s:t, s:t] = np.add(tbt[s:t, s:t, s:t, s:t],
                                             balance_tbt)

        tmp = feru.get_ferm_op(tbt[s:t, s:t, s:t, s:t], True)
        sparse_H_tmp = of.get_sparse_operator(tmp)
        tmp_E_min, t_sol = of.get_ground_state(sparse_H_tmp)

        # Adding the ground state elements to forma a Slater determinant
        st = sdstate(n_qubit=len(orbs))
        for i in range(len(t_sol)):
            if np.linalg.norm(t_sol[i]) > np.finfo(np.float32).eps:
                st += sdstate(s=i, coeff=t_sol[i])
        st.normalize()
        E_st = st.exp(tmp)
        E_total += E_st

    return E_total, tbt
