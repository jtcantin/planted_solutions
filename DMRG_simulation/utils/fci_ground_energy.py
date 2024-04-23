import openfermion as of
import numpy as np
from CAS_Cropping.sdstate import *
import CAS.ferm_utils as feru

def get_fci_ground_energy(Hf, spin_orbs):
    sparse = of.linalg.get_sparse_operator(Hf, n_qubits=spin_orbs)
    ground_energy, gs = of.linalg.get_ground_state(
        sparse, initial_guess=None
    )
    return ground_energy, gs


def get_ground_state_manually(two_body_tensor, k, ne_each):
    tbt = two_body_tensor
    E_total = 0
    for i in range(len(k)):
        orbs = k[i]

        s = orbs[0]
        t = orbs[-1] + 1
        norbs = len(orbs)

        balance_tbt = np.zeros([norbs, norbs, norbs, norbs, ])
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
