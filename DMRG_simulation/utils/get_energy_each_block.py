import random
import CAS.ferm_utils as feru
from CAS_Cropping.sdstate import *
from DMRG_simulation.utils.fci_ground_energy import get_fci_ground_energy

def get_energy_from_each_block(two_body_tensor, k, ne_each):
    tbt = two_body_tensor
    E_total = []
    fci_each_block = []
    for i in range(len(k)):
        orbs = k[i]

        s = orbs[0]
        t = orbs[-1] + 1

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
        E_total.append(E_st)

        fci_each_block.append(get_fci_ground_energy(tmp))

    return E_total, fci_each_block
