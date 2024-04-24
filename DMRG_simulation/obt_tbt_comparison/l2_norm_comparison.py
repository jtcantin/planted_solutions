import numpy as np
import scipy
import DMRG_simulation.data_repository.catalysts_loader as catalyst_loader
from DMRG_simulation.format_dmrg.format_tensor import physicist_to_chemist
import CAS_Cropping.ferm_utils as feru
from DMRG_simulation.data_repository.cas_converter import cas_from_tensor
import openfermion as of
import CAS_Cropping.csa_utils as csau
from CAS_Cropping.matrix_utils import construct_random_sz_unitary
from scipy.stats import unitary_group
def get_several_tbt(load_result):

    # Every tbt is in spin orbital
    one_body_phy = load_result['one_body_tensor']
    two_body_phy = load_result['two_body_tensor']
    spin_orbs = load_result['num_spin_orbitals']

    tbt_combined = load_result['combined_chem_tbt']
    one_body, two_body = physicist_to_chemist(
        one_body_phy, two_body_phy, spin_orbs)

    symmetrized_tensor = symmetrize_tensors(one_body_phy, 0.5 * two_body_phy)

    return two_body, tbt_combined, symmetrized_tensor

def symmetrize_tensors(h_phy_obt, g_phy_tbt):
    """

    Args:
        phy_obt: Physicist Obt in spin orbital
        phy_tbt: Physicist Tbt in spin orbital

    Returns:

    """
    g_aa, g_bb, g_ab, g_ba = get_spin_tensor(g_phy_tbt)
    h_a, h_b = get_spin_one_tensor(h_phy_obt)
    g_obt = get_obt_from_tbt(g_phy_tbt)
    g_a, g_b = get_spin_one_tensor(g_obt)
    f_a = h_a - g_a
    f_b = h_b - g_b

    A_aa = feru.onebody_to_twobody(f_a)
    A_bb = feru.onebody_to_twobody(f_b)

    tbt_aa = 2 * A_aa + g_aa
    tbt_bb = 2 * A_bb + g_bb
    tbt_ab = g_ab
    tbt_ba = g_ba

    return (tbt_aa + tbt_bb + tbt_ba + tbt_ab) / (4 * 2)

def calculate_l2_norm(tbt_1, tbt_2):
    """

    Args:
        tbt_1: First tbt
        tbt_2: Second tbt

    Returns: The L2 norm between tbt_1 and tbt_2

    """

    l2_norm = scipy.linalg.norm(tbt_1 - tbt_2, ord=None)
    return l2_norm


def get_spin_tensor(tensor):
    """
    Everything is in the spin orbital and chemical indices
    Args:
        tensor: tensor in spin orbital

    Returns: Spin tensors correspond to each

    """
    n = tensor.shape[0]
    tensor_aa = np.zeros((n, n, n, n))
    tensor_ab = np.zeros((n, n, n, n))
    tensor_ba = np.zeros((n, n, n, n))
    tensor_bb = np.zeros((n, n, n, n))
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    if p % 2 == 0 and q % 2 ==0 and r % 2 == 0 and s % 2 == 0:
                        tensor_aa[p, q, r, s] = tensor[p, q, r, s]
                    elif p % 2 == 1 and q % 2 == 1 and r % 2 == 1 and s % 2 == 1:
                        tensor_bb[p, q, r, s] = tensor[p, q, r, s]
                    elif p % 2 == 0 and q % 2 == 0 and r % 2 == 1 and s % 2 == 1:
                        tensor_ab[p, q, r, s] = tensor[p, q, r, s]
                    elif p % 2 == 1 and q % 2 == 1 and r % 2 == 0 and s % 2 == 0:
                        tensor_ba[p, q, r, s] = tensor[p, q, r, s]

    return tensor_aa, tensor_bb, tensor_ab, tensor_ba


def get_spin_one_tensor(tensor):
    """

    Args:
        tensor:

    Returns:

    """
    n = tensor.shape[0]
    tensor_a = np.zeros((n, n))
    tensor_b = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            if p % 2 == 0 and q % 2 == 0:
                tensor_a[p, q] = tensor[p][q]
            elif p % 2 == 1 and q % 2 == 1:
                tensor_b[p, q] = tensor[p][q]

    return tensor_a, tensor_b


def get_obt_from_tbt(g_tensor):
    """

    Args:
        g_tensor:

    Returns:

    """
    n = g_tensor.shape[0]
    one_body_tensor = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            total = 0
            for i in range(n):
                total += g_tensor[p, i, i, q]
            one_body_tensor[p, q] = total

    return one_body_tensor


def add_killer_hidden(tensor, killer):
    spin_orbs = tensor.shape[0]
    killer_tbt = feru.get_chemist_tbt(killer, spin_orbs, spin_orb=True)
    killer_one_body = of.normal_ordered(
        killer - feru.get_ferm_op(killer_tbt, spin_orb=True))
    killer_one_body_matrix = feru.get_obt(
        killer_one_body, n=spin_orbs, spin_orb=True)
    killer_one_body_tbt = feru.onebody_to_twobody(killer_one_body_matrix)
    killer_op = np.add(killer_tbt, killer_one_body_tbt)
    # unitary = construct_random_sz_unitary(spin_orbs)
    unitary = unitary_group.rvs(spin_orbs)

    tbt_with_killer = np.add(tensor, killer_op)
    tbt_hidden_rotated = csau.cartan_orbtransf(tensor, unitary,
                                               complex=False)
    killer_tbt_hidden = csau.cartan_orbtransf(tbt_with_killer, unitary,
                                              complex=False)

    return tensor, tbt_hidden_rotated, tbt_with_killer, killer_tbt_hidden


def get_random_hiding_unitary(spin_orbs):
    random_unitary = unitary_group.rvs(spin_orbs // 2)
    m = np.zeros((spin_orbs, spin_orbs), dtype='complex')
    for i in range(spin_orbs // 2):
        for k in range(spin_orbs // 2):
            m[2 * i, 2 * k] = random_unitary[i, k]
            m[2 * k, 2 * i] = random_unitary[k, i]
    return m


if __name__ == '__main__':
    path = "../data/"
    fcidump_file = "fcidump.2_co2_6-311++G**"
    load_result = catalyst_loader.load_tensor(path + fcidump_file)
    tbt = load_result['two_body_tensor']
    e_num_actual = load_result['num_electrons']

    setting_dict = {
        "num_e_block": 4,
        "block_size": 6,
        "actual_e_num": e_num_actual
    }
    print(tbt.shape)
    l2_norm = scipy.linalg.norm(tbt, ord=None)
    print("L2 norm Original PHY TBT in Spin orbital:", l2_norm)
    two_body, tbt_combined, symmetrized_tensor = get_several_tbt(load_result)
    l2_norm_1 = scipy.linalg.norm(two_body, ord=None)
    l2_norm_2 = scipy.linalg.norm(tbt_combined, ord=None)
    l2_norm_3 = scipy.linalg.norm(symmetrized_tensor, ord=None)
    l2_norm_1_2 = scipy.linalg.norm(two_body - tbt_combined, ord=None)
    l2_norm_2_3 = scipy.linalg.norm(tbt_combined - symmetrized_tensor, ord=None)
    l2_norm_3_1 = scipy.linalg.norm(symmetrized_tensor - two_body, ord=None)

    print("All TBT in Chemist Spin Orbitals")
    print("1: Original TBT only, 2: absorb_h1e TBT, 3: Solution 2")
    print(f"L2 norm each 1: {l2_norm_1}, 2: {l2_norm_2}, 3: {l2_norm_3}")
    print(f"L2 norm difference 1-2:{l2_norm_1_2}, 2-3:{l2_norm_2_3}, 3-1:{l2_norm_3_1}")

    planted_sol_1 = cas_from_tensor(two_body, setting_dict)
    planted_sol_2 = cas_from_tensor(tbt_combined, setting_dict)
    planted_sol_3 = cas_from_tensor(symmetrized_tensor, setting_dict)

    tbt_cas_1 = planted_sol_1["cas_tbt"]
    killer1 = planted_sol_1["killer"]
    tbt_cas_2 = planted_sol_2["cas_tbt"]
    killer2 = planted_sol_2["killer"]
    tbt_cas_3 = planted_sol_3["cas_tbt"]
    killer3 = planted_sol_3["killer"]

    (tbt_1, tbt_1_hidden, tbt_1_killer, tbt_1_killer_hidden) = (
        add_killer_hidden(tbt_cas_1, killer1))
    (tbt_2, tbt_2_hidden, tbt_2_killer, tbt_2_killer_hidden) = (
        add_killer_hidden(tbt_cas_2, killer1))
    (tbt_3, tbt_3_hidden, tbt_3_killer, tbt_3_killer_hidden) = (
        add_killer_hidden(tbt_cas_3, killer1))

    l2_norm_1_2_hidden = scipy.linalg.norm(tbt_1_hidden - tbt_2_hidden, ord=None)
    l2_norm_2_3_hidden = scipy.linalg.norm(tbt_2_hidden - tbt_3_hidden, ord=None)
    l2_norm_3_1_hidden = scipy.linalg.norm(tbt_3_hidden - tbt_1_hidden, ord=None)

    l2_norm_1_2_killer = scipy.linalg.norm(tbt_1_killer - tbt_2_killer,
                                           ord=None)
    l2_norm_2_3_killer = scipy.linalg.norm(tbt_2_killer - tbt_3_killer,
                                           ord=None)
    l2_norm_3_1_killer = scipy.linalg.norm(tbt_3_killer - tbt_1_killer,
                                           ord=None)

    l2_norm_1_2_killer_h = scipy.linalg.norm(tbt_1_killer_hidden - tbt_2_killer_hidden,
                                           ord=None)
    l2_norm_2_3_killer_h = scipy.linalg.norm(tbt_2_killer_hidden - tbt_3_killer_hidden,
                                           ord=None)
    l2_norm_3_1_killer_h = scipy.linalg.norm(tbt_3_killer_hidden - tbt_1_killer_hidden,
                                           ord=None)

    print(
        f"Original L2 norm difference 1-2:{l2_norm_1_2}, 2-3:{l2_norm_2_3}, 3-1:{l2_norm_3_1}")
    print(
        f"Hidden L2 norm difference 1-2:{l2_norm_1_2_hidden}, 2-3:{l2_norm_2_3_hidden}, 3-1:{l2_norm_3_1_hidden}")
    print(
        f"Killer L2 norm difference 1-2:{l2_norm_1_2_killer}, 2-3:{l2_norm_2_3_killer}, 3-1:{l2_norm_3_1_killer}")
    print(
        f"Killer Hidden L2 norm difference 1-2:{l2_norm_1_2_killer_h}, 2-3:{l2_norm_2_3_killer_h}, 3-1:{l2_norm_3_1_killer_h}")




