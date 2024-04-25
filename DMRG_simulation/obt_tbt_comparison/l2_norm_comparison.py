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
    """
    Get original TBT, absorb_h1e TBT, and symmetrized TBT
    Args:
        load_result: The load result of the catalyst data

    Returns: Three TBTs

    """

    # Every tbt is in spin orbital
    one_body_phy = load_result['one_body_tensor']
    two_body_phy = load_result['two_body_tensor']
    spatial_obt_phy = load_result['one_body_tensor_spatial']
    spatial_tbt_phy = load_result['two_body_tensor_spatial']
    spin_orbs = load_result['num_spin_orbitals']

    tbt_combined = load_result['combined_chem_tbt']
    one_body, two_body = physicist_to_chemist(
        one_body_phy.copy(), two_body_phy.copy(), spin_orbs)

    # symmetrized_tensor = symmetrize_tensors(one_body_phy.copy(), two_body_phy.copy())
    symmetrized_tensor = symmetrize_tensors(spatial_obt_phy.copy(),
                                            spatial_tbt_phy.copy())

    return two_body, tbt_combined, symmetrized_tensor

def symmetrize_tensors(h_phy_obt, g_phy_tbt):
    """
    Symmetried h_pq, and g_pqrs tensors in spin orbital, physicist
    Args:
        h_phy_obt: Physicist Obt in spin orbital (h_pq)
        g_phy_tbt: Physicist Tbt in spin orbital (g_pqrs)

    Returns: Symmetrized tensor of chemist notation in spin orbital

    """
    # g_aa, h_a , g_a are in spatial orbital
    # g_aa, g_bb, g_ab, g_ba = get_spin_tensor(g_phy_tbt.copy())
    g_aa = g_phy_tbt.copy()
    g_bb = g_phy_tbt.copy()
    g_ab = g_phy_tbt.copy()
    g_ba = g_phy_tbt.copy()

    g_obt_a = get_obt_from_tbt(g_phy_tbt.copy())
    g_obt_b = get_obt_from_tbt(g_phy_tbt.copy())

    f_a = h_phy_obt.copy() - g_obt_a
    f_b = h_phy_obt.copy() - g_obt_b

    # A_pqrs in spin orbital
    A_pqrs_aa = feru.onebody_to_twobody(f_a)
    A_pqrs_bb = feru.onebody_to_twobody(f_b)

    # Extract the spatial orbital tensors
    # A_aa, A_bb, A_ab, A_ba = get_spin_tensor(A_pqrs_aa)

    tbt_aa = 2 * A_pqrs_aa + g_aa
    tbt_bb = 2 * A_pqrs_bb + g_bb
    tbt_ab = g_ab
    tbt_ba = g_ba

    return (tbt_aa + tbt_bb + tbt_ba + tbt_ab) / (4 * 2)

def get_spin_tensor(tensor):
    """
    Everything is in the spin orbital and chemical indices
    Args:
        tensor: tensor in spin orbital (g_pqrs)

    Returns: Spin tensors correspond to each spin combination

    """
    spin_orbs = tensor.shape[0]
    spatial = tensor.shape[0] // 2
    tensor_aa = np.zeros((spatial, spatial, spatial, spatial))
    tensor_ab = np.zeros((spatial, spatial, spatial, spatial))
    tensor_ba = np.zeros((spatial, spatial, spatial, spatial))
    tensor_bb = np.zeros((spatial, spatial, spatial, spatial))

    for p in range(spin_orbs):
        for q in range(spin_orbs):
            for r in range(spin_orbs):
                for s in range(spin_orbs):
                    if p % 2 == 0 and q % 2 == 0 and r % 2 == 0 and s % 2 == 0:
                        tensor_aa[p//2, q//2, r//2, s//2] = tensor[p, q, r, s]
                    elif p % 2 == 1 and q % 2 == 1 and r % 2 == 1 and s % 2 == 1:
                        tensor_bb[(p-1)//2, (q-1)//2, (r-1)//2, (s-1)//2] = tensor[p, q, r, s]
                    elif p % 2 == 0 and q % 2 == 0 and r % 2 == 1 and s % 2 == 1:
                        tensor_ab[p//2, q//2, (r-1)//2, (s-1)//2] = tensor[p, q, r, s]
                    elif p % 2 == 1 and q % 2 == 1 and r % 2 == 0 and s % 2 == 0:
                        tensor_ba[(p-1)//2, (q-1)//2, r//2, s//2] = tensor[p, q, r, s]

    return tensor_aa, tensor_bb, tensor_ab, tensor_ba


def get_spin_one_tensor(tensor):
    """
    Get one body tensor for each spin combination
    Args:
        tensor:

    Returns:

    """
    spin_orb = tensor.shape[0]
    spatial = spin_orb // 2
    tensor_a = np.zeros((spatial, spatial))
    tensor_b = np.zeros((spatial, spatial))
    for p in range(spin_orb):
        for q in range(spin_orb):
            if p % 2 == 0 and q % 2 == 0:
                tensor_a[p//2, q//2] = tensor[p, q]
            elif p % 2 == 1 and q % 2 == 1:
                tensor_b[(p-1)//2, (q-1)//2] = tensor[p, q]

    return tensor_a, tensor_b


def get_obt_from_tbt(g_tensor):
    """
    Get the one body chemist from the two body physicist tensor
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

    return 0.5 * one_body_tensor


def add_killer_hidden(tensor, killer):
    """
    Adding killer and also hiding by random unitary rotation
    Args:
        tensor: The tensor to be modified
        killer: The killer operator to be added.

    Returns:

    """
    spin_orbs = tensor.shape[0]
    killer_tbt = feru.get_chemist_tbt(killer, spin_orbs, spin_orb=True)
    killer_one_body = of.normal_ordered(
        killer - feru.get_ferm_op(killer_tbt, spin_orb=True))
    killer_one_body_matrix = feru.get_obt(
        killer_one_body, n=spin_orbs, spin_orb=True)
    killer_one_body_tbt = feru.onebody_to_twobody(killer_one_body_matrix)
    killer_op = np.add(killer_tbt, killer_one_body_tbt)
    unitary = construct_random_sz_unitary(spin_orbs)
    # tbt_hiding_directly = np.matmul(np.matrix(unitary), tensor)

    tbt_with_killer = np.add(tensor, killer_op)
    tbt_hidden_rotated = csau.cartan_orbtransf(tensor, unitary,
                                               complex=False)
    killer_tbt_hidden = csau.cartan_orbtransf(tbt_with_killer, unitary,
                                              complex=False)

    return tensor, tbt_hidden_rotated, tbt_with_killer, killer_tbt_hidden


if __name__ == '__main__':
    path = "../data/"
    fcidump_file = "fcidump.2_co2_6-311++G**"
    load_result = catalyst_loader.load_tensor(path + fcidump_file)
    tbt = load_result['two_body_tensor']
    e_num_actual = load_result['num_electrons']

    setting_dict = {
        "num_e_block": 8,
        "block_size": 12,
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

    l2_norm_1_original = scipy.linalg.norm(tbt_1, ord=None)
    l2_norm_1_hidden = scipy.linalg.norm(tbt_1_hidden, ord=None)
    l2_norm_1_killer = scipy.linalg.norm(tbt_1_killer, ord=None)
    l2_norm_1_killer_hidden = scipy.linalg.norm(tbt_1_killer_hidden, ord=None)

    l2_norm_2_original = scipy.linalg.norm(tbt_2, ord=None)
    l2_norm_2_hidden = scipy.linalg.norm(tbt_2_hidden, ord=None)
    l2_norm_2_killer = scipy.linalg.norm(tbt_2_killer, ord=None)
    l2_norm_2_killer_hidden = scipy.linalg.norm(tbt_2_killer_hidden, ord=None)

    l2_norm_3_original = scipy.linalg.norm(tbt_3, ord=None)
    l2_norm_3_hidden = scipy.linalg.norm(tbt_3_hidden, ord=None)
    l2_norm_3_killer = scipy.linalg.norm(tbt_3_killer, ord=None)
    l2_norm_3_killer_hidden = scipy.linalg.norm(tbt_3_killer_hidden, ord=None)

    print(f"L2 norms for case 1: Original:{l2_norm_1_original} Hidden: {l2_norm_1_hidden} Killer: {l2_norm_1_killer} Killer Hidden: {l2_norm_1_killer_hidden}")
    print(
        f"L2 norms for case 2: Original:{l2_norm_2_original} Hidden: {l2_norm_2_hidden} Killer: {l2_norm_2_killer} Killer Hidden: {l2_norm_2_killer_hidden}")
    print(
        f"L2 norms for case 3: Original:{l2_norm_3_original} Hidden: {l2_norm_3_hidden} Killer: {l2_norm_3_killer} Killer Hidden: {l2_norm_3_killer_hidden}")
    # print(
    #     f"Original L2 norm difference 1-2:{l2_norm_1_2}, 2-3:{l2_norm_2_3}, 3-1:{l2_norm_3_1}")
    print(
        f"Hidden L2 norm difference 1-2:{l2_norm_1_2_hidden}, 2-3:{l2_norm_2_3_hidden}, 3-1:{l2_norm_3_1_hidden}")
    print(
        f"Killer L2 norm difference 1-2:{l2_norm_1_2_killer}, 2-3:{l2_norm_2_3_killer}, 3-1:{l2_norm_3_1_killer}")
    print(
        f"Killer Hidden L2 norm difference 1-2:{l2_norm_1_2_killer_h}, 2-3:{l2_norm_2_3_killer_h}, 3-1:{l2_norm_3_1_killer_h}")




