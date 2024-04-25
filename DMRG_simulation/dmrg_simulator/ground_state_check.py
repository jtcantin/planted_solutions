from DMRG_simulation.obt_tbt_comparison.l2_norm_comparison import symmetrize_tensors
from DMRG_simulation.data_repository.cas_converter import cas_from_tensor
import CAS_Cropping.ferm_utils as feru
import openfermion as of
import CAS_Cropping.csa_utils as csau
import numpy as np
from CAS_Cropping.matrix_utils import construct_random_sz_unitary
from DMRG_simulation.dmrg_simulator.dmrg_chem_tbt import get_dmrg_from_chem
from DMRG_simulation.format_dmrg.format_dmrg_param import get_dmrg_param, get_dmrg_process_param
import DMRG_simulation.data_repository.catalysts_loader as catalyst_loader
from DMRG_simulation.test.symmetry_test import test_two_body_symmetry_pq_rs, test_two_body_symmetry_conj_srqp, test_two_body_symmetry_conj_pqrs, test_one_body_symmetry
from DMRG_simulation.format_dmrg.format_tensor import physicist_to_chemist
from CAS.dmrghandler.src.dmrghandler.pyscf_wrappers import two_body_tensor_orbital_to_spin_orbital

default_final_bond_dim = 100
default_sweep_schedule_bond_dims = [default_final_bond_dim] * 4 + [
    default_final_bond_dim
] * 4
default_sweep_schedule_noise = [1e-4] * 4 + [1e-5] * 4 + [0]
default_sweep_schedule_davidson_threshold = [1e-10] * 8




def get_ground_state_dmrg(load_result, setting_dict):
    """

    Returns:

    """

    # Loaded the h_pq, g_pqrs in physicist notation
    one_body_phy = load_result['one_body_tensor']
    two_body_phy = load_result['two_body_tensor']
    spatial_obt_phy = load_result['one_body_tensor_spatial']
    spatial_tbt_phy = load_result['two_body_tensor_spatial']
    spin_orbs = load_result['num_spin_orbitals']
    e_num_actual = load_result['num_electrons']
    two_S = load_result['two_S']
    two_Sz = load_result['two_Sz']
    setting_dict["actual_e_num"] = e_num_actual

    # one_body, two_body = physicist_to_chemist(one_body_phy, two_body_phy,
    #                                           load_result["num_spin_orbitals"])
    # onebody_tbt = feru.onebody_to_twobody(one_body)
    # # htbt_here is in chemist notation (One body two body combined)
    # htbt_here_new = np.add(two_body, onebody_tbt)


    num_orbitals = spin_orbs // 2
    num_electrons = e_num_actual
    num_spin_orbitals = spin_orbs
    num_unpaired_electrons = two_Sz
    multiplicity = 1 + two_S

    init_state_bond_dimension = 50
    max_num_sweeps = 200
    energy_convergence_threshold = 1e-8
    sweep_schedule_bond_dims = default_sweep_schedule_bond_dims
    sweep_schedule_noise = default_sweep_schedule_noise
    sweep_schedule_davidson_threshold = (
        default_sweep_schedule_davidson_threshold
    )
    dmrg_process_param = get_dmrg_process_param(
        init_state_bond_dimension, max_num_sweeps, energy_convergence_threshold,
        sweep_schedule_bond_dims, sweep_schedule_noise,
        sweep_schedule_davidson_threshold)
    dmrg_param = get_dmrg_param(
        num_orbitals, num_electrons, num_unpaired_electrons,
        multiplicity, dmrg_process_param)

    print("Shape of symmetrized:",
          symmetrize_tensors(spatial_obt_phy, spatial_tbt_phy).shape)
    tbt_cas = two_body_tensor_orbital_to_spin_orbital(
        symmetrize_tensors(spatial_obt_phy, spatial_tbt_phy))

    planted_sol = cas_from_tensor(tbt_cas, setting_dict)

    #tbt_cas = planted_sol["cas_tbt"]
    #tbt_cas2 = planted_sol2["cas_tbt"]
    killer = planted_sol["killer"]
    cas_x = planted_sol["cas_x"]
    k = planted_sol["k"]
    tbt_cas = csau.get_cas_matrix(cas_x, spin_orbs, k)

    # tbt_cas = load_result['combined_chem_tbt']

    spin_orbs = tbt_cas.shape[0]
    killer_tbt = feru.get_chemist_tbt(killer, spin_orbs, spin_orb=True)
    killer_one_body = of.normal_ordered(
        killer - feru.get_ferm_op(killer_tbt, spin_orb=True))
    killer_one_body_matrix = feru.get_obt(
        killer_one_body, n=spin_orbs, spin_orb=True)
    killer_one_body_tbt = feru.onebody_to_twobody(killer_one_body_matrix)
    killer_op = np.add(killer_tbt, killer_one_body_tbt)
    unitary = construct_random_sz_unitary(spin_orbs)

    tbt_with_killer = np.add(tbt_cas, killer_op)
    tbt_hidden_rotated = csau.cartan_orbtransf(tbt_cas, unitary,
                                               complex=False)
    killer_tbt_hidden = csau.cartan_orbtransf(tbt_with_killer, unitary,
                                              complex=False)

    # H_cas = feru.get_ferm_op(htbt_here_new, spin_orb=True)
    # fci_emin, sol = of.get_ground_state(of.get_sparse_operator(H_cas))
    # print(fci_emin)

    dmrg_result= get_dmrg_from_chem(tbt_cas, spin_orbs, dmrg_param)
    dmrg_result_killer = get_dmrg_from_chem(tbt_with_killer,
                                                        spin_orbs, dmrg_param)
    # dmrg_result_hidden, fci_hidden = get_dmrg_from_chem(tbt_hidden_rotated, spin_orbs, dmrg_param)
    # dmrg_result_killer_hidden, fci_killer_hidden = get_dmrg_from_chem(tbt_cas, spin_orbs, dmrg_param)
    result = {}
    result['dmrg_result'] = dmrg_result
    # result['fci'] = fci
    # result['fci_sol'] = fci_sol
    result['dmrg_result_killer'] = dmrg_result_killer
    # result['fci_killer'] = fci_killer
    # result['fci_sol_killer'] = fci_sol_killer
    # result['dmrg_result_hidden'] = dmrg_result_hidden
    # result['fci_hidden'] = fci_hidden
    # result['dmrg_result_killer_hidden'] = dmrg_result_killer_hidden
    # result['fci_killer_hidden'] = fci_killer_hidden

    return result


if __name__ == '__main__':
    path = "../data/"
    fcidump_file = "fcidump.2_co2_6-311++G**"
    load_result = catalyst_loader.load_tensor(path + fcidump_file)

    setting_dict = {
        "num_e_block": 8,
        "block_size": 12,
    }

    result = get_ground_state_dmrg(load_result, setting_dict)
    print("DMRG result no hidden:", result['dmrg_result']["dmrg_ground_state_energy"])
    # print(result['fci'])
    print(np.linalg.norm(np.matrix(result['fci_sol'])))
    print(result['dmrg_result_killer']["dmrg_ground_state_energy"])
    print(result['fci_killer'])
    # print(result['dmrg_result_hidden'])
    # print(result['fci_hidden'])
    # print(result['dmrg_result_killer_hidden'])
    # print(result['fci_killer_hidden'])


