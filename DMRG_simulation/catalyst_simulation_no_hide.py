import CAS_Cropping.ferm_utils as feru
import CAS_Cropping.csa_utils as csau
import CAS_Cropping.var_utils as varu
from DMRG_simulation.dmrg_simulator.dmrg_phy_tensor import get_dmrg_from_phy
from CAS.dmrghandler.src.dmrghandler.qchem_dmrg_calc import (
    single_qchem_dmrg_calc)
from CAS.dmrghandler.src.dmrghandler.dmrg_calc_prepare import (
    check_spin_symmetry, spinorbitals_to_orbitals)
from CAS_Cropping.matrix_utils import construct_random_sz_unitary
import pickle
import os
from DMRG_simulation.format_dmrg.format_dmrg_param import (
    get_dmrg_param, get_dmrg_process_param)
from DMRG_simulation.format_dmrg.format_tensor import get_correct_permutation
from DMRG_simulation.validator.symmetry_check import (
    check_permutation_symmetries_complex_orbitals)
from DMRG_simulation.utils.fci_ground_energy import (get_fci_ground_energy,
                                                     get_ground_state_manually)
from CAS_Cropping.sdstate import *
from DMRG_simulation.format_dmrg.format_tensor import physicist_to_chemist

def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace


default_final_bond_dim = 100
default_sweep_schedule_bond_dims = [default_final_bond_dim] * 4 + [
    default_final_bond_dim
] * 4
default_sweep_schedule_noise = [1e-4] * 4 + [1e-5] * 4 + [0]
default_sweep_schedule_davidson_threshold = [1e-10] * 8

os.environ["PYDEVD_USE_CYTHON"] = "NO"
os.environ["PYDEVD_USE_FRAME_EVAL"] = "NO"


def construct_Hamiltonian_with_solution(path, file_name, key = ""):
    """
    Load the planted_soluiton Hamiltonian from the planted_solution,
    with key to be the block structure of encoding the Hamiltonian.
    """
    with open(path + file_name, 'rb') as handle:
        dic = pickle.load(handle)
    if not key:
        key = list(dic.keys())[0]
    E_min = dic[key]["E_min"]
    cas_x = dic[key]["cas_x"]
    cas_obt_x = dic[key]["cas_one_body"]
    cas_tbt_x = dic[key]["cas_two_body"]
    k = dic[key]["k"]
    spin_orbs = dic[key]["spin_orbs"]
    e_nums = dic[key]["e_nums"]
    sol = dic[key]["sol"]
    e_num_actual = dic[key]["e_num_actual"]
    htbt_before_truncate = dic[key]["tbt_before_truncate"]
    obt_phy = dic[key]["original_one_body"]
    tbt_phy = dic[key]["original_two_body"]

    print(k)
    two_body_term = csau.get_cas_matrix(cas_x, spin_orbs, k)
    cas_one_body = csau.get_cas_matrix(cas_obt_x, spin_orbs, k)
    cas_two_body = csau.get_cas_matrix(cas_tbt_x, spin_orbs, k)

    return (two_body_term, k, sol, e_nums, E_min, spin_orbs, e_num_actual,
            htbt_before_truncate, obt_phy, tbt_phy, cas_one_body, cas_two_body)


def cas_to_dmrg(tbt, spin_orb, dmrg_param):
    """
    Given a CAS hamiltonian file, we extract its structure and
    pass it to the DMRG calculator.
    Args:
        tbt:
        spin_orb:
        dmrg_param:

    Returns: (DMRG result, FCI result)

    """
    obt = np.zeros((spin_orb, spin_orb))
    # Get the correct permutation of tensors for Block2
    one_body_tensor, two_body_tensor = (
        get_correct_permutation(obt, tbt, spin_orbs))
    new_one_body_tensor, new_two_body_tensor, spin_symm_broken = (
        spinorbitals_to_orbitals(one_body_tensor, two_body_tensor))

    print("Spin symmetry broken:", spin_symm_broken)

    dmrg_result = single_qchem_dmrg_calc(new_one_body_tensor, new_two_body_tensor,
                                         dmrg_param)

    return dmrg_result["dmrg_ground_state_energy"]


def cas_each_block_to_dmrg(tbt, ne_each, k, unpaired_electrons):
    """
    Given a CAS hamiltonian file, we extract its structure and
    pass it to the DMRG calculator.
    Args:
        path: PATH to the hamiltonian folder
        file_name: file name of each hamiltonian
        key:

    Returns: (DMRG result, FCI result)

    """
    E_total = []
    for i in range(len(k)):
        orbs = k[i]

        s = orbs[0]
        t = orbs[-1] + 1

        spin_orbs_here = len(orbs)
        e_num = ne_each[i]

        init_state_bond_dimension = 100
        max_num_sweeps = 200
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = default_sweep_schedule_bond_dims
        sweep_schedule_noise = default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            default_sweep_schedule_davidson_threshold
        )

        num_orbitals = spin_orbs_here // 2
        num_electrons = ne_each[i]
        num_spin_orbitals = spin_orbs_here
        num_unpaired_electrons = (ne_each[i] % 2)
        multiplicity = 1 + (ne_each[i] % 2) * 2

        dmrg_process_param = get_dmrg_process_param(
            init_state_bond_dimension, max_num_sweeps,
            energy_convergence_threshold,
            sweep_schedule_bond_dims, sweep_schedule_noise,
            sweep_schedule_davidson_threshold)
        dmrg_param = get_dmrg_param(
            num_orbitals, num_electrons, num_unpaired_electrons,
            multiplicity, dmrg_process_param)

        # tbt is in chemist
        zero_matrix = np.zeros((len(orbs), len(orbs)))
        # print("Check permutation Symmetry Before correction",
        #       check_permutation_symmetries_complex_orbitals(
        #       zero_matrix, tbt[s:t, s:t, s:t, s:t]))
        # print("Check the Spin Symmetry", check_spin_symmetry(zero_matrix, tbt[s:t, s:t, s:t, s:t]))
        one_body_tensor, two_body_tensor = (
            get_correct_permutation(np.zeros((len(orbs), len(orbs))), tbt[s:t, s:t, s:t, s:t], len(orbs)))

        new_one_body_tensor, new_two_body_tensor, spin_symm_broken = (
            spinorbitals_to_orbitals(one_body_tensor, two_body_tensor))

        print("Spin symmetry broken", spin_symm_broken)

        result = single_qchem_dmrg_calc(new_one_body_tensor, new_two_body_tensor,
                                        dmrg_param)
        print("DMRG result for the block:", result["dmrg_ground_state_energy"])
        E_total.append(result["dmrg_ground_state_energy"])

    return E_total, E_min


if __name__ == "__main__":
    # sys.settrace(trace)
    ps_path = "../DMRG_simulation/planted_solutions/"
    # File name in ps_path folder
    file_name = "2_co2_6-311++G**.pkl"
    (two_body_original, k, sol, e_nums, E_min, spin_orbs, e_num_actual,
     htbt_before_truncate, phy_obt, phy_tbt, cas_obt_x, cas_tbt_x) \
        = construct_Hamiltonian_with_solution(ps_path, file_name)

    one_body, two_body = physicist_to_chemist(phy_obt, phy_tbt,
                                              spin_orbs)

    onebody_tbt = feru.onebody_to_twobody(one_body)
    Htbt_added = np.add(two_body, onebody_tbt)
    k = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    e_nums = [8]

    print("Manually found ground state:", get_ground_state_manually(two_body_original, k, e_nums)[0])
    print("CAS obt shape: {}".format(cas_obt_x.shape))
    print("Actual # of electrons: {}, # of partitioning: {}".format(e_num_actual, e_nums))

    Hf = feru.get_ferm_op(Htbt_added, spin_orb=True)

    onebody_matrix = np.zeros((spin_orbs, spin_orbs))
    ground_energy_tbt, gs_fci = get_fci_ground_energy(Hf)
    gs_matrix_fci = np.matrix(gs_fci)
    print("FCI Ground energy tbt", ground_energy_tbt)
    print("FCI Ground state norm:", np.matmul(gs_matrix_fci, gs_matrix_fci.H))

    init_state_bond_dimension = 100
    max_num_sweeps = 200
    energy_convergence_threshold = 1e-8
    sweep_schedule_bond_dims = default_sweep_schedule_bond_dims
    sweep_schedule_noise = default_sweep_schedule_noise
    sweep_schedule_davidson_threshold = (
        default_sweep_schedule_davidson_threshold
    )

    num_orbitals = spin_orbs // 2
    num_electrons = sum(e_nums)
    num_spin_orbitals = spin_orbs
    num_unpaired_electrons = 0
    multiplicity = 1

    dmrg_process_param = get_dmrg_process_param(
        init_state_bond_dimension, max_num_sweeps,
        energy_convergence_threshold,
        sweep_schedule_bond_dims, sweep_schedule_noise,
        sweep_schedule_davidson_threshold)
    dmrg_param = get_dmrg_param(
        num_orbitals, num_electrons, num_unpaired_electrons,
        multiplicity, dmrg_process_param)

    # We want a transformation to transform the chemical tbt to obt
    #
    transformed_obt = feru.tbt_to_obt(cas_obt_x)
    print("PHY CAS to DMRG:", get_dmrg_from_phy(phy_obt, phy_tbt, dmrg_param)["dmrg_ground_state_energy"])
    print("Chem CAS to DMRG:", cas_to_dmrg(Htbt_added, spin_orbs, dmrg_param))
    # result = cas_to_dmrg(two_body_original, phy_tbt, spin_orbs, dmrg_param)
    # print("DMRG Ground energy tbt", result)
