import CAS_Cropping.ferm_utils as feru
import CAS_Cropping.csa_utils as csau
import CAS_Cropping.var_utils as varu
import openfermion as of
import numpy as np
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
from DMRG_simulation.utils.adding_balance import  add_balance
from DMRG_simulation.utils.get_energy_each_block import get_energy_from_each_block
from DMRG_simulation.dmrg_calc import get_dmrg_energy
import sys

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
    killer = dic[key]["killer"]
    k = dic[key]["k"]
    upnum = dic[key]["upnum"]
    spin_orbs = dic[key]["spin_orbs"]
    e_nums = dic[key]["e_nums"]
    sol = dic[key]["sol"]
    print(k)
    two_body_term = csau.get_cas_matrix(cas_x, spin_orbs, k)

    ## Adding killer
    killer_tbt = feru.get_chemist_tbt(killer, spin_orbs, spin_orb=True)
    killer_one_body = of.normal_ordered(
        killer - feru.get_ferm_op(killer_tbt, spin_orb=True))
    killer_onebody_matrix = feru.get_obt(
        killer_one_body, n = spin_orbs, spin_orb=True)
    killer_onebody_tbt = feru.onebody_to_twobody(killer_onebody_matrix)
    Htbt_killer = np.add(killer_tbt, killer_onebody_tbt)
    Htbt_with_killer = np.add(two_body_term, Htbt_killer)

    U = construct_random_sz_unitary(spin_orbs)

    Htbt_hidden = csau.cartan_orbtransf(Htbt_with_killer, U, complex=False)
    tbt_hidden = csau.cartan_orbtransf(two_body_term, U, complex=False)

    return (two_body_term, tbt_hidden, Htbt_with_killer, k,
            Htbt_hidden, sol, e_nums, E_min, spin_orbs)


def cas_to_dmrg(tbt, spin_orb, electron_each):
    """
    Given a CAS hamiltonian file, we extract its structure and
    pass it to the DMRG calculator.
    Args:
        tbt:
        spin_orb:
        electron_each:

    Returns: (DMRG result, FCI result)

    """

    init_state_bond_dimension = 60
    max_num_sweeps = 200
    energy_convergence_threshold = 1e-8
    sweep_schedule_bond_dims = default_sweep_schedule_bond_dims
    sweep_schedule_noise = default_sweep_schedule_noise
    sweep_schedule_davidson_threshold = (
        default_sweep_schedule_davidson_threshold
    )

    multiplicity = 1
    for block in electron_each:
        multiplicity += 2 * (block % 2)

    num_orbitals = spin_orb // 2
    num_electrons = sum(electron_each)
    num_spin_orbitals = spin_orb
    num_unpaired_electrons = (multiplicity - 1) // 2
    multiplicity = multiplicity


    dmrg_process_param = get_dmrg_process_param(
        init_state_bond_dimension, max_num_sweeps,
        energy_convergence_threshold,
        sweep_schedule_bond_dims, sweep_schedule_noise,
        sweep_schedule_davidson_threshold)
    dmrg_param = get_dmrg_param(
        num_orbitals, num_electrons, num_unpaired_electrons,
        multiplicity, dmrg_process_param)

    one_body_matrix = np.zeros((spin_orb, spin_orb))
    # Check symmetries
    print("Check the Spin Symmetry", check_spin_symmetry(one_body_matrix, tbt))
    # Get the correct permutation of tensors for Block2
    one_body_tensor, two_body_tensor = (
        get_correct_permutation(one_body_matrix, tbt, spin_orb))

    new_one_body_tensor, new_two_body_tensor, spin_symm_broken = (
        spinorbitals_to_orbitals(one_body_tensor, two_body_tensor))

    print("Spin symmetry broken", spin_symm_broken)

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
    ps_path = "../CAS_Cropping/planted_solutions/"
    # File name in ps_path folder
    file_name = "2_co2_6-311++G___12_e1d41ac3-ed50-42dc-a697-2d89d2275a2b.pkl"
    (two_body_original, tbt_hidden, Htbt_with_killer, k, Htbt_hidden, sol, e_nums,
     E_min, spin_orbs) = construct_Hamiltonian_with_solution(ps_path, file_name)
    print("Manually found ground state:", get_ground_state_manually(two_body_original, k, e_nums)[0])
    print("# of Electrons:", e_nums)
    # E_loaded, tbt_balance = add_balance(tbt, k, ne_each)
    # print("Ground state energy loaded:", E_loaded)

    Hf = feru.get_ferm_op(two_body_original, spin_orb=True)
    # result = get_dmrg_energy(two_body_original, spin_orbs, dmrg_param)
    # print(result)

    onebody_matrix = np.zeros((spin_orbs, spin_orbs))
    ground_energy_tbt = get_fci_ground_energy(Hf)
    print("FCI Ground energy tbt", ground_energy_tbt)
    print("Manually found energy for each block tbt", get_energy_from_each_block(two_body_original, k, e_nums)[0])
    print("FCI energy for each block tbt",
          get_energy_from_each_block(two_body_original, k, e_nums)[1])
    # unpaired_electrons = [3, 0, 1]
    # result_each = cas_each_block_to_dmrg(two_body_original, e_nums, k, unpaired_electrons)
    # print("Each block result:", result_each[0])
    # print("DMRG Ground energy from blocks", sum(result_each[0]))
    result = cas_to_dmrg(two_body_original, spin_orbs, e_nums)
    print("DMRG Ground energy tbt", result)
