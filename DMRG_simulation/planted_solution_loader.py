import CAS.ferm_utils as feru
import CAS_Cropping.csa_utils as csau
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
from DMRG_simulation.utils.fci_ground_energy import get_fci_ground_energy


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
    tbt = csau.get_cas_matrix(cas_x, spin_orbs, k)

    ## Adding killer
    killer_tbt = feru.get_chemist_tbt(killer, spin_orbs, spin_orb=True)
    killer_one_body = of.normal_ordered(
        killer - feru.get_ferm_op(killer_tbt, spin_orb=True))
    killer_onebody_matrix = feru.get_obt(
        killer_one_body, n = spin_orbs, spin_orb=True)
    killer_onebody_tbt = feru.onebody_to_twobody(killer_onebody_matrix)
    Htbt_killer = np.add(killer_tbt, killer_onebody_tbt)
    Htbt_with_killer = np.add(tbt, Htbt_killer)

    U = construct_random_sz_unitary(spin_orbs)

    Htbt_hidden = csau.cartan_orbtransf(Htbt_with_killer, U, complex=False)
    tbt_hidden = csau.cartan_orbtransf(tbt, U, complex=False)

    return (tbt, tbt_hidden, Htbt_with_killer,
            Htbt_hidden, sol, e_nums, E_min, spin_orbs)


def cas_to_dmrg(onebody_matrix, tbt):
    """
    Given a CAS hamiltonian file, we extract its structure and
    pass it to the DMRG calculator.
    Args:
        path: PATH to the hamiltonian folder
        file_name: file name of each hamiltonian
        key:

    Returns: (DMRG result, FCI result)

    """

    # Check symmetries
    print("Check the Spin Symmetry", check_spin_symmetry(onebody_matrix, tbt))

    # Get the correct permutation of tensors for Block2
    one_body_tensor, two_body_tensor = (
        get_correct_permutation(onebody_matrix, tbt, spin_orbs))
    print("Check permutation Symmetry After correction",
          check_permutation_symmetries_complex_orbitals(
              one_body_tensor, two_body_tensor))

    new_one_body_tensor, new_two_body_tensor, spin_symm_broken = (
        spinorbitals_to_orbitals(one_body_tensor, two_body_tensor))

    print("Spin symmetry broken", spin_symm_broken)

    result = single_qchem_dmrg_calc(new_one_body_tensor, new_two_body_tensor,
                                    dmrg_param)
    return result["dmrg_ground_state_energy"], E_min


if __name__ == "__main__":
    # sys.settrace(trace)
    ps_path = "../CAS_Cropping/planted_solutions/"
    # File name in ps_path folder
    file_name = "2_co2_6-311++G___12_9d464efb-b312-45f8-b0ba-8c42663059dc.pkl"
    (tbt, tbt_hidden, Htbt_with_killer, Htbt_hidden, sol, e_nums,
     E_min, spin_orbs) = construct_Hamiltonian_with_solution(ps_path, file_name)

    init_state_bond_dimension = 50
    max_num_sweeps = 200
    energy_convergence_threshold = 1e-8
    sweep_schedule_bond_dims = default_sweep_schedule_bond_dims
    sweep_schedule_noise = default_sweep_schedule_noise
    sweep_schedule_davidson_threshold = (
        default_sweep_schedule_davidson_threshold
    )

    num_orbitals = spin_orbs // 2
    num_electrons = 8
    num_spin_orbitals = spin_orbs
    num_unpaired_electrons = 8
    multiplicity = 1

    dmrg_process_param = get_dmrg_process_param(
        init_state_bond_dimension, max_num_sweeps, energy_convergence_threshold,
        sweep_schedule_bond_dims, sweep_schedule_noise,
        sweep_schedule_davidson_threshold)
    dmrg_param = get_dmrg_param(
        num_orbitals, num_electrons, num_unpaired_electrons,
        multiplicity, dmrg_process_param)

    # Set which one_body_tenro and two_body_tensor we use
    onebody_matrix = np.zeros((spin_orbs, spin_orbs))
    two_body_tensor = Htbt_with_killer

    tbt_to_ferm = feru.get_ferm_op(tbt, spin_orb=spin_orbs)

    ground_energy_tbt = get_fci_ground_energy(tbt)
    ground_energy_tbt_hidden = get_fci_ground_energy(tbt_hidden)
    ground_energy_Htbt_hidden = get_fci_ground_energy(Htbt_hidden)
    ground_energy_Htbt_hidden_killer = get_fci_ground_energy(Htbt_with_killer)

    print("Tyoe of tbt:", type(tbt))
    print("FCI Ground energy tbt", ground_energy_tbt)
    print("FCI Ground energy tbt hidden", ground_energy_tbt_hidden)
    print("FCI Ground energy Htbt hidden", ground_energy_Htbt_hidden)
    print("FCI Ground energy Htbt_hidden and Killer",
          ground_energy_Htbt_hidden_killer)
    print("Loaded Ground energy", E_min)

    # result = cas_to_dmrg(onebody_matrix, two_body_tensor)
    # print("DMRG Ground energy", result[0])
