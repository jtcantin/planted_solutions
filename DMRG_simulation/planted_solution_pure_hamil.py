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
from DMRG_simulation.utils.fci_ground_energy import get_fci_ground_energy
from CAS_Cropping.sdstate import *
from DMRG_simulation.utils.adding_balance import add_balance
from DMRG_simulation.utils.get_energy_each_block import get_energy_from_each_block
from planted_solution_loader import *

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



def cas_to_dmrg(onebody_matrix, tbt, spin_orb, k):
    """
    Given a CAS hamiltonian file, we extract its structure and
    pass it to the DMRG calculator.
    Args:
        path: PATH to the hamiltonian folder
        file_name: file name of each hamiltonian
        key:

    Returns: (DMRG result, FCI result)

    """

    # Htbt is in a^ a a^ a format
    # Htbt = feru.get_chemist_tbt(tbt, spin_orb, spin_orb=True)
    # Htbt is in a^ a^ a a format

    Htbt = feru.get_two_body_tensor(tbt, spin_orb)
    print("Check permutation Symmetry Before correction",
          check_permutation_symmetries_complex_orbitals(
              onebody_matrix, Htbt))

    # Get one body difference
    one_body = varu.get_one_body_correction_from_tbt(tbt,
                                                   feru.get_chemist_tbt(tbt))

    # Get the one body tensor from the difference
    # onebody_matrix = feru.get_obt(one_body, n=spin_orb, spin_orb=True)
    onebody_matrix = np.zeros((spin_orb, spin_orb))
    # Check symmetries
    print("Check the Spin Symmetry", check_spin_symmetry(onebody_matrix, Htbt))
    # Get the correct permutation of tensors for Block2
    # one_body_tensor, two_body_tensor = (
    #     get_correct_permutation(onebody_matrix, Htbt, spin_orbs))
    one_body_tensor, two_body_tensor = (
        get_correct_permutation(onebody_matrix, Htbt, spin_orbs))
    print("Check permutation Symmetry After correction",
          check_permutation_symmetries_complex_orbitals(
              one_body_tensor, two_body_tensor))

    new_one_body_tensor, new_two_body_tensor, spin_symm_broken = (
        spinorbitals_to_orbitals(onebody_matrix, 2 * Htbt))

    print("Spin symmetry broken", spin_symm_broken)

    result = single_qchem_dmrg_calc(new_one_body_tensor, new_two_body_tensor,
                                    dmrg_param)

    return result["dmrg_ground_state_energy"], E_min



def cas_each_block_to_dmrg(onebody_matrix, tbt, spin_orb, k):
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

        tmp = feru.get_ferm_op(tbt[s:t, s:t, s:t, s:t], True)
        # Htbt = feru.get_two_body_tensor(tmp, len(orbs))
        Htbt = feru.get_chemist_tbt(tmp, len(orbs), spin_orb=True)
        onebody_matrix = np.zeros((len(orbs), len(orbs)))
        print("Check permutation Symmetry Before correction",
              check_permutation_symmetries_complex_orbitals(
                  onebody_matrix, Htbt))

        # Get one body difference
        one_body = varu.get_one_body_correction_from_tbt(tmp,
                                                       feru.get_chemist_tbt(tmp))

       #  Get the one body tensor from the difference
        onebody_matrix = feru.get_obt(one_body, n=len(orbs), spin_orb=True)
        # onebody_matrix = np.zeros((len(orbs), len(orbs)))
        # Check symmetries
        print("Check the Spin Symmetry",
              check_spin_symmetry(onebody_matrix, Htbt))
        one_body_tensor, two_body_tensor = (
            get_correct_permutation(onebody_matrix, Htbt, len(orbs)))
        print("Check permutation Symmetry After correction",
              check_permutation_symmetries_complex_orbitals(
                  one_body_tensor, two_body_tensor))

        new_one_body_tensor, new_two_body_tensor, spin_symm_broken = (
            spinorbitals_to_orbitals(onebody_matrix, 2 * Htbt))

        print("Spin symmetry broken", spin_symm_broken)

        result = single_qchem_dmrg_calc(new_one_body_tensor,
                                        new_two_body_tensor,
                                        dmrg_param)
        E_total.append(result["dmrg_ground_state_energy"])

    return E_total, E_min


if __name__ == "__main__":
    # sys.settrace(trace)
    ps_path = "../CAS_Cropping/planted_solutions/"
    # File name in ps_path folder
    file_name = "2_co2_6-311++G___12_9d464efb-b312-45f8-b0ba-8c42663059dc.pkl"
    (tbt, tbt_hidden, Htbt_with_killer, k, Htbt_hidden, sol, e_nums,
     E_min, spin_orbs) = construct_Hamiltonian_with_solution(ps_path, file_name)

    ne_each = e_nums

    init_state_bond_dimension = 50
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
    num_unpaired_electrons = sum(e_nums)
    multiplicity = 1

    dmrg_process_param = get_dmrg_process_param(
        init_state_bond_dimension, max_num_sweeps, energy_convergence_threshold,
        sweep_schedule_bond_dims, sweep_schedule_noise,
        sweep_schedule_davidson_threshold)
    dmrg_param = get_dmrg_param(
        num_orbitals, num_electrons, num_unpaired_electrons,
        multiplicity, dmrg_process_param)

    onebody_matrix = np.zeros((spin_orbs, spin_orbs))
    tbt_in_ferm = feru.get_ferm_op(tbt, spin_orb=spin_orbs)

    ground_energy_tbt = get_fci_ground_energy(tbt)

    print("FCI Ground energy tbt", ground_energy_tbt)
    print("FCI energy for each block tbt",
          get_energy_from_each_block(tbt, k, ne_each)[0])
    result_each = cas_each_block_to_dmrg(onebody_matrix, tbt, spin_orbs, k)
    print("Each block result:", result_each[0])
    result = cas_to_dmrg(onebody_matrix, tbt_in_ferm, spin_orbs, k)
    print("DMRG Ground energy from blocks", sum(result_each[0]))
    print("DMRG Ground energy tbt", result[0])
