import numpy as np
from DMRG_simulation.planted_solution_loader import construct_Hamiltonian_with_solution
from DMRG_simulation.utils.adding_balance import add_balance
from DMRG_simulation.format_dmrg.format_tensor import get_correct_permutation
from CAS.dmrghandler.src.dmrghandler.dmrg_calc_prepare import check_spin_symmetry

import CAS_Cropping.ferm_utils as feru
import CAS_Cropping.var_utils as varu
import openfermion as of
import sys
import CAS.saveload_utils as sl

sys.path.append("../CAS/")

def test_one_body_symmetry(one_body_tensor):
    num_orbitals = one_body_tensor.shape[0]
    symm_check_passed = True
    for p in range(num_orbitals):
        for q in range(num_orbitals):
            if not np.allclose(
                    one_body_tensor[p, q],
                    (one_body_tensor[q, p]).conj()):
                symm_check_passed = False
    assert symm_check_passed, "One body symmetry check failed."


def test_two_body_symmetry_pq_rs(two_body_tensor):
    num_spin_orbitals = two_body_tensor.shape[0]
    num_orbitals = num_spin_orbitals // 2
    symm_check_passed = True
    for p in range(num_orbitals):
        for q in range(num_orbitals):
            for r in range(num_orbitals):
                for s in range(num_orbitals):
                    if (not np.allclose(
                            two_body_tensor[2 * p, 2 * q, 2 * r, 2 * s],
                            two_body_tensor[
                                2 * p + 1, 2 * q + 1, 2 * r, 2 * s
                            ],
                        )):
                        symm_check_passed = False

    assert symm_check_passed, "pq, rs not exchangeable. Symmetry check failed."


def test_two_body_symmetry_conj_pqrs(two_body_tensor):
    num_spin_orbitals = two_body_tensor.shape[0]
    num_orbitals = num_spin_orbitals // 2
    symm_check_passed = True
    for p in range(num_orbitals):
        for q in range(num_orbitals):
            for r in range(num_orbitals):
                for s in range(num_orbitals):
                    if (not np.allclose(
                            two_body_tensor[2 * p, 2 * q, 2 * r, 2 * s],
                            two_body_tensor[
                                2 * p, 2 * q, 2 * r + 1, 2 * s + 1
                            ],
                        )):
                        symm_check_passed = False

    assert symm_check_passed, "pqrs <=> qpsr conj not exchangeable. Symmetry check failed."


def test_two_body_symmetry_conj_srqp(two_body_tensor):
    num_spin_orbitals = two_body_tensor.shape[0]
    num_orbitals = num_spin_orbitals // 2
    symm_check_passed = True
    for p in range(num_orbitals):
        for q in range(num_orbitals):
            for r in range(num_orbitals):
                for s in range(num_orbitals):
                    if (not np.allclose(
                            two_body_tensor[2 * p, 2 * q, 2 * r, 2 * s],
                            two_body_tensor[
                                2 * p + 1,
                                2 * q + 1,
                                2 * r + 1,
                                2 * s + 1,
                            ],
                        )):
                        symm_check_passed = False

    assert symm_check_passed, "pqrs <=> srqp conj not exchangeable. Symmetry check failed."


def test_spin_symmetry(obt, tbt):
    assert check_spin_symmetry(obt, tbt), "Spin symmetry check failed"


def test_sz_symmetry(H_cas, spin_orbs):
    spatial_orbs = spin_orbs // 2
    Sz = of.hamiltonians.sz_operator(spatial_orbs)
    assert of.FermionOperator.zero() == of.normal_ordered(
        of.commutator(Sz, H_cas)), "Sz symmetry broken"


def test_s2_symmetry(H_cas, spin_orbs):
    spatial_orbs = spin_orbs // 2
    S2 = of.hamiltonians.s_squared_operator(spatial_orbs)
    assert of.FermionOperator.zero() == of.normal_ordered(
        of.commutator(S2, H_cas)), "S2 symmetry broken"


if __name__ == '__main__':
    # sys.settrace(trace)
    ps_path = "../../CAS_Cropping/planted_solutions/"
    # File name in ps_path folder
    file_name = "2_co2_6-311++G___12_9d464efb-b312-45f8-b0ba-8c42663059dc.pkl"


    (tbt, tbt_hidden, Htbt_with_killer, k, Htbt_hidden, sol, e_nums,
     E_min, spin_orbs) = construct_Hamiltonian_with_solution(ps_path, file_name)
    tbt_copy = tbt
    # Adding a balance term to the loaded hamiltonian.
    ne_each = e_nums
    E_loaded, tbt_balance = add_balance(tbt, k, ne_each)

    print("Start test for tbt without any change")
    one_body_tensor = np.zeros((4, 4))
    test_one_body_symmetry(one_body_tensor)
    test_two_body_symmetry_pq_rs(tbt_copy)
    test_two_body_symmetry_pq_rs(tbt_balance)
    test_two_body_symmetry_conj_pqrs(tbt_copy)
    test_two_body_symmetry_conj_pqrs(tbt_balance)
    test_two_body_symmetry_conj_srqp(tbt_copy)
    test_two_body_symmetry_conj_srqp(tbt_balance)
    test_spin_symmetry(one_body_tensor, tbt_balance)

    # Get the Fermion Operator representation of the Hamiltonian
    tbt_in_ferm = feru.get_ferm_op(tbt_copy, spin_orb=spin_orbs)

    # Get the chemist notation of the Hamiltonian
    Htbt = feru.get_two_body_tensor(tbt_in_ferm, spin_orbs)
    # Htbt = feru.get_chemist_tbt(tbt_in_ferm, spin_orbs, spin_orb=True)
    print("Start test for tbt transformed to chemist tbt")

    test_two_body_symmetry_pq_rs(Htbt)
    test_two_body_symmetry_conj_pqrs(Htbt)
    test_two_body_symmetry_conj_srqp(Htbt)
    test_spin_symmetry(one_body_tensor, Htbt)

    print("Test for Sz symmetry broken")
    test_sz_symmetry(tbt_in_ferm, spin_orbs)

    print("Test for S2 symmetry broken")
    test_s2_symmetry(tbt_in_ferm, spin_orbs)

    print("Start test for one body difference")

    # Getting the one body difference of the Hamiltonian and the
    # Chemist notation of the Hamiltonian
    one_body = varu.get_one_body_correction_from_tbt(
        tbt_in_ferm, feru.get_chemist_tbt(tbt_in_ferm))

    # Get the one body tensor from the difference
    onebody_matrix = feru.get_obt(one_body, n=spin_orbs, spin_orb=True)
    # test_one_body_symmetry(onebody_matrix)

    print("Start test for corrected permutation")
    zero_matrix = np.zeros((spin_orbs, spin_orbs))
    one_body_tensor, two_body_tensor = (
        get_correct_permutation(zero_matrix, Htbt, spin_orbs))

    # After correcting the permutation, the one body symmetry still fails
    test_one_body_symmetry(one_body_tensor)
    test_two_body_symmetry_pq_rs(two_body_tensor)
    test_two_body_symmetry_conj_pqrs(two_body_tensor)
    test_two_body_symmetry_conj_srqp(two_body_tensor)
    test_spin_symmetry(one_body_tensor, two_body_tensor)
