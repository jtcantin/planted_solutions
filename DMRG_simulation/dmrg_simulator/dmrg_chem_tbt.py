import CAS_Cropping.ferm_utils as feru
import CAS_Cropping.var_utils as varu
import openfermion as of
from CAS.dmrghandler.src.dmrghandler.qchem_dmrg_calc import single_qchem_dmrg_calc
from CAS.dmrghandler.src.dmrghandler.dmrg_calc_prepare import check_spin_symmetry, spinorbitals_to_orbitals
from DMRG_simulation.validator.symmetry_check import check_permutation_symmetries_complex_orbitals
import numpy as np
from DMRG_simulation.format_dmrg.format_tensor import get_correct_permutation, physicist_to_chemist
from DMRG_simulation.test.symmetry_test import test_two_body_symmetry_pq_rs, test_two_body_symmetry_conj_srqp, test_two_body_symmetry_conj_pqrs, test_one_body_symmetry
from openfermion import FermionOperator


def get_dmrg_from_chem(chem_tbt, spin_orb: int, dmrg_param: dict):
    """
    Get DMRG result from chemistry indices convention
    Args:
        chem_tbt:
        spin_orb: number of spin orbitals
        dmrg_param:

    Returns:

    """

    zero_matrix = np.zeros((spin_orb, spin_orb))
    # Get the physics indices convention h_pq, g_pqrs
    one_body_tensor, two_body_tensor = (
        get_correct_permutation(zero_matrix, chem_tbt, spin_orb))

    # new_one_body_tensor and new_two_body_tensor in physicist notation
    new_one_body_tensor, new_two_body_tensor, spin_symm_broken = (
        spinorbitals_to_orbitals(one_body_tensor, two_body_tensor))


    # chem_obt_2, chem_tbt_2 = physicist_to_chemist(new_one_body_tensor, new_two_body_tensor, new_one_body_tensor.shape[0])



    fci_emin, sol = None, None
    result = single_qchem_dmrg_calc(
        new_one_body_tensor, new_two_body_tensor, dmrg_param)
    return result


def get_dmrg_from_chem_original(chem_obt, chem_tbt, spin_orb: int, dmrg_param: dict):
    """
    Get DMRG result from chemistry indices convention
    Args:
        Hf: Hamiltonian in Fermion Operator
        spin_orb: number of spin orbitals

    Returns:

    """
    # Get the correct permutation of tensors for Block2
    one_body_tensor, two_body_tensor = (
        get_correct_permutation(chem_obt, chem_tbt, spin_orb))

    new_one_body_tensor, new_two_body_tensor, spin_symm_broken = (
        spinorbitals_to_orbitals(one_body_tensor, two_body_tensor))

    result = single_qchem_dmrg_calc(
        new_one_body_tensor, new_two_body_tensor, dmrg_param)
    return result["dmrg_ground_state_energy"], one_body_tensor, two_body_tensor


def get_ferm_op_two_phy(tbt, spin_orb):
    '''
    Return the corresponding fermionic operators based on tbt (two body tensor)
    This tensor can index over spin-orbtals or orbitals
    '''
    n = tbt.shape[0]
    op = FermionOperator.zero()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if not spin_orb:
                        for a in range(2):
                            for b in range(2):
                                op += FermionOperator(
                                    term = (
                                        (2*i+a, 1), (2*k+b, 1),
                                        (2*l+b, 0), (2*j+a, 0)
                                    ), coefficient=tbt[i, j, k, l]
                                )
                    else:
                        op += FermionOperator(
                            term=(
                                (i, 1), (k, 1),
                                (l, 0), (j, 0)
                            ), coefficient=tbt[i, j, k, l]
                        )
    return op

def get_ferm_op_one(obt, spin_orb):
    '''
    Return the corresponding fermionic operators based on one body tensor
    '''
    n = obt.shape[0]
    op = FermionOperator.zero()
    for i in range(n):
        for j in range(n):
            if not spin_orb:
                for a in range(2):
                    op += FermionOperator(
                        term = (
                            (2*i+a, 1), (2*j+a, 0)
                        ), coefficient=obt[i, j]
                    )
            else:
                op += FermionOperator(
                    term = (
                        (i, 1), (j, 0)
                    ), coefficient=obt[i, j]
                )
    return op
