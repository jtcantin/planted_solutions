import DMRG_simulation.data_repository.cas_converter as cas_converter
import DMRG_simulation.data_repository.catalysts_loader as catalyst_loader
import CAS_Cropping.csa_utils as csau
import CAS_Cropping.ferm_utils as feru
import openfermion as of
import numpy as np
from CAS_Cropping.matrix_utils import construct_random_sz_unitary
from pyscf.tools.fcidump import from_integrals
from DMRG_simulation.format_dmrg.format_tensor import get_correct_permutation
from CAS.dmrghandler.src.dmrghandler.dmrg_calc_prepare import (
    check_spin_symmetry, spinorbitals_to_orbitals)
import DMRG_simulation.test.symmetry_test as symmetry_test


def fcidump_tensor_formatting(obt: np.ndarray, tbt: np.ndarray):
    """

    Args:
        obt:
        tbt:

    Returns:

    """
    obt_original, tbt_original = get_correct_permutation(obt,
                                                         tbt,
                                                         obt.shape[0])
    new_one_body_tensor, new_two_body_tensor, spin_symm_broken = (
        spinorbitals_to_orbitals(obt_original, tbt_original))

    return new_one_body_tensor, new_two_body_tensor


def get_planted_solution_fcidump(path, catalyst_file, block_size, num_e_block, param) -> None:
    """
    Get the planted solution in FCIDump format
    Args:
        catalyst_file: the path to the catalyst file
        block_partitioning: The block partitioning of the hamiltonian
        param: Other parameters stored as a dictionary format

    Return None
    """
    # 1, Loading data # ✓

    load_result = catalyst_loader.load_tensor(path + catalyst_file)
    setting_dict = {
        "block_size": block_size,
        "num_e_block": num_e_block
    }
    e_num_total = load_result["num_electrons"]

    # 2, Convert data to CAS format # ✓
    planted_sol = cas_converter.convert_data_to_cas(load_result, setting_dict)

    cas_x = planted_sol["cas_x"]
    spin_orbs = planted_sol["spin_orbs"]
    killer = planted_sol["killer"]
    E_min = planted_sol["E_min"]
    killer_Emin = planted_sol["killer_Emin"]
    ferm_H_cas = planted_sol["H_cas"]
    block_partitioning = planted_sol["k"]
    tbt_original = csau.get_cas_matrix(cas_x, spin_orbs, block_partitioning)

    # 3, Adding killer and hiding # ✓

    killer_tbt = feru.get_chemist_tbt(killer, spin_orbs, spin_orb=True)
    killer_one_body = of.normal_ordered(
        killer - feru.get_ferm_op(killer_tbt, spin_orb=True))
    killer_one_body_matrix = feru.get_obt(
        killer_one_body, n=spin_orbs, spin_orb=True)
    killer_one_body_tbt = feru.onebody_to_twobody(killer_one_body_matrix)
    killer_op = np.add(killer_tbt, killer_one_body_tbt)
    unitary = construct_random_sz_unitary(spin_orbs)


    tbt_with_killer = np.add(tbt_original, killer_op)
    tbt_hidden_rotated = csau.cartan_orbtransf(tbt_original, unitary, complex=False)
    killer_tbt_hidden = csau.cartan_orbtransf(tbt_with_killer, unitary, complex=False)

    symmetry_test.test_sz_symmetry(feru.get_ferm_op(tbt_original),
                                   spin_orbs)
    symmetry_test.test_s2_symmetry(feru.get_ferm_op(tbt_original),
                                   spin_orbs)
    print("Tbt original symmetry test passed")
    # symmetry_test.test_s2_symmetry(feru.get_ferm_op(tbt_hidden_rotated), spin_orbs)
    # symmetry_test.test_sz_symmetry(feru.get_ferm_op(tbt_hidden_rotated),
    #                                spin_orbs)

    # 4, Transform each hamiltonian into physicist notation #

    zero_matrix = np.zeros((spin_orbs, spin_orbs))
    obt_original, tbt_original = fcidump_tensor_formatting(zero_matrix, tbt_original)
    obt_hidden, tbt_hidden = fcidump_tensor_formatting(zero_matrix,
                                                           tbt_hidden_rotated)

    print(type(obt_original))
    print(type(obt_hidden))
    # For hidden and killer ones, the spin symmetries fail

    # For the killer one, the permutation symmetry fails
    obt_with_killer_correct, tbt_with_killer_correct = fcidump_tensor_formatting(zero_matrix,
                                                           tbt_with_killer)
    print(type(obt_with_killer_correct))
    obt_killer_hidden, tbt_killer_hidden = fcidump_tensor_formatting(zero_matrix,
                                                           killer_tbt_hidden)

    # 5, Generate FCIDump files #

    filename = catalyst_file.split(".")[1]
    print(type(obt_hidden))
    try:
        from_integrals("fcidump." + filename + "planted_original", h1e=obt_original, h2e=tbt_original,
                       nmo=spin_orbs//2, nelec=e_num_total)
        from_integrals("fcidump." + filename + "planted_hidden", obt_hidden, tbt_hidden,
                       nmo=spin_orbs//2, nelec=e_num_total)
        from_integrals("fcidump." + filename + "planted_with_killer", obt_with_killer_correct, tbt_with_killer_correct,
                       nmo=spin_orbs//2, nelec=e_num_total)
        from_integrals("fcidump." + filename + "planted_killer_hidden", obt_killer_hidden, tbt_killer_hidden,
                       nmo=spin_orbs//2, nelec=e_num_total)
    except ValueError:
        print(Exception("Saving to FCIDump failed"))

    except TypeError:
        print(Exception("Type error happening during FCIDump generation"))

    except RuntimeError:
        print(Exception("Run time error during FCIDump generation"))


if __name__ == '__main__':
    # spin_orbs = 12
    # unitary = construct_random_sz_unitary(6)
    # symmetry_test.test_sz_symmetry(feru.get_ferm_op(unitary), spin_orbs)
    # symmetry_test.test_s2_symmetry(feru.get_ferm_op(unitary), spin_orbs)
    path = "./data/"
    fcidump_file = "fcidump.2_co2_6-311++G**"
    block_size = 6
    num_e_block = 4
    extra_param = {}
    get_planted_solution_fcidump(path, fcidump_file, block_size, num_e_block, extra_param)

