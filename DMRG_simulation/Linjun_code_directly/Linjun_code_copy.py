import CAS_Cropping.CAS_cropping as cas_cropping
import CAS_Cropping.csa_utils as csau
import CAS_Cropping.ferm_utils as feru
import openfermion as of
import numpy as np
from CAS_Cropping.sdstate import *
from CAS_Cropping.matrix_utils import construct_orthogonal
from DMRG_simulation.format_dmrg.format_dmrg_param import get_dmrg_param, get_dmrg_process_param
from CAS.dmrghandler.src.dmrghandler.pyscf_wrappers import two_body_tensor_orbital_to_spin_orbital
from DMRG_simulation.obt_tbt_comparison.l2_norm_comparison import symmetrize_tensors
from DMRG_simulation.dmrg_simulator.dmrg_chem_tbt import get_dmrg_from_chem
import DMRG_simulation.data_repository.catalysts_loader as catalyst_loader

default_final_bond_dim = 100
default_sweep_schedule_bond_dims = [default_final_bond_dim] * 4 + [
    default_final_bond_dim
] * 4
default_sweep_schedule_noise = [1e-4] * 4 + [1e-5] * 4 + [0]
default_sweep_schedule_davidson_threshold = [1e-10] * 8


balance_strength = 2
ne_range = 2

def get_cas_dict(Htbt, setting_dict):
    spin_orbs = Htbt.shape[0]
    spatial_orbs = Htbt.shape[0] // 2
    block_size = setting_dict["block_size"]
    ne_per_block = setting_dict["ne_per_block"]
    k = cas_cropping.construct_blocks(block_size, spin_orbs)
    print(f"orbital splliting: {k}")
    upnum, casnum, pnum = csau.get_param_num(spin_orbs, k, complex=False)

    cas_tbt, cas_x = cas_cropping.get_truncated_cas_tbt(Htbt, k, casnum)
    #     cas_tbt_tmp = copy.deepcopy(cas_tbt)
    e_nums, states, E_cas = cas_cropping.solve_enums(cas_tbt, k, ne_per_block=ne_per_block,
                                        ne_range=ne_range,
                                        balance_t=balance_strength)
    #     assert np.allclose(cas_tbt_tmp, cas_tbt), "changed"
    print(f"e_nums:{e_nums}")
    print(f"E_cas: {E_cas}")
    #         sd_sol = sdstate()

    #         for st in states:
    #             sd_sol = sd_sol.concatenate(st)
    # The following code segment checks the state energy for the full Hamiltonian, takes exponential space
    # and time with respect to the number of blocks
    #     E_sol = sd_sol.exp(cas_tbt)
    #     print(f"Double check ground state energy: {E_sol}")

    # Checking ground state with FCI
    # Warning: This takes exponential time to run
    #     Checking H_cas symmetries
    FCI = False
    # Checking symmetries of the planted Hamiltonian, very costly
    check_symmetry = False

    if check_symmetry or FCI:
        H_cas = feru.get_ferm_op(cas_tbt, True)
    if check_symmetry:
        Sz = of.hamiltonians.sz_operator(spatial_orbs)
        S2 = of.hamiltonians.s_squared_operator(spatial_orbs)
        assert of.FermionOperator.zero() == of.normal_ordered(
            of.commutator(Sz, H_cas)), "Sz symmetry broken"
        assert of.FermionOperator.zero() == of.normal_ordered(
            of.commutator(S2, H_cas)), "S2 symmetry broken"

    if FCI:
        E_min, sol = of.get_ground_state(of.get_sparse_operator(H_cas))
        print(f"FCI Energy: {E_min}")
        tmp_st = sdstate(n_qubit=spin_orbs)
        for s in range(len(sol)):
            if sol[s] > np.finfo(np.float32).eps:
                tmp_st += sdstate(s, sol[s])
        #         print(bin(s))
        print(tmp_st.norm())
        tmp_st.normalize()
        print(tmp_st.exp(H_cas))

    cas_killer = cas_cropping.construct_killer(k, e_nums, n=spin_orbs)
    if check_symmetry:
        assert of.FermionOperator.zero() == of.normal_ordered(
            of.commutator(Sz, cas_killer)), "Killer broke Sz symmetry"
        assert of.FermionOperator.zero() == of.normal_ordered(
            of.commutator(S2, cas_killer)), "S2 symmetry broken"

    # Checking: if FCI of killer gives same result. Warning; takes exponential time
    if FCI:
        sparse_with_killer = of.get_sparse_operator(cas_killer + H_cas)
        killer_Emin, killer_sol = of.get_ground_state(sparse_with_killer)
        print(f"FCI Energy solution with killer: {killer_Emin}")
        sd_Emin = killer_sol.exp(cas_tbt) + killer_sol.exp(cas_killer)
        print(f"difference with CAS energy: {sd_Emin - killer_Emin}")

    # Checking: if killer does not change ground state
    #         killer_error = sd_sol.exp(cas_killer)
    #         print(f"Solution Energy shift by killer: {killer_error}")
    #     killer_E_sol = sd_sol.exp(H_cas + cas_killer)
    #     print(f"Solution Energy with killer: {killer_E_sol}")

    planted_sol = {}
    planted_sol["E_min"] = E_cas
    planted_sol["e_nums"] = e_nums
    planted_sol["sol"] = states
    planted_sol["killer"] = cas_killer
    planted_sol["cas_x"] = cas_x
    planted_sol["k"] = k
    planted_sol["casnum"] = casnum
    planted_sol["pnum"] = pnum
    planted_sol["upnum"] = upnum
    planted_sol["spin_orbs"] = spin_orbs

    return planted_sol


def construct_hamiltonian_with_solution(planted_sol):

    E_min = planted_sol["E_min"]
    cas_x = planted_sol["cas_x"]
    killer = planted_sol["killer"]
    k = planted_sol["k"]
    upnum = planted_sol["upnum"]
    spin_orbs = planted_sol["spin_orbs"]
    #     Number of electrons in each CAS block
    e_nums = planted_sol["e_nums"]
    #     Solution in a list of sdstates, each sdstate represent the ground state with in each CAS block.
    # This can be generalized into the general solution with concatenate() in sdstates, but not recommended as it
    # requires exponential time and space.
    sol = planted_sol["sol"]

    tbt = csau.get_cas_matrix(cas_x, spin_orbs, k)
    #     Construct hidden 2e tensor and Hamiltonian
    killer_tbt = feru.get_chemist_tbt(killer, spin_orbs, spin_orb=True)
    killer_one_body = of.normal_ordered(
        killer - feru.get_ferm_op(killer_tbt, spin_orb=True))
    killer_onebody_matrix = feru.get_obt(killer_one_body, n=spin_orbs,
                                         spin_orb=True)
    killer_onebody_tbt = feru.onebody_to_twobody(killer_onebody_matrix)
    Htbt_killer = np.add(killer_tbt, killer_onebody_tbt)
    Htbt_with_killer = np.add(tbt, Htbt_killer)
    #     Set up random unitary to hide 2e tensor, the parameters here are initally set as random unitary rotations
    random_uparams = np.random.rand(upnum)
    U = construct_orthogonal(spin_orbs, random_uparams)
    #     Hide 2e etensor with random unitary transformation
    Htbt_hidden = csau.cartan_orbtransf(Htbt_with_killer, U, complex=False)
    #     Hidden 2e tensor without killer
    tbt_hidden = csau.cartan_orbtransf(tbt, U, complex=False)
    return tbt, tbt_hidden, Htbt_with_killer, Htbt_hidden, sol, e_nums, E_min, spin_orbs


def conduct_dmrg(load_result, setting_dict):
    one_body_phy = load_result['one_body_tensor']
    two_body_phy = load_result['two_body_tensor']
    spatial_obt_phy = load_result['one_body_tensor_spatial']
    spatial_tbt_phy = load_result['two_body_tensor_spatial']
    spin_orbs = load_result['num_spin_orbitals']
    e_num_actual = load_result['num_electrons']
    two_S = load_result['two_S']
    two_Sz = load_result['two_Sz']

    Htbt = two_body_tensor_orbital_to_spin_orbital(
        symmetrize_tensors(spatial_obt_phy, spatial_tbt_phy))

    planted_sol = get_cas_dict(Htbt, setting_dict=setting_dict)

    tbt, tbt_hidden, Htbt_with_killer, Htbt_hidden, sol, e_nums, E_min, spin_orbs = construct_hamiltonian_with_solution(planted_sol)

    print("E_nums:", e_nums)
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

    result = get_dmrg_from_chem(tbt, spin_orbs, dmrg_param)
    # result_hidden = get_dmrg_from_chem(tbt_hidden, spin_orbs, dmrg_param)
    result_killer = get_dmrg_from_chem(Htbt_with_killer, spin_orbs, dmrg_param)
    # result_killer_hidden = get_dmrg_from_chem(Htbt_hidden, spin_orbs, dmrg_param)

    return result, result_killer, E_min


if __name__ == '__main__':
    path = "../data/"
    fcidump_file = "fcidump.2_co2_6-311++G**"
    load_result = catalyst_loader.load_tensor(path + fcidump_file)

    setting_dict = {
        "ne_per_block": 8,
        "block_size": 12,
    }

    result, result_killer, E_min = conduct_dmrg(load_result, setting_dict)
    print("Planted e_min found:", E_min)
    print("DMRG result:", result["dmrg_ground_state_energy"])
    print("DMRG killer:", result_killer["dmrg_ground_state_energy"])
