import CAS_Cropping.ferm_utils as feru
import CAS_Cropping.csa_utils as csau
from CAS_Cropping.sdstate import *
from itertools import product
import random
import h5py
import os
import pickle
import DMRG_simulation.test.symmetry_test as symmetry_test
from DMRG_simulation.data_repository.catalysts_loader import load_tensor
from DMRG_simulation.format_dmrg.format_tensor import physicist_to_chemist
from CAS.dmrghandler.src.dmrghandler.pyscf_wrappers import two_body_tensor_orbital_to_spin_orbital, one_body_tensor_orbital_to_spin_orbital


def partition_num_elec(ne_block, e_total, total_block):
    e_partitioning = []
    total = 0
    while total + ne_block < e_total:
        e_partitioning.append(ne_block)
        total += ne_block
    e_partitioning.append(e_total - total)
    while len(e_partitioning) < total_block:
        e_partitioning.append(0)
    return e_partitioning


def construct_blocks(b: int, spin_orbs: int):
    """
    Constructs a CAS block partitioning
    Args:
        b: size of each block
        spin_orbs: total number of spin orbitals

    Returns:

    """
    k = []
    tmp = [0]
    for i in range(1, spin_orbs):
        if i % b == 0:
            k.append(tmp)
            tmp = [i]
        else:
            tmp.append(i)
    if len(tmp) != 0:
        k.append(tmp)
    return k


def get_truncated_cas_tbt(Htbt, k, casnum):
    """
    Constructs a CAS block two body tensor
    Args:
        Htbt: The Hamiltonian to be truncated
        k: block partitioning
        casnum: the total number of elements of the CAS hamiltonian

    Returns: Truncated tbt and 1D cas_x array

    """
    cas_tbt = np.zeros(Htbt.shape)
    cas_x = np.zeros(casnum)
    idx = 0
    for block in k:
        for a in block:
            for b in block:
                for c in block:
                    for d in block:
                        cas_tbt[a, b, c, d] = Htbt[a, b, c, d]
                        cas_x[idx] = Htbt[a, b, c, d]
                        idx += 1
    return cas_tbt, cas_x


def solve_enums(cas_tbt, k, ne_per_block):
    """
    Solve for number of electrons in each CAS block with FCI within the block
    Parameters:
        cas_tbt: CAS Hamiltonian
        k: block partitioning
        ne_per_block: number of electrons in each CAS block
        ne_range:
        balance_t:
    """
    e_nums = []
    states = []
    E_min_each = []
    E_cas = 0
    ne_range = 0
    balance_t = 10

    # orbs is each CAS block
    for i in range(len(k)):
        spinorbs = k[i]
        ne = ne_per_block[i]

        s = spinorbs[0]
        t = spinorbs[-1] + 1
        # print("s,t:", s, t)

        # Number of spin orbitals in the block
        nspinorbs = len(spinorbs)

        # Number of electrons in the block
        # ne = min(ne_per_block + random.randint(-ne_range, ne_range), norbs - 1)

        # print(f"Ne within current block: {ne}")
        # Construct (Ne^-ne)^2 terms in matrix, to enforce structure of states
        balance_tbt = np.zeros([nspinorbs, nspinorbs, nspinorbs, nspinorbs, ])
        if ne_per_block != 0:
            for p, q in product(range(nspinorbs), repeat = 2):
                balance_tbt[p,p,q,q] += 1
            for p in range(len(spinorbs)):
                balance_tbt[p,p,p,p] -= 2 * ne
#             Construct 2e tensor to enforce the Ne in the ground state.
            #
#             tmp_tbt = np.add(tmp_tbt, balance_tbt)
        flag = True
        while flag:
            # balance_tbt = np.zeros([norbs, norbs, norbs, norbs, ])
            # balance_tbt is added to each element of CAS
            cas_tbt[s:t, s:t, s:t, s:t] = np.add(cas_tbt[s:t, s:t, s:t, s:t], balance_tbt)
            tmp = feru.get_ferm_op(cas_tbt[s:t, s:t, s:t, s:t], True)
            sparse_H_tmp = of.get_sparse_operator(tmp)
            tmp_E_min, t_sol = of.get_ground_state(sparse_H_tmp)

            # Adding the ground state elements to form a Slater determinant
            st = sdstate(n_qubit=len(spinorbs))
            for i in range(len(t_sol)):
                if np.linalg.norm(t_sol[i]) > np.finfo(np.float32).eps:
                    st += sdstate(s=i, coeff=t_sol[i])
            st.normalize()
            E_st = st.exp(tmp)
            E_min_each.append(E_st)
            flag = False
            # for sd in st.dic:
            #     ne_computed = bin(sd)[2:].count('1')
            #     if ne_computed != ne:
            #         flag = True
            #         break
        # print(f"Make sure E_min:{tmp_E_min} = current state Energy: {E_st}")
        E_cas += E_st
        states.append(st)
        e_nums.append(ne)
    return e_nums, states, E_cas, cas_tbt, E_min_each


def construct_killer(k, e_nums, tot_spin_orbs=0, const=1e-2, t=1e2, n_killer=5):
    """
    Construct a killer operator for CAS Hamiltonian, based on cas block structure of k and the size of killer is
    given in k, the number of electrons in each CAS block of the ground state
    is specified by e_nums. t is the strength of quadratic balancing terms for the killer with respect to k,
    n_killer specifies the number of operators O to choose.
    Args:
        k: block partitioning
        e_nums: number of electrons in each CAS block
        tot_spin_orbs: spin orbitals
        const:
        t:
        n_killer:

    Returns:

    """
    if not tot_spin_orbs:
        tot_spin_orbs = max([max(orbs) for orbs in k])
    killer = of.FermionOperator.zero()
    for i in range(len(k)):
        spinorb = k[i]
        outside_orbs = [j for j in range(tot_spin_orbs) if j not in spinorb]
        Ne = sum([of.FermionOperator("{}^ {}".format(i, i)) for i in spinorb])
    #     Construct O, for O as combination of Epq which preserves Sz and S2
        if len(outside_orbs) >= 4:
            tmp = 0
            while tmp < n_killer:
                p, q = random.sample(outside_orbs, 2)
                if abs(p - q) > 1:
                    O = of.FermionOperator.zero()
                    if p % 2 != 0:
                        p -= 1
                    if q % 2 != 0:
                        q -= 1
                    ferm_op = of.FermionOperator("{}^ {}".format(p, q)) + of.FermionOperator("{}^ {}".format(q, p))
                    O += ferm_op
                    O += of.hermitian_conjugated(ferm_op)
                    ferm_op = of.FermionOperator("{}^ {}".format(p + 1, q + 1)) + of.FermionOperator("{}^ {}".format(q + 1, p + 1))
                    O += ferm_op
                    O += of.hermitian_conjugated(ferm_op)
                    killer += (1 + np.random.rand()) * const * O * (Ne - e_nums[i])
                    tmp += 1
        killer += t * (1 + np.random.rand()) * const * ((Ne - e_nums[i]) ** 2)
    return killer


def convert_data_to_cas(load_result, setting_dict):
    """
    Converts the data into CAS Hamiltonian
    Args:
        load_result:
        setting_dict:
        file_name:

    Returns:

    """

    # 1, Extracting & formatting the data #
    ne_block = setting_dict['num_e_block']
    block_size = setting_dict['block_size']

    # These tensors are in physicist ordering, spin orbitals
    one_body_phy = load_result['one_body_tensor']
    two_body_phy = load_result['two_body_tensor']
    spin_orbs = load_result['num_spin_orbitals']
    e_num_actual = load_result['num_electrons']
    # Chemist ordering of a combined tensor
    tbt_combined = load_result['combined_chem_tbt']

    block_partitioning = construct_blocks(block_size, spin_orbs)
    ne_per_block = partition_num_elec(
        ne_block, e_num_actual, len(block_partitioning))

    # Convert the physicist notation to chemist notation spin orbitals
    one_body, two_body = physicist_to_chemist(
        one_body_phy, two_body_phy, spin_orbs)

    # obt_in_tbt is chemist notation
    obt_in_tbt = feru.onebody_to_twobody(one_body)
    # Htbt_added in chemist notation
    Htbt_added = np.add(two_body, obt_in_tbt)

    print("Shape of Htbt:", Htbt_added.shape)
    print("Shape of tbt combined:", tbt_combined.shape)
    print(f"orbital splliting: {block_partitioning}")

    # 2, Truncate the Hamiltonian

    upnum, casnum, pnum = csau.get_param_num(
        spin_orbs, block_partitioning, complex=False)
    # Construct the truncated CAS Hamiltonian
    cas_tbt, cas_x = get_truncated_cas_tbt(
        tbt_combined, block_partitioning, casnum)

    # H_original = feru.get_ferm_op(cas_tbt)
    # E_min, sol = of.get_ground_state(of.get_sparse_operator(H_original))

    # 3, Get the planted eigenvalues for the truncated Hamiltonian
    e_nums, states, E_cas, cas_tbt_with_b, E_min_each= solve_enums(
        cas_tbt, block_partitioning, ne_per_block=ne_per_block,)

    # 4, Check for symmetry breaking & creating a killer operator
    check_symmetry = True
    FCI = True
    H_cas = feru.get_ferm_op(cas_tbt, True)
    cas_killer = construct_killer(block_partitioning, ne_per_block, tot_spin_orbs=spin_orbs)

    if check_symmetry:
        symmetry_test.test_sz_symmetry(H_cas, spin_orbs)
        symmetry_test.test_s2_symmetry(H_cas, spin_orbs)
        symmetry_test.test_sz_symmetry(cas_killer, spin_orbs)
        symmetry_test.test_s2_symmetry(cas_killer, spin_orbs)

    if FCI:
        E_min, sol = of.get_ground_state(of.get_sparse_operator(H_cas))
        print(f"FCI Energy: {E_min} = E_cas: {E_cas}")
        tmp_st = sdstate(n_qubit=spin_orbs)
        for s in range(len(sol)):
            if sol[s] > np.finfo(np.float32).eps:
                tmp_st += sdstate(s, sol[s])
        tmp_st.normalize()
        print(f"FCI energy with SD: {tmp_st.exp(H_cas)}")

        sparse_with_killer = of.get_sparse_operator(cas_killer + H_cas)
        killer_Emin, killer_sol = of.get_ground_state(sparse_with_killer)
        print(f"FCI Energy solution with killer: {killer_Emin}")

        tmp_st_killer = sdstate(n_qubit=spin_orbs)
        for s in range(len(killer_sol)):
            if killer_sol[s] > np.finfo(np.float32).eps:
                tmp_st_killer += sdstate(s, killer_sol[s])
        tmp_st_killer.normalize()
        sd_Emin = tmp_st.exp(H_cas) + tmp_st_killer.exp(cas_killer)
        print(f"SD Energy with Killer: {sd_Emin}")
        print(f"difference with CAS energy: {sd_Emin - killer_Emin}")

    # 5, Return or save the data
    planted_sol = {}
    planted_sol["E_min"] = E_cas
    planted_sol["killer_Emin"] = killer_Emin
    planted_sol["e_nums"] = e_nums
    planted_sol["sol"] = states
    planted_sol["killer"] = cas_killer
    planted_sol["cas_x"] = cas_x
    planted_sol["H_cas"] = H_cas
    planted_sol["cas_tbt_with_b"] = cas_tbt_with_b
    planted_sol["k"] = block_partitioning
    planted_sol["casnum"] = casnum
    planted_sol["pnum"] = pnum
    planted_sol["upnum"] = upnum
    planted_sol["spin_orbs"] = spin_orbs
    planted_sol["two_S"] = load_result["two_S"]
    planted_sol["two_Sz"] = load_result["two_Sz"]
    planted_sol["e_num_actual"] = e_num_actual
    planted_sol["tbt_before_truncate"] = Htbt_added
    planted_sol["original_one_body"] = one_body_phy
    planted_sol["original_two_body"] = two_body_phy

    return planted_sol
    # ps_path = "../planted_solutions/"
    # f_name = file_name.split(".")[1] + ".pkl"
    # print("Saved file path:", ps_path + f_name)
    #
    # l = list(map(len, block_partitioning))
    # l = list(map(str, l))
    # key = "-".join(l)
    # print(key)
    # if os.path.exists(ps_path + f_name):
    #     with open(ps_path + f_name, 'rb') as handle:
    #         dic = pickle.load(handle)
    # else:
    #     dic = {}
    #
    # with open(ps_path + f_name, 'wb') as handle:
    #     dic[key] = planted_sol
    #     pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)


def cas_from_tensor(combined_tbt, setting_dict):


    # 1, Extracting & formatting the data #
    ne_block = setting_dict['num_e_block']
    block_size = setting_dict['block_size']
    e_num_actual = setting_dict['actual_e_num']

    spin_orbs = combined_tbt.shape[0]

    block_partitioning = construct_blocks(block_size, spin_orbs)
    ne_per_block = partition_num_elec(
        ne_block, e_num_actual, len(block_partitioning))

    print(f"block partitioning: {block_partitioning}")

    # 2, Truncate the Hamiltonian

    upnum, casnum, pnum = csau.get_param_num(
        spin_orbs, block_partitioning, complex=False)
    # Construct the truncated CAS Hamiltonian
    cas_tbt, cas_x = get_truncated_cas_tbt(
        combined_tbt, block_partitioning, casnum)

    # H_original = feru.get_ferm_op(cas_tbt)
    # E_min, sol = of.get_ground_state(of.get_sparse_operator(H_original))

    # 3, Get the planted eigenvalues for the truncated Hamiltonian
    e_nums, states, E_cas, cas_tbt_with_b, actual_e_num = solve_enums(
        cas_tbt, block_partitioning, ne_per_block=ne_per_block, )

    # 4, Check for symmetry breaking & creating a killer operator
    check_symmetry = False
    FCI = False
    H_cas = feru.get_ferm_op(cas_tbt, True)
    cas_killer = construct_killer(block_partitioning, ne_per_block,
                                  tot_spin_orbs=spin_orbs)

    if check_symmetry:
        symmetry_test.test_sz_symmetry(H_cas, spin_orbs)
        symmetry_test.test_s2_symmetry(H_cas, spin_orbs)
        symmetry_test.test_sz_symmetry(cas_killer, spin_orbs)
        symmetry_test.test_s2_symmetry(cas_killer, spin_orbs)

    if FCI:
        E_min, sol = of.get_ground_state(of.get_sparse_operator(H_cas))
        tmp_st = sdstate(n_qubit=spin_orbs)
        for s in range(len(sol)):
            if sol[s] > np.finfo(np.float32).eps:
                tmp_st += sdstate(s, sol[s])
        tmp_st.normalize()

        sparse_with_killer = of.get_sparse_operator(cas_killer + H_cas)
        killer_Emin, killer_sol = of.get_ground_state(sparse_with_killer)

        tmp_st_killer = sdstate(n_qubit=spin_orbs)
        for s in range(len(killer_sol)):
            if killer_sol[s] > np.finfo(np.float32).eps:
                tmp_st_killer += sdstate(s, killer_sol[s])
        tmp_st_killer.normalize()
        sd_Emin = tmp_st.exp(H_cas) + tmp_st_killer.exp(cas_killer)

    # 5, Return or save the data
    planted_sol = {}
    planted_sol["E_min"] = E_cas
    planted_sol["e_nums"] = e_nums
    planted_sol["sol"] = states
    planted_sol["killer"] = cas_killer
    planted_sol["cas_x"] = cas_x
    planted_sol["H_cas"] = H_cas
    planted_sol["cas_tbt_with_b"] = cas_tbt_with_b
    planted_sol["cas_tbt"] = cas_tbt
    planted_sol["k"] = block_partitioning
    planted_sol["casnum"] = casnum
    planted_sol["pnum"] = pnum
    planted_sol["upnum"] = upnum
    planted_sol["spin_orbs"] = spin_orbs

    return planted_sol


if __name__ == "__main__":
    # print(construct_blocks(6, 14))
    # print(partition_num_elec(4, 6, len(construct_blocks(6, 14))))
    setting_dict = {
        "num_e_block": 4,
        "block_size": 6
    }
    file_name = "fcidump.2_co2_6-311++G**"
    load_result = load_tensor("../data/fcidump.2_co2_6-311++G**")
    convert_data_to_cas(load_result, setting_dict)
