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


def construct_blocks(b: int, spin_orbs: int):
    """
    Constructs a CAS block partitioning
    Args:
        b:
        spin_orbs:

    Returns:

    """
    b = b * 2
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
        Htbt:
        k:
        casnum:

    Returns:

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
    E_cas = 0
    ne_range = 0
    balance_t = 10

    # orbs is each CAS block
    for i in range(len(k)):
        orbs = k[i]
        ne = ne_per_block[i]

        s = orbs[0]
        t = orbs[-1] + 1
        print("s,t:", s, t)

        # Number of orbitals in the block
        norbs = len(orbs)

        # Number of electrons in the block
        # ne = min(ne_per_block + random.randint(-ne_range, ne_range), norbs - 1)

        print(f"Ne within current block: {ne}")
        # Construct (Ne^-ne)^2 terms in matrix, to enforce structure of states
        balance_tbt = np.zeros([norbs, norbs, norbs, norbs, ])
        if ne_per_block != 0:
            for p, q in product(range(norbs), repeat = 2):
                balance_tbt[p,p,q,q] += 1
            for p in range(len(orbs)):
                balance_tbt[p,p,p,p] -= 2 * ne
#             Construct 2e tensor to enforce the Ne in the ground state.
            #
#             tmp_tbt = np.add(tmp_tbt, balance_tbt)
        flag = True
        while flag:
            # balance_tbt = np.zeros([norbs, norbs, norbs, norbs, ])
            # balance_tbt is added to each element of CAS
            balance_tbt *= balance_t
            cas_tbt[s:t, s:t, s:t, s:t] = np.add(cas_tbt[s:t, s:t, s:t, s:t], balance_tbt)
            tmp = feru.get_ferm_op(cas_tbt[s:t, s:t, s:t, s:t], True)
            sparse_H_tmp = of.get_sparse_operator(tmp)
            tmp_E_min, t_sol = of.get_ground_state(sparse_H_tmp)

            # Adding the ground state elements to forma a Slater determinant
            st = sdstate(n_qubit = len(orbs))
            for i in range(len(t_sol)):
                if np.linalg.norm(t_sol[i]) > np.finfo(np.float32).eps:
                    st += sdstate(s = i, coeff = t_sol[i])
            st.normalize()
            E_st = st.exp(tmp)
            flag = False
            # for sd in st.dic:
            #     ne_computed = bin(sd)[2:].count('1')
            #     if ne_computed != ne:
            #         flag = True
            #         break
        print(f"E_min: {tmp_E_min} for orbs: {orbs}")
        print(f"current state Energy: {E_st}")
        E_cas += E_st
        states.append(st)
        e_nums.append(ne)
    return e_nums, states, E_cas, cas_tbt


def construct_killer(k, e_nums, n=0, const=1e-2, t=1e2, n_killer=5):
    """
    Construct a killer operator for CAS Hamiltonian, based on cas block structure of k and the size of killer is
    given in k, the number of electrons in each CAS block of the ground state
    is specified by e_nums. t is the strength of quadratic balancing terms for the killer with respect to k,
    n_killer specifies the number of operators O to choose.
    Args:
        k:
        e_nums:
        n:
        const:
        t:
        n_killer:

    Returns:

    """
    if not n:
        n = max([max(orbs) for orbs in k])
    killer = of.FermionOperator.zero()
    for i in range(len(k)):
        orbs = k[i]
        outside_orbs = [j for j in range(n) if j not in orbs]
    #     Construct Ne
        Ne = sum([of.FermionOperator("{}^ {}".format(i, i)) for i in orbs])
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


def convert_data_to_cas(load_result, setting_dict, file_name: None):
    """
    Converts the data into CAS Hamiltonian
    Args:
        load_result:
        setting_dict:
        file_name:

    Returns:

    """
    ne_per_block = setting_dict['ne_per_block']
    block_partitioning = setting_dict['block_partitioning']

    # These tensors are in physicist ordering
    one_body_phy = load_result['one_body_tensor']
    two_body_phy = load_result['two_body_tensor']
    spin_orbs = load_result['num_spin_orbitals']
    e_num_actual = load_result['num_electrons']

    # Convert the physicist notation to chemist notation
    one_body, two_body = physicist_to_chemist(
        one_body_phy, two_body_phy, spin_orbs)

    # obt_in_tbt is chemist notation
    obt_in_tbt = feru.onebody_to_twobody(one_body)
    # Htbt in chemist notation
    Htbt_added = np.add(two_body, obt_in_tbt)

    print("Shape of Htbt:", Htbt_added.shape)
    print(f"orbital splliting: {block_partitioning}")

    upnum, casnum, pnum = csau.get_param_num(spin_orbs, block_partitioning, complex=False)
    # Construct the truncated CAS Hamiltonian
    cas_tbt, cas_x = get_truncated_cas_tbt(Htbt_added, block_partitioning, casnum)

    # print("Finding the Minimum Energy")
    # H_original = feru.get_ferm_op(cas_tbt)
    # E_min, sol = of.get_ground_state(of.get_sparse_operator(H_original))
    # print("Original Emin:", E_min)
    e_nums, states, E_cas, cas_tbt_with_b = solve_enums(
        cas_tbt, block_partitioning, ne_per_block=ne_per_block,)
    # print(f"e_nums:{e_nums}")
    # print(f"E_cas: {E_cas}")

    check_symmetry = False
    FCI = True
    H_cas = feru.get_ferm_op(cas_tbt, True)
    cas_killer = construct_killer(block_partitioning, ne_per_block, n=spin_orbs)

    if check_symmetry:
        symmetry_test.test_sz_symmetry(H_cas, spin_orbs)
        symmetry_test.test_s2_symmetry(H_cas, spin_orbs)
        symmetry_test.test_sz_symmetry(cas_killer, spin_orbs)
        symmetry_test.test_s2_symmetry(cas_killer, spin_orbs)

    if FCI:
        E_min, sol = of.get_ground_state(of.get_sparse_operator(H_cas))
        print(f"FCI Energy: {E_min}")
        tmp_st = sdstate(n_qubit=spin_orbs)
        for s in range(len(sol)):
            if sol[s] > np.finfo(np.float32).eps:
                tmp_st += sdstate(s, sol[s])
        tmp_st.normalize()
        print(f"FCI energy with SD: {tmp_st.exp(H_cas)}")

    planted_sol = {}
    planted_sol["E_min"] = E_cas
    planted_sol["e_nums"] = e_nums
    planted_sol["sol"] = states
    planted_sol["killer"] = cas_killer
    planted_sol["cas_x"] = cas_x
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
    # l = list(map(len, k))
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


if __name__ == "__main__":
    setting_dict = {
        "ne_per_block": [5, 3],
        "block_partitioning": [[i for i in range(8)], [j+8 for j in range(4)]]
    }
    file_name = "fcidump.2_co2_6-311++G**"
    load_result = load_tensor("../data/fcidump.2_co2_6-311++G**")
    convert_data_to_cas(load_result, setting_dict, file_name)
