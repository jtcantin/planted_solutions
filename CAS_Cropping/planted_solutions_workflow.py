import copy

"""Construct CAS Hamiltonians with cropping
"""
import CAS_Cropping.ferm_utils as feru
from CAS_Cropping.sdstate import *
from itertools import product
import random
from CAS_Cropping.matrix_utils import construct_orthogonal
import pickle
import tensorflow as tf

### Parameters
tol = 1e-5
balance_strength = 2
save = False
# Number of spatial orbitals in a block
block_size = 3
# Number of electrons per block
ne_per_block = 6
# +- difference in number of electrons per block
ne_range = 0
# Number of killer operators for each CAS block
n_killer = 3
# Running Full CI to check compute the ground state, takes exponentially amount of time to execute
FCI = False
# Checking symmetries of the planted Hamiltonian, very costly
check_symmetry = False
# Concatenate the states in each block to compute the ground state solution
check_state = False


def construct_blocks(size: int, spin_orbs: int, spin_orb = False):
    """Construct CAS blocks with size for spin_orbs/spatial number of orbitals"""
    if spin_orb:
        size = size * 2
    blocks = []
    tmp = [0]
    for i in range(1, spin_orbs):
        if i % size == 0:
            blocks.append(tmp)
            tmp = [i]
        else:
            tmp.append(i)
    if len(tmp) != 0:
        blocks.append(tmp)
    return blocks

def get_truncated_cas_tbt(H, blocks, casnum):
    """
    Trunctate the original Hamiltonian two body tensor into the cas block structures
    Args:
        H: hamiltonian in tuple
        blocks: The block partitioning
        casnum:

    Returns: cas_obt, cas_tbt, cas_x

    """
    Hobt, Htbt = H
    n = Htbt.shape[0]
    cas_tbt = np.zeros([n, n, n, n])
    cas_obt = np.zeros([n, n])
    cas_x = np.zeros(casnum)
    idx = 0
    for block in blocks:
        for p, q in product(block, repeat = 2):
            cas_obt[p,q] = Hobt[p,q]
            cas_x[idx] = Hobt[p,q]
            idx += 1
        for a, b, c, d in product(block, repeat = 4):
            cas_tbt [a,b,c,d] = Htbt [a,b,c,d]
            cas_x[idx] = Htbt[a,b,c,d]
            idx += 1
    return cas_obt, cas_tbt, cas_x

def in_orbs(term, orbs):
    """Return if the term is a local excitation operator within orbs"""
    if len(term) == 2:
        return term[0][0] in orbs and term[1][0] in orbs
    elif len(term) == 4:
        return term[0][0] in orbs and term[1][0] in orbs and term[2][0] in orbs and term[3][0] in orbs
    return False

def solve_enums(H, blocks, e_total, ne_per_block = 0, ne_range = 0, balance_t = 10,):
    """Solve for number of electrons in each CAS block with FCI within the block,
    H = (obt, tbt) as the Hamiltonian in spatial orbitals.
    Notice that some quadratic terms (Ne-ne)^2 are added to ensure the correct number
    of electrons in the ground state of each block
    """
    cas_obt = H[0]
    cas_tbt = H[1]
    e_nums = []
    states = []
    E_cas = 0
    for index in range(len(blocks)):
        orbs = blocks[index]
        s = orbs[0]
        t = orbs[-1] + 1
        norbs = len(orbs)

        ne = min(ne_per_block + random.randint(-ne_range, ne_range), norbs * 2)

        if e_total - ne_per_block * (index + 1) >= 0:
            ne = ne_per_block
        elif e_total - ne_per_block * index >= 0:
            ne = e_total - ne_per_block * index
        else:
            ne = 0

        print(f"Ne within current block: {ne}")
#         Construct (Ne^-ne)^2 terms in matrix, to enforce structure of states
        if ne_per_block != 0:
            balance_obt = np.zeros([norbs, norbs])
            balance_tbt = np.zeros([norbs, norbs,  norbs, norbs])
            for p, q in product(range(norbs), repeat = 2):
                balance_tbt[p,p,q,q] += 1
            for p in range(len(orbs)):
                balance_obt[p,p] -= 2 * ne
#             Construct 2e tensor to enforce the Ne in the ground state.
            strength = balance_t * (1 + random.random())
#             tmp_tbt = np.add(tmp_tbt, balance_tbt)
        flag = True
        while flag:
            strength *= 2
            cas_tbt[s:t, s:t, s:t, s:t] = np.add(cas_tbt[s:t, s:t, s:t, s:t], strength * balance_tbt)
            cas_obt[s:t, s:t] = np.add(cas_obt[s:t, s:t], strength * balance_obt)
#             Set spin_orb to False to represent spatial orbital basis Hamiltonian
            tmp = feru.get_ferm_op(cas_tbt[s:t, s:t, s:t, s:t], spin_orb = False)
            tmp += feru.get_ferm_op(cas_obt[s:t, s:t], spin_orb = False)
            sparse_H_tmp = of.get_sparse_operator(tmp)
            tmp_E_min, t_sol = of.get_ground_state(sparse_H_tmp)
            st = sdstate(n_qubit = len(orbs) * 2)
            for i in range(len(t_sol)):
                if np.linalg.norm(t_sol[i]) > np.finfo(np.float32).eps:
                    st += sdstate(s = i, coeff = t_sol[i])
#             print(f"state norm: {st.norm()}")
            st.normalize()
            E_st = st.exp(tmp)
            flag = False
            # for sd in st.dic:
            #     ne_computed = bin(sd)[2:].count('1')
            #     if ne_computed != ne:
            #         print("Not enough balance, adding more terms")
            #         print(bin(sd))
            #         flag = True
            #         break
        print(f"E_min: {tmp_E_min} for orbs: {orbs}")
        print(f"current state Energy: {E_st}")
        E_cas += E_st
        states.append(st)
        e_nums.append(ne)
    return e_nums, states, E_cas


def H_to_sparse(H: of.FermionOperator, n):
    """ Construct the sparse tensor representation of the Hamiltonian, represented by a constant term, a
    """
    h1e_keys = []
    h1e_vals = []
    h2e_keys = []
    h2e_vals = []
    for key, val in H.terms.items():
        if len(key) == 2:
            h1e_keys.append([key[0][0], key[1][0]])
            h1e_vals.append(val)
        elif len(key) == 4:
            h2e_keys.append([key[0][0], key[1][0], key[2][0], key[3][0]])
            h2e_vals.append(val)
    sparse_h1 = tf.sparse.SparseTensor(indices = h1e_keys, values = h1e_vals, dense_shape = [n,n])
    sparse_h2 = tf.sparse.SparseTensor(indices = h2e_keys, values = h2e_vals, dense_shape = [n,n,n,n])
    return sparse_h1, sparse_h2

def sparse_to_H(H_sparse):
    H = of.FermionOperator.zero()
    for _, (term, value) in enumerate(zip(H_sparse.indices, H_sparse.values)):
        index = [int(i) for i in term.numpy()]
        val = value.numpy()
        if len(index) == 2:
            H += of.FermionOperator(
                term = (
                    (index[0], 1), (index[1], 0)
                ),
                coefficient = val
            )
        elif len(index) == 4:
            H += of.FermionOperator(
                term = (
                    (index[0], 1), (index[1], 0),
                    (index[2], 1), (index[3], 0)
                ),
                coefficient = val
            )
    return H
# Killer Construction
def construct_killer(k, e_nums, n = 0, const = 1e-2, t = 1e2, n_killer = 3):
    """ Construct a killer operator for CAS Hamiltonian, based on cas block structure of k and the size of killer is
    given in k, the number of electrons in each CAS block of the ground state
    is specified by e_nums. t is the strength of quadratic balancing terms for the killer with respect to k,
    n_killer specifies the number of operators O to choose.
    """
    if not n:
        n = max([max(orbs) for orbs in k])
    killer = of.FermionOperator.zero()
    for i in range(len(k)):
        orbs = k[i]
        outside_orbs = [j for j in range(n) if j not in orbs]
        # Construct Ne
        Ne = sum([of.FermionOperator("{}^ {}".format(i, i)) for i in orbs])
        # Construct O, for O as combination of Epq which preserves Sz and S2
        if len(outside_orbs) >= 4:
            tmp = 0
            while tmp < n_killer:
                p, q = random.sample(outside_orbs, 2)
                if abs(p - q) > 1:
                    # Constructing symmetry conserved killers
                    O = of.FermionOperator.zero()
                    ferm_op = of.FermionOperator("{}^ {}".format(p, q)) + of.FermionOperator("{}^ {}".format(q, p))
                    O += ferm_op
                    O += of.hermitian_conjugated(ferm_op)
                    k_const = const * (1 + np.random.rand())
                    killer += k_const * O * (Ne - e_nums[i])
                    tmp += 1
        killer += t * (1 + np.random.rand()) * const * ((Ne - e_nums[i]) ** 2)
    killer_obt, killer_tbt = H_to_sparse(killer, n)
    # Killer constant term
    c = killer.terms[()]
    return c, killer_obt, killer_tbt

def get_param_num(n, k, complex = False):
    """
    Counting the parameters needed, where k is the number of orbitals occupied by CAS Fragments,
    and n-k orbitals are occupied by the CSA Fragments
    """
    if not complex:
        upnum = int(n * (n - 1) / 2)
    else:
        upnum = n * (n - 1)
    casnum = 0
    for block in k:
        casnum += len(block) ** 4 + len(block) ** 2
    pnum = upnum + casnum
    return upnum, casnum, pnum


def chem_spatial_orb_to_phys_spatial_orb(obt, tbt):
    """
    Converts the spatial orbital chemist notation into physcicist notation
    Args:
        obt:
        tbt:

    Returns:

    """
    phy_obt = obt + np.einsum("prrq->pq", tbt)
    phy_tbt = 2 * tbt
    return phy_obt, phy_tbt

def get_cas_matrix(cas_x, n, k):
    obt = np.zeros([n, n])
    tbt = np.zeros([n, n, n, n])
    idx = 0
    for orbs in k:
        for p, q in product(orbs, repeat = 2):
            obt[p,q] = cas_x[idx]
            idx += 1
        for p, q, r, s in product(orbs, repeat = 4):
            tbt[p,q,r,s] = cas_x[idx]
            idx += 1
    return obt, tbt

def orbtransf(tensor, U, complex = False):
    """Return applying UHU* for the tensor representing the 1e or 2e tensor"""
    if len(tensor.shape) == 4:
        p = np.einsum_path('ak,bl,cm,dn,klmn->abcd', U, U, U, U, tensor)[0]
        return np.einsum('ak,bl,cm,dn,klmn->abcd', U, U, U, U, tensor, optimize = p)
    elif len(tensor.shape) == 2:
        p = np.einsum_path('ap,bq, pq->ab', U, U, tensor)[0]
        return np.einsum('ap,bq, pq->ab', U, U, tensor, optimize = p)
def load_Hamiltonian_with_solution(path, file_name):
    """Load the precomputed Hamiltonian with the given file_name. Return the Hamiltonians and the state
    Hamiltonian is represented in spatial orbital basis as three terms: (c, h_pq, g_pqrs)
    Returns:
    U: Arbitrary Unitary Rotation, in spatial orbital basis
    H_cas: Unhidden CAS Fragements
    H_hidden：U H_cas U*
    H_with_killer: H_cas + killer
    H_killer_hidden: U H_with_killer U*
    """
    with open(path + file_name, 'rb') as handle:
        dic = pickle.load(handle)
        key = list(dic.keys())[0]
        dic = dic[key]
    E_min = dic["E_min"]
    cas_x = dic["cas_x"]
    killer = dic["killer"]
    killer_c = killer[0]
    sol = dic["sol"]
    k_obt = tf.sparse.reorder(killer[1])
    k_tbt = tf.sparse.reorder(killer[2])
    k = dic["k"]
    upnum = dic["upnum"]
    spatial_orbs = dic["spatial_orbs"]
    obt = dic["cas_obt"]
    tbt = dic["cas_tbt"]

#     CAS 2e tensor
    # obt, tbt = get_cas_matrix(cas_x, spatial_orbs, k)
    H_cas = (0, obt, tbt)
    H_with_killer = (killer_c, obt+tf.sparse.to_dense(k_obt), tbt+tf.sparse.to_dense(k_tbt))
#     Set up random unitary to hide 2e tensor
    random_uparams = np.random.rand(upnum)
    U = construct_orthogonal(spatial_orbs, random_uparams)
#     Hide 2e etensor with random unitary transformation
    H_hidden = (0, orbtransf(obt, U), orbtransf(tbt, U))
    H_killer_hidden = (killer_c, orbtransf(H_with_killer[1], U), orbtransf(H_with_killer[2], U))
    return U, H_cas, H_hidden, H_with_killer, H_killer_hidden, sol, E_min
