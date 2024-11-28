import datetime
import numpy as np
from itertools import product
import openfermion as of
import tensorflow as tf
from matrix_utils import construct_orthogonal
import scipy
import dmrghandler.dmrg_calc_prepare
import pyscf
import copy


import logging
import urllib.request
import json
import jsonschema
from pathlib import Path
import hashlib
import uuid


# log = logging.getLogger("{Path(config_file_name).stem}")
log = logging.getLogger("dmrghandler")
log.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler("dmrghandler.log")
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - Line %(lineno)d - %(message)s"
)
fh.setFormatter(formatter)
# add the handlers to the log
log.addHandler(fh)


def construct_blocks(size: int, total_orbs: int, spin_orb=False):
    """Construct CAS blocks with size for spin_orbs/spatial number of orbitals"""
    if spin_orb:
        size = size * 2
    blocks = []
    tmp = [0]

    # Here we form a list of length size until we reach
    for i in range(1, total_orbs):
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
        for p, q in product(block, repeat=2):
            cas_obt[p, q] = Hobt[p, q]
            cas_x[idx] = Hobt[p, q]
            idx += 1
        for a, b, c, d in product(block, repeat=4):
            cas_tbt[a, b, c, d] = Htbt[a, b, c, d]
            cas_x[idx] = Htbt[a, b, c, d]
            idx += 1
    return cas_obt, cas_tbt, cas_x


def solve_enums(
    H,
    blocks,
    total_number_of_electrons,
    ne_per_block=0,
    ne_range=0,
    balance_t=50,
    rng_obj=None,
):
    """Solve for number of electrons in each CAS block with FCI within the block,
    H = (obt, tbt) as the Hamiltonian in spatial orbitals.
    Notice that some quadratic terms (Ne-ne)^2 are added to ensure the correct number
    of electrons in the ground state of each block
    """
    if rng_obj is None:
        rng_obj = np.random.default_rng()
        raise ValueError("rng_obj must be provided")

    # Ensure the total number of electrons per block is even
    if ne_per_block % 2 != 0:
        raise ValueError("Number of electrons per block must be even")

    cas_obt = H[0]
    cas_tbt = H[1]
    e_nums = []
    states = []
    E_cas = 0
    core_energy_balance = 0
    total_elec_count = 0
    for index in range(len(blocks)):
        orbs = blocks[index]
        s = orbs[0]
        t = orbs[-1] + 1
        norbs = len(orbs)

        if ne_per_block + total_elec_count > total_number_of_electrons:
            ne = total_number_of_electrons - total_elec_count
        else:
            ne = ne_per_block
        total_elec_count += ne

        if ne != 0:

            # Construct (Ne-ne)^2 = (E_pp-ne)^2 = E_ppE_qq -2*ne*E_pp + ne**2 terms in matrix,
            # to enforce structure of states for each CAS block in sd basis.

            balance_const = ne**2
            balance_obt = np.zeros([norbs, norbs], dtype=np.float64)
            balance_tbt = np.zeros([norbs, norbs, norbs, norbs], dtype=np.float64)

            for p in range(len(orbs)):
                balance_obt[p, p] -= 2 * ne
                for q in range(len(orbs)):
                    balance_tbt[p, p, q, q] += 1

            # Coefficient for (Ne-ne)^2 term
            strength = balance_t * (1 + rng_obj.uniform(0, 1))

            # Add the balance term to the truncated hamiltonian separately for one-body and two-body
            cas_tbt[s:t, s:t, s:t, s:t] = np.add(
                cas_tbt[s:t, s:t, s:t, s:t], strength * balance_tbt
            )
            cas_obt[s:t, s:t] = np.add(cas_obt[s:t, s:t], strength * balance_obt)
            core_energy_block = strength * balance_const
            core_energy_balance += core_energy_block

            # Get FCI ground state energy for each CAS block
            cisolver = pyscf.fci.direct_spin1.FCI()
            cisolver.max_cycle = 100  # Max. iterations for diagonalization
            cisolver.conv_tol = 1e-13  # Convergence tolerance for diagonalization

            num_alpha_electrons = int(ne // 2 + ne % 2)
            num_beta_electrons = int(ne // 2)

            local_cas_obt = copy.deepcopy(cas_obt[s:t, s:t])
            local_cas_tbt = copy.deepcopy(cas_tbt[s:t, s:t, s:t, s:t])
            local_cas_obt_phys, local_cas_tbt_phys = (
                chem_spatial_orb_to_phys_spatial_orb(local_cas_obt, local_cas_tbt)
            )
            block_e_min_fci, block_fcivec = cisolver.kernel(
                local_cas_obt_phys,
                local_cas_tbt_phys,
                ecore=core_energy_block,
                norb=norbs,
                nelec=(num_alpha_electrons, num_beta_electrons),
            )

        else:
            block_e_min_fci = 0
            block_fcivec = np.zeros(2**norbs)
        print(f"E_min: {block_e_min_fci} for orbs: {orbs}")

        # The ground state energy of the whole hamiltonian is the sum of each
        # CAS block minimum energy
        E_cas += block_e_min_fci
        states.append(block_fcivec)
        e_nums.append(ne)

    # Ensure the total number of electrons is correct
    assert (
        sum(e_nums) == total_number_of_electrons
    ), f"Total number of electrons is not correct: {sum(e_nums)} != {total_number_of_electrons}"
    assert (
        total_elec_count == total_number_of_electrons
    ), f"Total number of electrons is not correct: {total_elec_count} != {total_number_of_electrons}"

    return e_nums, states, E_cas, core_energy_balance


def H_to_sparse(H: of.FermionOperator, n):
    """Construct the sparse tensor representation of the Hamiltonian, represented by a constant term, a"""
    h1e_keys = []
    h1e_vals = []
    h2e_keys = []
    h2e_vals = []
    c = 0.0
    for key, val in H.terms.items():
        if len(key) == 2:
            h1e_keys.append([key[0][0], key[1][0]])
            h1e_vals.append(val)
        elif len(key) == 4:
            h2e_keys.append([key[0][0], key[1][0], key[2][0], key[3][0]])
            h2e_vals.append(val)
        elif len(key) == 0:
            c += val
    sparse_h1 = tf.sparse.SparseTensor(
        indices=h1e_keys, values=h1e_vals, dense_shape=[n, n]
    )
    sparse_h2 = tf.sparse.SparseTensor(
        indices=h2e_keys, values=h2e_vals, dense_shape=[n, n, n, n]
    )
    return sparse_h1, sparse_h2, c


def construct_killer(k, e_nums, n=0, const=1e-2, t=1e2, n_killer=3, rng_obj=None):
    """Construct a killer operator for CAS Hamiltonian, based on cas block structure of k and the size of killer is
    given in k, the number of electrons in each CAS block of the ground state
    is specified by e_nums. t is the strength of quadratic balancing terms for the killer with respect to k,
    n_killer specifies the number of operators O to choose.
    """
    if rng_obj is None:
        rng_obj = np.random.default_rng()
        raise ValueError("rng_obj must be provided")
    if not n:
        n = max([max(orbs) for orbs in k])

    killer = of.FermionOperator.zero()
    for i in range(len(k)):
        orbs = k[i]
        outside_orbs = [j for j in range(n) if j not in orbs]

        # Define Ne
        Ne = sum([of.FermionOperator("{}^ {}".format(i, i)) for i in orbs])

        # Construct O, for O as combination of Epq which preserves Sz and S2
        if len(outside_orbs) >= 4:
            tmp = 0
            while tmp < n_killer:
                p, q = rng_obj.choice(a=outside_orbs, size=2, replace=False)
                if abs(p - q) > 1:

                    # Constructing symmetry conserved killers
                    O = of.FermionOperator.zero()
                    ferm_op = of.FermionOperator(
                        "{}^ {}".format(p, q)
                    ) + of.FermionOperator("{}^ {}".format(q, p))
                    O += ferm_op

                    k_const = const * (1 + rng_obj.uniform(0, 1))
                    killer_local = k_const * O * (Ne - e_nums[i])

                    killer += killer_local
                    tmp += 1

    killer_obt, killer_tbt, c = H_to_sparse(killer, n)

    killer_obt = tf.sparse.reorder(killer_obt)
    killer_tbt = tf.sparse.reorder(killer_tbt)
    killer_obt = tf.sparse.to_dense(killer_obt)
    killer_tbt = tf.sparse.to_dense(killer_tbt)
    return c, killer_obt, killer_tbt


def get_param_num(n, k, complex=False):
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


def check_for_incorrect_spin_terms(tbt_to_check):
    num_incorrect_terms = 0
    # Check no incorrect spin terms present
    num_spin_orbitals = tbt_to_check.shape[0]
    no_incorrect_terms = True
    for piter in range(num_spin_orbitals):
        for qiter in range(num_spin_orbitals):
            for riter in range(num_spin_orbitals):
                for siter in range(num_spin_orbitals):
                    if (
                        piter % 2 == 0
                        and qiter % 2 == 0
                        and riter % 2 == 0
                        and siter % 2 == 0
                    ):
                        continue
                    if (
                        piter % 2 == 1
                        and qiter % 2 == 1
                        and riter % 2 == 1
                        and siter % 2 == 1
                    ):
                        continue
                    if (
                        piter % 2 == 0
                        and qiter % 2 == 0
                        and riter % 2 == 1
                        and siter % 2 == 1
                    ):
                        continue
                    if (
                        piter % 2 == 1
                        and qiter % 2 == 1
                        and riter % 2 == 0
                        and siter % 2 == 0
                    ):
                        continue
                    if not np.isclose(tbt_to_check[piter, qiter, riter, siter], 0.0):
                        # print(f"Incorrect spin term present in two body tensor at indices {piter}, {qiter}, {riter}, {siter}: {tbt_to_check[piter, qiter, riter, siter]}")
                        no_incorrect_terms = False
                        num_incorrect_terms += 1

    return no_incorrect_terms, num_incorrect_terms


def check_hamiltonian(obt_to_check, tbt_to_check, spatial_orbitals):
    return_dict = {}
    if not spatial_orbitals:
        print("!!!!!!!!!Assuming Hamiltonian is in spin orbitals!!!!!!!!!!")
        no_incorrect_terms, num_incorrect_terms = check_for_incorrect_spin_terms(
            tbt_to_check
        )
        return_dict["no_incorrect_spin_terms"] = no_incorrect_terms
        return_dict["num_incorrect_spin_terms"] = num_incorrect_terms
        print(f"No incorrect spin terms present: {no_incorrect_terms}")
        print(f"Number of incorrect terms: {num_incorrect_terms}")

        spin_symm_check_passed = dmrghandler.dmrg_calc_prepare.check_spin_symmetry(
            one_body_tensor=obt_to_check, two_body_tensor=tbt_to_check
        )
        return_dict["spin_symm_check_passed"] = spin_symm_check_passed
        print(f"Spin symmetry check passed: {spin_symm_check_passed}")
    else:
        print("!!!!!!!!!Assuming Hamiltonian is in spatial orbitals!!!!!!!!!!")

    permutation_symmetries_complex_orbitals_check_passed = (
        dmrghandler.dmrg_calc_prepare.check_permutation_symmetries_complex_orbitals(
            obt_to_check, tbt_to_check
        )
    )
    return_dict["permutation_symmetries_complex_orbitals_check_passed"] = (
        permutation_symmetries_complex_orbitals_check_passed
    )
    print(
        f"Permutation symmetries complex orbitals check passed: {permutation_symmetries_complex_orbitals_check_passed}"
    )

    permutation_symmetries_real_orbitals_check_passed = (
        dmrghandler.dmrg_calc_prepare.check_permutation_symmetries_real_orbitals(
            obt_to_check, tbt_to_check
        )
    )
    return_dict["permutation_symmetries_real_orbitals_check_passed"] = (
        permutation_symmetries_real_orbitals_check_passed
    )
    print(
        f"Permutation symmetries real orbitals check passed: {permutation_symmetries_real_orbitals_check_passed}"
    )
    return return_dict


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
        for p, q in product(orbs, repeat=2):
            obt[p, q] = cas_x[idx]
            idx += 1
        for p, q, r, s in product(orbs, repeat=4):
            tbt[p, q, r, s] = cas_x[idx]
            idx += 1
    return obt, tbt


def orbtransf(tensor, U, complex=False):
    """Return applying UHU* for the tensor representing the 1e or 2e tensor"""
    if len(tensor.shape) == 4:
        p = np.einsum_path("ak,bl,cm,dn,klmn->abcd", U, U, U, U, tensor)[0]
        return np.einsum("ak,bl,cm,dn,klmn->abcd", U, U, U, U, tensor, optimize=p)
    elif len(tensor.shape) == 2:
        p = np.einsum_path("ap,bq, pq->ab", U, U, tensor)[0]
        return np.einsum("ap,bq, pq->ab", U, U, tensor, optimize=p)


def unitary_rotation_obfuscation(
    obt,
    tbt,
    k_obt,
    k_tbt,
    E_min,
    killer_c,
    upnum,
    spatial_orbs,
    rng_obj=None,
    core_energy=0.0,
    scaling_factor=1.0,
):
    """Generate orbital rotations to hide the Hamiltonian Structure and generate different
    versions of the Hamiltonian.
    Returns:
    U: Arbitrary Unitary Rotation, in spatial orbital basis
    H_cas: Unhidden CAS Fragements
    H_hidden：U H_cas U*
    H_with_killer: H_cas + killer
    H_killer_hidden: U H_with_killer U*
    """
    if rng_obj is None:
        rng_obj = np.random.default_rng()
        raise ValueError("rng_obj must be provided")

    obt = np.array(obt, dtype=np.float64)
    tbt = np.array(tbt, dtype=np.float64)
    k_obt = np.array(k_obt, dtype=np.float64)
    k_tbt = np.array(k_tbt, dtype=np.float64)

    # CAS 2e tensor
    H_cas = (
        core_energy,
        np.array(obt, dtype=np.float64),
        np.array(tbt, dtype=np.float64),
    )
    H_with_killer = (
        core_energy + killer_c,
        np.array(obt + k_obt, dtype=np.float64),
        np.array(tbt + k_tbt, dtype=np.float64),
    )

    # Set up random unitary to hide 2e tensor
    random_uparams = np.array(
        rng_obj.uniform(0, 2 * np.pi * scaling_factor, size=upnum), dtype=np.float64
    )
    U = np.array(construct_orthogonal(spatial_orbs, random_uparams), dtype=np.float64)
    # Hide 2e etensor with random unitary transformation
    H_hidden = (core_energy, orbtransf(obt, U), orbtransf(tbt, U))
    H_killer_hidden = (
        core_energy + killer_c,
        orbtransf(H_with_killer[1], U),
        orbtransf(H_with_killer[2], U),
    )
    return U, H_cas, H_hidden, H_with_killer, H_killer_hidden, E_min


def ensure_rotation_invariance(
    H_cas,
    H_hidden,
    block_balanced_H_ij,
    block_balanced_rotated_H_ij,
    block_balanced_G_ijkl,
    block_balanced_rotated_G_ijkl,
    H_with_killer,
    H_killer_hidden,
    block_balanced_killer_H_ij,
    block_balanced_killer_rotated_H_ij,
    block_balanced_killer_G_ijkl,
    block_balanced_killer_rotated_G_ijkl,
):
    # Ensure rotation did not change the L2 norm of the Hamiltonian
    L2_norm_diff_obt_chem = scipy.linalg.norm(H_cas[1], ord=None) - scipy.linalg.norm(
        H_hidden[1], ord=None
    )
    L2_norm_diff_tbt_chem = scipy.linalg.norm(H_cas[2], ord=None) - scipy.linalg.norm(
        H_hidden[2], ord=None
    )
    L2_norm_diff_obt_phys = scipy.linalg.norm(
        block_balanced_H_ij, ord=None
    ) - scipy.linalg.norm(block_balanced_rotated_H_ij, ord=None)
    L2_norm_diff_tbt_phys = scipy.linalg.norm(
        block_balanced_G_ijkl, ord=None
    ) - scipy.linalg.norm(block_balanced_rotated_G_ijkl, ord=None)

    assert np.isclose(L2_norm_diff_obt_chem, 0)
    assert np.isclose(L2_norm_diff_tbt_chem, 0)
    assert np.isclose(L2_norm_diff_obt_phys, 0)
    assert np.isclose(L2_norm_diff_tbt_phys, 0)

    # Ensure rotation did not change the L2 norm of the Hamiltonian plus killer
    L2_norm_diff_obt_chem = scipy.linalg.norm(
        H_with_killer[1], ord=None
    ) - scipy.linalg.norm(H_killer_hidden[1], ord=None)
    L2_norm_diff_tbt_chem = scipy.linalg.norm(
        H_with_killer[2], ord=None
    ) - scipy.linalg.norm(H_killer_hidden[2], ord=None)
    L2_norm_diff_obt_phys = scipy.linalg.norm(
        block_balanced_killer_H_ij, ord=None
    ) - scipy.linalg.norm(block_balanced_killer_rotated_H_ij, ord=None)
    L2_norm_diff_tbt_phys = scipy.linalg.norm(
        block_balanced_killer_G_ijkl, ord=None
    ) - scipy.linalg.norm(block_balanced_killer_rotated_G_ijkl, ord=None)

    assert np.isclose(L2_norm_diff_obt_chem, 0)
    assert np.isclose(L2_norm_diff_tbt_chem, 0)
    assert np.isclose(L2_norm_diff_obt_phys, 0)
    assert np.isclose(L2_norm_diff_tbt_phys, 0)


def construct_killer_directly(
    k, e_nums, n=0, const=1e-2, t=1e2, n_killer=3, rng_obj=None
):
    """Construct a killer operator for CAS Hamiltonian, based on cas block structure of k and the size of killer is
    given in k, the number of electrons in each CAS block of the ground state
    is specified by e_nums. t is the strength of quadratic balancing terms for the killer with respect to k,
    n_killer specifies the number of operators O to choose.
    """
    if rng_obj is None:
        rng_obj = np.random.default_rng()
        raise ValueError("rng_obj must be provided")
    if not n:
        n = max([max(orbs) for orbs in k])

    killer_obt = np.zeros([n, n], dtype=np.float64)
    killer_tbt = np.zeros([n, n, n, n], dtype=np.float64)
    const_term = 0

    for i in range(len(k)):
        orbs = k[i]
        outside_orbs = [j for j in range(n) if j not in orbs]

        if len(outside_orbs) >= 4:
            tmp = 0
            while tmp < n_killer:

                p, q = rng_obj.choice(a=outside_orbs, size=2, replace=False)
                if abs(p - q) > 1:

                    k_const = const * (1 + rng_obj.uniform(0, 1))

                    # Constant part
                    const_term += 0

                    # OBT part
                    to_add = -1 * k_const * (e_nums[i])
                    killer_obt[p, q] += to_add
                    killer_obt[q, p] += to_add

                    # TBT part
                    to_add = k_const * 0.5
                    for orb_i in orbs:
                        killer_tbt[p, q, orb_i, orb_i] += to_add
                        killer_tbt[q, p, orb_i, orb_i] += to_add
                        killer_tbt[orb_i, orb_i, p, q] += to_add
                        killer_tbt[orb_i, orb_i, q, p] += to_add

                    tmp += 1

    return const_term, killer_obt, killer_tbt


problem_instance_json_schema_url_default = "https://raw.githubusercontent.com/isi-usc-edu/qb-gsee-benchmark/refs/heads/main/schemas/problem_instance.schema.0.0.1.json"
contact_info_temp = [
    {
        "name": "temp",
        "email": "temp",
        "institution": "temp",
    }
]

requirements_default = {
    "probability_of_success": 0.99,
    "time_limit_seconds": 172800,
    "accuracy": 1.0,
    "energy_units": "millihartree",
    "energy_target": 0.99,
}


def ensure_required_in_dict(dictionary: dict, required_keys: list[str]):
    """
    Ensures that a dictionary has the required keys.

    Args:
        dictionary: The dictionary to check.
        required_keys: A list of the required keys.

    Raises:
        KeyError: If a required key is not in the dictionary.
    Code  taken from qoptbench (https://github.com/zapatacomputing/bobqat-qb-opt-benchmark)
    """
    for key in required_keys:
        if key not in dictionary.keys():
            raise KeyError(
                f"Required key {key} not in dictionary."
                + f" Required keys are: {required_keys}."
                + f"Current keys are: {dictionary.keys()}."
                + "Most keys allow values of None or 'N/A'."
            )


def gen_json_files(
    filename_json,
    uuid_string_instance,
    uuid_string_fcidump,
    short_name,
    filename_fcidump,
    parameter_dict,
    generation_code_url,
    source_fcidump,
    fcidump_permanent_storage_location,
    # source_uuid,
    status="in_development",
    contact_info=contact_info_temp,
    superseded_by=None,
    problem_type="GSEE",
    application_domain="QC",
    requirements=requirements_default,
    problem_instance_json_schema_url=problem_instance_json_schema_url_default,
):
    """
    This generates two json files, one to be placed in a benchmark for the performer
    and one for the benchmark proctor. The performer json file contains the information
    needed to run the benchmark, but does not contain information about the generation method.
    This reduces the likihood of the performer being able to cheat.
    The proctor json file contains all the information needed to generate the planted solution.
    """
    schema_filepath = Path("temp_schema.json")
    # Download schema
    urllib.request.urlretrieve(problem_instance_json_schema_url, schema_filepath.name)
    schema = json.load(open(schema_filepath))

    # Generate the performer json file

    digestobj = hashlib.new("sha1")
    with open(filename_fcidump, "rb") as fileobj:
        digestobj.update(fileobj.read())

    ensure_required_in_dict(
        dictionary=parameter_dict,
        required_keys=[
            "multiplicity",
            "num_electrons",
            "num_orbitals",
            "utility_scale",
            "block_size",
            "ne_per_block",
            "balance_strength",
            "ne_range",
            "n_killer",
            "rng_global_seed",
            "github_repository_url",
            "github_commit_sha",
            "paper_reference_doi",
            "known_ground_state_energy_hartrees",
            "killer_coefficient",
            "orbital_rotation_angle_scaling_factor",
        ],
    )

    multiplicity = parameter_dict["multiplicity"]
    num_electrons = parameter_dict["num_electrons"]
    num_orbitals = parameter_dict["num_orbitals"]
    utility_scale = parameter_dict["utility_scale"]

    performer_json_dict = {
        "$schema": problem_instance_json_schema_url,
        "references": ["https://github.com/jtcantin/planted_solutions"],
        "problem_instance_uuid": uuid_string_instance,
        "creation_timestamp": datetime.datetime.now().isoformat(),
        "calendar_due_date": None,
        "short_name": short_name,
        "license": {
            "name": "Apache 2.0",
            "url": "http://www.apache.org/licenses/LICENSE-2.0",
        },
        "contact_info": contact_info,
        "status": status,
        "superseded_by": superseded_by,
        "problem_type": problem_type,
        "application_domain": application_domain,
        "tasks": [
            {
                "task_uuid": str(uuid.uuid4()),
                "supporting_files": [
                    {
                        "instance_data_object_uuid": uuid_string_fcidump,
                        "instance_data_object_url": fcidump_permanent_storage_location
                        + f"{Path(filename_fcidump).name}",
                        "instance_data_checksum": str(digestobj.hexdigest()),
                        "instance_data_checksum_type": "sha1sum",
                    }
                ],
                "requirements": requirements,
                "features": {
                    "multiplicity": multiplicity,
                    "num_electrons": num_electrons,
                    "num_orbitals": num_orbitals,
                    "utility_scale": utility_scale,
                },
            },
        ],
    }
    Path(filename_json).parent.mkdir(parents=True, exist_ok=True)
    save_json(filename_json, performer_json_dict)

    print(f"Validating {filename_json} against schema")
    jsonschema.validate(json.load(open(filename_json)), schema)
    print("Validated")

    # Generate the proctor json file
    filename_json_proctor = str(filename_json).split(".json")[0] + "_proctor.json"

    proctor_dict = copy.deepcopy(parameter_dict)
    proctor_dict["source_fcidump"] = source_fcidump

    proctor_dict["generation_code_url"] = generation_code_url

    performer_json_dict["tasks"][0]["features"].update(proctor_dict)

    save_json(filename_json_proctor, performer_json_dict)

    print(f"Validating {filename_json_proctor} against schema")
    jsonschema.validate(json.load(open(filename_json_proctor)), schema)


def save_json(json_filename, dict_to_save):
    with open(json_filename, "w") as json_file:
        json.dump(dict_to_save, json_file, indent=4)
