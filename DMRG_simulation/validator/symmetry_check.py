import numpy as np


def check_permutation_symmetries_complex_orbitals(one_body_tensor, two_body_tensor):
    """
    Checks the permutation symmetry for one body tensor and two body tensor with
    respect to block2 permutation convention.
    Args:
        one_body_tensor:
        two_body_tensor:

    Returns: Whether they pass the test or not.

    """
    # Works for both spin-orbital and orbital tensors
    symm_check_passed = True
    num_orbitals = one_body_tensor.shape[0]
    for p in range(num_orbitals):
        for q in range(num_orbitals):
            if not np.allclose(
                    one_body_tensor[p, q],
                    (one_body_tensor[q, p]).conj()):
                symm_check_passed = False
                print("One body tensor symmetry broken")

            for r in range(num_orbitals):
                for s in range(num_orbitals):
                    if (not np.allclose(
                            two_body_tensor[p, q, r, s],
                            two_body_tensor[r, s, p, q])):
                        symm_check_passed = False
                        print("two_body_tensor[p, q, r, s], "
                              "two_body_tensor[r, s, p, q]")
                    elif (not np.allclose(
                            two_body_tensor[p, q, r, s],
                            two_body_tensor[q, p, s, r].conj(),
                        )):
                        symm_check_passed = False
                        print("two_body_tensor[p, q, r, s], "
                              "two_body_tensor[q, p, s, r].conj()")
                    elif (not np.allclose(
                            two_body_tensor[p, q, r, s],
                            two_body_tensor[s, r, q, p].conj(),
                        )):

                        symm_check_passed = False
                        print("Two body tensor symmetry broken")
    return symm_check_passed
