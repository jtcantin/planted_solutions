import numpy as np


def obt_to_tbt_chem(obt, spin_orb):
    """

    Args:
        obt: One body tensor in chemistry notation
        spin_orb:

    Returns:

    """
    n = spin_orb

    N = spin_orb
    D, U = np.linalg.eigh(obt)

    U = U.T  # note that my implementation of orbital rotations is U.T @ X @ U, so this line is needed
    tbt = np.zeros([N, N, N, N])
    for p in range(N):
        tbt[p, p, p, p] = D[p]

    tmp_path = np.einsum_path('pqrs,pa,qb,rc,sd->abcd', tbt, U, U, U, U)[0]
    chemisttbt = np.einsum('pqrs,pa,qb,rc,sd->abcd', tbt, U, U, U, U,
                           optimize=tmp_path)

    return chemisttbt


if __name__ == '__main__':
    obt = np.array([[1, 4], [2, 3]])
    chem_tbt = obt_to_tbt_chem(obt, 2)
    print(chem_tbt)
