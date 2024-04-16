import numpy as np
from pyscf import __config__
from DMRG_simulation.planted_solution_loader import construct_Hamiltonian_with_solution

DEFAULT_FLOAT_FORMAT = getattr(__config__, 'fcidump_float_format', ' %.16g')
TOL = getattr(__config__, 'fcidump_write_tol', 1e-15)

MOLPRO_ORBSYM = getattr(__config__, 'fcidump_molpro_orbsym', False)

ORBSYM_MAP = {
    'D2h': (1,         # Ag
            4,         # B1g
            6,         # B2g
            7,         # B3g
            8,         # Au
            5,         # B1u
            3,         # B2u
            2),        # B3u
    'C2v': (1,         # A1
            4,         # A2
            2,         # B1
            3),        # B2
    'C2h': (1,         # Ag
            4,         # Bg
            2,         # Au
            3),        # Bu
    'D2' : (1,         # A
            4,         # B1
            3,         # B2
            2),        # B3
    'Cs' : (1,         # A'
            2),        # A"
    'C2' : (1,         # A
            2),        # B
    'Ci' : (1,         # Ag
            2),        # Au
    'C1' : (1,)
}


def write_head(fout, nmo, nelec, ms=0, orbsym=None):
    if not isinstance(nelec, (int, np.number)):
        ms = abs(nelec[0] - nelec[1])
        nelec = nelec[0] + nelec[1]
    fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo, nelec, ms))
    if orbsym is not None and len(orbsym) > 0:
        fout.write('  ORBSYM=%s\n' % ','.join([str(x) for x in orbsym]))
    else:
        fout.write('  ORBSYM=%s\n' % ('1,' * nmo))
    fout.write('  ISYM=1,\n')
    fout.write(' &END\n')


if __name__ == '__main__':
    ps_path = "planted_solutions/"
    # File name in ps_path folder
    file_name = "2_co2_6-311++G___12_9d464efb-b312-45f8-b0ba-8c42663059dc.pkl"
    tbt, tbt_hidden, Htbt_with_killer, Htbt_hidden, sol, e_nums, E_min, spin_orbs = construct_Hamiltonian_with_solution(
        ps_path, file_name)
    one_body_tensor = np.zeros((spin_orbs, spin_orbs))
