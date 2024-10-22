def get_dmrg_param(num_orbitals, num_electrons, num_unpaired_electrons,
                   multiplicity, process_param):
    """
    Get parameters for dmrg calculation
    Args:
        num_orbitals:
        num_electrons:
        num_unpaired_electrons:
        multiplicity:
        process_param: Parameters specific to dmrg process

    Returns:

    """
    nuc_rep_energy = 0
    dmrg_param = {"factor_half_convention": True,
                  "symmetry_type": "SZ",
                  "num_threads": 1,
                  "n_mkl_threads": 1,
                  "num_orbitals": num_orbitals,
                  "num_spin_orbitals": 2 * num_orbitals,
                  "num_electrons": num_electrons,
                  "two_S": num_unpaired_electrons,
                  "two_Sz": int((multiplicity - 1) / 2),
                  "orb_sym": None,
                  "temp_dir": "./tests/temp",
                  "stack_mem": 1073741824,
                  "restart_dir": "./tests/restart",
                  "core_energy": nuc_rep_energy,
                  "reordering_method": "none",
                  "init_state_seed": 64241,  # 0 means random seed
                  "initial_mps_method": "random",
                  "init_state_bond_dimension":
                      process_param["init_state_bond_dimension"],
                  "occupancy_hint": None,
                  "full_fci_space_bool": True,
                  "init_state_direct_two_site_construction_bool": False,
                  "max_num_sweeps":
                      process_param["max_num_sweeps"],
                  "energy_convergence_threshold":
                      process_param["energy_convergence_threshold"],
                  "sweep_schedule_bond_dims":
                      process_param["sweep_schedule_bond_dims"],
                  "sweep_schedule_noise": process_param["sweep_schedule_noise"],
                  "sweep_schedule_davidson_threshold":
                      process_param["sweep_schedule_davidson_threshold"],
                  "davidson_type": None,  # Default is None, for "Normal"
                  "eigenvalue_cutoff": 1e-120,
                  "davidson_max_iterations": 4000,  # Default is 4000
                  "davidson_max_krylov_subspace_size": 50,  # Default is 50
                  "lowmem_noise_bool": False,
                  "sweep_start": 0,  # Default is 0, where to start sweep
                  "initial_sweep_direction": None,
                  "stack_mem_ratio": 0.4,  # Default is 0.4
                  }
    return dmrg_param


def get_dmrg_process_param\
                (init_state_bond_dimension, max_num_sweeps,
                 energy_convergence_threshold, sweep_schedule_bond_dims,
                 sweep_schedule_noise, sweep_schedule_davidson_threshold):
    """
    Get parameters for dmrg process
    Returns:

    """
    dmrg_process_param = {
        "init_state_bond_dimension": init_state_bond_dimension,
        "max_num_sweeps": max_num_sweeps,
        "energy_convergence_threshold": energy_convergence_threshold,
        "sweep_schedule_bond_dims": sweep_schedule_bond_dims,
        "sweep_schedule_noise": sweep_schedule_noise,
        "sweep_schedule_davidson_threshold": sweep_schedule_davidson_threshold
    }

    return dmrg_process_param
