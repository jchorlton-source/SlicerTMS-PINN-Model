from simnibs import sim_struct, run_simnibs
import os
from multiprocessing import Pool
import itertools

coilPosList = [
    'Fp1', 'Fpz', 'Fp2',
    'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
    'Iz'
]

coilDirList = [
    'Fp1', 'Fpz', 'Fp2',
    'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
    'Iz'
]

# Get random position and direction combinations
# Use a fixed seed so results are reproducible across runs
allCombinations = list(itertools.product(coilPosList, coilDirList))
randomCombinations = [(i, pos, direc) for i, (pos, direc) in enumerate(allCombinations)]
numSims = len(randomCombinations)

baseDir = "simOutput"

def run_single_sim(args):
    """Run a single TMS simulation for a given coil position and direction."""
    idx, posName, directName = args

    # Each sim gets its own uniquely named output folder
    # Index included to avoid clashes if pos+dir combination is repeated
    outDir = os.path.join(baseDir, f"{idx}_{posName}_{directName}")
    os.makedirs(outDir, exist_ok=True)

    try:
        # Initialise session
        s = sim_struct.SESSION()

        # Simulation settings
        s.fields = "EDs"
        s.open_in_gmsh = False
        s.map_to_surf = False
        s.map_to_fsavg = False
        s.map_to_vol = True
        s.map_to_MNI = False

        # Name of head mesh
        s.subpath = "/scratch/pawsey1242/jchorlton/Thesis/SlicerTMS-PINN-Model/DataCreation/m2m_ernie"

        # Output folder
        s.pathfem = outDir

        # Initialise list of TMS simulations
        tmslist = s.add_tmslist()

        # Select coil
        tmslist.fnamecoil = os.path.join('legacy_and_other', 'Magstim_70mm_Fig8.ccd')

        # Initialise a coil position
        pos = tmslist.add_position()

        # Select coil centre
        pos.centre = posName

        # Select coil direction
        pos.pos_ydir = directName

        # Run simulation
        run_simnibs(s)
        print(f"[{idx+1}/{numSims}] Completed: {posName} -> {directName}", flush=True)

    except Exception as e:
        # Log the error but don't crash the whole pool
        print(f"[{idx+1}/{numSims}] FAILED: {posName} -> {directName} | Error: {e}", flush=True)


if __name__ == '__main__':
    # Read number of CPUs from Slurm environment, default to 1 if not set
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    print(f"Starting {numSims} simulations across {num_workers} workers...", flush=True)

    with Pool(processes=num_workers) as pool:
        pool.map(run_single_sim, randomCombinations)

    print("All simulations complete.", flush=True)
