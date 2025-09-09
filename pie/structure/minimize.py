from pathlib import Path
from tqdm import tqdm
from openmm.app import *
from openmm import *
from openmm.unit import *

# Preload forcefield once (avoid reloading for every file)
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

def minimize_pdb(pdb_file: Path, output_file: Path):
    """Minimize a single PDB file and save the result."""
    pdb = PDBFile(str(pdb_file))
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=NoCutoff,
        constraints=None
    )

    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    # Energy minimization
    simulation.minimizeEnergy()

    # Save minimized structure
    positions = simulation.context.getState(getPositions=True).getPositions()
    with output_file.open("w") as f:
        PDBFile.writeFile(simulation.topology, positions, f)

def minimize_all_pdbs(input_dir: str | Path, output_dir: str | Path):
    """
    Minimize all PDB structures in input_dir and save them in output_dir
    with '_min.pdb' suffix. If a file fails, print the error and continue.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = list(input_dir.glob("*.pdb"))

    for pdb_file in tqdm(pdb_files, desc="Minimizing structures"):
        out_file = output_dir / f"{pdb_file.stem}_min.pdb"
        try:
            minimize_pdb(pdb_file, out_file)
        except Exception as e:
            print(f"PDB {pdb_file.name} could not be minimized: {e}")

    print("All minimizations done.")