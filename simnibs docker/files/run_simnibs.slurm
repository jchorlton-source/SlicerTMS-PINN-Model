#!/bin/bash
# ============================================================
#  SimNIBS 4.5.0 — SLURM batch script for Setonix (Pawsey)
#  Edit the VALUES TO CHANGE sections before submitting.
# ============================================================

#SBATCH --job-name=simnibs
#SBATCH --account=pawsey1242              # ← YOUR PROJECT ID
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # SimNIBS FEM solver is multithreaded
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── VALUES TO CHANGE ─────────────────────────────────────────
SIF=/scratch/${PAWSEY_PROJECT}/${USER}/containers/simnibs-4.5.0.sif
SCRIPT=/scratch/${PAWSEY_PROJECT}/${USER}/Thesis/dataSim.py
M2M_DIR=/scratch/${PAWSEY_PROJECT}/${USER}/Thesis/data/m2m_ernie
OUTPUT_DIR=/scratch/${PAWSEY_PROJECT}/${USER}/Thesis/simOutput
# ─────────────────────────────────────────────────────────────

module load singularity/3.8.6-nompi

mkdir -p "${OUTPUT_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export APPTAINERENV_OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SINGULARITYENV_OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

singularity exec \
  --bind "${M2M_DIR}":"${M2M_DIR}" \
  --bind "${OUTPUT_DIR}":"${OUTPUT_DIR}" \
  --bind "$(dirname ${SCRIPT})":"$(dirname ${SCRIPT})" \
  "${SIF}" \
  python "${SCRIPT}"
