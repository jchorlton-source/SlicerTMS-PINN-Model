# SimNIBS 4.5.0 — Docker / Singularity container

Packages SimNIBS 4.5.0 for local testing (Docker) and production runs on
Setonix/Pawsey (Singularity). Your scripts and data are always mounted at
runtime — nothing is baked into the image.

---

## Local testing (Docker)

### 1. Build

```bash
docker build -t simnibs:4.5.0 .
```

First build takes ~10–15 min (downloads ~4 GB of conda packages).

### 2. Run your script

Mount the folder containing your script and m2m subject folder into `/data`:

```bash
docker run --rm \
  -v /path/to/your/m2m_ernie:/data/m2m_ernie \
  -v /path/to/your/scripts:/data/scripts \
  -v /path/to/output:/data/output \
  simnibs:4.5.0 \
  python /data/scripts/dataSim.py
```

The Python interpreter inside the container is the SimNIBS-managed one —
equivalent to selecting the SimNIBS `activate` interpreter locally.

### 3. Interactive shell (useful for debugging)

```bash
docker run --rm -it \
  -v /path/to/your/m2m_ernie:/data/m2m_ernie \
  simnibs:4.5.0 \
  bash
```

Then inside:
```bash
python -c "import simnibs; print(simnibs.__version__)"
python /data/scripts/dataSim.py
```

---

## Setonix (Singularity/Apptainer)

Setonix cannot build containers — build locally, push to Docker Hub, pull on Setonix.

### 1. Push to Docker Hub

```bash
docker tag simnibs:4.5.0 yourdockerid/simnibs:4.5.0
docker push yourdockerid/simnibs:4.5.0
```

### 2. Pull on Setonix (run once on login node)

```bash
module load singularity/3.8.6-nompi

singularity pull \
  /scratch/${PAWSEY_PROJECT}/${USER}/containers/simnibs-4.5.0.sif \
  docker://yourdockerid/simnibs:4.5.0
```

The `.sif` is ~5–6 GB — store it in `/scratch`, not `$HOME`.

### 3. Submit with SLURM

Edit `run_simnibs.slurm` and submit:

```bash
sbatch run_simnibs.slurm
```

---

## How paths work inside the container

| Host path | Mount point | Variable in script |
|-----------|-------------|-------------------|
| your m2m folder | `/data/input` | `s.subpath` |
| your output dir | `/data/output` | `s.pathfem` / `baseDir` |
| your script | `/data/scripts` | passed to `python` |

The `--bind` (Singularity) or `-v` (Docker) flags connect host paths to these mount points.
