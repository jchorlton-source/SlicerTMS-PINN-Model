# ============================================================
#  SimNIBS 4.5.0 — Docker image (Singularity/Apptainer compatible)
#
#  Singularity compatibility:
#   - SimNIBS installed under /opt/ (Singularity remounts $HOME
#     at runtime, hiding anything written there at build time)
#   - No USER switching
#   - Permissions set world-readable for non-root runtime user
#   - PATH forwarded via SINGULARITYENV_ and APPTAINERENV_
# ============================================================
FROM continuumio/miniconda3:24.1.2-0

LABEL version="4.5.0" \
      description="SimNIBS 4.5.0 — TMS/TES brain stimulation simulation"

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        gcc \
        g++ \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
        libxext6 \
        libx11-6 \
        libgomp1 \
        libstdc++6 \
        ca-certificates \
        locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/* \
    && chmod 1777 /tmp /var/tmp

# ── Install SimNIBS into /opt/conda/envs/simnibs ─────────────
RUN wget -q \
        https://raw.githubusercontent.com/simnibs/simnibs/v4.5.0/environment_linux.yml \
        -O /tmp/environment_linux.yml \
    && conda env create -f /tmp/environment_linux.yml -n simnibs \
    && conda clean -afy \
    && rm /tmp/environment_linux.yml

RUN /opt/conda/envs/simnibs/bin/pip install --no-cache-dir \
    https://github.com/simnibs/simnibs/releases/download/v4.5.0/simnibs-4.5.0-cp311-cp311-linux_x86_64.whl

# ── Post-install: write SimNIBS data files to /opt/simnibs ───
RUN mkdir -p /opt/simnibs \
    && /opt/conda/envs/simnibs/bin/postinstall_simnibs \
       --no-links -d /opt/simnibs \
    || true

# ── Also install tqdm (used in dataSim.py) ───────────────────
RUN /opt/conda/envs/simnibs/bin/pip install --no-cache-dir tqdm

# ── World-readable so Singularity's non-root user can run ────
RUN chmod -R a+rX /opt/conda/envs/simnibs /opt/simnibs

# ── Bind-mount point for data/scripts at runtime ─────────────
RUN mkdir -p /data && chmod 777 /data

# ── Environment ──────────────────────────────────────────────
ENV PATH="/opt/conda/envs/simnibs/bin:$PATH" \
    SIMNIBS_DIR="/opt/simnibs" \
    LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8" \
    LD_PRELOAD="/opt/conda/envs/simnibs/lib/libstdc++.so.6" \
    # Forward PATH into Singularity (3.x) and Apptainer (>=1.1)
    SINGULARITYENV_PATH="/opt/conda/envs/simnibs/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    SINGULARITYENV_SIMNIBS_DIR="/opt/simnibs" \
    SINGULARITYENV_LD_PRELOAD="/opt/conda/envs/simnibs/lib/libstdc++.so.6" \
    APPTAINERENV_PATH="/opt/conda/envs/simnibs/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    APPTAINERENV_SIMNIBS_DIR="/opt/simnibs" \
    APPTAINERENV_LD_PRELOAD="/opt/conda/envs/simnibs/lib/libstdc++.so.6"

WORKDIR /data

# ── Smoke-test ───────────────────────────────────────────────
RUN python -c "import simnibs; print('SimNIBS', simnibs.__version__, 'ready')"

CMD ["bash"]
