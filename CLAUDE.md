# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning research project focused on predicting electric field (E-field) distributions in brain tissue using transcranial magnetic stimulation (TMS) data. The project now implements NVIDIA PhysicsNeMo's Fourier Neural Operator (FNO) architecture that directly maps from magnetic field derivatives (dA/dt) and tissue conductivity to E-field distributions without encoding/decoding.

## Core Architecture

### Key Components
- **Data Simulation (`dataSim.py`)**: Generates TMS simulation data using SimNIBS with random coil positions and directions across EEG electrode locations
- **Dataset Handler (`newDataset.py`)**: PyTorch Dataset class for loading and preprocessing 3D NIfTI brain imaging data with center cropping to 96³ voxels and input normalization
- **Model Architecture (`newModel.py`)**: PhysicsNeMo FNO model that directly processes 4-channel input (3 dA/dt + 1 conductivity) to predict 3-channel E-field output using Fourier domain operations
- **Training Pipeline (`newTrain.py`)**: Training setup with weighted MSE loss, mixed precision training, optimized for FNO architecture
- **Physics Loss (`PhysicsLoss.py`)**: Optional physics-informed loss function implementing Maxwell's equations (divergence, Faraday's law, conductivity consistency)

### Data Flow
1. SimNIBS generates TMS simulations → NIfTI files (dA/dt, conductivity, E-field)
2. Dataset loads, crops to 96³ voxels, and normalizes inputs → PyTorch tensors (CDHW format)
3. FNO directly processes concatenated 4-channel input → 3-channel E-field prediction via Fourier operations
4. No encoding/decoding steps - maintains full spatial resolution throughout

## Development Commands

### Python Environment
- Python 3.12.3 is used
- No explicit dependency management files found - dependencies are imported directly

### Key Dependencies
- PyTorch (with CUDA support)
- nvidia-physicsnemo (FNO implementation)
- nibabel (NIfTI file handling)
- simnibs (TMS simulation)
- tqdm (progress bars)
- numpy

### Running Components

**Data Generation:**
```bash
python dataSim.py
```
Generates random TMS simulations in `simOutput/` directory with position_direction naming.

**Training:**
```bash
python newTrain.py
```
Trains the model using hardcoded data paths. Model checkpoints saved to `checkpoints/` directory.

**Individual Testing:**
```bash
# Test dataset loading
python -c "from newDataset import make_dataloaders; print('Dataset module working')"

# Test model architecture
python -c "from newModel import DualBranchResNet3D; print('Model module working')"
```

## Project Structure

```
cameronscode2/
├── dataSim.py          # SimNIBS TMS simulation generator
├── newDataset.py       # PyTorch dataset and data loading
├── newModel.py         # 3D dual-branch ResNet architecture
├── newTrain.py         # Training loop with weighted loss
└── simOutput/          # Generated simulation data
    ├── C1_AF8/         # Example simulation output
    └── P2_AF4/         # Example simulation output
```

## Model Architecture Details

- **Input**: dA/dt (3 channels) + conductivity (1 channel) concatenated → 4-channel input, 96³ voxels
- **FNO Architecture**: 3D Fourier Neural Operator with 16 Fourier modes, 64 latent channels, 4 layers
- **Processing**: Direct Fourier domain operations - no spatial downsampling/upsampling
- **Output**: 3-channel E-field prediction at full 96³ resolution
- **Loss Options**: 
  - Default: Custom weighted MSE emphasizing higher magnitude E-field regions
  - Optional: Physics-informed loss with Maxwell's equations constraints

## Important Notes

- Training uses mixed precision (AMP) and gradient scaling for memory efficiency  
- FNO handles variable input sizes naturally - no padding required
- Reduced batch size (8) recommended due to FNO memory requirements on Colab
- Hardcoded data paths in `newTrain.py` point to Google Colab structure - update for local use
- SimNIBS head model path in `dataSim.py` uses Windows path - update for Linux/Mac
- No explicit testing framework - verification done through manual imports
- Physics loss available in `PhysicsLoss.py` - easily switchable via commented lines in `newTrain.py`