# A Framework for Evaluating 3DGS Geometry via Mesh-based Novel View Synthesis

## Contents
2 Python scripts: Load a mesh and render RGB images with nvdiffrast (mesh.py), rasterizer output changed to 3D world positions and debug (3d_intersections.py)

Different training scripts: Basic training with meshes (training.py), training with NeRF Synthetic ground truth meshes (training_nerf.py), training with NeRF Synthetic dataset (training_nerf_synthetic.py), test script for debugging (training_nerf_test.py)

TensoRF utils (tensorf_appearance.py) and training script with TensoRF as appearance model (training_tensorf.py)

## Installation
```bash
git clone https://github.com/marcoseibold/bachelorarbeit.git

conda env create --file environment.yml
conda activate meshing_splats
```

## Training
```bash
python training.py --mesh <path to Mesh> --data_root <path to dataset>/<scene> --model <voxel_grid or tensorf> 
```

## Geometry metric calculation
```bash
python metrics.py <path to reconstructed mesh> <path to GT mesh>
```

