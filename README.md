# A Framework for Evaluating 3DGS Geometry via Mesh-based Novel View Synthesis

## Contents
Training script to train the selected appearance model (Voxel grid or TensoRF) and perform NVS afterwards. Geometry metric calculation script for NeRF-Synthetic. Tanks & Temples standard evaluation script in eval_tnt. TensoRF utils (tensorf_appearance.py) and dataset loader (dataset_loader.py) for NeRF-Synthetic, MipNeRF 360, Tanks & Temples

Old scripts: Load a mesh and render RGB images with nvdiffrast (mesh.py), rasterizer output changed to 3D world positions and debug (3d_intersections.py), and some old training versions

## Installation
```bash
git clone https://github.com/marcoseibold/bachelorarbeit.git

conda env create --file environment.yml
conda activate meshing_splats
```
For TNT evaluation, a 2DGS/GOF environment can be used.

## Training
```bash
python training.py --mesh <path to Mesh> --data_root <path to dataset>/<scene> --model <voxel_grid or tensorf> 
```

## Geometry metric calculation
```bash
python metrics.py <path to reconstructed mesh> <path to GT mesh>
```

