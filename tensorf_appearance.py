import torch
import torch.nn as nn
import torch.nn.functional as F

# From https://github.com/apchenstu/TensoRF
# Spherical Harmonic normalization constants
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def eval_sh_bases(deg, dirs):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param deg: int SH max degree. Currently, 0-4 supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., (deg+1) ** 2)
    """
    assert deg <= 4 and deg >= 0
    result = torch.empty((*dirs.shape[:-1], (deg + 1) ** 2), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = C0
    if deg > 0:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -C1 * y;
        result[..., 2] = C1 * z;
        result[..., 3] = -C1 * x;
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = C2[0] * xy;
            result[..., 5] = C2[1] * yz;
            result[..., 6] = C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = C2[3] * xz;
            result[..., 8] = C2[4] * (xx - yy);

            if deg > 2:
                result[..., 9] = C3[0] * y * (3 * xx - yy);
                result[..., 10] = C3[1] * xy * z;
                result[..., 11] = C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = C3[5] * z * (xx - yy);
                result[..., 15] = C3[6] * x * (xx - 3 * yy);

                if deg > 3:
                    result[..., 16] = C4[0] * xy * (xx - yy);
                    result[..., 17] = C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result

# Render view dependent color out of view direction and SH coefficients
def SHRender(xyz_sampled, viewdirs, features, sh_degree=2):
    sh_mult = eval_sh_bases(sh_degree, viewdirs)[:, None]   # v = viewdirs / (viewdirs.norm(dim=-1, keepdim=True) + 1e-8)
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb

# Contract 3D points to a bounded domain
def mipnerf_360_contraction(xyz_sampled: torch.Tensor, t : float = 0.5) -> torch.Tensor:
    xyz_contracted = torch.zeros_like(xyz_sampled)
    norm = torch.norm(xyz_sampled, dim=-1, keepdim=True).repeat(1, 3)
    norm_inv = 1.0 / norm
    # Close points
    close_mask = norm <= 1.0
    xyz_contracted[close_mask] = xyz_sampled[close_mask] * t
    # Far away points
    far_away_mask = norm > 1.0
    far_away_value = (2.0 - norm_inv[far_away_mask]) * xyz_sampled[far_away_mask] * norm_inv[far_away_mask]
    xyz_contracted[far_away_mask] = (1.0 - t) * (far_away_value - 1.0) + t

    return xyz_contracted

# Compute grid res out of bbox and number of voxels
def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

# Total Variation Loss (Measure neighbor difference)
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        count_w = max(count_w, 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
# Base class for TensoRF
class FieldBase(nn.Module):
    def __init__(self, n_voxels=3000**3, device='cuda', use_mlp=False, app_dim=27, app_n_comp=16, **kargs):
        super(FieldBase, self).__init__()
        print(f"[INFO] scene_extent: {kargs['scene_extent']}")
        self.device = device
        self.scene_extent = kargs["scene_extent"]
        self.aabb = - self.scene_extent * torch.ones(3), self.scene_extent * torch.ones(3)
        
        # TensoRF parameters
        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.app_dim = app_dim
        self.app_n_comp = app_n_comp
        self.gridSize = N_to_reso(n_voxels, self.aabb)
        self.use_mlp = use_mlp
        print(f"[INFO] gridSize: {self.gridSize}")

class ColorFieldVM(FieldBase):
    def __init__(self, n_voxels=3000**3, device='cuda', use_mlp=False, app_dim=27, app_n_comp=16, sh_degree=2, **kargs):
        self.sh_degree = sh_degree
        super(ColorFieldVM, self).__init__(n_voxels, device, use_mlp, app_dim, app_n_comp, **kargs)
        self.define_modules()
        self.init_svd_volume(self.gridSize[0], self.device)
        print(f"[WARNING] Model doesn't use normal directions")

    # Normalize to [-1, 1]
    def normalize_coord(self, xyz_sampled: torch.Tensor) -> torch.Tensor:
        xyz_normalized = xyz_sampled / self.scene_extent
        xyz_contracted = mipnerf_360_contraction(xyz_normalized, t=0.5) 
        #print(f"Min: {xyz_normalized.min().item()}, Max: {xyz_normalized.max().item()}")
        return torch.clamp(xyz_contracted, -1.0, 1.0)

    # Pick renderer (Only SH)
    def define_modules(self):
        self.reg = TVLoss()
        self.render_module = SHRender   # only use SH
        print(f"[INFO] Render Model: {self.render_module}")
    
    # Initialize TensoRF-VM parameter
    def init_svd_volume(self, res, device):
        self.plane_coef = torch.nn.Parameter(
            0.2 * torch.randn((3, self.app_n_comp, res, res), device=device)
        )
        self.line_coef = torch.nn.Parameter(
            0.2 * torch.randn((3, self.app_n_comp, res, 1), device=device)
        )
        n_sh = (self.sh_degree +1) **2
        self.basis_mat = torch.nn.Linear(self.app_n_comp * 3, 3* n_sh, bias=False).to(device)
    
    # Set different learning rates 
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz}, {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        return grad_vars
    
    # Sample VM-parameters, project to coefficients
    def compute_appfeature(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        return app_features
    
    def TV_loss(self):
        total = 0
        total = total + self.reg(self.plane_coef) * 1e-2 #+ reg(self.app_line[idx]) * 1e-3
        return total


    def forward(self, xyz_sampled, view_directions, normal_directions=None):
        features = self.compute_appfeature(xyz_sampled)
        return self.render_module(xyz_sampled, view_directions, features, sh_degree=self.sh_degree)