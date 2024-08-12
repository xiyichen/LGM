import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS

from core.unet import UNet
from core.options import Options
from core.gs import GaussianRenderer

def get_ortho_rays(origins, directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3
    assert origins.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = torch.matmul(c2w[:, :3, :3], directions[:, :, None]).squeeze()  # (N_rays, 3)
        rays_o = torch.matmul(c2w[:, :3, :3], origins[:, :, None]).squeeze()  # (N_rays, 3)
        rays_o = c2w[:,:3,3].expand(rays_d.shape) + rays_o  
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = torch.matmul(c2w[None, None, :3, :3], directions[:, :, :, None]).squeeze()  # (H, W, 3)
            rays_o = torch.matmul(c2w[None, None, :3, :3], origins[:, :, :, None]).squeeze()  # (H, W, 3)
            rays_o = c2w[None, None,:3,3].expand(rays_d.shape) + rays_o  
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = torch.matmul(c2w[:,None, None, :3, :3], directions[None, :, :, :, None]).squeeze()  # # (B, H, W, 3)
            rays_o = torch.matmul(c2w[:,None, None, :3, :3], origins[None, :, :, :, None]).squeeze()  # # (B, H, W, 3)
            rays_o = c2w[:,None, None, :3,3].expand(rays_d.shape) + rays_o  

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d

def get_ortho_ray_directions_origins(W, H, use_pixel_centers=False):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    origins = torch.stack([(i/W-0.5)*2, (j/H-0.5)*2, torch.zeros_like(i)], dim=-1) # W, H, 3
    directions = torch.stack([torch.zeros_like(i), torch.zeros_like(j), -torch.ones_like(i)], dim=-1) # W, H, 3

    return origins, directions

def compute_fovy(focal, h):
    """
    Compute the vertical field of view (fovy) given the focal length and image height.

    Parameters:
    focal (float): The focal length.
    h (float): The height of the image.

    Returns:
    float: The vertical field of view (fovy) in degrees.
    """
    fovy_rad = 2 * np.arctan(h / (2 * focal))
    fovy_deg = np.rad2deg(fovy_rad)
    return fovy_deg

def invert_matrix(matrix):
    """
    Invert a 4x4 transformation matrix.
    
    Parameters:
    matrix (numpy.ndarray): 4x4 matrix to be inverted
    
    Returns:
    numpy.ndarray: Inverted 4x4 matrix
    """
    # Extract the rotation matrix (upper-left 3x3 part)
    R = matrix[:3, :3]
    
    # Extract the translation vector (last column, excluding the bottom element)
    T = matrix[:3, 3]
    
    # Compute the inverse rotation (transpose of the rotation matrix)
    R_inv = R.T
    
    # Compute the transformed translation vector
    T_inv = -R_inv @ T
    
    # Construct the inverse matrix
    inverse_matrix = np.eye(4)
    inverse_matrix[:3, :3] = R_inv
    inverse_matrix[:3, 3] = T_inv
    
    return inverse_matrix

def inv_RT(RT):
    RT_h = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
    RT_inv = np.linalg.inv(RT_h)

    return RT_inv[:3, :]

def RT_opengl2opencv(RT):
     # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    R = RT[:3, :3]
    t = RT[:3, 3]

    R_bcam2cv = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)

    R_world2cv = R_bcam2cv @ R
    t_world2cv = R_bcam2cv @ t

    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)

    return RT

def generate_rx_90():
    """
    Generates a 4x4 rotation matrix for a 90-degree rotation around the x-axis.

    Returns:
        numpy.ndarray: The 4x4 rotation matrix.
    """
    rx_90 = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    return rx_90

class LGM(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        # unet
        self.unet = UNet(
            9, 14, 
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
        )

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)


    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict


    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays, get_rays_focal, get_rays_K
        # cam_poses_c2w = np.stack([
        #     invert_matrix(orbit_camera(elevation, 0, radius=self.opt.cam_radius)), # front [0,0,1.5] -> [0,-1.5,0]
        #     invert_matrix(orbit_camera(elevation, 90, radius=self.opt.cam_radius)), # right [1.5,0,0] -> [1.5,0,0]
        #     invert_matrix(orbit_camera(elevation, 180, radius=self.opt.cam_radius)), # back [0,0,-1.5] -> [0,1.5,0]
        #     invert_matrix(orbit_camera(elevation, 270, radius=self.opt.cam_radius)), # left [-1.5,0,0] -> [-1.5,0,0]
        # ], axis=0) # [4, 4, 4]
        # cam_poses_c2w = torch.from_numpy(cam_poses_c2w)
        # cam_poses = np.stack([o
        #     (orbit_camera(elevation, 0, radius=self.opt.cam_radius)),
        #     (orbit_camera(elevation, 90, radius=self.opt.cam_radius)),
        #     (orbit_camera(elevation, 180, radius=self.opt.cam_radius)),
        #     (orbit_camera(elevation, 270, radius=self.opt.cam_radius)),
        # ], axis=0) # [4, 4, 4]
        
        from scipy.spatial.transform import Rotation as Rot
        # rotation_matrix = np.array([[1, 0, 0, 0],  [0, 0, 1, 0],  [0, -1, 0, 0], [0, 0, 0, 1]])
        # cam_pos_origin = orbit_camera(elevation, 0, radius=self.opt.cam_radius)[:3,3] # c2w
        # cam_pos_rotated = rotation_matrix[:3,:3]@cam_pos_origin
        # # cam_rot_new = Rot.from_matrix(rotation_matrix[:3,:3]@orbit_camera(elevation, 0, radius=self.opt.cam_radius)[:3,:3].T).as_euler('xyz', True)
        # cam_rot_new = Rot.from_matrix(rotation_matrix[:3,:3]@orbit_camera(elevation, 270, radius=self.opt.cam_radius)[:3,:3].T).as_euler('xyz', True)
        
        # w2c
        # Ks = np.zeros((4,3,3))
        # Ks[0] = np.array([[296.555 ,   0.    , 139.6875],
        # [  0.    , 296.555 , 125.9375],
        # [  0.    ,   0.    ,   1.    ]])
        # Ks[1] = np.array([[296.555 ,   0.    , 124.1875],
        # [  0.    , 296.555 , 125.9375],
        # [  0.    ,   0.    ,   1.    ]])
        # Ks[2] = np.array([[296.555 ,   0.    , 107.4375],
        # [  0.    , 296.555 ,  94.9375],
        # [  0.    ,   0.    ,   1.    ]])
        # Ks[3] = np.array([[296.555 ,   0.    , 140.6875],
        # [  0.    , 296.555 , 113.6875],
        # [  0.    ,   0.    ,   1.    ]])
        # cam_poses = np.zeros((4,4,4))
        
        # cam_poses[0] = [[ 0.0741335 , -0.996402  , -0.0410817 , -0.0354861 ],
        # [ 0.132818  , -0.030963  ,  0.990657  ,  0.02833058],
        # [-0.988364  , -0.0788973 ,  0.130045  ,  1.02277032],
        # [0,0,0,1]] # front
        
        # cam_poses[1] = [[-0.999448  , -0.029158  , -0.0159137 ,  0.00685991],
        # [-0.0120158 , -0.129291  ,  0.991534  ,  0.02803919],
        # [-0.0309686 ,  0.991178  ,  0.128869  ,  1.03400435],
        # [0,0,0,1]] # right
        
        
        # cam_poses[2] = [[-0.00310857,  0.999836  ,  0.0178431 ,  0.06578426],
        # [-0.263284  , -0.0180319 ,  0.96455   ,  0.12433737],
        # [ 0.964713  , -0.00169943,  0.263297  ,  1.09644205],
        # [0,0,0,1]] # back
        
        
        # cam_poses[3] = [[ 0.994929  ,  0.0991203 , -0.0170984 , -0.03746843],
        # [ 0.00644421,  0.106826  ,  0.994257  ,  0.05642239],
        # [ 0.100378  , -0.989325  ,  0.105646  ,  1.0232148 ],
        # [0,0,0,1]] # left
        
        Ks = np.zeros((4,3,3))
        cam_poses = np.zeros((4,4,4))
        
        cam_poses[0] = [[ 0.0741335 , -0.996402  , -0.0410817 , -0.0354861 ],
        [ 0.132818  , -0.030963  ,  0.990657  ,  0.02833058],
        [-0.988364  , -0.0788973 ,  0.130045  ,  1.02277032],
        [0,0,0,1]] # front
        Ks[0] = np.array([[253.06026667,   0.        , 110.45333333],
        [  0.        , 253.06026667,  99.78666667],
        [  0.        ,   0.        ,   1.        ]])
    
        cam_poses[1] = [[-0.999448  , -0.029158  , -0.0159137 ,  0.00685991],
        [-0.0120158 , -0.129291  ,  0.991534  ,  0.02803919],
        [-0.0309686 ,  0.991178  ,  0.128869  ,  1.03400435],
        [0,0,0,1]] # right
        Ks[1] = np.array([[253.06026667,   0.        , 124.74666667],
        [  0.        , 253.06026667, 126.24      ],
        [  0.        ,   0.        ,   1.        ]])
        
        
        cam_poses[2] = [[-0.00310857,  0.999836  ,  0.0178431 ,  0.06578426],
        [-0.263284  , -0.0180319 ,  0.96455   ,  0.12433737],
        [ 0.964713  , -0.00169943,  0.263297  ,  1.09644205],
        [0,0,0,1]] # back
        Ks[2] = np.array([[253.06026667,   0.        , 110.45333333],
        [  0.        , 253.06026667,  99.78666667],
        [  0.        ,   0.        ,   1.        ]])
        
        
        cam_poses[3] = [[ 0.994929  ,  0.0991203 , -0.0170984 , -0.03746843],
        [ 0.00644421,  0.106826  ,  0.994257  ,  0.05642239],
        [ 0.100378  , -0.989325  ,  0.105646  ,  1.0232148 ],
        [0,0,0,1]] # left
        Ks[3] = np.array([[253.06026667,   0.        , 138.82666667],
        [  0.        , 253.06026667, 115.78666667],
        [  0.        ,   0.        ,   1.        ]])
        
        
        # cam_poses[0] = invert_matrix(cam_poses[0]) # [2.03, 0.07, -1.10] x y z -> y z x
        # cam_poses[1] = invert_matrix(cam_poses[1]) # [0.08, -2.07, -1.10]
        # cam_poses[2] = invert_matrix(cam_poses[2]) # [-2.04, -0.15, -1.59]
        # cam_poses[3] = invert_matrix(cam_poses[3]) # [-0.13, 1.99, -1.10]
        import trimesh, pdb
        # Load your mesh
        mesh = trimesh.load('/cluster/scratch/xiychen/100050/smplx/smplx_mesh/000000.obj')
        vertices = np.array(mesh.vertices)
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        center = ((min_coords + max_coords) / 2) * 0
        for i in range(4):
            cam_poses[i] = invert_matrix(cam_poses[i])
            cam_poses[i][:3,3] -= center
            c2w_blender = kiui.cam.convert(cam_poses[i].copy(), 'opencv', 'blender')
            c2w_blender[:3,1] *= -1
            c2w_blender[:3,2] *= -1
            R_x = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
            
            # c2w_opengl = kiui.cam.convert(c2w_blender.copy(), 'blender', 'opengl')
            c2w_opengl = kiui.cam.convert(R_x@c2w_blender.copy(), 'blender', 'opengl')
            c2w_blender = kiui.cam.convert(c2w_opengl.copy(), 'opengl', 'blender')
            # pdb.set_trace()
            # c2w_opencv = kiui.cam.convert(c2w_blender.copy(), 'blender', 'opencv')
            cam_poses[i] = c2w_opengl
        
        # import pdb
        # pdb.set_trace()
        
        
        # import pdb
        # pdb.set_trace()
        
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        # origins, dirs = get_ortho_ray_directions_origins(W=self.opt.input_size, H=self.opt.input_size)
        for i in range(cam_poses.shape[0]):
            import pdb
            # pdb.set_trace()
            # rays_o, rays_d = get_ortho_rays(origins.float(), dirs.float(), cam_poses[i].float(), True)
            fovy = compute_fovy(296.555, 256)
            # pdb.set_trace()
            # rays_o, rays_d = get_rays(cam_poses[i].float(), self.opt.input_size, self.opt.input_size, self.opt.fovy)
            # rays_o, rays_d = get_rays(cam_poses[i].float(), self.opt.input_size, self.opt.input_size, fovy) # [h, w, 3]
            rays_o, rays_d = get_rays_K(cam_poses[i].float(), self.opt.input_size, self.opt.input_size, Ks[i])
            # pdb.set_trace()
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
        

    def forward_gaussians(self, images):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x = self.unet(images) # [B*4, 14, h, w]
        x = self.conv(x) # [B*4, 14, h, w]

        x = x.reshape(B, 4, 14, self.opt.splat_size, self.opt.splat_size)
        
        ## visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians

    
    def forward(self, data, step_ratio=1):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        images = data['input'] # [B, 4, 9, h, W], input features
        
        # use the first view to predict gaussians
        gaussians = self.forward_gaussians(images) # [B, N, 14]

        results['gaussians'] = gaussians

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        
        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results
