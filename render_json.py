# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use 
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #
# python render_json.py -m ../3dgs_scenes/fish_real -s ../3dgs_scenes/fish_real/input --out ../3dgs_scenes/fish_real/output --white_background

import os
from os import makedirs
from tqdm import tqdm
import torch
import torchvision
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import math
import json
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View, getWorld2View2, getProjectionMatrix, focal2fov


def interpolate_camera_path(camera_poses, num_frames):
    """
    Interpolate camera path between given camera poses.
    """
    rotations = []
    translations = []
    for pose in camera_poses:
        R = np.array(pose['rotation'])
        T = np.array(pose['position'])
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R
        Rt[:3, 3] = T
        Rt[3, 3] = 1.0
        W2C = np.linalg.inv(Rt)
        R = W2C[:3, :3].transpose()
        T = W2C[:3, 3]
        rotations.append(R)
        translations.append(T)
    rotations = Rotation.from_matrix(rotations)
    translations = np.array(translations)

    # Convert rotation matrices to Rotation objects
    # rotations = Rotation.from_matrix([np.array(pose['rotation']) for pose in camera_poses])
    # translations = np.array([pose['position'] for pose in camera_poses])
    
    key_times = np.linspace(0, 1, len(camera_poses))
    interp_times = np.linspace(0, 1, num_frames)
    interp_times = interp_times[:-1]
    
    # Create rotation interpolator
    slerp = Slerp(key_times, rotations)
    trans_interpolator = interp1d(key_times, translations, axis=0)
    
    interpolated_poses = []
    for t in interp_times:
        R = slerp(t).as_matrix()
        T = trans_interpolator(t)
        interpolated_poses.append({
            'rotation': R.tolist(),
            'position': T.tolist(),
            'fy': camera_poses[0]['fy'],
            'fx': camera_poses[0]['fx'],
            'width': camera_poses[0]['width'],
            'height': camera_poses[0]['height']
        })
    
    return interpolated_poses


def process_cameras_json(input_path, output_path, frames_between=30):
    """
    Read cameras.json, interpolate poses, and save to cameras2.json
    
    Args:
        input_path: Path to input cameras.json
        output_path: Path to output cameras2.json
        frames_between: Number of frames to interpolate between each pair
    """
    # Read original cameras
    with open(input_path, 'r') as f:
        cameras = json.load(f)
    with open(input_path, 'w') as f:
        json.dump(cameras, f, indent=2)
    
    # Interpolate
    cameras = cameras + [cameras[0]]
    interpolated_cameras = interpolate_camera_path(cameras, 
                                                 num_frames=(len(cameras)-1)*frames_between + 1)
    
    # Save interpolated cameras
    with open(output_path, 'w') as f:
        json.dump(interpolated_cameras, f, indent=2)


def img2gif(in_filenames, out_filename, d=40):
    images = [Image.open(str(in_filename)) for in_filename in in_filenames]
    images[0].save(str(out_filename), save_all=True, append_images=images[1:], optimize=False, loop=0, duration=d)


class MyCamera:
    def __init__(self, R, T, FoVx, FoVy, W, H):
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        self.R = R
        self.T = T
        # self.world_view_transform = torch.tensor(getWorld2View(R, T)).transpose(0, 1).cuda()
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0,0,0]), 1.0)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.image_width = W
        self.image_height = H
        self.FoVx = FoVx
        self.FoVy = FoVy


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, render_path : str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        model_path = dataset.model_path
        iteration = scene.loaded_iter

        with open(os.path.join(render_path, "cameras.json")) as json_file:
            contents = json.load(json_file)
            views = []
            for content in contents:
                R = np.array(content["rotation"], "float32")
                T = np.array(content["position"], "float32")
                # Rt = np.zeros((4, 4))
                # Rt[:3, :3] = R
                # Rt[:3, 3] = T
                # Rt[3, 3] = 1.0
                # C2W = np.linalg.inv(Rt)
                # R = C2W[:3, :3].transpose()
                # T = C2W[:3, 3]
                W = np.array(content["width"], "int")
                H = np.array(content["height"], "int")
                FoVx = focal2fov(np.array(content["fx"], "float32"), W)
                FoVy = focal2fov(np.array(content["fy"], "float32"), H)

                view = MyCamera(R, T, FoVx, FoVy, W, H)
                views.append(view)


        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background)["render"]
            # rendering = rendering[..., rendering.shape[-1] // 2:]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        renders = sorted(list(Path(render_path).glob("*.png")))
        img2gif(renders, os.path.join(render_path, "ani.gif"), d=30)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out", help="output camera json file")
    parser.add_argument("--frames", default=30, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    print(args)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Generate new cameras
    makedirs(args.out, exist_ok=True)
    process_cameras_json(os.path.join(args.model_path, "cameras.json"), os.path.join(args.out, "cameras.json"), frames_between=args.frames)
    
    # Rendering
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.out)

