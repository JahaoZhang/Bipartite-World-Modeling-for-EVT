import habitat
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from habitat.config.default_structured_configs import AgentConfig
from habitat.tasks.nav.nav import NavigationEpisode
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import trange
import math
from PIL import Image, ImageDraw
import os
import torch
import time
torch.set_num_threads(8)
torch.set_num_interop_threads(4)
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)
import os.path as osp
import imageio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from experiments.robot.openvla_utils import (
    get_action_head,
    get_path_point_token_MLP,
    get_path_point_head,
    get_position_token_MLP,
    get_position_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    normalize_position,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from habitat.tasks.utils import get_angle


def get_target_rel_polar_position(robot_agent, target_pos: np.ndarray) -> np.ndarray:
    """
    Compute target position in robot agent frame as polar coords [angle, distance].

    Same convention as TrackVLA/evt_bench/additional_action.py (get_r_xita):
      - angle: rad, left negative / right positive
      - distance: meters in xz plane
    """
    robot_pos = np.array(robot_agent.base_pos, dtype=np.float64)
    target_pos = np.array(target_pos, dtype=np.float64)

    base_T = robot_agent.base_transformation
    forward = np.array([1.0, 0.0, 0.0])
    robot_forward = np.array(base_T.transform_vector(forward))[[0, 2]]

    rel_pos = (target_pos - robot_pos)[[0, 2]]
    distance = float(np.linalg.norm(rel_pos))
    if distance < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)

    angle = float(get_angle(robot_forward, rel_pos))
    is_left = np.cross(robot_forward, rel_pos) > 0
    angle = angle if is_left else -angle
    return np.array([angle, distance], dtype=np.float32)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/home/mmc_jhzhang/z_outputs/outputs_11images/pathpoints/224px/pathpoints_pad_action_next_position_envrebuild_targrtrebuild_lang_stt_dt_224px_Decoupling_envmask_63_onestate/configs+hm3dpathpadsttdt224px_dataset+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter--track--multi--pathpoints_pad_action_next_position_envrebuild_targrtrebuild_lang_stt_dt_224px_Decoupling_envmask----18000_chkpt"     # Pretrained checkpoint path
    # pretrained_checkpoint: Union[str, Path] = "model"     # Pretrained checkpoint path
    
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_minivlm: bool = True                         # If True, uses minivlm
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)
    unnorm_key: Union[str, Path] = "hm3dpathpadsttdt224px_dataset"                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    env_img_res: int = 160                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    save_version: str = "vla-adapter"                # version of 
    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    use_pro_version: bool = True                     # encourage to use the pro models we released.
    seed: int = 7                                    # Random Seed (for reproducibility)

def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)
        path_point_token_MLP = get_path_point_token_MLP(cfg, model.llm_dim)
        path_point_head = get_path_point_head(cfg, model.llm_dim)
        position_token_MLP = get_position_token_MLP(cfg, model.llm_dim)
        position_head = get_position_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        # check_unnorm_key(cfg, model)

    return model, action_head, path_point_token_MLP, path_point_head, position_token_MLP, position_head, proprio_projector, noisy_action_projector, processor

def evaluate_agent(config, dataset_split, save_path) -> None:
    cfg = GenerateConfig()
    
    model, action_head, path_point_token_MLP, path_point_head, position_token_MLP, position_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    position_norm_stats = model.norm_stats[cfg.unnorm_key]["position"]

    # robot definition
    robot_config = GTBBoxAgent(save_path)
    
    first_init = True
    with habitat.TrackEnv(
        config=config,
        dataset=dataset_split
    ) as env:
        sim = env.sim
        robot_config.reset()
        
        num_episodes = len(env.episodes)
        for _ in trange(num_episodes):
            obs = env.reset()
            light_setup = [
                LightInfo(
                    vector=[10.0, -2.0, 0.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[-10.0, -2.0, 0.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[0.0, -2.0, 10.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[0.0, -2.0, -10.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
            ]
            sim.set_light_setup(light_setup)

            result = {}
            record_infos = []

            if first_init:
                instruction = env.current_episode.info['instruction']
                first_init = False

            action_dict = dict()
            finished = False
            
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent

            iter_step = 0
            followed_step = 0
            human_no_move = 0
            too_far_count = 0
            status = 'Normal'
            info = env.get_metrics()
            image_feature_memory = []
            last_action = 0

            while not env.episode_over:
                record_info = {}
                
                obs = sim.get_sensor_observations()
                rgb = obs["agent_1_articulated_agent_jaw_rgb"]

                # Target (main human) position relative to robot, in agent polar frame [angle(rad), dist(m)]
                target_world_pos = np.array(humanoid_agent_main.base_pos)
                target_rel_position_raw = get_target_rel_polar_position(robot_agent, target_world_pos)
                target_rel_position = normalize_position(target_rel_position_raw, position_norm_stats)

                rgb_ = rgb[:, :, :3]
                
                h, w, _ = rgb_.shape
                side = min(h, w)

                top = (h - side) // 2
                left = (w - side) // 2
                rgb_crop = rgb_[top:top+side, left:left+side]
                import cv2

                rgb_resized = cv2.resize(rgb_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
                observation = {
                    "full_image": rgb_resized,
                    "target_rel_position": target_rel_position,
                }
                task_description = instruction

                time_start = time.time()
                action, predicted_path_points, predicted_position, image_feature_memory= get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    path_point_token_MLP=path_point_token_MLP,
                    path_point_head=path_point_head,
                    position_token_MLP=position_token_MLP,
                    position_head=position_head,
                    proprio_projector=None,
                    noisy_action_projector=None,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm,
                    image_feature_memory=image_feature_memory
                )
                action = action[0]

                turn_speed_weighted = action[0]
                tangent_speed_weighted = action[1]
                forward_speed_weighted = action[2]

                forward_speed_weighted = forward_speed_weighted * 0.7 + last_action * 0.3

                last_action = forward_speed_weighted
                action_use = [forward_speed_weighted, tangent_speed_weighted, -turn_speed_weighted]


                n_waypoints = 8
                polar_coords = predicted_path_points.reshape(n_waypoints, 2)
                polar_coords = np.vstack([np.array([[0.0, 0.0]], dtype=np.float32), polar_coords])

                angles = polar_coords[:, 0]
                distances = polar_coords[:, 1]
                distances = np.clip(distances, 0.0, None)

                cartesian_coords = np.zeros((n_waypoints, 2), dtype=np.float32)
                for i in range(n_waypoints):
                    angle = float(angles[i])
                    dist = float(distances[i])

                    cartesian_coords[i, 0] = dist * math.cos(angle)    
                    cartesian_coords[i, 1] = -dist * math.sin(angle)   
                cartesian_coords[0, 0] = 0.0
                cartesian_coords[1, 0] = 0.5
                pil_img = Image.fromarray(rgb_crop)

                # ===== Visualization =====
                try:
                    pil_img = Image.fromarray(rgb_crop)
                    draw = ImageDraw.Draw(pil_img)

                    w, h = pil_img.size
                    base_x = w // 2
                    base_y = int(h * 0.86)
                    meters_to_pixels = h / 3.0 *0.3

                    def to_pxxy(traj_xy):
                        pts = []
                        for i in range(traj_xy.shape[0]):
                            x, y = float(traj_xy[i, 0]), float(traj_xy[i, 1])
                            px = base_x - int(y * meters_to_pixels)
                            py = base_y - int(x * meters_to_pixels)
                            pts.append((px, py))
                        return pts

                    pts_pred = to_pxxy(cartesian_coords)

                    if len(pts_pred) > 1:
                        for i in range(1, len(pts_pred)):
                            draw.line([pts_pred[i-1], pts_pred[i]], fill=(0, 0, 0), width=10)
                        for i in range(1, len(pts_pred)):
                            draw.line([pts_pred[i-1], pts_pred[i]], fill=(0, 255, 200), width=6)
                            r = 8 - 0.8 * i
                            sx, sy = pts_pred[i]
                            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(0, 255, 0))

                    if pts_pred:
                        r = 6
                        sx, sy = pts_pred[0]
                        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(0, 255, 0))
                except Exception as e:
                    print(f"[VIS] Error at step {iter_step}: {e}")
                    import traceback
                    traceback.print_exc()
                pil_img = pil_img.copy()

                draw = ImageDraw.Draw(pil_img)
                draw.text(
                    (10, 25),
                    str(predicted_position),
                    fill=(255, 255, 255)  
                )
                draw.text(
                    (10, 50),
                    f"Instr: {task_description}",
                    fill=(255, 255, 255),
                )
                draw.text(
                    (10, 75),
                    f"Action: {action_use}",
                    fill=(255, 255, 255),
                )
                
                detector = env.task._get_observations(env.current_episode)
                action_no_use = robot_config.act(obs, detector, env.current_episode.episode_id, pil_img)

                action_dict = {
                    "action": ("agent_0_humanoid_navigate_action", "agent_1_base_velocity", "agent_2_oracle_nav_randcoord_action_obstacle", "agent_3_oracle_nav_randcoord_action_obstacle", "agent_4_oracle_nav_randcoord_action_obstacle", "agent_5_oracle_nav_randcoord_action_obstacle"),
                    "action_args": {
                        "agent_1_base_vel" : action_use
                    }
                }
                
                iter_step += 1
                env.step(action_dict)

                info = env.get_metrics()
                if info['human_following'] == 1.0:
                    print("Followed")
                    followed_step += 1
                    too_far_count = 0
                else:
                    print("Lost")

                if np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos) > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        print("Too far from human!")
                        status = 'Lost'
                        finished = False
                        break

                record_info["step"] = iter_step
                record_info["target_rel_position_raw"] = target_rel_position_raw.tolist()
                record_info["target_rel_position"] = target_rel_position.tolist()
                record_info["dis_to_human"] = float(np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos))
                record_info["facing"] = info['human_following']
                record_infos.append(record_info)

                if info['human_collision'] == 1.0:
                    print("Collision detected!")
                    status = 'Collision'
                    finished = False
                    break
                
                print(f"========== ID: {env.current_episode.episode_id} Step now is: {iter_step} action is: {action} dis_to_main_human: {np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos)} ============")

            print("finished episode id: ", env.current_episode.episode_id)
            info = env.get_metrics()
            robot_config.reset(env.current_episode)

            if env.episode_over:
                finished = True
            
            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(record_infos, f, indent=2) 
            result['finish'] = finished
            result['status'] = status
            if iter_step < 300:
                result['success'] = info['human_following_success'] and info['human_following']
            else:
                result['success'] = info['human_following']
            result['following_rate'] = followed_step / iter_step
            result['following_step'] = followed_step
            result['total_step'] = iter_step
            result['collision'] = info['human_collision']
            with open(os.path.join(save_dir, "{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result, f, indent=2) 


class GTBBoxAgent(AgentConfig):
    def __init__(self, result_path):
        super().__init__()
        print("Initialize gtbbox agent")

        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        
        self.rgb_list = []
        self.rgb_box_list = []

        self.kp_t = 2
        self.kd_t = 0
        self.kp_f = 1
        self.kd_f = 0

        self.prev_error_t = 0
        self.prev_error_f = 0

        self.first_inside = True

        self.reset()

    def reset(self, episode: NavigationEpisode = None):
        if len(self.rgb_list) != 0:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
            imageio.mimsave(output_video_path, self.rgb_list)

            print(f"Successfully save the episode video with episode id {episode.episode_id}")

            self.rgb_list = []
        
        self.first_inside = True

    def act(self, observations, detector, episode_id, pil_img):
        self.episode_id = episode_id
        
        rgb = observations["agent_1_articulated_agent_jaw_rgb"]
        rgb_ = rgb[:, :, :3]
        image = np.asarray(rgb_[:, :, ::-1])
        height, width = image.shape[:2]
        
        if detector['agent_1_main_humanoid_detector_sensor']['facing']:
            box = detector['agent_1_main_humanoid_detector_sensor']['box']
            best_box =  np.array([(box[0]+box[2])/(2*width), (box[1]+box[3])/(2*height), (box[2]-box[0])/width, (box[3]-box[1])/height], dtype=np.float32)
            
            center_x = best_box[0]

            error_t = 0.5 - center_x
            error_f = (30000 - (box[2]-box[0])*(box[3]-box[1])) / 10000
            if abs(error_f) < 0.5:
                error_f = 0

            derivative_t = error_t - self.prev_error_t
            derivative_f = error_f - self.prev_error_f

            yaw_speed = self.kp_t * error_t + self.kd_t * derivative_t
            move_speed = self.kp_f * error_f + self.kd_f * derivative_f

            self.prev_error_t = error_t
            self.prev_error_f = error_f

            action = [move_speed, 0, yaw_speed]
        else:
            action = [0, 0, 0]
        
        self.last_action = action
        self.rgb_list.append(pil_img)

        return action
