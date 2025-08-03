import gymnasium as gym

class QuadrupedAgent(object):
    def __init__(self):
        self.env = gym.make(
            "Ant-v5",
            xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
            forward_reward_weight=1,  # kept the same as the 'Ant' environment
            ctrl_cost_weight=0.05,  # changed because of the stronger motors of `Go1`
            contact_cost_weight=5e-4,  # kept the same as the 'Ant' environment
            healthy_reward=1,  # kept the same as the 'Ant' environment
            main_body=1,  # represents the "trunk" of the `Go1` robot
            healthy_z_range=(0.195, 0.75),
            include_cfrc_ext_in_observation=True,
            exclude_current_positions_from_observation=False,
            reset_noise_scale=0.1,
            frame_skip=25,
            max_episode_steps=1000,
            render_mode='human',
        )

    def get_action(self, obs):

