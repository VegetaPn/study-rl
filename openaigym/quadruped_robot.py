import time

import numpy as np

import gymnasium as gym


# We want to tweak the `frame_skip` parameter to get `dt` to an acceptable value
# (typical values are `dt` ∈ [0.01, 0.1] seconds),

# Reminder: dt = frame_skip × model.opt.timestep, where `model.opt.timestep` is the integrator
# time step selected in the MJCF model file.

# The `Go1` model we are using has an integrator timestep of `0.002`, so by selecting
# `frame_skip=25` we can set the value of `dt` to `0.05s`.

# To avoid overfitting the policy, `reset_noise_scale` should be set to a value appropriate
# to the size of the robot, we want the value to be as large as possible without the initial
# distribution of states being invalid (`Terminal` regardless of control actions),
# for `Go1` we choose a value of `0.1`.

# And `max_episode_steps` determines the number of steps per episode before `truncation`,
# here we set it to 1000 to be consistent with the based `Gymnasium/MuJoCo` environments,
# but if you need something higher you can set it so.

# The arguments of interest are `terminate_when_unhealthy` and `healthy_z_range`.

# We want to set `healthy_z_range` to terminate the environment when the robot falls over,
# or jumps really high, here we have to choose a value that is logical for the height of the robot,
# for `Go1` we choose `(0.195, 0.75)`.
# Note: `healthy_z_range` checks the absolute value of the height of the robot,
# so if your scene contains different levels of elevation it should be set to `(-np.inf, np.inf)`

# We could also set `terminate_when_unhealthy=False` to disable termination altogether,
# which is not desirable in the case of `Go1`.

# For the arguments `forward_reward_weight`, `ctrl_cost_weight`, `contact_cost_weight` and `healthy_reward`
# we have to pick values that make sense for our robot, you can use the default `MuJoCo/Ant`
# parameters for references and tweak them if a change is needed for your environment.
# In the case of `Go1` we only change the `ctrl_cost_weight` since it has a higher actuator force range.

# For the argument `main_body` we have to choose which body part is the main body
# (usually called something like "torso" or "trunk" in the model file) for the calculation
# of the `forward_reward`, in the case of `Go1` it is the `"trunk"`
# (Note: in most cases including this one, it can be left at the default value).

def main():
    """Run the final Go1 environment setup."""
    env = gym.make(
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

    # Note: If you need a different reward function, you can write your own `RewardWrapper`
    # (see the documentation).

    # Note: If you need a different termination condition, you can write your own `TerminationWrapper`
    # (see the documentation).

    # Example of running the environment for a few steps
    obs, info = env.reset()

    for i in range(100):
        action = env.action_space.sample()  # Replace with your agent's action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step {i}, {action=}, {obs=}, {reward=}, {terminated=}, {truncated=}")
        time.sleep(0.05)

        if terminated or truncated:
            obs, info = env.reset()

    env.render()
    env.close()
    print("Environment tested successfully!")

    # Now you would typically:
    # 1. Set up your RL algorithm
    # 2. Train the agent
    # 3. Evaluate the agent's performance


if __name__ == "__main__":
    main()
