import argparse

# Hyperparameters:
def create_parser():
    parser = argparse.ArgumentParser(description='Modular Reinforcement Learning for Quadrotor UAV Control')
    parser.add_argument('--seed', default=789, type=int, metavar='N', help='Random seed of Gym, PyTorch and Numpy (default: 1234)') 
    parser.add_argument("--save_model", default=True, action="store_true", help='Save models and optimizer parameters (default: True)')
    parser.add_argument("--test_model", default=False, type=bool, help='Load and test trained models (default: False)')  
    parser.add_argument("--save_log", default=False, type=bool, help='Load trained models and save log(default: False)')      
    parser.add_argument('--render', default=False, type=bool, help='Simulation visualization (default: False)')

    # args of environment:
    parser.add_argument('--framework', default="CMP", help='Name of framework: NMP, DMP, or CMP ')
    parser.add_argument('--max_steps', default=4000, type=int, help='Maximum number of steps in each episode (default: 2000)')
    parser.add_argument('--max_timesteps', default=int(3e6), type=int, help='Number of total timesteps (default: 7e6)')
    parser.add_argument("--num_eval", type=float, default=10, help="Number of episodes to evaluate our trained model")
    parser.add_argument("--eval_freq", default=2e3, type=int, help='How often (time steps) evaluate our trained model (default: 1e4)')       
    parser.add_argument('--eval_max_steps', default=5, type=int, help='[sec] Maximum number of steps in each episode for evaluation (default: 5)')
    # Coefficients in reward function:
    # Agent1's reward:
    parser.add_argument('--Cx', default=6.0, type=float, metavar='G', help='Position coeff. (default: )')
    parser.add_argument('--CIx', default=0.1, type=float, metavar='G', help='Position integral coeff. (default: )')
    parser.add_argument('--Cv', default=0.4, type=float, metavar='G', help='Velocity coeff. (default: )')
    parser.add_argument('--Cw12', default=0.6, type=float, metavar='G', help='Angular velocity coeff. (default: )')
    parser.add_argument('--alpha', default=0.01, type=float, metavar='G', help='Position integral terms. (default: )')
    # Agent2's reward:
    parser.add_argument('--Cb1', default=6.0, type=float, metavar='G', help='Heading coeff. (default: )')
    parser.add_argument('--CIb1', default=0.1, type=float, metavar='G', help='Heading integral coeff. (default: )')
    parser.add_argument('--CW3', default=0.1, type=float, metavar='G', help='Angular velocity coeff. (default: )')
    parser.add_argument('--beta', default=0.05, type=float, metavar='G', help='Heading integral terms. (default: )')
    # Domain randomization: 
    parser.add_argument("--use_UDM", default=True, type=bool, help="Uniform domain randomization for sim-to-real")
    parser.add_argument("--UDM_percentage", default=10, type=float, help="± randomness 0 ~ 100[%]")

    # args of agents:
    parser.add_argument("--start_timesteps", default=int(5e5), type=int, help='Number of steps for uniform-random action selection (default: int(5e5))')
    parser.add_argument("--actor_hidden_dim", default=[16, 8], type=int, help='Number of nodes in hidden layers of actor net (default: 64)')
    parser.add_argument("--critic_hidden_dim", default=512, type=int, help='Number of nodes in hidden layers of critic net (default: 256)')
    parser.add_argument("--lr_a", default=[3e-4, 1e-4], type=float, help="Learning rate of actor, alpha (default: 1e-5)")
    parser.add_argument("--lr_c", default=[2e-4, 2e-4], type=float, help="Learning rate of critic, alpha (default: 1e-5)")
    parser.add_argument('--discount', default=0.99, type=float, metavar='G', help='discount factor, discount (default: 0.99)')
    parser.add_argument("--max_action", default=1., type=float, help="Max action")
    parser.add_argument("--use_equiv", default=False, type=bool, help="Train models with equivariant reinforcement learning")
    parser.add_argument("--use_clip_grad_norm", default=True, type=bool, help="Clips gradient norm of parameters")
    parser.add_argument("--grad_max_norm", default=100., type=float, help="max norm of the gradients")
    parser.add_argument("--use_explor_noise_decay", default=True, type=bool, help="Whether to decay the explor_noise_std")
    parser.add_argument("--explor_noise_std_init", default=0.3, type=float, help="Stddev of Gaussian noise for exploration")
    parser.add_argument("--explor_noise_std_min", default=0.05, type=float, help="Stddev of Gaussian noise for exploration")
    parser.add_argument("--explor_noise_decay_steps", default=5e6, type=float, help="How many steps before the explor_noise_std decays to the minimum")
    parser.add_argument('--tau', default=0.005, type=float, metavar='G', help='Target network update rate (default: 0.005)')
    parser.add_argument("--target_noise", default=0.2, type=float, help='Stddev for smoothing noise added to target policy (default: 0.2)')
    parser.add_argument("--noise_clip", default=0.5, type=float, help='Clipping range of target policy smoothing noise (default: 0.5)')
    parser.add_argument('--policy_update_freq', default=3, type=int, metavar='N', help='Frequency of “Delayed” policy updates (default: 2)')
    # Regularizing action policies for smooth control:
    parser.add_argument('--lam_T', default=0.4, type=int, metavar='N', help='Temporal Smoothness (default: 0.5~0.8)')
    parser.add_argument('--lam_S', default=0.3, type=int, metavar='N', help='Spatial Smoothness (default: 0.3~0.5)')
    parser.add_argument('--lam_M', default=0.6, type=int, metavar='N', help='Magnitude Smoothness (default: 0.2~0.5)')

    # args of replay buffer:
    parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='Batch size of actor and critic networks (default: 256)')
    parser.add_argument('--replay_buffer_size', default=int(1e6), type=int, metavar='N', help='Maximum size of replay buffer (default: 1e6)')

    return parser