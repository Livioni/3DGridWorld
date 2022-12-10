import argparse,os,random,time,gym,torch
from distutils.util import strtobool
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=3406,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Huawei-mathematical-modeling",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="GridWorld-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1e5,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--max_step_per_episode", type=int, default=1000,
        help="the interval of recording the agent's performances")
    parser.add_argument("--total-episodes", type=int, default=10,
        help="total timesteps of the experiments")

    args = parser.parse_args()
    # fmt: on
    return args

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0))
        return -p_log_p.sum(-1)

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(9, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(9, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 6), std=0.01),
        )

        self.checkpoint_path_1 = "models/" + "critic_seed_" + str(args.seed) + ".pth"
        self.checkpoint_path_2 = "models/" + "actor_seed_" + str(args.seed) + ".pth"

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action

    def save(self):
        torch.save(self.critic.state_dict(), self.checkpoint_path_1)
        torch.save(self.actor.state_dict(), self.checkpoint_path_2)

    def load(self):
        self.critic.load_state_dict(torch.load(self.checkpoint_path_1, map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(torch.load(self.checkpoint_path_2, map_location=lambda storage, loc: storage))


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    agent = Agent(env).to(device)
    agent.load()
    for ep in range(args.total_episodes):
        print("episode: ",ep)
        sum_reward = 0
        observation = torch.Tensor(env.reset()).to(device)
        for i in range(args.max_step_per_episode):
            action = agent.get_action_and_value(observation)
            observation, reward, done, info = env.step(action.numpy())
            observation = torch.Tensor(observation).to(device)
            # env.my_render()
            sum_reward += reward
            if done:
                print(f"episode={ep}, episodic_return={info['episode']['r']}," f"episodic_length={info['episode']['l']}")    
                print('Sum reward = ',sum_reward)
                env.my_render()
                break
        print('crashed',env.crashed)

    env.close()