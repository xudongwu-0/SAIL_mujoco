# rollout_buffer.py
import torch

class RolloutBuffer:
    """
    Minimal on-policy trajectory buffer for TRPO.
    Stores one episode (or one rollout segment) and then yields tensors.
    """
    def __init__(self, obs_dim, action_dim, device):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.obs = []
        self.act = []
        self.rew = []
        self.done = []

    def add(self, obs, act, rew, done):
        self.obs.append(torch.as_tensor(obs, dtype=torch.float32))
        self.act.append(torch.as_tensor(act, dtype=torch.float32))
        self.rew.append(torch.as_tensor([rew], dtype=torch.float32))
        self.done.append(torch.as_tensor([done], dtype=torch.float32))

    def get(self):
        return {
            "obs": torch.stack(self.obs).to(self.device),
            "act": torch.stack(self.act).to(self.device),
            "rew": torch.stack(self.rew).to(self.device),
            "done": torch.stack(self.done).to(self.device),
        }
