# 文件路径: agent/sac_revkl_npg.py
import numpy as np
import torch
import torch.nn.functional as F
import utils
import hydra

from agent import Agent


def compute_state_entropy(obs, full_obs, k):
    batch_size = 500
    with torch.no_grad():
        dists = []
        for idx in range(len(full_obs) // batch_size + 1):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            dist = torch.norm(obs[:, None, :] - full_obs[None, start:end, :], dim=-1, p=2)
            dists.append(dist)

        dists = torch.cat(dists, dim=1)
        knn_dists = torch.kthvalue(dists, k=k + 1, dim=1).values
        state_entropy = knn_dists
    return state_entropy.unsqueeze(1)


# -----------------------------
# Helpers for NPG
# -----------------------------
def _flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def _set_flat_params(model: torch.nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view_as(p))
        idx += n


def _flat_grad_from_params(params) -> torch.Tensor:
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)


def _conjugate_gradient(Avp, b, cg_iters=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(cg_iters):
        Avp_p = Avp(p)
        alpha = rdotr / (torch.dot(p, Avp_p) + 1e-8)
        x += alpha * p
        r -= alpha * Avp_p
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x


def _kl_diag_normal(mu0, std0, mu1, std1):
    """
    KL( N(mu0,std0) || N(mu1,std1) ), summed over action dims, returns [B,1].
    IMPORTANT: For SquashedNormal (tanh transform), KL is invariant under the same bijective transform,
    so we can compute KL on the base Normal safely.
    """
    var0 = std0.pow(2)
    var1 = std1.pow(2)
    kl = torch.log(std1 / std0) + (var0 + (mu0 - mu1).pow(2)) / (2.0 * var1) - 0.5
    return kl.sum(-1, keepdim=True)


class SACRevKL_NPG_Agent(Agent):
    """
    SAC critic + (optional) RevKL regularization,
    but actor update uses NPG (CG on Fisher) + fixed trust region scaling (max_kl).
    No line search. Optional single-step KL backoff for safety.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        critic_cfg,
        actor_cfg,
        discount,
        init_temperature,
        alpha_lr,
        alpha_betas,
        actor_lr,
        actor_betas,
        actor_update_frequency,
        critic_lr,
        critic_betas,
        critic_tau,
        critic_target_update_frequency,
        batch_size,
        learnable_temperature,
        normalize_state_entropy=True,
        revkl_gamma=0.0,
        # ---- NPG params ----
        max_kl=0.01,
        cg_iters=10,
        damping=0.1,
        # sampling strategy (recommended)
        use_recent_batch=True,
        recent_batch_size=4096,
        # safety valve (NOT line search)
        enable_kl_backoff=True,
        kl_backoff=0.5,
        kl_max_factor=2.0,
    ):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        # Keep configs for reset
        self.critic_cfg = critic_cfg
        self.actor_cfg = actor_cfg

        # store lrs/betas for reset
        self.critic_lr = critic_lr
        self.critic_betas = critic_betas
        self.actor_lr = actor_lr
        self.actor_betas = actor_betas
        self.alpha_lr = alpha_lr
        self.alpha_betas = alpha_betas

        self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=device)
        self.normalize_state_entropy = normalize_state_entropy

        self.init_temperature = init_temperature
        self.step = 0
        self.revkl_gamma = revkl_gamma

        # NPG params
        self.max_kl = float(max_kl)
        self.cg_iters = int(cg_iters)
        self.damping = float(damping)
        self.use_recent_batch = bool(use_recent_batch)
        self.recent_batch_size = int(recent_batch_size)

        self.enable_kl_backoff = bool(enable_kl_backoff)
        self.kl_backoff = float(kl_backoff)
        self.kl_max_factor = float(kl_max_factor)

        # Inject dims before instantiate
        actor_cfg.params.obs_dim = obs_dim
        actor_cfg.params.action_dim = action_dim
        critic_cfg.params.obs_dim = obs_dim
        critic_cfg.params.action_dim = action_dim

        # Networks
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        # Reference actor for RevKL
        self.actor_ref = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.actor_ref.load_state_dict(self.actor.state_dict())
        for p in self.actor_ref.parameters():
            p.requires_grad = False

        # Temperature
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim

        # Optimizers: critic + alpha only (actor uses NPG step)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=critic_betas)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=alpha_betas)

        self.train()
        self.critic_target.train()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def set_reference_policy_to_current(self):
        print(f"--- Step {self.step}: Updating reference policy (actor_ref). ---")
        self.actor_ref.load_state_dict(self.actor.state_dict())

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    # -----------------------------
    # Critic updates (unchanged SAC)
    # -----------------------------
    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step, print_flag=True):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_critic_state_ent(self, obs, full_obs, action, next_obs, not_done, logger, step, K=5, print_flag=True):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob

        state_entropy = compute_state_entropy(obs, full_obs, k=K)

        self.s_ent_stats.update(state_entropy)
        norm_state_entropy = state_entropy / self.s_ent_stats.std
        if self.normalize_state_entropy:
            state_entropy = norm_state_entropy

        if print_flag:
            logger.log("train_critic/entropy", state_entropy.mean(), step)

        target_Q = state_entropy + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    # -----------------------------
    # Actor update: NPG + TR scaling
    # -----------------------------
    def update_actor_and_alpha(self, obs, logger, step, print_flag=False):
        """
        Objective = SAC actor objective (+ optional RevKL(ref||current)).
        Update method = NPG (CG) + fixed trust region scaling (max_kl), no line search.
        """

        # Build loss (same as SAC)
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # RevKL(ref || current) on base normals (stable)
        if self.revkl_gamma > 0:
            with torch.no_grad():
                dist_ref = self.actor_ref(obs)
            mu_ref, std_ref = dist_ref.base_dist.loc, dist_ref.base_dist.scale
            mu_cur, std_cur = dist.base_dist.loc, dist.base_dist.scale
            kl_ref = _kl_diag_normal(mu_ref, std_ref, mu_cur, std_cur)  # KL(ref || cur)
            actor_loss = actor_loss + (self.revkl_gamma * kl_ref).mean()
            if print_flag:
                logger.log('train_actor/kl_div_ref', kl_ref.mean(), step)

        # Gradient of loss
        self.actor.zero_grad()
        actor_loss.backward()
        params = list(self.actor.parameters())
        g = _flat_grad_from_params(params).detach()

        # Save old base normal params for KL metric
        with torch.no_grad():
            dist_old = self.actor(obs)
            mu_old = dist_old.base_dist.loc.detach()
            std_old = dist_old.base_dist.scale.detach()

        # Fisher-vector product via KL Hessian
        def Fvp(v):
            dist_new = self.actor(obs)
            mu_new = dist_new.base_dist.loc
            std_new = dist_new.base_dist.scale
            kl = _kl_diag_normal(mu_old, std_old, mu_new, std_new).mean()
            grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
            flat_kl_grad = torch.cat([gg.view(-1) for gg in grads])
            kl_v = (flat_kl_grad * v).sum()
            grads2 = torch.autograd.grad(kl_v, self.actor.parameters())
            flat_hvp = torch.cat([gg.contiguous().view(-1) for gg in grads2]).detach()
            return flat_hvp + self.damping * v

        # Solve F x = g  => step_dir = -x
        with torch.enable_grad():
            x = _conjugate_gradient(Fvp, g, cg_iters=self.cg_iters)

        step_dir = -x

        # Scale step to satisfy quadratic KL approx
        F_step = Fvp(step_dir)
        sAs = torch.dot(step_dir, F_step)

        if (sAs <= 0) or torch.isnan(sAs) or torch.isinf(sAs):
            if print_flag:
                logger.log('train_actor/npg_bad_sAs', sAs, step)
            # safest behavior: skip actor update this iteration
            return

        scale = torch.sqrt((2.0 * self.max_kl) / (sAs + 1e-8))
        old_theta = _flat_params(self.actor)
        new_theta = old_theta + scale * step_dir
        _set_flat_params(self.actor, new_theta)

        # Real KL check (single backoff, NOT line search)
        with torch.no_grad():
            dist_post = self.actor(obs)
            mu_post = dist_post.base_dist.loc
            std_post = dist_post.base_dist.scale
            kl_real = _kl_diag_normal(mu_old, std_old, mu_post, std_post).mean()

        if self.enable_kl_backoff and (kl_real > self.kl_max_factor * self.max_kl):
            new_theta = old_theta + (scale * self.kl_backoff) * step_dir
            _set_flat_params(self.actor, new_theta)
            with torch.no_grad():
                dist_post2 = self.actor(obs)
                mu_post2 = dist_post2.base_dist.loc
                std_post2 = dist_post2.base_dist.scale
                kl_real = _kl_diag_normal(mu_old, std_old, mu_post2, std_post2).mean()

        if print_flag:
            logger.log('train_actor/loss', actor_loss, step)
            logger.log('train_actor/entropy', -log_prob.mean(), step)
            logger.log('train_actor/npg_scale', scale, step)
            logger.log('train_actor/kl_real', kl_real, step)

        self.actor.log(logger, step)

        # Alpha update: keep identical to SAC (optional to freeze by learnable_temperature=false)
        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            if print_flag:
                logger.log('train_alpha/loss', alpha_loss, step)
                logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    # -----------------------------
    # Public API used by Workspace
    # -----------------------------
    def reset_critic(self):
        self.critic = hydra.utils.instantiate(self.critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(self.critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, betas=self.critic_betas)

    def update(self, replay_buffer, logger, step, gradient_update=1):
        for index in range(gradient_update):
            # Prefer recent batch for NPG stability
            if self.use_recent_batch and hasattr(replay_buffer, "sample_recent"):
                obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample_recent(
                    self.recent_batch_size
                )
            else:
                obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)

            print_flag = (index == gradient_update - 1)
            if print_flag:
                logger.log('train/batch_reward', reward.mean(), step)

            self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step, print_flag)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def update_after_reset(self, replay_buffer, logger, step, gradient_update=1, policy_update=True):
        for index in range(gradient_update):
            if self.use_recent_batch and hasattr(replay_buffer, "sample_recent"):
                obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample_recent(
                    self.recent_batch_size
                )
            else:
                obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)

            print_flag = (index == gradient_update - 1)
            if print_flag:
                logger.log('train/batch_reward', reward.mean(), step)

            self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step, print_flag)

            if (index % self.actor_update_frequency == 0) and policy_update:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

            if index % self.critic_target_update_frequency == 0:
                utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def update_state_ent(self, replay_buffer, logger, step, gradient_update=1, K=5):
        for index in range(gradient_update):
            obs, full_obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample_state_ent(
                self.batch_size
            )

            print_flag = (index == gradient_update - 1)
            if print_flag:
                logger.log('train/batch_reward', reward.mean(), step)

            self.update_critic_state_ent(obs, full_obs, action, next_obs, not_done_no_max, logger, step, K=K,
                                         print_flag=print_flag)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), f'{model_dir}/actor_{step}.pt')
        torch.save(self.critic.state_dict(), f'{model_dir}/critic_{step}.pt')
        torch.save(self.critic_target.state_dict(), f'{model_dir}/critic_target_{step}.pt')

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load(f'{model_dir}/actor_{step}.pt', map_location=torch.device('cpu')))
        self.critic.load_state_dict(torch.load(f'{model_dir}/critic_{step}.pt', map_location=torch.device('cpu')))
        self.critic_target.load_state_dict(torch.load(f'{model_dir}/critic_target_{step}.pt', map_location=torch.device('cpu')))
