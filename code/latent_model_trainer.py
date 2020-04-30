import os
from collections import deque
import numpy as np
import torch
import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from memory import LazyMemory
from network import LatentNetwork, GaussianPolicy
from utils import calc_kl_divergence, update_params, RunningMeanStats
from PIL import Image


class LatentTrainer:
    def __init__(self, env, log_dir, num_steps=3000000,
                 initial_latent_steps=100000, batch_size=256,
                 latent_batch_size=32, num_sequences=8, lr=0.0003,
                 latent_lr=0.0001, feature_dim=256, latent1_dim=32,
                 latent2_dim=256, hidden_units=[256, 256], memory_size=1e5,
                 gamma=0.99, target_update_interval=1, tau=0.005,
                 entropy_tuning=True, ent_coef=0.2, leaky_slope=0.2,
                 grad_clip=None, updates_per_step=1, start_steps=10000,
                 training_log_interval=10, learning_log_interval=100,
                 eval_interval=50000, cuda=True, seed=0, colab='save'):
        self.env = env
        self.observation_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.action_repeat = self.env.action_repeat

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.latent = LatentNetwork(
            self.observation_shape, self.action_shape, feature_dim,
            latent1_dim, latent2_dim, hidden_units, leaky_slope
        ).to(self.device)

        self.policy = GaussianPolicy(
            num_sequences * feature_dim
            + (num_sequences - 1) * self.action_shape[0],
            self.action_shape[0], hidden_units).to(self.device)

        # Policy is updated without the encoder.
        self.latent_optim = Adam(self.latent.parameters(), lr=latent_lr)
        self.memory = LazyMemory(
            memory_size, num_sequences, self.observation_shape,
            self.action_shape, self.device)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.images_dir = os.path.join(log_dir, 'images')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_rewards = RunningMeanStats(training_log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.initial_latent_steps = initial_latent_steps
        self.num_sequences = num_sequences
        self.num_steps = num_steps
        self.tau = tau
        self.batch_size = batch_size
        self.latent_batch_size = latent_batch_size
        self.start_steps = start_steps
        self.gamma = gamma
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.training_log_interval = training_log_interval
        self.learning_log_interval = learning_log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
        self.colab = colab

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and \
               self.steps >= self.start_steps * self.action_repeat

    def reset_deque(self, state):
        state_deque = deque(maxlen=self.num_sequences)
        action_deque = deque(maxlen=self.num_sequences - 1)

        for _ in range(self.num_sequences - 1):
            state_deque.append(
                np.zeros(self.observation_shape, dtype=np.uint8))
            action_deque.append(
                np.zeros(self.action_shape, dtype=np.uint8))
        state_deque.append(state)

        return state_deque, action_deque

    def deque_to_batch(self, state_deque, action_deque):
        # Convert deques to batched tensor.
        state = np.array(state_deque, dtype=np.uint8)
        state = torch.ByteTensor( \
            state).unsqueeze(0).to(self.device).float() / 255.0
        with torch.no_grad():
            feature = self.latent.encoder(state).view(1, -1)

        action = np.array(action_deque, dtype=np.float32)
        action = torch.FloatTensor(action).view(1, -1).to(self.device)
        feature_action = torch.cat([feature, action], dim=-1)
        return feature_action

    def explore(self, state_deque, action_deque):
        # Act with randomness
        feature_action = self.deque_to_batch(state_deque, action_deque)
        with torch.no_grad():
            action, _, _ = self.policy.sample(feature_action)
        return action.cpu().numpy().reshape(-1)

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()
        self.memory.set_initial_state(state)
        state_deque, action_deque = self.reset_deque(state)

        while not done:
            if self.steps >= self.start_steps * self.action_repeat:
                action = self.explore(state_deque, action_deque)
            else:
                action = 2 * np.random.rand(*self.action_shape) - 1

            next_state, reward, done, _ = self.env.step(action)
            self.steps += self.action_repeat
            episode_steps += self.action_repeat
            episode_reward += reward

            self.memory.append(action, reward, next_state, done)

            if self.is_update():
                # First, train the latent model only.
                if self.learning_steps < self.initial_latent_steps:
                    print('-' * 60)
                    print('Learning the latent model only...')
                    for _ in tqdm(range(self.initial_latent_steps)):
                        self.learning_steps += 1
                        self.learn_latent()
                    print('Finish learning the latent model.')
                    print('-' * 60)

            state_deque.append(next_state)
            action_deque.append(action)

        if self.episodes % self.training_log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_rewards.get(), self.steps)

        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  ')

    def learn_latent(self):
        images_seq, actions_seq, rewards_seq, dones_seq = \
            self.memory.sample_latent(self.latent_batch_size)
        latent_loss = self.calc_latent_loss(
            images_seq, actions_seq, rewards_seq, dones_seq)
        update_params(
            self.latent_optim, self.latent, latent_loss, self.grad_clip)

        if self.learning_steps % self.learning_log_interval == 0:
            self.writer.add_scalar(
                'loss/latent', latent_loss.detach().item(),
                self.learning_steps)

    def calc_latent_loss(self, images_seq, actions_seq, rewards_seq,
                         dones_seq):
        features_seq = self.latent.encoder(images_seq)

        # Sample from posterior dynamics.
        (latent1_post_samples, latent2_post_samples), \
        (latent1_post_dists, latent2_post_dists) = \
            self.latent.sample_posterior(features_seq, actions_seq)
        # Sample from prior dynamics.
        (latent1_pri_samples, latent2_pri_samples), \
        (latent1_pri_dists, latent2_pri_dists) = \
            self.latent.sample_prior(actions_seq)

        # KL divergence loss.
        kld_loss = calc_kl_divergence(latent1_post_dists, latent1_pri_dists)

        # Log likelihood loss of generated observations.
        images_seq_dists = self.latent.decoder(
            [latent1_post_samples, latent2_post_samples])
        log_likelihood_loss = images_seq_dists.log_prob(
            images_seq).mean(dim=0).sum()

        latent_loss = \
            kld_loss - log_likelihood_loss

        if self.learning_steps % self.learning_log_interval == 0:
            reconst_error = (
                    images_seq - images_seq_dists.loc
            ).pow(2).mean(dim=(0, 1)).sum().item()
            self.writer.add_scalar(
                'stats/reconst_error', reconst_error, self.learning_steps)

        if self.learning_steps % self.learning_log_interval == 0:
            gt_images = images_seq[0].detach().cpu()
            post_images = images_seq_dists.loc[0].detach().cpu()

            with torch.no_grad():
                pri_images = self.latent.decoder(
                    [latent1_pri_samples[:1], latent2_pri_samples[:1]]
                ).loc[0].detach().cpu()
                cond_pri_samples, _ = self.latent.sample_prior(
                    actions_seq[:1], features_seq[:1, 0])
                cond_pri_images = self.latent.decoder(
                    cond_pri_samples).loc[0].detach().cpu()

            images = torch.cat(
                [gt_images, post_images, cond_pri_images, pri_images],
                dim=-2)

            for idx, img in enumerate(gt_images):
                Image.fromarray((img * 255).numpy().astype(np.uint8).transpose([1, 2, 0])).save(
                    os.path.join(self.images_dir, 'gt_image%03i' % idx + '.png'))
                Image.fromarray((post_images[idx] * 255).numpy().astype(np.uint8).transpose([1, 2, 0])).save(
                    os.path.join(self.images_dir, 'post_images%03i' % idx + '.png'))
                Image.fromarray((cond_pri_images[idx] * 255).numpy().astype(np.uint8).transpose([1, 2, 0])).save(
                    os.path.join(self.images_dir, 'cond_pri_image%03i' % idx + '.png'))
                Image.fromarray((pri_images[idx] * 255).numpy().astype(np.uint8).transpose([1, 2, 0])).save(
                    os.path.join(self.images_dir, 'pri_images%03i' % idx + '.png'))

            # Visualize multiple of 8 images because each row contains 8
            # images at most.
            self.writer.add_images(
                'images/gt_posterior_cond-prior_prior',
                images[:(len(images) // 8) * 8], self.learning_steps)

        return latent_loss

    def save_models(self):
        self.latent.encoder.save(os.path.join(self.model_dir, 'encoder.pth'))
        self.latent.save(os.path.join(self.model_dir, 'latent.pth'))
        self.policy.save(os.path.join(self.model_dir, 'policy.pth'))

    def save_images(self, images):
        pass

    def __del__(self):
        self.writer.close()
        self.env.close()
