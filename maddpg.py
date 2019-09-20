import numpy as np
import random
import copy
from collections import namedtuple, deque
from modelLocal import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
GAMMA = 0.99            # discount factor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, GAMMA=GAMMA, TAU=TAU,seed=0):
        super(MADDPG, self).__init__()
        self.agents = [Agent(24,2,48,4,seed), Agent(24,2,48,4,seed)]
        self.GAMMA = GAMMA
        self.tau = TAU

    def target_act(self, obs_all_agents):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [agent.actor_target(obs) for agent, obs in zip(self.agents, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """
        states, actions, rewards, next_states, dones = samples
        
        #THISLOOKSOK
        obs_full = states.view(-1,48).to(device)
        actions_full = actions.view(-1,4).to(device)
        rewards = torch.transpose(rewards,0,1).to(device)
        next_obs_full = next_states.view(-1,48).to(device)
        dones = torch.transpose(dones,0,1).to(device)
        states = torch.transpose(states, 0, 1).to(device)
        next_states = torch.transpose(next_states, 0, 1).to(device)        

        agent = self.agents[agent_number]
        
        # CRITIC LOSS
        target_actions = self.target_act(next_states)        
        target_actions = torch.cat(target_actions, dim=1)
        with torch.no_grad():
            q_targets_next = agent.critic_target(next_obs_full,target_actions)
        q_targets = rewards[agent_number].view(-1, 1) + \
                    self.GAMMA * q_targets_next * (1 - dones[agent_number].view(-1, 1))

        q_expected = agent.critic_local(obs_full, actions_full)
        critic_loss = F.mse_loss(q_expected, q_targets.detach())        
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
#         torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()
        
        # ACTOR LOSS
        actions_pred = [ self.agents[i].actor_local(ob) if i == agent_number \
                   else self.agents[i].actor_local(ob).detach()
                   for i, ob in enumerate(states) ]
                
        actions_pred = torch.cat(actions_pred, dim=1)
        actor_loss = -agent.critic_local(obs_full, actions_pred).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
#         torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()


    def update_targets(self,agent_number):
        """soft update targets"""
        agent = self.agents[agent_number]
        self.soft_update(agent.actor_target, agent.actor_local, self.tau)
        self.soft_update(agent.critic_target, agent.critic_local, self.tau)
       
    def soft_update(self,target, source, tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for update
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, local_state_size, action_size, global_state_size, global_action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            local_state_size (int): dimension of each local state
            action_size (int): dimension of each action
            global_state_size (int): dimension of each global state
            global_action_size (int): dimension of each global action
            seed (int): random seed
        """
        self.local_state_size = local_state_size
        self.action_size = action_size
        self.global_state_size = global_state_size
        self.global_action_size = global_action_size
        self.seed = random.seed(seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(local_state_size, action_size, seed).to(device)
        self.actor_target = Actor(local_state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(global_state_size, global_action_size, seed).to(device)
        self.critic_target = Critic(global_state_size, global_action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, seed)
        
        # Make sure the Actor Target Network has the same weight values as the Local Network
        for target, local in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target.data.copy_(local.data)
            
        # Make sure the Critic Target Network has the same weight values as the Local Network
        for target, local in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target.data.copy_(local.data)

    def act(self, state, noisefactor, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += noisefactor*self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.tensor([e.states for e in experiences if e is not None]).float()
        actions = torch.tensor([e.actions for e in experiences if e is not None]).float()
        rewards = torch.tensor([e.rewards for e in experiences if e is not None]).float()
        next_states = torch.tensor([e.next_states for e in experiences if e is not None]).float()
        dones = torch.tensor([e.dones for e in experiences if e is not None]).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)