import os
import numpy as np
import torch
import datetime
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from rltorch.memory import MultiStepMemory

from model import TwinnedQNetwork, GaussianPolicy
from model_safe import QLearningAgent
from utils import grad_false, hard_update, soft_update, to_batch,\
    update_params, RunningMeanStats
from log import CheckpointableData, Log, TabularLog

from refitepsilon import RefitEpsilon


class VESQRL:

    def __init__(self, env, env1, env_name, log_dir, num_steps=1000, pretrain_steps=100,
                 batch_size=256, lr=0.0003, hidden_units=[256, 256], memory_size=1e6,
                 gamma=0.99, tau=0.005, entropy_tuning=True, ent_coef=1,
                 multi_step=1, grad_clip=None, updates_per_step=1,
                 start_steps=1000, log_interval=10, target_update_interval=1,
                 eval_interval=1000, cuda=True, seed=0, min_epsilon=0.1, train_steps=500000):
        
        
        self.env = env # Env for pretraining
        self.env1 = env1 # Env for training

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        torch.backends.cudnn.deterministic = True  # It harms a performance.
        torch.backends.cudnn.benchmark = False
        
        

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.policy = GaussianPolicy(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)
        self.critic = TwinnedQNetwork(
            self.env.observation_space.shape[0],self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)
        self.critic_target = TwinnedQNetwork(
            self.env.observation_space.shape[0],self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device).eval()
        
        self.qsafe = QLearningAgent(
            self.env.observation_space.shape[0]+self.env.action_space.shape[0],
            1,
            hidden_units=hidden_units,
            learning_rate=0.7,
            gamma=0.9,
            device=self.device,
            )
        
        self.emodel = None
        

        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)

        if entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(torch.Tensor(
                self.env.action_space.shape).to(self.device)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)

        # Safety coefficient
        self.nu = torch.zeros(1, requires_grad=True, device=self.device)
        self.nu_optim = Adam([self.nu], lr=lr)
        
        self.memory = MultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step)

        #Doffline
        self.offline_memory = MultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step)

        #Dsafe
        self.safe_memory = MultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step)
        

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.log_dir = os.path.join(log_dir, 'logs')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.data = CheckpointableData()
        self.log = Log(env_name, self.log_dir)
        self.offline_log = TabularLog(self.log_dir, 'offline.csv')
        self.online_log = TabularLog(self.log_dir, 'online.csv')
        self.training_log = TabularLog(self.log_dir, 'training.csv')

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_rewards = RunningMeanStats(log_interval)

        self.steps = 0
        self.off_steps = 0
        self.on_steps = 0
        self.pretrain_steps = pretrain_steps
        self.learning_steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.max_episode_steps = 1000
        self.tau = tau
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
        self.min_epsilon = min_epsilon
        self.train_steps = train_steps
        self.train_steps = 1000000
        self.violations = 0
        self.episodes_for_epsilon = []

    def log_offline(self, row):
        for k, v in row.items():
            self.data.append(k, v, verbose=True)
        self.offline_log.row(row)

    def log_online(self, row):
        for k, v in row.items():
            self.data.append(k, v, verbose=True)
        self.online_log.row(row)

    def log_training(self, row):
        for k, v in row.items():
            self.data.append(k, v, verbose=True)
        self.training_log.row(row)

    def numpyify(self,x):
        if isinstance(x, np.ndarray):
            return x
        elif torch.is_tensor(x):
            return x.cpu().numpy()
        else:
            return np.array(x)

    def pretrain(self):
        for i in range(self.pretrain_steps):
            print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Starting pretraining step {i+1:<4}\n')
            self.log.write(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Starting pretraining step {i+1:<4}\n')
            torch.cuda.empty_cache()
            self.violations = 0
            self.episodes = 0
            self.steps = 0
            while self.episodes < self.num_steps:
                self.offpolicy_pretrain()

            # Unprocessed epsiodes left after offline pretraining
            if len(self.episodes_for_epsilon) > 0:
                self.update_epsilons(self.episodes_for_epsilon)

            self.save_models()

            self.episodes=0
            self.violations=0
            self.steps = 0
            while self.episodes < self.num_steps:
                self.onpolicy_rollouts()

    def train(self):
        self.violations = 0
        self.steps = 0
        self.episodes = 0
        policy_loss = None
        entropy_loss = None
        safety_loss = None
        while self.steps < self.train_steps:
            print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Starting training episode {self.episodes+1:<4}\n')
            self.log.write(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Starting training episode {self.episodes+1:<4}\n')
            episode_reward = 0.
            episode_steps = 0
            violations = 0
            self.episodes += 1
    
            state = self.env1.reset()
            for step in range(self.max_episode_steps):
                action = self.act_safe(state)

                # Reached a very dangerous state
                if action is None:
                    self.violations += 1
                    violations +=1
                    next_state = self.env1.reset()
                else:
                    self.steps += 1
                    next_state, reward, done, info = self.env1.step(action)
                    violation = info['violation']
                    self.steps += 1
                    episode_steps += 1
                    episode_reward += reward
                    self.offline_memory.append(state, action, reward, next_state, violation,
                        episode_done=violation)

                    policy_loss, entropy_loss, safety_loss = self.training_learn(state,action)
                    if violation:
                        self.violations += 1
                        violations +=1
                        next_state = self.env1.reset()
                    
                    if done:
                        break

                    if self.steps >= self.train_steps:
                        break

            reward_per_step = episode_reward/episode_steps if episode_steps > 0 else 0

            self.log.write(f'Episode Steps:{episode_steps:<4}\tTotal Steps:{self.steps:<4}\n'
                           f'Episode Reward:{episode_reward:<5.1f}\tReward per step:{reward_per_step:<5.1f}\n'
                           f'Episode Violation:{bool(violations)}\tTotal Violations:{self.violations:<4}\n')
        
            print(f'Episode Steps:{episode_steps:<4}\tTotal Steps:{self.steps:<4}\n'
                f'Episode Reward:{episode_reward:<5.1f}\tReward per step:{reward_per_step:<5.1f}\n'
                f'Episode Violation:{bool(violations)}\tTotal Violations:{self.violations:<4}\n')
            
            self.log.write(f'Policy Loss: {policy_loss:<4}\n'
              f'Entropy Coefficient Loss: {entropy_loss:<4}\n'
              f'Safety Critic Loss: {safety_loss:<5.5f}\n')

            print(f'Policy Loss: {policy_loss:<4}\n'
              f'Entropy Coefficient Loss: {entropy_loss:<4}\n'
              f'Safety Critic Loss: {safety_loss:<5.5f}\n')
            
            self.log_training({
                'Episodes': self.episodes+1,
                'Total Violations': self.violations,
                'Total Steps': self.steps,
                'Reward': episode_reward,
                'Episode Steps': episode_steps,
                'Episode Violations': violations,
                'Policy Loss': policy_loss,
                'Entropy Loss': entropy_loss,
                'Safety Critic Loss': safety_loss,
                **self.evaluate()
            })


    def is_update(self):
        return len(self.memory) > self.batch_size and\
            self.steps >= self.start_steps

    def act(self, state):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        # act with randomness
        state = torch.unsqueeze(state, dim=0)
        state = state.to(self.device)

        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        # act without randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, action = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q

    def offpolicy_pretrain(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        episode_violations = 0
        done = False
        state = self.env.reset()

        E = []
        
        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Starting offline pretraining episode {self.episodes:<4}\n')
        self.log.write(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Starting offline pretraining episode {self.episodes:<4}\n')
        while not done:
            action = self.act(state)

            next_state, reward, done, info = self.env.step(action)


            violation = info['violation']
            self.steps += 1
            episode_steps += 1
            episode_reward += reward
           
            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            # We need to give true done signal with addition to masked done
            # signal to calculate multi-step rewards.
            self.memory.append(
                    state, action, reward, next_state, masked_done,
                    episode_done=done)
                
            self.offline_memory.append(
                    state, action, reward, next_state, violation,
                    episode_done=violation)

            #save E
            E.append((state, action, reward, next_state, done, violation))

            #applying SAC update
            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            if self.steps % self.eval_interval == 0:
                self.save_models()

            #line 13
            if violation:
                self.violations += 1
                episode_violations += 1
                # Take random state
                next_state = self.env.reset()
                # Include only those episodes that end in an unsafe state
                self.episodes_for_epsilon.append(E)
                E = []

            if episode_steps >= self.max_episode_steps:
                break

            state = next_state

        #update qsafe network
        if self.safe_memory.__len__() > 0:
            self.learn_safe()

        # We log running mean of training rewards.
        self.train_rewards.append(episode_reward)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_rewards.get(), self.steps)
            
        reward_per_step = episode_reward/episode_steps if episode_steps > 0 else 0

        self.log.write(f'Episode Steps:{episode_steps:<4}\tTotal Steps:{self.steps:<4}\n'
              f'Episode Reward:{episode_reward:<5.1f}\tReward per step:{reward_per_step:<5.1f}\n'
              f'Episode Violation:{bool(episode_violations)}\tTotal Violations:{self.violations:<4}\n')
        
        print(f'Episode Steps:{episode_steps:<4}\tTotal Steps:{self.steps:<4}\n'
              f'Episode Reward:{episode_reward:<5.1f}\tReward per step:{reward_per_step:<5.1f}\n'
              f'Episode Violation:{bool(episode_violations)}\tTotal Violations:{self.violations:<4}\n')
        
        self.log_offline({
                'Episodes': self.episodes,
                'Total Violations': self.violations,
                'Total Steps': self.steps,
                'Reward': episode_reward,
                'Episode Steps': episode_steps,
                'Episode Violations': episode_violations
            })
        
        if len(self.episodes_for_epsilon)>=10:
            self.update_epsilons(self.episodes_for_epsilon)
            self.episodes_for_epsilon=[]
        


    def act_safe(self, state):
        if self.safe_memory.__len__() == 0:
            epsilon = self.min_epsilon
        else:
            epsilon = self.emodel.predict(np.array([state.tolist()]))
        action = self.qsafe.select_action(state,epsilon, self.policy)
        return action

    def onpolicy_rollouts(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()
        violations = 0

        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Starting online pretraining episode {self.episodes:<4}\n')
        self.log.write(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Starting online pretraining episode {self.episodes:<4}\n')

        while not done:
            action = self.act_safe(state)
            if action is None:
                self.violations += 1
                violations += 1
                break

            next_state, reward, done, info = self.env.step(action)
            violation = info['violation']
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            # We need to give true done signal with addition to masked done
            # signal to calculate multi-step rewards.
            self.memory.append(
                    state, action, reward, next_state, masked_done,
                    episode_done=done)
                
            self.safe_memory.append(
                    state, action, reward, next_state, violation,
                    episode_done=violation)

            if self.steps % self.eval_interval == 0:
                self.save_models()

            #line 13
            if violation:
                self.violations += 1
                violations += 1
                #take random state
                next_state = self.env.reset()
            
            if episode_steps >= self.max_episode_steps:
                break

            state = next_state

        # We look at running mean of training rewards.
        self.train_rewards.append(episode_reward)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_rewards.get(), self.steps)
        reward_per_step = episode_reward/episode_steps if episode_steps > 0 else 0
        self.log.write(f'Episode Steps:{episode_steps:<4d}\tTotal Steps:{self.steps:<4d}\n'
                       f'Episode Reward:{episode_reward:<5.1f}\tReward per step:{reward_per_step:<5.1f}\n'
                       f'Episode Violation:{bool(violations)}\tTotal Violations:{self.violations:<4}\n')
        
        print(f'Episode Steps:{episode_steps:<4}\tTotal Steps:{self.steps:<4}\n'
              f'Episode Reward:{episode_reward:<5.1f}\tReward per step:{reward_per_step:<5.1f}\n'
              f'Episode Violation:{bool(violations)}\tTotal Violations:{self.violations:<4}\n')
        


        self.log_online({
                'Episodes': self.episodes,
                'Total Violations': self.violations,
                'Total Steps': self.steps,
                'Reward': episode_reward,
                'Episode Steps': episode_steps,
                'Episode Violations': violations
            })

    def update_epsilons(self,episodes):
        print('Updating epsilons...')
        totalstates = np.empty((0,len(episodes[0][0][0]))) # Episodes[0][0][0] is the size of a state
        totalepsilons = np.empty((0,1))
        for episode in episodes:
            if not episode[-1][5]:
                continue
            states = [transition[0].tolist() for transition in episode]
            epsilons = torch.zeros(len(episode))
            epsilons[0] = 1.0
            i = 0
            for i in range(1,len(episode)):
                state = episode[i][0]
                action = episode[i][1]
                if self.safe_memory.__len__() > 0:   
                    epsilons[i] = self.qsafe.get_value(state.detach(), action)
                else:
                    epsilons[i] = self.min_epsilon

                if epsilons[i] < self.min_epsilon:
                    epsilons[i] = self.min_epsilon

            states_epsilons = self.numpyify(epsilons.view(-1,1).float())
            statesflipped = np.array(states[::-1][:len(episode)])
            totalstates = np.vstack([totalstates,statesflipped])
            totalepsilons = np.vstack([totalepsilons,states_epsilons])

        if self.emodel is not None:
            self.emodel.add_samples(totalstates,totalepsilons)   
        else:
            self.emodel = RefitEpsilon(self.min_epsilon,statesflipped,states_epsilons)
        self.emodel.fit()
        print('Done,continuing...')
    def learn_safe(self):
        batch = self.safe_memory.sample(self.batch_size)
        weights = 1.0

        states, actions, rewards, next_states, violations = batch

        # Convert NumPy arrays to PyTorch tensors
        states = torch.Tensor(states).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        violations = torch.Tensor(violations).to(self.device)

        self.qsafe.update_q_network(states,actions,rewards,next_states, violations, self.policy)


    def training_learn(self, state, action):
        batch = self.memory.sample(self.batch_size)
        weights = 1.
        policy_loss, entropies = self.calc_policy_loss(batch, weights)

        # Fine tuning the policy
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip)
        policy_loss = policy_loss.detach().item()
        self.writer.add_scalar(
                'loss/policy', policy_loss,
                self.learning_steps)

        # Fine tuning the entropy
        entropy_loss = self.calc_entropy_loss(entropies, weights)
        update_params(self.alpha_optim, None, entropy_loss)
        self.alpha = self.log_alpha.exp()
        self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(), self.steps)
        self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)
        
        # Fine tuning the safety
        safety_loss = self.calc_safety_loss(state, action, weights)
        update_params(self.nu_optim, None, safety_loss)
        self.writer.add_scalar(
                'loss/nu', safety_loss.detach().item(), self.steps)

        return policy_loss, entropy_loss, safety_loss.detach().item()


    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        batch = self.memory.sample(self.batch_size)
        weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)

        update_params(
            self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip, retain_graph=True)
        update_params(
            self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip, retain_graph=True)
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip, retain_graph=True)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.policy.sample(states)
        # expectations of Q with clipped double Q technique
        q1, q2 = self.critic(states, sampled_action)
        q = torch.min(q1, q2)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = torch.mean((- q - self.alpha * entropy) * weights)
        return policy_loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
            * weights)
        return entropy_loss
    
    def calc_safety_loss(self, state, action, weights):
        if self.emodel is not None:
            epsilon = self.emodel.predict(np.array([state.tolist()]))[0]
        else:
            epsilon = self.min_epsilon
        safety_loss = torch.mean(
            self.nu * (epsilon - self.qsafe.get_value(state, action)) * weights)
        return safety_loss

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)
        lens = np.zeros((episodes,))

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.
            done = False
            length = 0
            while not done:
                length += 1
                action = self.act_safe(state)
                if action is None:
                    break
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            returns[i] = episode_reward
            lens[i] = length

        mean_return = np.mean(returns)
        std_returns = np.std(returns)
        mean_length = np.mean(lens)
        std_length = np.std(lens)

        return {
            'Eval return mean': mean_return,
            'Eval return std': std_returns,
            'Eval length mean': mean_length,
            'Eval length std': std_length
        }

    def save_models(self):
        self.policy.save(os.path.join(self.model_dir, 'policy.pth'))
        self.critic.save(os.path.join(self.model_dir, 'critic.pth'))
        self.critic_target.save(os.path.join(self.model_dir, 'critic_target.pth'))
        # self.qsafe.save(os.path.join(self.model_dir, 'qsafe.pth'))

    def __del__(self):
        # self.writer.close()
        self.env.close()
