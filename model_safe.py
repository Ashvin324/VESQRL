import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetworkSafe(nn.Module):
    def __init__(self, input_size, output_size, hidden_units):
        super(QNetworkSafe, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QLearningAgent:
    def __init__(self, input_size, output_size, hidden_units, learning_rate, gamma, device):
        self.device = device
        self.q_network = QNetworkSafe(input_size, output_size, hidden_units[0]).to(device)
        self.target_q_network = QNetworkSafe(input_size, output_size, hidden_units[1]).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma

    def get_value(self, state, action):
        state = state.unsqueeze(0).to(self.device)
        action = torch.Tensor(action).unsqueeze(0).to(self.device)
        if state.ndim < action.ndim:
            action = action[:,-1,:]
        q_safe_value = self.target_q_network(torch.cat((state, action), dim=1)).item()
        return max(0, min(q_safe_value,1))

    def select_action(self, state, epsilon, policy):
        ori_state=state
        ori_state = ori_state.unsqueeze(0).to(self.device)

        state = state.unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, _  = policy.sample(state)
        q_safe_value = self.target_q_network(torch.cat((ori_state, action), dim=1)).item()
        
        steps = 0
        while q_safe_value > epsilon and steps < 100:
            steps += 1
            with torch.no_grad():
                action, _, _  = policy.sample(state)
            q_safe_value = self.target_q_network(torch.cat((ori_state, action), dim=1)).item()
            
            if q_safe_value <= epsilon:
                return action
            
        if q_safe_value <= epsilon:
            return action
        else:
            return None   

    def update_q_network(self, states, actions, rewards, next_states, violations, policy):       
        q_value = self.q_network(torch.cat((states,actions), dim=1))

        next_q_value = self.target_q_network(torch.cat((next_states,torch.FloatTensor(policy.sample(states)[0])), dim=1)).to(self.device)

        expected_q_value = violations + (1-violations)*self.gamma*next_q_value

        loss = self.loss_fn(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())