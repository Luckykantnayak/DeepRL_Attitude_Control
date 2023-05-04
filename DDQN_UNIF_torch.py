import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

home_dir = os.path.expanduser('~')+"/DeepRL_Attitude_Control/"
    
class DeepQnetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir) -> None:
        super(DeepQnetwork, self).__init__()
        self.chkpt_dir = os.path.join(home_dir, chkpt_dir)
        self.checkpoint_file = os.path.join(self.chkpt_dir, name)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions
    
    def save_checkpoint(self):
        print('... saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class DDQNAgent():
    def __init__(self, gamma, max_epsilon, lr, input_dims, batch_size, n_actions,fc1_dims=256, fc2_dims=256, max_mem_size=100000, eps_end=0.01, epsilon_decay=5e-6, replace_target=100, chkpt_dir='ddqn',name='lunar_lander_ddqn_q') :
        self.gamma = gamma
        self.epsilon = max_epsilon
        self.eps_min = eps_end
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.replace_target = replace_target

        self.Q_online = DeepQnetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims, name=name+'_online',
                                   chkpt_dir=chkpt_dir).double()
        self.Q_target = DeepQnetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims, name=name+'_target',
                                   chkpt_dir=chkpt_dir).double()

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float64)
        self.new_state_memoey = np.zeros((self.mem_size, input_dims), dtype=np.float64)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float64)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
       

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memoey[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
    
    def choose_action(self, observation ):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array(observation)).to(self.Q_online.device)
            actions = self.Q_online.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    def replace_target_network(self):
        if self.mem_cntr % self.replace_target ==0:
            self.Q_target.load_state_dict(self.Q_online.state_dict())
    
    def save_models(self):
        self.Q_online.save_checkpoint()
        self.Q_target.save_checkpoint()

    def load_models(self):
        self.Q_online.load_checkpoint()
        self.Q_target.load_checkpoint()

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_online.optimizer.zero_grad()
        self.replace_target_network()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_online.device)
        new_state_batch = T.tensor(self.new_state_memoey[batch]).to(self.Q_online.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_online.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_online.device)

        action_batch = self.action_memory[batch]

        q_online = self.Q_online.forward(state_batch)[batch_index, action_batch]
        q_online_next = self.Q_target.forward(new_state_batch)
        max_actions = T.argmax(q_online_next, dim=1)

        q_target_next = self.Q_target.forward(new_state_batch)
        q_target_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*q_target_next[batch_index,max_actions]

        loss = self.Q_online.loss(q_target, q_online).to(self.Q_online.device)
        

        loss.backward()
        self.Q_online.optimizer.step()

        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.eps_min else self.eps_min

        
           





