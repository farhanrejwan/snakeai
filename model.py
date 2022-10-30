import os
import torch
import torch.nn as nn
import torch.nn.functional as nf
import torch.optim as op

class Linear_QNet(nn.Module) :
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size) :
        super().__init__()

        self.linear_1 = nn.Linear(input_layer_size, hidden_layer_size)
        self.linear_2 = nn.Linear(hidden_layer_size, output_layer_size)
    
    def forward(self, x) :
        x = nf.relu(self.linear_1(x))
        x = self.linear_2(x)

        return x
    
    def save(self, model_file_name='model.pth') :
        folder_path = './model'

        if not os.path.exists(folder_path) :
            os.makedirs(folder_path)
        
        file_name = os.path.join(folder_path, model_file_name)
        torch.save(self.state_dict(), file_name)
    
    def load(self, model_file_name='model.pth') :
        torch.load(self.state_dict(), model_file_name)

class QTrainer :
    def __init__(self, model, lr, gamma) :
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = op.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done) :
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1 :
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done,)
        
        prediction = self.model(state)
        target = prediction.clone()

        for index in range(len(done)) :
            Q_new = reward[index]

            if not done[index] :
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
            
            target[index][torch.argmax(action).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        
        self.optimizer.step()