import gymnasium as gym
import torch
from torch import nn
from torch import distributions
from collections import deque
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.hidden_nodes = 128
        self.fc1 = nn.Linear(observation_dim, self.hidden_nodes)
        self.fc2 = nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.fc3 = nn.Linear(self.hidden_nodes, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        #確率の対数を返す　self.fc2(x) = (バッチサイズ, action_dim) ← -1で、action_dimに対して
        return torch.log_softmax(self.fc3(x), -1)  
    

class Agent():
    def __init__(self, observation_dim, action_dim):
        self.policy_net = PolicyNet(observation_dim, action_dim) 
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.rewards_history = [] #各ステップの報酬を記憶
        self.actions_log_probs = [] #各ステップでとった行動の対数確立を記憶
        self.all_log_probs = [] #各ステップのすべての行動の対数確立
        self.entropy_coeff = 0.001
        self.gamma = 0.99
    
    def select_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0) #stateにバッチ次元を追加
        log_softmax_output = self.policy_net(state_tensor) #対数確立を得る
        dist = distributions.Categorical(logits=log_softmax_output) #対数確立から確率に変換
        action = dist.sample() #確率をもとに行動を決定 バッチサイズ１のテンソル(1, )
        self.actions_log_probs.append(dist.log_prob(action)) #選んだ行動の対数確立を保存
        self.all_log_probs.append(log_softmax_output) #ある状態のすべての対数確立を保存
        return action.item() #行動のテンソルから行動の値を取得してreturn
    
    def test_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        
        log_probs = self.policy_net(state_tensor)
        probs = torch.exp(log_probs)
        #print(log_probs, probs, self.select_action(state))
        return torch.argmax(probs).item()

    def store_reward(self, reward):
        self.rewards_history.append(reward)

    def learn(self):
        #1エピソードの最後から報酬を累積
        returns = [] #報酬リスト
        R = 0 #未来の割引報酬(終端では未来がないので０)
        for reward in reversed(self.rewards_history):
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        #報酬の標準化(エピソードごとの報酬のばらつきを抑える)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        #基本loss(実際にとった行動のloss)
        policy_loss = []
        for log_prob, r in zip(self.actions_log_probs, returns):
            # 勾配は -log_prob(対数確立) * return(rewardと割引報酬の合計) に比例
            policy_loss.append(-log_prob * r) #1ステップごとのlossのテンソルを追加
        policy_loss_tensor = torch.stack(policy_loss) #全ステップの基本lossを持ったリストのテンソルに変換
        
        #エントロピーの計算(log_all_probsを使用)
        episode_entropy = torch.tensor(0.0) #初期化
        if len(self.all_log_probs) > 0:
            #各ステップでの行動確率のエントロピーを計算
            entropies = [distributions.Categorical(logits=output.squeeze(0)).entropy() for output 
                        in self.all_log_probs]
            episode_entropy = torch.stack(entropies).sum()

        #最終的な損失
        final_loss = policy_loss_tensor.sum() - self.entropy_coeff * episode_entropy

        self.optimizer.zero_grad()
        final_loss.backward()
        self.optimizer.step()

        self.rewards_history = []
        self.actions_log_probs = []
        self.all_log_probs = [] # エントロピー計算用の履歴もリセット

env_name = "CartPole-v1"
env = gym.make(env_name)
agent = Agent(env.observation_space.shape[0], env.action_space.n)
rewards_list = deque(maxlen=100) #100エピソードの報酬
def train(train_episode):
    for episode in range(train_episode):
        step = 0
        total_reward = 0
        state, _ = env.reset()
        while(True):
            step += 1
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            agent.store_reward(reward)
            state = next_state

            if done or truncated:
                break
        agent.learn() # エピソード終了後に方策を更新
        rewards_list.append(total_reward)
        print(f"\repisode:{episode+1}  reward:{total_reward}  mean_reward:{np.mean(rewards_list):.2f} step:{step}", end="")
        if np.mean(rewards_list) > 475:
            break

def test(test_episode):
    print(f"\n---TEST---")
    env = gym.make(env_name, render_mode = "human")
    for episode in range(test_episode):
        step = 0
        total_reward = 0
        state, _ = env.reset()
        while(True):
            step += 1
            action = agent.test_action(state)
            #4ステップ左右どちらかに揺らす
            if step % 100 >= 96:action = step // 100 % 2
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
            if done or truncated:
                break
        print(f"episode:{episode+1}  reward:{total_reward}  step:{step}")


train(10000)
test(5)