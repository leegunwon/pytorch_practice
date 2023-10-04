import random
from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time



# Hyperparameters
lr_mu = 0.0005
lr_q = 0.001
gamma = 0.99
batch_size = 32
buffer_limit = 50000
tau = 0.005  # for target network soft update

# DDPG와 DQN의 다른점
# DQN의 경우 action이 discrete하고, DDPG의 경우 action이 continuous하다.
# 따라서 DQN은 action을 선택할 때 argmax를 사용하지만, DDPG는 action을 선택할 때 mu 함수를 사용한다.
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
            torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(17, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * 2  # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(17, 64)
        self.fc_a = nn.Linear(6, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

# Ornstein-Uhlenbeck Noise를 사용하는 이유 : action을 선택할 때 exploration을 하기 위해서
# 기존의 DQN의 경우 action을 선택할 때 epsilon-greedy를 사용했다.
# DDPG의 경우 action이 continuous하기 때문에 epsilon-greedy를 사용할 수 없다.
# epsilon greedy의 경우 epsilon한 확률로 랜덤아하게 action을 선택하는데 이러한 방법은
# 탐색범위가 넓은 continuous action에 적합하지 않다.
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

        # 이전 x + theta * (mu - 이전 x) * dt + sigma * sqrt(dt) * N(0,1) 랜덤
        # 장기 평균에서 많이 벗어날 경우 평균회귀성에 의해 다시 평균에 가까워짐

        # 위 코드에서는 평균을 mu로 하는 N(0,1) 랜덤을 사용함
        # 이전 노이즈 값이 mu와 얼마나 떨어졌는지를 반영하여 다음 노이즈를 생성함

        self.x_prev = x
        return x


def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    # reinforce 알고리즘과는 다르게 DDPG의 경우 Replay Buffer를 사용한다.
    s, a, r, s_prime, done_mask = memory.sample(batch_size)

    # q 네트워크와 mu 네트워크를 함께 학습 시킨다.
    # 기존의 DQN 업데이트 식 사용
    # mu 네트워크의 경우 q value에 -값을 손실함수로 정의 했다.
    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()  #
    q_loss.backward()
    q_optimizer.step()
    # mu 네트워크는 상태 s를 받아서 action을 내뱉어주는 함수
    # mu 네트워크의 경우 q value에 -값을 손실함수로 정의 했다. why?
    mu_loss = -q(s, mu(s)).mean()  # mean을 왜 하는지 모르겠다.
    # 앞에서 Replay Buffer로 부터 32개의 데이터를 뽑았기 때문에 32개의 action이 존재
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def main():
    env = HalfCheetahEnv()
    memory = ReplayBuffer()
    score_list = []
    max_score_list = []
    max_score = -10000

    q, q_target = QNet(), QNet()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(10000):
        env.render()
        s = env.reset()
        done = False

        count = 0
        while count < 1000 and not done:
            a = mu(torch.from_numpy(s).float())   # reinforcement learning의 policy network와 동일하다.
            # mu network의 로 부터 action을 받아온다.
            a = [a[i].item() + ou_noise()[0] for i in range(len(a))]
            s_prime, r, done, truncated = env.step(a)
            memory.put((s, a, r, s_prime, done))
            score += r
            s = s_prime
            count += 1
            if score > max_score:
                max_score = score

        if memory.size() > 2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q, q_target)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score_list.append(score / print_interval)
            max_score_list.append(max_score)
            max_score = -1000
            score = 0.0

    xml_model = env.model.get_xml()
    with open('model.xml', 'w') as f:
        f.write(xml_model)

    env.close()





if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print("time :", t2 - t1)