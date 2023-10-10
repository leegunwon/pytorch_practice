import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time
import os
from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv


# Hyperparameters
learning_rate = 0.0005
gamma = 0.8
lmbda = 0.8
eps_clip = 0.3
K_epoch = 8
rollout_len = 10
buffer_size = 32
minibatch_size = 32


os.makedirs('./model_weights', exist_ok=True)
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(17, 128)
        self.fc_mu = nn.Linear(128, 6)
        self.fc_std = nn.Linear(128, 6)
        self.fc_v = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        mu = 1.0 * torch.tanh(self.fc_mu(x))     # 평균 생성
        std = F.softplus(self.fc_std(x))        # 표준 편차
        return mu, std

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition

                    s_lst.append(s)
                    a_lst.append(a)
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append(prob_a)
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)

            mini_batch = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch, dtype=torch.float), \
                torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
                torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            data.append(mini_batch)

        return data

    def sample_action(self, mu, std):

        dist = Normal(mu, std)
        a = dist.sample()
        log_prob = dist.log_prob(a).sum(-1, keepdim=True)
        a = a.detach().numpy()
        log_prob = log_prob.detach().numpy()
        return a, log_prob


    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                # neural network가 생각하는 s_prime의 value에 대한 예측값에 현재 얻은 reward를 더해줌
                td_target = r + gamma * self.v(s_prime) * done_mask
                # neural network가 생각하는 s에서 s_prime으로 갈 때의 advantage
                delta = td_target - self.v(s)
            delta = delta.numpy()  # tensor를 numpy로 바꿔줌

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

    def train_net(self):
        if len(self.data) == minibatch_size * buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)
            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    # 새로운 정책 네트워크에서 생성된 액션 확률 분포 생성
                    mu, std = self.pi(s)
                    dist = Normal(mu, std)

                    # 새로운 액션 확률 분포 내에서 과거의 사용한 action이 도출될 확률 log_prob 계산
                    log_prob = dist.log_prob(a).sum(-1, keepdim=True)
                    log_prob = log_prob.detach().numpy()

                    # 기존 액션 분포와 새로운 액션 분포 사이의 차이 계산
                    ratio = torch.exp(torch.from_numpy(log_prob).float() - old_log_prob)  # a/b == exp(log(a)-log(b))

                    # 확률 분포 사이의 차이와 advantage를 곱해서
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target)

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1



def main():
    env = HalfCheetahEnv()
    model = PPO()
    score = 0.0
    print_interval = 20
    rollout = []

    for n_epi in range(5000):
        # env.render()
        s = env.reset()
        done = False
        count = 0
        while count < 1000 and not done:
            for t in range(rollout_len):
                mu, std = model.pi(torch.from_numpy(s).float())
                a, log_prob = model.sample_action(mu, std[0])
                s_prime, r, done, truncated = env.step(a)

                rollout.append((s, a, r, s_prime, log_prob, done))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []

                s = s_prime
                score += r
                count += 1

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0


    torch.save(model.state_dict(), './model_weights/agent_'+str(n_epi))
    env.close()


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print(t2 - t1)
