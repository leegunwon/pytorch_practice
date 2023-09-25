import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import plotly.graph_objects as go

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3
T_horizon = 20


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(3, 128)
        self.fc_mu = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)
        self.fc_v = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a


    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):  # k_epoch 만큼 반복 학습
            # 똑같은 데이터로 학습을 k_epoch 만큼 반복하는 이유는?
            # k_epoch 만큼 반복하면서 학습을 하면 더 안정적으로 학습이 가능하다고 함

            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)   # temporal difference 오차를 뜻함
            delta = delta.detach().numpy()
            # GAE 방법 어드벤티지 추정값을 이용하여 학습
            advantage_lst = []   # advantage 값으로 업데이트 하기 때문에 필요
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)


            mu, std = self.pi(s, softmax_dim=1)
            log_prob = Normal(mu, std).log_prob(a)

            ratio = torch.exp(torch.log(log_prob - prob_a) - torch.log(log_prob - prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach()) # 정책 신경망 loss + 가치 신경망 loss
            # min을 하는 이유 min을 할 경우 결국 2개 중의 작은 것을 선택한다는 것이 최소를 보장
            # 결국 loss가 surr1보다 작거나, 같은 경우 두 가지가 된다. lowr bound를 보장
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def main():
    env = gym.make('Pendulum-v1')
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        count = 0

        while count < 200 and not done:

            for t in range(T_horizon):   # T_horizon 만큼 반복
                mu, std = model.pi(torch.from_numpy(s).float())    # pi 네트워크로 부터 확률 얻음
                dist = Normal(mu, std)
                a = dist.sample()
                log_prob = dist.log_prob(a) # 주어진 샘플이 분포에서 어떤 확률 밀도를 가지는지를 계산
                s_prime, r, done, truncated, info = env.step([a.item()])
                model.put_data((s, a, r / 100.0, s_prime, log_prob.item(), done))  # 메모리에 저장
                s = s_prime

                score += r
                count += 1

            model.train_net() # T_horizon 만큼 반복 후 학습 에피소드 단위가 아님? why?
            # T_horizon 만큼 반복 후 학습의 이점은?
            # 에피소드 단위 학습이 아닌 그보다 짧은 단위의 학습을 해서 추정치의 분산을 줄일 수 있음


        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()