import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98

# continuous action space를 탐색하기 위해서 policy gradient를 사용한다.
# policy gradient는 action을 취한 후에 그 action이 좋은 action인지 나쁜 action인지를 평가하는 뉴럴넷이 필요하다.
#

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]: #
            R = r + gamma * R  # Gt를 구하는 과정
            loss = -torch.log(prob) * R  # loss를 구하는 과정 (policy gradient)
            loss.backward()   # action이 어떤 것인지 관계 없음, action이 좋은지 나쁜지만 관심
        self.optimizer.step()
        self.data = []   # 데이터 리셋 즉 한 에피소드의 데이터를 학습한 후 계속 리셋

# pi(torch.from_numpy(s).float()) 코드를 통해서 상태 s에서 선택 가능한 action의 확률을 구한다.
# 즉 s에서 action 확률 분포를 얻을 수 있다.

def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False

        while not done:  # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())  # s에서 action 확률 분포를 얻을 수 있는 코드.
            m = Categorical(prob)  # Categorical 함수를 통해서 범주형 확률 분포를 만들 수 있다.
            a = m.sample()  # 범주형 확률 분포에서 샘플링을 통해서 action을 선택한다. (확률에 따른 랜덤 선택)
            s_prime, r, done, truncated, info = env.step(a.item())  # item()을 통해서 tensor를 숫자로 바꿔준다.
            # 환경에 숫자 형태로 action을 넣어줘서 다음 상태를 받아 온다.
            pi.put_data((r, prob[a]))  # r은 reward, prob[a]는 action의 확률을 의미한다.
            # ? 왜 상태 s는 저장안하지?
            s = s_prime  # 다음 상태로 넘어간다.
            score += r

        pi.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()