import numpy as np
import collections
import random
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 1
gamma = 0.98
buffer_limit = 25000
batch_size = 16   # MC이므로 한 에피소드 사용


class SingleMachine():
    def __init__(self):
        self.a_left = 10
        self.b_left = 10
        self.c_left = 10
        self.oper_time = 0

    def step(self, a):

        if a[-1] == 0:
            if a[-2] == 1:
                self.oper_time += 5
            elif a[-2] == 2:
                self.oper_time += 5
            self.a_left -= 1
            self.oper_time += 10

        elif a[-1] == 1:
            if a[-2] == 0:
                self.oper_time += 10
            elif a[-2] == 2:
                self.oper_time += 5
            self.b_left -= 1
            self.oper_time += 20

        elif a[-1] == 2:
            if a[-2] == 0:
                self.oper_time += 5
            elif a[-2] == 1:
                self.oper_time += 5
            self.c_left -= 1
            self.oper_time += 30

        r, done = self.done()
        if a == [2, 2, 2, 2]:
            r += 20


        s = np.array([self.a_left, self.b_left, self.c_left, self.oper_time])

        return s, r, done

    def reset(self):
        self.a_left = 10
        self.b_left = 10
        self.c_left = 10
        self.oper_time = 0
        s = np.array(
            [self.a_left, self.b_left, self.c_left, self.oper_time])  # [남은 A 작업 수, 남은 B 작업 수, 남은 C 작업 수, 현재 시간]
        return s

    def done(self):
        if self.oper_time == 100:
            return 1, True

        elif self.oper_time > 100:
            return 0, True
        else:
            return 1, False


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, done_mask_lst, score_lst = [], [], [], [], []

        for episode in mini_batch:
            for transition in episode:
                s, a, r, done_mask = transition
                s_lst.append(s)
                a_lst.append([a])
                r_lst.append([r])
                done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 2)
        else:
            return out.argmax().item()


def train(q, memory, optimizer):

    for i in range(10):
        s, a, r, done_mask = memory.sample(1)
        # MC이므로 한 에피소드 사용 - 충족
        # MC 경우 한 에피소드를 한번에 불러와서 학습 - 충족
        # MC의 경우 에피소드 끝에서 리턴 값을 바탕으로 학습한 뒤 한단계씩 앞으로 가면서 학습 - 충족
        q_out = q(s)
        q_a = torch.gather(q_out, 1, a)       # q_out tensor에서 a 자리에 있는 열들 중 a값에 해당하는 위치를 인덱싱해서 뽑아옴
        sum_reward = [0]*len(r)
        sum = 0
        for i in range(len(r)):
            sum += r[i]
            sum_reward[i] = sum
        sum_reward = torch.tensor(sum_reward).unsqueeze(1)
        loss = F.smooth_l1_loss(q_a, sum_reward)   # smooth_l1_loss 함수는 Huber loss 함수와 같음

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = SingleMachine()
    q = Qnet()
    memory = ReplayBuffer()

    print_interval = 20
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)  # optimizer 설정 Adam 사용
    sc_history = []

    for n_epi in range(10000):
        epsilon = max(0.01, 0.7 - 0.03 * (n_epi/ 200))
        s = env.reset()
        score = 0.0
        done = False
        a_history = [2]
        history = collections.deque()
        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            a_history.append(a)
            s_prime, r, done = env.step(a_history)
            done_mask = 0.0 if done else 1.0
            score += r
            history.appendleft([s, a, r, done_mask])
            s = s_prime


            if done:
                break
        sc_history.append(score)
        memory.put(history)

        if memory.size() > 500:
            train(q, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}% sqs : {}, {}".format(
                 n_epi, score, memory.size(), epsilon * 100, a_history, q.forward(torch.tensor([10,10,10,0]).float())))


    score = 0.0
    epsilon = 0
    s = env.reset()
    done = False
    a_history = [2]
    while not done:
        a = q.sample_action(torch.from_numpy(s).float(), epsilon)
        a_history.append(a)
        s_prime, r, done = env.step(a_history)
        s = s_prime

        score += r
        if done:
            break

    print("score : {:.1f}, n_buffer : {}, eps : {:.1f}% sqs : {}, {}".format(
        score, memory.size(), epsilon * 100, a_history, q.forward(torch.tensor([10,10,10,0]).float())))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=np.arange(len(sc_history)), y=sc_history, name='score'))

if __name__ == '__main__':
    main()