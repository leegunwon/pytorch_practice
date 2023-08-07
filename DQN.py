import numpy as np
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 1
gamma = 0.98
buffer_limit = 10000
batch_size = 16

class SingleMachine():
    def __init__(self):
        self.a_left = 10
        self.b_left = 10
        self.c_left = 10
        self.oper_time = 0

    def step(self, a):

        if a[-1] ==0:
            if a[-2] == 1:
                self.oper_time += 5
            elif a[-2] == 2:
                self.oper_time += 5
            self.a_left -= 1
            self.oper_time += 10

        elif a[-1] ==1:
            if a[-2] == 0:
                self.oper_time += 10
            elif a[-2] == 2:
                self.oper_time += 5
            self.b_left -= 1
            self.oper_time += 20

        elif a[-1] ==2:
            if a[-2] == 0:
                self.oper_time += 10
            elif a[-2] == 1:
                self.oper_time += 5
            self.c_left -= 1
            self.oper_time += 30

        r, done = self.done()

        if done:
            if a.count(2) == 4:
                r += 20

        s = np.array([self.a_left, self.b_left, self.c_left, self.oper_time])

        return s, r, done

    def reset(self):
        self.a_left = 10
        self.b_left = 10
        self.c_left = 10
        self.oper_time = 0
        s = np.array([self.a_left, self.b_left, self.c_left, self.oper_time]) # [남은 A 작업 수, 남은 B 작업 수, 남은 C 작업 수, 현재 시간]
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
        self.buffer = collections.deque(maxlen=buffer_limit) # double-ended queue 양방향에서 데이터를 처리할 수 있는 queue형 자료구조를 의미한다

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

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


def train(q, q_target, memory, optimizer):
    for i in range(20):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)


        q_out = q(s)
        q_a = torch.gather(q_out, 1, a)       # q_out tensor에서 a 자리에 있는 열들 중 a값에 해당하는 위치를 인덱싱해서 뽑아옴
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)  # unsqueeze 함수 1일 경우 차원이 하나씩 증가함
        # max(i) 함수 안에 i는 차원 수를 의미 ex) 3차원 텐서에서 0이면 1차원에서 가장 큰 값들을 뽑아옴
        # i+1 숫자 만큼 대괄호를 걷어내고 나서 가장 큰 값들을 뽑아옴
        target = r + gamma * max_q_prime * done_mask   #
        loss = F.smooth_l1_loss(q_a, target)   # smooth_l1_loss 함수는 Huber loss 함수와 같음


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = SingleMachine()
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)     # optimizer 설정 Adam 사용

    for n_epi in range(5000):
        epsilon = max(0.01, 0.8 - 0.01 * n_epi / 50)
        s = env.reset()
        done = False
        a_history = [3]

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            a_history.append(a)
            s_prime, r, done = env.step(a_history)
            done_mask = 0.0 if done else 1.0

            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())    # q_target 업데이트 20번에 한번 씩
            # print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
               #  n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0

    score = 0.0
    epsilon = 0
    s = env.reset()
    done = False
    a_history = [3]
    while not done:
        a = q.sample_action(torch.from_numpy(s).float(), epsilon)
        a_history.append(a)
        s_prime, r, done = env.step(a_history)
        done_mask = 0.0 if done else 1.0
        memory.put((s, a, r / 100.0, s_prime, done_mask))
        s = s_prime

        score += r
        if done:
            break
    print(score)

if __name__ == '__main__':
    for i in range(7):
        main()