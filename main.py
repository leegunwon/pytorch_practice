import collections
from collections import Counter
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import warnings

warnings.filterwarnings('ignore')

'''
machine : 1개
Job type : A, B, C
Job number : 각 10개, 총 30개
목적함수 : 가산점 + Throughput(생산 갯수) 최대화
is_done : 110초
processing time : A(10), B(20), C(30)
초기 setup : C
Setup 시간 : C->A:5, C->B:5, A->B:10, A->C:10, B->A:5, B->C:10
가산점 : Throughput에 C가 3개 이상있으면 20점
'''
learning_rate = 0.001
gamma = 0.99
buffer_limit = 20000
batch_size = 32
num_episodes = 2500


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)  # 스칼라 값이 아닌 벡터 값이 나올 수 있기 때문
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)  # 스칼라 값이 아닌 벡터 값이 나올 수 있기 때문
            done_mask_lst.append([done_mask])
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(
            s_prime_lst, dtype=torch.float), torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Score_Single_machine():
    def __init__(self):
        self.x = [0, 0, 0, 0, 0, 0]
        self.jobs = {'A': 10, 'B': 10, 'C': 10}
        self.job_processing_time = {'A': 10, 'B': 20, 'C': 30}
        self.all_setup_times = {
            'C': {'A': 5, 'B': 5},
            'A': {'B': 10, 'C': 10},
            'B': {'A': 5, 'C': 10}
        }
        self.current_job_type = 'C'
        self.total_jobs = 30
        self.select_job = []
        self.stop = 0
        self.total_processing_time = 0
        self.total_setup_time = 0
        self.setup_changes = 0
        self.final_score = 0

    def step(self, a):
        a = 'A' if a == 0 else "B" if a == 1 else "C"
        if a != self.current_job_type:
            setup_time = self.change_setup(a)
            self.total_setup_time += setup_time
            self.setup_changes += 1
        self.select_job.append(a)

        # processing_time 계산
        processing_time = self.job_processing_time[a]
        self.total_processing_time += processing_time
        self.stop += processing_time

        done = self.is_done()
        # 가산점
        C_num = 10 - self.jobs['C']
        bonus_points = 20 if C_num >= 3 else 0
        number_of_jobs_produced = self.total_jobs - sum(self.jobs.values())
        reward = bonus_points + 1
        if done == True:
            self.final_score = number_of_jobs_produced + bonus_points
        else:
            self.jobs[a] -= 1

        self.x = [bonus_points, C_num, sum(self.jobs.values()), self.total_setup_time, self.setup_changes,
                  100 - self.stop]
        return self.x, reward, done, self.final_score

    def change_setup(self, a):
        setup_time = self.all_setup_times[self.current_job_type][a]
        self.stop += setup_time
        self.current_job_type = a
        return setup_time

    def is_done(self):
        if self.stop >= 110:
            return True
        else:
            return False

    def reset(self):
        self.x = [0, 0, 0, 0, 0, 0]
        self.jobs = {'A': 10, 'B': 10, 'C': 10}
        self.current_job_type = 'C'
        self.select_job = []
        self.stop = 0
        self.total_processing_time = 0
        self.total_setup_time = 0
        self.setup_changes = 0
        self.final_score = 0
        return self.x


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 2)
        else:
            return out.argmax().item()

    def select_action(self, obs):
        out = self.forward(obs)
        return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):  # 1번의 업데이트에 32개의 데이터가 사용됨, 한 에피소드가 끝날 때마다 버퍼에서 총 320개의 데이터를 뽑아서 사용함
        s, a, r, s_prime, done_mask = memory.sample(batch_size)  # 리플레이 버퍼에서 미니 배치 추출
        q_out = q(s)
        q_a = q_out.gather(1, a)  # 실제 선택된 액션의 q값을 의미 # a에 해당하는 q값 추출
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)  # q_target 네트워크는 정답지를 계산할 떄 쓰이는 네트워크로 학습 대상이 아니다.
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)  # 손실 함수인 MSE 계산
        optimizer.zero_grad()  # 현재 배치에 대한 그라디언트를 계산하기 전에 모델의 매개변수 그라디언트를 초기화하는 역할
        loss.backward()  # loss에 대한 그라디언트 계산이 일어남
        optimizer.step()  # Qnet의 파라미터의 업데이트가 일어남


def main():
    env = Score_Single_machine()
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(num_episodes):
        epsilon = max(0.01, 0.4 - 0.05 * (n_epi / 200))
        s = env.reset()
        s = np.array(s)
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, final_score = env.step(a)
            s = np.array(s)
            s_prime = np.array(s_prime)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime
            if done:
                break

        # s = env.reset()
        # s = np.array(s)
        # done = False
        # while not done:
        #     a = q.select_action(torch.from_numpy(s).float())
        #     s_prime, r, done, final_score1 = env.step(a)
        #     s = np.array(s)
        #     s_prime = np.array(s_prime)
        #     s = s_prime
        #     if done:
        #         break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode : {}, final_score : {}, n_buffer : {}, eps : {:.1f}%  {} ".format(n_epi, final_score,
                                                                                          memory.size(), epsilon * 100, q.forward(torch.tensor([0, 0, 0, 0, 0, 0]).float())))
            # print("n_episode : {}, final_score : {}".format(n_epi, final_score1))

    s = env.reset()
    s = np.array(s)
    done = False
    act_set = []
    while not done:
        a = q.select_action(torch.from_numpy(s).float())
        s_prime, r, done, final_score2 = env.step(a)
        s = np.array(s)
        s_prime = np.array(s_prime)
        s = s_prime
        a = 'A' if a == 0 else "B" if a == 1 else "C"
        act_set.append(a)
        if done:
            break
    return act_set, final_score2
    # return final_score2


if __name__ == '__main__':
    act_set, final_score2 = main()
    print('Action : {}, Score : {}'.format(act_set, final_score2))
    # score_set = []
    # ten = 0
    # twenty_four = 0
    # for i in range(100):
    #     final_score2 = main()
    #     score_set.append(final_score2)
    #     if final_score2 == 10:
    #         ten += 1
    #     elif final_score2 == 24:
    #         twenty_four += 1
    #     print('{} 회 완료'.format(i+1))
    # print(score_set)
    # print('10 : {}, 24 : {}'.format(ten, twenty_four))

    # number_count = Counter(score_set)
    # # 결과 출력
    # for number, count in number_count.items():
    #     print(f"{number}: {count}개")
