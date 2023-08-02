import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# pip install torch==2.0.0
# pytorch tutorial 자료 'https://tutorials.pytorch.kr/beginner/nlp/sequence_models_tutorial.html?highlight=lstm'
# LSTM(Long Short-Term Memory)의 구성 요소
# 1. forget gate    : 이전 cell state에서 어떤 정보를 버릴지 결정
# 2. input gate     : 현재 input에서 어떤 정보를 cell state에 추가할지 결정
# 3. output gate    : cell state를 어떤 값으로 내보낼지 결정
# 4. temp cell state: 현재 input을 반영한 cell state
# 5. cell state     : 이전 cell state에 현재 input을 반영한 값
# 6. hidden state   : 현재 cell state를 tanh 함수를 통해 걸러낸 값

# LSTM 함수의 경우 neural network 함수의 차원을 설정하는 것만 신경쓰면 됨
# LSTM 기본적인 사용 구조
lstm = nn.LSTM(3, 3)    # Input dim 3, output dim 3  document 주소'https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM'
inputs = [torch.randn(1, 1, 3) for _ in range(5)]   # input data [1,3]shape tensor 5개 리스트
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))   # hidden state [1,1,3]shape tensor 2개 튜플

for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)   # view()함수를 사용하여 3차원으로 변환 사용하지 않고 넣을 경우 (1,3)shape tensor가 들어가기 때문에 오류 발생

"""
위의 for문과 동일한 결과를 내는 코드
inputs = torch.cat(inputs).view(len(inputs), 1, -1)  # torch.cat()함수를 사용하여 input을 이어 붙임 (5,1,3)shape tensor로 변환
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  
out, hidden = lstm(inputs, hidden)
"""


# 학습 데이터 생성 및 전처리
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}

for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # word_to_ix에 word를 key로, len(word_to_ix)를 value로 추가
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # 품사 태그에 대한 고유한 인덱스를 부여

# usually 32 or 64 dimensional.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMTagger(nn.Module):   # LSTMTagger 클래스 정의 nn.Module을 상속받음 뉴럴 네트워크라는 표시

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)   # look up table을 생성

        # embedding_dim은 input_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # LSTM의 hidden state를 선형 네트워크 함수를 적용하여 tagset_size만큼의 차원으로 변환
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence) # look up table sentence만큼 indexing
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)   # SGD(Stochastic Gradient Descent) optimizer 사용

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)   # pre-training 결과

for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()   # gradient를 0으로 초기화


        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(sentence_in)  # forward 함수를 통해 tag score를 계산

        loss = loss_function(tag_scores, targets)
        loss.backward()    # loss를 통해 parameter의 gradient를 계산
        optimizer.step()   # optimizer를 통해 parameter를 업데이트

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # "the dog ate the apple".  태깅 결과
    # 가장 높은 값의 index를 통해 태깅 결과를 확인할 수 있음
    # 0번째 단어는 DET, 1번째 단어는 NN, 2번째 단어는 V, 3번째 단어는 DET, 4번째 단어는 NN
    # 0,1,2,0,1의 index를 가지고 있음
    print(tag_scores)   # training 결과