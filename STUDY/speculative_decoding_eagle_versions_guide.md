# Speculative Decoding EAGLE 버전별 쉬운 정리

이 문서는 speculative decoding의 EAGLE 계열을 버전별로 쉽게 이해하기 위한 노트다.
핵심은 EAGLE-1, EAGLE-2, EAGLE-3가 모두 "draft를 만들고 target LLM이 검증한다"는
speculative decoding 틀 안에 있지만, draft를 만드는 방식과 tree를 다루는 방식이 점점 바뀌었다는 점이다.

기준으로 확인한 자료:

- EAGLE 공식 repo: <https://github.com/SafeAILab/EAGLE>
- EAGLE 논문: <https://arxiv.org/abs/2401.15077>
- EAGLE-2 논문: <https://arxiv.org/abs/2406.16858>
- EAGLE-3 논문: <https://arxiv.org/abs/2503.01840>
- Hugging Face paper pages:
  - <https://huggingface.co/papers/2401.15077>
  - <https://huggingface.co/papers/2406.16858>
  - <https://huggingface.co/papers/2503.01840>
- SGLang 문서: `docs/advanced_features/speculative_decoding.md`
- SGLang 코드:
  - `python/sglang/srt/speculative/spec_info.py`
  - `python/sglang/srt/server_args.py`

## 1. 한 줄 요약

| 버전 | 핵심 변화 | 쉽게 말하면 |
|---|---|---|
| EAGLE / EAGLE-1 | token이 아니라 target LLM의 feature를 예측 | "단어를 바로 찍지 말고, 큰 모델 머릿속 표현을 먼저 예측하자" |
| EAGLE-2 | 고정 draft tree를 문맥별 dynamic draft tree로 변경 | "맞을 것 같은 가지를 더 똑똑하게 펼치자" |
| EAGLE-3 | feature prediction 제약을 제거하고 direct token prediction + multi-layer feature fusion 사용 | "feature를 억지로 맞히지 말고, 여러 층 정보를 보고 토큰을 직접 잘 맞히자" |

SGLang 실행 관점에서는 다음처럼 보면 된다.

| 논문/개념 | SGLang args |
|---|---|
| EAGLE-2 | `--speculative-algorithm EAGLE` |
| EAGLE-3 | `--speculative-algorithm EAGLE3` |
| NEXTN | 내부적으로 `EAGLE` alias로 처리되는 경로 |

즉 SGLang에서 `EAGLE`이라고만 적으면 EAGLE-3가 아니라 보통 EAGLE-2 계열로 이해하면 된다.
EAGLE-3는 반드시 `EAGLE3`로 명시한다.

## 2. 먼저 speculative decoding 감각 잡기

일반 autoregressive decoding은 큰 target LLM이 토큰을 하나씩 만든다.

```text
target LLM -> token 1
target LLM -> token 2
target LLM -> token 3
target LLM -> token 4
```

문제는 매 토큰마다 큰 모델 전체를 다시 돌려야 한다는 것이다. GPU는 병렬 계산을 잘하지만,
decode 단계에서는 한 토큰씩 순차적으로 진행되므로 latency가 커진다.

Speculative decoding은 작은 draft side가 여러 토큰 후보를 먼저 만든다.

```text
draft side -> token 1, token 2, token 3, token 4 후보 생성
target LLM -> 후보들을 한 번에 검증
accept     -> 맞은 후보는 그대로 사용
reject     -> 틀린 지점부터 target LLM 결과로 보정
```

이 방식이 빠른 이유는 target LLM이 한 번 호출될 때 여러 후보를 병렬로 검증할 수 있기 때문이다.
초안이 많이 맞을수록 target LLM을 덜 자주 호출하게 되어 속도가 오른다.

중요한 점은 "검증 규칙"이다. speculative sampling의 accept/reject 규칙을 제대로 쓰면,
최종 출력 분포가 target LLM을 그냥 autoregressive decoding으로 돌린 것과 같게 유지된다.
그래서 EAGLE 논문들은 이를 lossless acceleration이라고 부른다.

## 3. EAGLE-1: token 대신 feature를 예측한다

EAGLE-1의 출발점은 다음 관찰이다.

```text
token sequence를 직접 예측하는 것보다
target LLM의 feature sequence를 예측하는 것이 더 쉽다.
```

여기서 feature는 보통 target LLM의 LM head 직전 hidden state, 논문 표현으로는
second-to-top-layer feature 또는 top-layer feature 근처의 representation을 의미한다.

일반 작은 draft LLM은 다음 토큰을 직접 예측한다.

```text
prefix tokens -> draft token distribution -> draft token
```

EAGLE-1은 다음처럼 한다.

```text
target LLM이 이미 계산한 feature들
+ 한 step 앞선 token들
-> EAGLE draft model
-> 다음 feature 예측
-> target LLM의 LM head로 token distribution 생성
-> draft token sample
```

### 왜 feature를 쓰나

자연어 token은 불연속적이고 문맥에 따라 갈림길이 많다. 반면 hidden feature는 모델 내부의
연속적인 표현이라 더 규칙적일 수 있다. EAGLE 논문은 feature-level autoregression이 token-level
autoregression보다 더 잘 맞는다고 주장한다.

쉽게 비유하면 다음과 같다.

```text
token 직접 예측:
  "다음 단어가 Paris인지 London인지 바로 맞혀"

feature 예측:
  "큰 모델이 다음 단어를 고르기 직전의 생각 벡터를 먼저 맞혀"
```

feature를 잘 맞히면 target LLM의 LM head를 그대로 써서 token distribution을 만들 수 있다.
그래서 target LLM의 vocabulary/head와 잘 정렬된 draft를 만들 수 있다.

### 왜 shifted token이 필요한가

feature만 예측하면 문제가 있다. 샘플링 결과에 따라 다음 feature가 달라진다.

예를 들어 현재 token이 `I`라면 다음 token으로 `am`이 나올 수도 있고 `always`가 나올 수도 있다.
두 경우 뒤따르는 feature 경로는 달라진다.

그래서 EAGLE-1은 draft model 입력에 feature sequence만 넣지 않고, 한 step 앞선 token sequence도
같이 넣는다.

```text
feature: f1, f2, f3
token:      t2, t3, t4
```

이렇게 하면 "실제로 어떤 token이 샘플되었는지"를 draft model이 알 수 있어서 feature prediction의
불확실성이 줄어든다.

### EAGLE-1의 draft tree

EAGLE-1은 tree attention을 이용해 후보를 tree 형태로 만든다.

```text
root
├── 후보 A
│   ├── 후보 A1
│   └── 후보 A2
└── 후보 B
    ├── 후보 B1
    └── 후보 B2
```

chain draft는 중간 token이 reject되면 뒤 후보를 모두 버려야 한다. tree draft는 한 가지가
틀려도 다른 branch를 검증할 수 있어 average acceptance length를 늘릴 수 있다.

EAGLE-1의 요약:

- target LLM은 수정하지 않는다.
- 작은 autoregression head만 추가 학습한다.
- token-level draft보다 feature-level draft가 더 잘 맞는다는 관찰을 사용한다.
- shifted token 입력으로 sampling uncertainty를 줄인다.
- tree attention으로 여러 후보 branch를 검증한다.
- 논문 기준 LLaMA2-Chat 70B에서 2.7x-3.5x latency speedup을 보고했다.

## 4. EAGLE-2: static tree에서 dynamic tree로

EAGLE-2는 EAGLE-1의 draft model 자체를 크게 바꾸는 방법이 아니다.
핵심은 draft tree를 어떻게 만들고 고를지를 바꾸는 것이다.

EAGLE-1, Medusa 같은 방법은 대체로 정해진 tree 모양을 사용한다.

```text
항상 같은 폭과 깊이로 후보를 펼친다.
```

하지만 실제 문맥은 매번 다르다.

```text
"10 + 2 ="      -> 다음 token은 매우 쉬움
"The answer is" -> 여러 후보가 있을 수 있음
```

쉬운 문맥에서는 한 branch를 깊게 파는 것이 좋고, 어려운 문맥에서는 여러 branch를 넓게 보는 것이
좋다. EAGLE-2는 이 차이를 이용한다.

### confidence score를 acceptance rate 근사로 사용

EAGLE-2의 중요한 관찰은 draft model의 confidence score가 실제 acceptance rate와 꽤 잘 맞는다는 것이다.

```text
draft model confidence 높음
-> target LLM도 accept할 가능성이 높음

draft model confidence 낮음
-> reject될 가능성이 높음
```

target LLM을 호출해서 accept probability를 직접 알면 비용이 너무 크다. 그래서 EAGLE-2는
draft model이 이미 계산한 confidence를 사용해 "이 branch가 살아남을 가능성"을 추정한다.

각 node의 값은 대략 그 node까지 오는 경로의 confidence 곱으로 볼 수 있다.

```text
value(node) ~= path에 있는 token confidence들의 곱
```

깊은 node는 앞 token들이 모두 accept되어야 도달하므로 path 전체 확률이 중요하다.

### expansion phase

EAGLE-2는 현재 layer의 모든 node를 무작정 확장하지 않는다.
가장 promising한 node만 골라 다음 layer를 만든다.

```text
현재 layer nodes
-> value가 높은 top-k node 선택
-> 선택된 node만 draft model로 확장
```

이렇게 하면 reject될 가능성이 큰 branch에 draft compute를 낭비하지 않는다.

### reranking phase

확장만 하면 깊은 node 위주로 tree가 만들어질 수 있다. 그런데 어떤 shallow node는 깊은 node보다
value가 높을 수 있다. 그래서 EAGLE-2는 전체 후보 node를 다시 정렬해서 target LLM에 검증시킬
최종 draft tokens를 고른다.

```text
확장된 전체 후보
-> value 기준 rerank
-> top speculative_num_draft_tokens 선택
-> tree attention mask 구성
-> target LLM verification
```

SGLang 문서에서도 EAGLE-2 구현은 draft tree를 configured steps만큼 확장한 뒤,
`speculative_num_draft_tokens`개 final node를 rerank해서 고른다고 설명한다.

EAGLE-2의 요약:

- EAGLE-1의 feature-level draft 아이디어는 유지한다.
- static draft tree 대신 context-aware dynamic draft tree를 사용한다.
- draft model confidence score로 acceptance rate를 근사한다.
- promising branch를 확장하고, 전체 후보를 rerank한다.
- 별도의 tree predictor를 새로 학습하지 않아도 된다.
- 논문 기준 3.05x-4.26x speedup, EAGLE-1 대비 20-40% 개선을 보고했다.

## 5. EAGLE-3: feature prediction 제약을 버린다

EAGLE-3는 EAGLE-1/2보다 더 큰 방향 전환이다.

EAGLE-1/2는 draft model이 다음 feature를 잘 맞히도록 학습한다. 하지만 실제 목표는 feature 자체가
아니라 target LLM이 accept할 token을 잘 제안하는 것이다.

EAGLE-3 논문은 feature prediction loss가 오히려 draft model의 표현력을 제한한다고 본다.

```text
진짜 목표:
  좋은 draft token을 만들어 target LLM에게 accept되기

EAGLE-1/2의 중간 목표:
  target LLM의 다음 feature를 회귀로 맞히기

문제:
  feature를 맞히는 제약이 token prediction 성능을 막을 수 있음
```

그래서 EAGLE-3는 feature prediction objective를 제거하고 direct token prediction으로 간다.

### direct token prediction

EAGLE-3는 draft model 출력이 target LLM의 top-layer feature와 같아야 한다고 강제하지 않는다.
대신 draft model이 token distribution을 직접 잘 만들도록 학습한다.

```text
target model에서 뽑은 여러 layer feature
-> EAGLE-3 draft model
-> token distribution 직접 예측
-> draft token 생성
```

이렇게 하면 draft model 출력 공간이 더 자유로워진다. feature regression이라는 보조 목표에 묶이지
않고, acceptance에 직접 도움이 되는 방향으로 학습할 수 있다.

### multi-layer feature fusion

EAGLE-1/2는 주로 top-layer feature에 의존한다. top-layer feature는 LM head 바로 앞이므로
다음 token prediction에는 좋지만, 여러 token 뒤를 예측하기에는 정보가 너무 "다음 token"에
특화되어 있을 수 있다.

EAGLE-3는 low, middle, high layer feature를 함께 사용한다.

```text
low layer    -> 표면적/형태적 정보
middle layer -> 구문/문맥 정보
high layer   -> 다음 token에 가까운 의미 정보
```

실제 layer별 의미를 이렇게 딱 잘라 말할 수는 없지만, 감각적으로는 여러 깊이의 정보를 섞어 더
풍부한 draft 입력을 만든다고 이해하면 된다.

### training-time test

EAGLE-3의 또 다른 핵심은 training-time test다.

일반적으로 draft model은 훈련할 때 깨끗한 target feature나 정답 token을 입력으로 받는다. 하지만
추론할 때는 자기 자신이 방금 만든 draft token 또는 draft output을 다시 입력으로 사용한다.

즉 train과 test가 다르다.

```text
training:
  깨끗한 target-side 정보 기반으로 다음 token 학습

inference:
  draft model이 만든 이전 예측을 다시 입력으로 사용
```

EAGLE-3는 훈련 중에도 실제 추론처럼 multi-step generation 상황을 시뮬레이션한다.
그래서 draft model이 "자기 예측이 입력으로 다시 들어오는 상황"에 익숙해진다.

```text
훈련 중 step 1 예측
-> 그 예측을 다음 step 입력에 반영
-> step 2 예측
-> 반복
```

논문에서는 이 덕분에 draft position이 뒤로 갈수록 acceptance rate가 크게 떨어지는 문제를 줄였다고
보고한다.

EAGLE-3의 요약:

- feature prediction loss를 제거한다.
- token을 직접 예측한다.
- top-layer feature만 쓰지 않고 low/mid/high layer feature를 fusion한다.
- training-time test로 train/test mismatch를 줄인다.
- 더 많은 training data를 넣었을 때 성능이 더 잘 오른다.
- 논문 기준 최대 6.5x speedup, EAGLE-2 대비 약 1.4x 개선을 보고했다.
- SGLang 환경에서는 LLaMA-Instruct 3.1 8B, batch size 64에서 1.38x throughput improvement를 보고했다.

## 6. 세 버전의 차이를 한 번에 보기

| 항목 | EAGLE-1 | EAGLE-2 | EAGLE-3 |
|---|---|---|---|
| draft 기본 단위 | feature prediction | feature prediction | direct token prediction |
| target LLM 수정 | 없음 | 없음 | 없음 |
| draft model 학습 | autoregression head 학습 | EAGLE-1 draft 사용, 추가 tree 모델 학습 없음 | EAGLE3 draft 학습 필요 |
| tree 구조 | static tree | dynamic tree | 구현에 따라 tree/chain 설정 가능 |
| 핵심 개선 | feature-level drafting | confidence 기반 dynamic draft tree | feature constraint 제거 + multi-layer fusion + training-time test |
| 장점 | token draft보다 높은 acceptance | 같은 drafter로 더 높은 accept length | 더 높은 acceptance, 데이터 scaling 효과 |
| 한계 | feature regression 제약 | feature regression 제약 유지 | EAGLE3 checkpoint/training 필요 |
| 대표 성능 | 2.7x-3.5x on LLaMA2-Chat 70B | 3.05x-4.26x | 최대 6.5x |

## 7. 쉬운 비유

### EAGLE-1

```text
큰 모델이 최종 답을 쓰기 직전의 메모를 작은 모델이 흉내 낸다.
그 메모를 큰 모델의 LM head에 넣어 다음 단어 후보를 만든다.
```

### EAGLE-2

```text
작은 모델이 여러 후보 가지를 만들 때,
아무 가지나 똑같이 펼치지 않고
"이 가지는 맞을 것 같다" 싶은 쪽을 더 펼친다.
```

### EAGLE-3

```text
이제 큰 모델의 메모를 글자 그대로 베끼려 하지 않는다.
대신 큰 모델의 여러 층에서 나온 힌트를 보고
작은 모델이 바로 다음 token 후보를 잘 쓰도록 훈련한다.
```

## 8. SGLang에서 기억할 점

SGLang 문서 기준 추천은 다음과 같다.

```text
최고 속도/품질 추천:
  --speculative-algorithm EAGLE3

호환성 좋은 기본 선택:
  --speculative-algorithm EAGLE
```

대표적인 EAGLE-2 실행 예:

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16
```

대표적인 EAGLE-3 실행 예:

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16
```

주요 튜닝 파라미터:

| 파라미터 | 의미 |
|---|---|
| `--speculative-num-steps` | draft model이 autoregressive하게 몇 step 앞까지 볼지 |
| `--speculative-eagle-topk` | 각 step에서 branch 후보를 몇 개 뽑을지 |
| `--speculative-num-draft-tokens` | target LLM이 한 번에 검증할 draft token 수 |
| `--speculative-draft-model-path` | EAGLE/EAGLE3 draft model checkpoint |

너무 공격적으로 잡으면 GPU memory와 verification cost가 커지고, reject가 많으면 오히려 느려질 수 있다.
보수적으로는 다음처럼 시작해서 점진적으로 올리는 방식이 안전하다.

```bash
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

## 9. 언제 어떤 버전을 생각하면 좋은가

### EAGLE-2가 맞는 경우

- SGLang에서 널리 검증된 EAGLE draft checkpoint가 있다.
- EAGLE3 checkpoint가 아직 없다.
- 호환성을 우선한다.
- 기존 EAGLE 계열 설정을 크게 바꾸고 싶지 않다.

### EAGLE-3가 맞는 경우

- target model에 맞는 EAGLE3 draft checkpoint가 있다.
- 작은 batch latency뿐 아니라 production batch throughput도 보고 싶다.
- draft training을 새로 할 수 있거나 SpecForge 같은 툴을 쓸 수 있다.
- 최신 EAGLE 계열 성능을 우선한다.

### 둘 다 조심해야 하는 경우

- batch size가 매우 커서 이미 GPU가 잘 차 있는 경우
- draft model 로딩으로 memory headroom이 부족한 경우
- target model과 draft model tokenizer/chat template/config가 맞지 않는 경우
- workload가 draft training data와 너무 다른 경우
- acceptance length가 낮게 나오는 경우

Speculative decoding은 "무조건 빨라지는 스위치"라기보다, draft 품질과 시스템 병목이 맞아야 빨라지는
가속 기법이다. 특히 batch가 커질수록 target LLM의 parallelism이 이미 좋아지기 때문에 이득이 줄 수 있다.
EAGLE-3는 이 구간에서도 EAGLE-2보다 더 버티는 결과를 논문과 SGLang 실험에서 보였다.

## 10. 최종 정리

```text
EAGLE-1:
  feature를 예측해서 좋은 draft token을 만든다.

EAGLE-2:
  feature drafter는 유지하고, draft tree를 문맥별로 똑똑하게 만든다.

EAGLE-3:
  feature를 맞히라는 제약을 버리고,
  여러 layer feature를 이용해 token을 직접 예측한다.
```

실전 감각으로는 다음 한 줄이면 된다.

```text
EAGLE-1은 "무엇을 예측할까"를 바꿨고,
EAGLE-2는 "후보 tree를 어떻게 고를까"를 바꿨고,
EAGLE-3는 "draft 모델을 어떤 목표와 입력으로 학습할까"를 다시 설계했다.
```
