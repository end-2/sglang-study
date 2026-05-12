# Speculative Decoding, EAGLE, MTP/NEXTN, DFlash 쉬운 정리

이 문서는 이번 세션에서 정리한 내용을 한 번에 읽을 수 있게 풀어 쓴 노트다.
SGLang에서 speculative decoding을 켤 때 나오는 `EAGLE`, `EAGLE3`, `MTP`,
`NEXTN`, `DFLASH`가 각각 무엇이고, 어떤 args로 실행하는지 쉽게 이해하는 것이 목표다.

기준:

- 현재 repository 코드: `python/sglang/srt/server_args.py`,
  `python/sglang/srt/speculative/spec_info.py`,
  `python/sglang/srt/models/deepseek_nextn.py`,
  `python/sglang/srt/models/dflash.py`
- SGLang 공식 문서: <https://docs.sglang.io/docs/advanced_features/speculative_decoding>
- 논문:
  - Speculative Decoding: <https://proceedings.mlr.press/v202/leviathan23a>
  - EAGLE: <https://arxiv.org/abs/2401.15077>
  - EAGLE-2: <https://arxiv.org/abs/2406.16858>
  - EAGLE-3: <https://arxiv.org/abs/2503.01840>
  - MTP: <https://arxiv.org/abs/2404.19737>
  - DFlash: <https://arxiv.org/abs/2602.06036>

## 1. 한 줄 요약

Speculative decoding은 "빠른 쪽이 초안을 쓰고, 큰 target 모델이 한 번에 검사해서 맞는
부분을 채택하는" decode 가속 방법이다.

| 이름 | 쉬운 설명 | SGLang 실행 관점 |
|---|---|---|
| Speculative decoding | 전체 기법 이름 | draft 생성 + target verify + accept/reject |
| EAGLE | 학습된 EAGLE draft 모델로 후보를 만드는 방식 | `--speculative-algorithm EAGLE` 또는 `EAGLE3` |
| MTP | 모델 안에 여러 미래 토큰 예측 모듈이 들어 있는 구조 | 보통 EAGLE speculative 경로를 사용 |
| NEXTN | MTP/Next-N layer를 가리키는 이름 또는 alias | 현재 코드에서는 `NEXTN`을 내부적으로 `EAGLE`로 바꿈 |
| DFlash | diffusion-style draft 모델이 토큰 block을 병렬로 만드는 방식 | `--speculative-algorithm DFLASH` |

가장 중요한 감각은 이것이다.

```text
EAGLE, MTP/NEXTN, DFlash는 모두 speculative decoding을 위한 draft 생성 방식이다.
서로 다른 점은 "초안을 누가, 어떤 구조로, 몇 토큰 단위로 쓰는가"다.
```

## 2. 왜 speculative decoding이 필요한가

일반 autoregressive decoding은 한 번에 다음 토큰 하나만 만든다.

```text
target LLM -> token 1
target LLM -> token 2
target LLM -> token 3
target LLM -> token 4
```

이 방식은 품질은 좋지만, 매 토큰마다 큰 모델을 다시 돌려야 한다. 특히 batch가 작거나
interactive latency가 중요한 상황에서는 GPU가 가진 병렬 계산 능력을 충분히 쓰지 못할 수 있다.

Speculative decoding은 더 빠른 draft side를 붙인다.

```text
draft side  -> token 1, token 2, token 3, token 4 후보를 빠르게 생성
target LLM  -> 후보 4개를 한 번에 검증
accept      -> 맞은 후보는 그대로 사용
reject      -> 틀린 지점부터 target 결과로 보정
```

비유하면, junior writer가 초안을 빠르게 쓰고 senior reviewer가 한 번에 빨간펜 검토를 하는
것과 비슷하다. 초안이 많이 맞을수록 빨라진다.

## 3. Speculative decoding의 기본 흐름

### Draft

draft 모델 또는 draft 모듈이 앞으로 나올 토큰 후보를 만든다.

예:

```text
현재 문맥: "The capital of France is"
draft 후보: " Paris ."
```

### Verify

target 모델이 draft 후보 전체를 한 번에 본다. target 모델은 원래 모델이므로 최종 품질을
결정하는 기준이다.

### Accept / Reject

draft token이 target 모델의 결과와 맞으면 accept된다. 어느 지점에서 틀리면 그 뒤 후보는 버리고
target 모델의 토큰으로 이어간다.

그래서 speculative decoding의 속도는 대체로 다음에 달려 있다.

- draft가 얼마나 싸게 여러 토큰을 만드는가
- draft 후보가 target 모델과 얼마나 자주 맞는가
- verify를 한 번에 얼마나 효율적으로 하는가
- KV cache, CUDA graph, scheduler가 이 흐름을 얼마나 잘 받쳐주는가

## 4. EAGLE

EAGLE은 일반적인 작은 draft LLM을 붙이는 것보다 target 모델의 hidden feature를 더 잘 활용하려는
speculative decoding 계열이다.

EAGLE 논문의 핵심 관찰은 token 자체보다 target 모델의 상위 hidden feature를 예측하는 편이 더
규칙적일 수 있다는 점이다. EAGLE은 target 모델의 feature 정보를 활용해서 더 잘 맞는 draft를
만들고, target 모델은 이를 검증한다.

### EAGLE-2와 EAGLE-3

SGLang 문서와 코드 기준으로 다음처럼 보면 된다.

| 이름 | args | 설명 |
|---|---|---|
| EAGLE-2 | `--speculative-algorithm EAGLE` | broad compatibility용 기본 EAGLE 경로 |
| EAGLE-3 | `--speculative-algorithm EAGLE3` | EAGLE-3 draft 모델 사용 |

중요:

```text
--speculative-algorithm EAGLE
```

이것은 EAGLE-3가 아니다. EAGLE-3를 쓰려면 반드시 다음처럼 명시한다.

```text
--speculative-algorithm EAGLE3
```

### EAGLE 실행 예시

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16 \
  --mem-fraction-static 0.7 \
  --cuda-graph-max-bs 8 \
  --log-level warning
```

각 옵션의 뜻:

| 옵션 | 쉬운 설명 |
|---|---|
| `--model` | 최종 답을 책임지는 target 모델 |
| `--speculative-algorithm EAGLE` | EAGLE 계열 speculative decoding 사용 |
| `--speculative-draft-model-path` | EAGLE draft 모델 checkpoint |
| `--speculative-num-steps 3` | draft를 autoregressive하게 3 step 앞까지 굴림 |
| `--speculative-eagle-topk 4` | 각 step에서 후보 token branch를 4개까지 봄 |
| `--speculative-num-draft-tokens 16` | target verify 단계에서 담을 수 있는 draft token 수 |
| `--mem-fraction-static 0.7` | SGLang static memory pool 비율. OOM이면 낮춤 |
| `--cuda-graph-max-bs 8` | CUDA graph capture의 최대 batch size 쪽 설정 |

### EAGLE3 실행 예시

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16 \
  --mem-fraction-static 0.7 \
  --cuda-graph-max-bs 8 \
  --dtype float16 \
  --log-level warning
```

EAGLE3는 EAGLE3용 draft checkpoint가 필요하다. EAGLE2용 draft 모델을 넣고
`--speculative-algorithm EAGLE3`로 돌리는 식으로 섞으면 안 된다.

## 5. EAGLE args를 이해하는 법

### `--speculative-num-steps`

draft가 몇 단계 앞까지 token을 만들어 볼지 정한다.

```text
num_steps=1 -> 한 단계만 미리 봄
num_steps=3 -> 세 단계 앞까지 미리 봄
num_steps=5 -> 더 멀리 봄
```

값을 키우면 한 번에 더 많이 accept될 가능성이 생긴다. 대신 draft 계산량과 reject cascade 위험도
커진다. 너무 크게 잡으면 오히려 느려질 수 있다.

### `--speculative-eagle-topk`

각 draft step에서 후보 branch를 몇 개 둘지 정한다.

```text
topk=1 -> 선형 후보 하나만 감
topk=4 -> step마다 여러 후보 branch를 만듦
```

`topk`가 커지면 target이 받아들일 후보를 찾을 확률이 올라갈 수 있다. 대신 tree가 커지고
메모리와 verify 비용이 늘어난다.

SpecV2 overlap scheduler를 쓰려면 현재 문서/코드 기준으로 `topk=1`이 안전하다.

### `--speculative-num-draft-tokens`

verify할 수 있는 draft token의 최대 개수다.

`topk=1`이면 SGLang이 다음 관계를 맞추려고 한다.

```text
speculative_num_draft_tokens = speculative_num_steps + 1
```

예:

```text
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

현재 코드에도 `topk == 1`이고 `num_draft_tokens != num_steps + 1`이면
`num_steps + 1`로 보정하는 로직이 있다.

### `topk > 1`이고 후보가 `num_draft_tokens`보다 많으면?

`topk > 1`에서는 draft가 선형 chain이 아니라 작은 tree처럼 후보를 만든다. 그래서 실제로 생성 가능한
후보 노드 수가 `--speculative-num-draft-tokens`보다 훨씬 많아질 수 있다.

예를 들어:

```text
--speculative-num-steps 3
--speculative-eagle-topk 4
--speculative-num-draft-tokens 8
```

이론적으로 완전한 tree를 pruning 없이 전부 펼치면 후보 수는 다음처럼 늘어난다.

```text
step 1 후보: 4개
step 2 후보: 4 x 4 = 16개
step 3 후보: 4 x 4 x 4 = 64개
```

다만 SGLang 구현은 완전한 tree를 끝까지 모두 materialize하지 않는다. 각 step 뒤에 누적 score 기준으로
active branch를 다시 `topk`개만 남기는 beam-style pruning을 한다.

그래서 실제 구현에서 step마다 확장해 비교하는 후보 수는 다음처럼 보면 된다.

```text
step 1: topk = 4개를 만든다.
step 2: active branch 4개 x child topk 4개 = 16개를 만들고, 다시 상위 4개만 남긴다.
step 3: active branch 4개 x child topk 4개 = 16개를 만들고, 다시 상위 4개만 남긴다.
```

즉 "완전한 tree 기준 후보 수"는 `4 + 16 + 64`처럼 커질 수 있지만, SGLang의 실제 draft loop에서
모으는 후보 노드는 대략 `4 + 16 + 16 = 36`개 쪽에 가깝다. 이후 verify 직전에 이 후보들 중
`speculative_num_draft_tokens - 1`개만 다시 고른다.

하지만 target verify에 넣을 수 있는 token 수는 `speculative_num_draft_tokens`로 제한된다.
그리고 verify 입력에는 이미 확정된 현재 token 1개가 포함된다.

따라서 실제 draft 후보는 다음 개수만 고른다.

```text
speculative_num_draft_tokens - 1
```

위 예시에서는:

```text
verify 전체 슬롯: 8개
현재 verified token: 1개
실제 draft 후보: 7개
```

SGLang은 이 7개를 **누적 draft score가 가장 높은 후보 노드**로 고른다. 단순히 앞 depth를 먼저
고르거나, 생성 순서대로 앞에서 자르는 방식이 아니다.

코드 흐름은 대략 이렇다.

```python
# 2번째 step부터 path score를 누적한다.
expand_scores = scores.unsqueeze(2) * topk_p.reshape(-1, topk, topk)

# 다음 step으로 이어갈 active branch는 topk개만 유지한다.
topk_cs_p, topk_cs_index = fast_topk(
    expand_scores.flatten(start_dim=1), topk, dim=-1
)

# verify에 넣을 최종 draft 후보는 전체 후보 score 중 num_draft_tokens - 1개만 고른다.
top_scores = torch.topk(
    score_list, speculative_num_draft_tokens - 1, dim=-1
)
```

여기서 `scores`는 path의 누적 확률에 가깝다.

```text
path_score = p(token1) * p(token2 | token1) * p(token3 | token1, token2) ...
```

즉 `topk > 1`일 때 SGLang은 다음 두 번의 pruning을 한다고 보면 된다.

1. **draft 진행 중 pruning**
   - 매 step마다 가능한 `topk * topk` 확장 중 누적 score 상위 `topk` branch만 다음 step으로 이어간다.
   - 관련 코드: `select_top_k_tokens()`

2. **verify 직전 pruning**
   - 지금까지 모은 후보 노드 전체에서 누적 score 상위 `speculative_num_draft_tokens - 1`개만 고른다.
   - 관련 코드: `organize_draft_results()` 또는 `draft_forward()`의 `torch.topk(...)`

후보를 고른 뒤에는 `top_scores_index`를 다시 sort한다. 이 sort는 score 순서로 verify하려는 목적이
아니라, 선택된 후보들을 tree 구조에 맞는 index 순서로 정리하기 위한 것이다.

그 다음 `build_tree_kernel_efficient()`가 다음 정보를 만든다.

| 값 | 의미 |
|---|---|
| `tree_mask` | 각 draft token이 어떤 ancestor token을 볼 수 있는지 나타내는 attention mask |
| `positions` | 각 draft token의 실제 position |
| `retrieve_index` | verify 결과에서 token을 찾아갈 index |
| `retrieve_next_token` | accept 후 다음 child 후보 |
| `retrieve_next_sibling` | 같은 parent 아래의 sibling 후보 |

검증 단계에서는 target model이 낸 token과 tree 후보를 비교한다. 현재 child가 target token과 맞으면
accept하고, 틀리면 sibling으로 넘어간다. 맞는 child/sibling이 없으면 그 지점에서 speculative accept가
끝나고 target token으로 보정한다.

부모가 선택되지 않은 child가 생기는 경우는 일반적으로 드물다. 누적 확률 곱 구조에서는 child score가
parent score보다 커지기 어렵기 때문이다. 그래도 NaN이나 tie 등으로 이상한 tree가 생기면 CUDA kernel
쪽에서 parent 없는 token을 경고하고 무시하는 방어 로직이 있다.

### `--speculative-draft-model-path`

EAGLE/EAGLE3에서는 보통 필수다. target 모델과 맞는 draft model을 지정해야 한다.

예:

```bash
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--speculative-algorithm EAGLE3 \
--speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B
```

MTP/NextN이 모델 내부에 포함된 경우에는 생략 가능한 모델들이 있다. 이 차이가 EAGLE과
MTP/NEXTN을 헷갈리게 만드는 핵심이다.

## 6. MTP와 NEXTN

MTP는 Multi-Token Prediction이다. 모델이 훈련될 때 "다음 토큰 하나"만 맞히는 것이 아니라,
"다음 여러 토큰"을 함께 예측하는 보조 목표나 모듈을 둔다.

MTP 논문에서는 shared trunk 위에 여러 output head를 두고, 각 위치에서 앞으로의 `n`개 토큰을
예측하도록 학습한다. DeepSeek-V3 같은 모델은 checkpoint 안에 MTP module을 포함한다.

### SGLang에서 MTP는 별도 algorithm인가?

현재 코드 기준으로는 별도 `--speculative-algorithm MTP`가 없다.

대신 MTP/NextN 모듈을 speculative decoding의 draft source처럼 쓰고, 실행 경로는 EAGLE 계열을
재사용한다.

그래서 MTP 예제도 보통 이렇게 생겼다.

```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --tp 8 \
  --trust-remote-code \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

### NEXTN은 무엇인가

`NEXTN`은 "다음 N개 토큰 예측"이라는 의미로 MTP/NextN layer를 가리키는 이름이다.

SGLang CLI choices에는 `NEXTN`이 있다.

```text
DFLASH
EAGLE
EAGLE3
NEXTN
STANDALONE
NGRAM
```

하지만 현재 `server_args.py`에서는 다음처럼 처리한다.

```python
if self.speculative_algorithm == "NEXTN":
    self.speculative_algorithm = "EAGLE"
```

즉 `NEXTN`은 사용자가 의도를 드러내기 위한 alias에 가깝고, 실제 내부 worker 선택은 EAGLE 쪽으로
간다.

요약:

```text
--speculative-algorithm NEXTN
-> server args 처리 중 EAGLE로 변환
-> MTP/NextN layer를 EAGLE speculative decoding 경로에서 사용
```

### DeepSeek MTP 실행 예시

```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --tp 8 \
  --trust-remote-code \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

옵션 설명:

| 옵션 | 쉬운 설명 |
|---|---|
| `--model-path deepseek-ai/DeepSeek-V3-0324` | MTP module이 들어 있는 target checkpoint |
| `--tp 8` 또는 `--tp-size 8` | tensor parallel size. 이 checkout에서는 `--tp-size`가 더 안전할 수 있음 |
| `--trust-remote-code` | HF remote code가 필요한 모델에서 사용 |
| `--speculative-algorithm EAGLE` | MTP를 EAGLE speculative 경로로 사용 |
| `--speculative-num-steps 3` | 3 step ahead draft |
| `--speculative-eagle-topk 1` | linear draft. MTP에서는 보통 이 값이 안정적 |
| `--speculative-num-draft-tokens 4` | `num_steps + 1`에 맞춘 verify capacity |

DeepSeek류 MTP는 별도의 `--speculative-draft-model-path`가 꼭 필요하지 않은 경우가 많다.
현재 코드에서는 일부 architecture에 대해 draft path가 없으면 target model path를 draft path로
자동 설정한다.

### Qwen3.x / Qwen3.6 계열 NEXTN 예시

이 repository의 다른 문서인 `STUDY/qwen36_35b_a3b_mtp_sglang_guide.md`에 더 자세한 실행
가이드가 있다. 여기서는 핵심만 다시 적는다.

```bash
SGLANG_ENABLE_SPEC_V2=1 python -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-35B-A3B-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --tp-size 1 \
  --mem-fraction-static 0.8 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mamba-scheduler-strategy extra_buffer \
  --page-size 64
```

주의:

- 문서나 모델 카드에서 `--speculative-algo NEXTN`처럼 짧은 alias가 보일 수 있다.
- 현재 checkout의 argparse를 코드로 확인하면 안전한 이름은 `--speculative-algorithm`이다.
- 실행이 안 되면 `--speculative-algo` 대신 `--speculative-algorithm`을 먼저 사용한다.

## 7. DFlash

DFlash는 EAGLE/MTP와 성격이 꽤 다르다. EAGLE/MTP는 대체로 autoregressive하게 후보를 늘리거나
tree를 만들지만, DFlash는 lightweight block diffusion draft model이 한 번에 token block을 만든다.

DFlash 논문의 설명을 쉽게 말하면 다음과 같다.

```text
기존 speculative decoding:
  draft도 여러 token을 순서대로 만든다.

DFlash:
  draft 모델이 diffusion-style로 token block을 병렬 생성한다.
  target 모델은 그 block을 검증한다.
```

SGLang의 DFlash draft model 구현도 일반 CausalLM과 다르다.

- draft model 자체에는 token embedding과 lm_head가 없다.
- target model의 embedding/lm_head를 사용한다.
- target model의 여러 hidden feature를 받아 draft hidden으로 projection한다.
- DFlash attention은 draft block 안에서 non-causal attention을 사용한다.

현재 코드의 `python/sglang/srt/models/dflash.py` 상단에도 DFlash가 target embedding/lm_head를
사용한다고 적혀 있다.

### DFlash 실행 예시

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat \
  --speculative-dflash-block-size 16
```

옵션 설명:

| 옵션 | 쉬운 설명 |
|---|---|
| `--speculative-algorithm DFLASH` | DFlash speculative decoding 경로 사용 |
| `--speculative-draft-model-path` | DFlash 전용 draft checkpoint. 필수 |
| `--speculative-dflash-block-size` | 한 번에 verify할 linear block 길이 |
| `--speculative-num-draft-tokens` | DFlash에서는 block size와 같은 의미로 취급 가능 |
| `--speculative-dflash-draft-window-size` | DFlash draft KV cache sliding window 크기 |

`--speculative-dflash-block-size`와 `--speculative-num-draft-tokens`를 둘 다 주면 값이 같아야 한다.
다르면 현재 코드 기준으로 error가 난다.

### DFlash 제약

현재 코드 기준 DFlash는 제약이 있다.

| 제약 | 이유 또는 의미 |
|---|---|
| `--speculative-draft-model-path` 필수 | DFlash 전용 draft checkpoint가 필요 |
| `--enable-dp-attention` 미지원 | 코드에서 명시적으로 error |
| `pp_size == 1` 필요 | pipeline parallel과 함께 쓰지 못함 |
| SpecV2 overlap scheduler 미지원 | DFlash 사용 시 overlap scheduler를 끔 |
| mixed chunked prefill 비활성화 | DFlash 사용 시 코드가 끔 |
| `speculative_num_steps`는 1로 강제 | DFlash의 자연 단위는 step tree가 아니라 block |
| `speculative_eagle_topk`는 1로 강제 | DFlash verify는 linear block 중심 |

그래서 DFlash는 EAGLE처럼 `num_steps`, `topk`를 튜닝하는 느낌보다는, DFlash draft checkpoint와
block size/window를 맞추는 느낌으로 접근하는 편이 좋다.

## 8. 세 방식의 차이점

| 비교 기준 | EAGLE | MTP/NEXTN | DFlash |
|---|---|---|---|
| draft source | 별도 EAGLE draft model | target checkpoint 내부 MTP/NextN layer | 별도 DFlash draft model |
| 별도 draft checkpoint | 보통 필요 | 모델에 내장되어 있으면 불필요 | 필수 |
| 내부 실행 경로 | EAGLE worker | 현재 SGLang에서는 EAGLE 경로 재사용 | DFlash worker |
| 후보 구조 | tree 또는 linear chain | 보통 linear chain에 가깝게 사용 | linear block |
| 대표 args | `EAGLE`, `EAGLE3` | `EAGLE` 또는 `NEXTN` | `DFLASH` |
| 주 튜닝값 | `num_steps`, `topk`, `num_draft_tokens` | `num_steps`, `topk=1`, `num_draft_tokens` | `block_size`, `draft_window_size` |
| SpecV2 | 가능, 단 `topk=1` | 가능할 수 있음, 보통 `topk=1` | 현재 미지원 |
| 장점 | 범용적이고 문서/구현 성숙 | 모델 내장 MTP면 추가 draft model 부담 감소 | block 병렬 draft로 큰 가속 가능성 |
| 주의점 | target과 맞는 draft model 필요 | 모델별 지원 여부 확인 필요 | 제약 많고 matching DFlash checkpoint 필요 |

## 9. 어떤 것을 고르면 좋나

### EAGLE/EAGLE3를 고르는 경우

- target 모델에 맞는 EAGLE draft checkpoint가 있다.
- SGLang 문서나 Hugging Face에 해당 모델용 EAGLE/EAGLE3 draft가 공개되어 있다.
- 범용적인 speculative decoding을 먼저 시도하고 싶다.
- EAGLE3 draft가 있다면 EAGLE3부터 고려할 만하다.

### MTP/NEXTN을 고르는 경우

- target 모델 자체가 MTP/NextN weight를 포함한다.
- DeepSeek V3/V3.2/R1, Qwen3.x Next/MTP 계열처럼 문서에 MTP 예시가 있다.
- 별도 draft checkpoint 없이 모델 내장 speculative path를 쓰고 싶다.

이 경우 처음에는 보통 아래 조합이 무난하다.

```bash
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

### DFlash를 고르는 경우

- target 모델에 맞는 DFlash draft checkpoint가 있다.
- DFlash 문서나 HF 모델 카드가 target/draft 조합을 명확히 안내한다.
- DP attention, pipeline parallel, SpecV2 overlap scheduler 같은 제약을 피할 수 있다.

## 10. 실전 튜닝 순서

처음부터 많은 값을 바꾸지 말고 다음 순서로 보는 것이 좋다.

### 1단계: 기본 조합으로 켜기

EAGLE/MTP:

```bash
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

EAGLE tree를 실험하고 싶을 때:

```bash
--speculative-num-steps 3 \
--speculative-eagle-topk 4 \
--speculative-num-draft-tokens 16
```

DFlash:

```bash
--speculative-dflash-block-size 16
```

### 2단계: acceptance length 보기

SGLang 로그나 server info에서 speculative acceptance length 관련 값을 확인한다.

감각:

```text
acceptance length가 높다 -> draft가 잘 맞고 있음
acceptance length가 낮다 -> draft 비용만 들고 이득이 작을 수 있음
```

### 3단계: batch size와 workload에 맞게 조정

Speculative decoding은 모든 workload에서 무조건 빨라지는 스위치가 아니다.

- 짧은 응답, 작은 batch, deterministic output에서는 이득이 클 수 있다.
- 매우 큰 batch에서는 target verify 비용과 scheduler 상황에 따라 이득이 줄 수 있다.
- creative sampling처럼 불확실성이 큰 workload에서는 acceptance가 낮아질 수 있다.

### 4단계: OOM이면 메모리부터 줄이기

자주 조정하는 값:

```bash
--mem-fraction-static 0.7
--cuda-graph-max-bs 8
--max-running-requests 48
```

EAGLE tree에서 OOM이면 먼저 `topk`와 `num_draft_tokens`를 낮춘다.

```bash
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

## 11. Args cheat sheet

### 공통

| args | 설명 |
|---|---|
| `--speculative-algorithm` | speculative 방식 선택 |
| `--speculative-draft-model-path` | draft checkpoint 경로 |
| `--speculative-num-steps` | draft depth |
| `--speculative-eagle-topk` | step별 branch 수 |
| `--speculative-num-draft-tokens` | verify 가능한 draft token 수 |
| `--speculative-draft-attention-backend` | draft model attention backend override |
| `--speculative-draft-model-quantization` | draft model quantization override |
| `--speculative-draft-load-format` | draft weight load format |
| `--speculative-token-map` | FR-Spec high-frequency token map. EAGLE-2용 |

### EAGLE/EAGLE3

| args | 추천 시작점 |
|---|---|
| `--speculative-algorithm EAGLE` | EAGLE-2 |
| `--speculative-algorithm EAGLE3` | EAGLE-3 |
| `--speculative-draft-model-path` | target에 맞는 EAGLE draft |
| `--speculative-num-steps` | `3` |
| `--speculative-eagle-topk` | `1` 또는 `4` |
| `--speculative-num-draft-tokens` | `4` 또는 `16` |

### MTP/NEXTN

| args | 추천 시작점 |
|---|---|
| `--speculative-algorithm EAGLE` | 문서와 코드 모두에서 안전한 MTP 경로 |
| `--speculative-algorithm NEXTN` | MTP 의도를 드러내는 alias. 내부적으로 EAGLE로 변경 |
| `--speculative-num-steps` | `3` |
| `--speculative-eagle-topk` | `1` |
| `--speculative-num-draft-tokens` | `4` |
| `--speculative-draft-model-path` | 모델에 내장 MTP가 있으면 생략 가능할 수 있음 |

### DFlash

| args | 추천 시작점 |
|---|---|
| `--speculative-algorithm DFLASH` | DFlash 사용 |
| `--speculative-draft-model-path` | 필수 |
| `--speculative-dflash-block-size` | `16`부터 시작 |
| `--speculative-dflash-draft-window-size` | 필요할 때만 설정 |
| `--speculative-num-steps` | 내부적으로 `1`로 강제 |
| `--speculative-eagle-topk` | 내부적으로 `1`로 강제 |

## 12. 자주 헷갈리는 질문

### `EAGLE`은 EAGLE3인가?

아니다.

```bash
--speculative-algorithm EAGLE
```

은 EAGLE-2 계열로 보면 된다.

EAGLE3는 이렇게 명시한다.

```bash
--speculative-algorithm EAGLE3
```

### `NEXTN`과 `EAGLE`은 같은가?

사용자 args로는 다르지만, 현재 코드에서는 `NEXTN`이 `EAGLE`로 바뀐다.

그래서 SGLang 내부 실행 경로만 보면 사실상 EAGLE speculative path를 탄다. 다만 문서나
운영 스크립트에서 `NEXTN`이라고 쓰면 "이 모델의 MTP/NextN layer를 쓰는 의도"가 더 잘 드러난다.

### MTP는 draft model이 필요 없나?

모델에 MTP/NextN layer가 들어 있으면 별도 draft checkpoint 없이 쓸 수 있는 경우가 많다.
하지만 모든 모델이 그런 것은 아니다. 모델 config와 SGLang 지원 여부를 확인해야 한다.

### DFlash도 EAGLE args를 튜닝하나?

아니다. DFlash는 `num_steps`, `topk`를 EAGLE처럼 튜닝하는 방식이 아니다. 현재 코드 기준으로
DFlash 사용 시 `num_steps=1`, `topk=1`로 강제된다. DFlash에서는 `block_size`와
`draft_window_size`가 더 중요하다.

### SpecV2는 언제 켜나?

EAGLE/EAGLE3/MTP 쪽에서 overlap scheduler를 쓰고 싶을 때 켠다.

```bash
SGLANG_ENABLE_SPEC_V2=1
```

단, `--speculative-eagle-topk 1`로 맞추는 것이 중요하다. DFlash는 현재 SpecV2를 지원하지 않는다.

## 13. 관련 STUDY 문서

- `STUDY/sglang_mtp_eagle_speculative_decoding_guide.md`
  - MTP와 EAGLE의 관계, `NEXTN` alias, 자동 기본값을 더 코드 중심으로 정리한 문서
- `STUDY/qwen36_35b_a3b_mtp_sglang_guide.md`
  - Qwen3.6 35B-A3B MTP 실행에 초점을 맞춘 문서
- `STUDY/dflash_speculative_decoding.md`
  - DFlash를 더 깊게 보는 문서
- `STUDY/sharegpt_bench_serving_env_args_guide.md`
  - serving benchmark args를 정리한 문서

## 14. 마지막 기억법

외울 것은 많아 보이지만, 아래 네 줄이면 대부분 정리된다.

```text
Speculative decoding = 초안 만들고 target이 검증하는 전체 기법
EAGLE/EAGLE3 = 별도 EAGLE draft 모델을 붙이는 방식
MTP/NEXTN = 모델 안의 next-n 예측 모듈을 draft처럼 쓰는 방식
DFlash = diffusion draft 모델이 token block을 병렬로 쓰는 방식
```

실행 args는 이렇게 시작하면 된다.

```bash
# EAGLE/EAGLE3
--speculative-algorithm EAGLE3 \
--speculative-draft-model-path <eagle3-draft> \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4

# MTP/NEXTN
--speculative-algorithm NEXTN \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4

# DFlash
--speculative-algorithm DFLASH \
--speculative-draft-model-path <dflash-draft> \
--speculative-dflash-block-size 16
```
