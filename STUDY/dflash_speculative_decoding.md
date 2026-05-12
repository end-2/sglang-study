# DFlash speculative decoding 쉽게 이해하기

검증 기준:

- 논문: [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- Hugging Face paper page: [huggingface.co/papers/2602.06036](https://huggingface.co/papers/2602.06036)
- DFlash GitHub: [z-lab/dflash](https://github.com/z-lab/dflash)
- DFlash HF draft model 예시: [z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16)
- vLLM Speculators 문서: [DFlash](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)
- 현재 SGLang 코드:
  - `python/sglang/srt/speculative/dflash_worker.py`
  - `python/sglang/srt/speculative/dflash_info.py`
  - `python/sglang/srt/speculative/dflash_utils.py`
  - `python/sglang/srt/models/dflash.py`
  - `python/sglang/srt/server_args.py`

## 1. 한 줄 요약

DFlash는 큰 LLM이 한 토큰씩 느리게 생성하는 문제를 줄이기 위해, 작은 block diffusion draft model이 다음 토큰 블록을 한 번에 제안하고, 큰 target LLM이 그 제안을 검증하는 speculative decoding 방법이다.

핵심은 이것이다.

```text
기존 방식:
큰 target LLM이 다음 토큰을 1개씩 직접 생성

일반 speculative decoding:
작은 draft model이 여러 토큰을 제안하고, 큰 target LLM이 검증

DFlash:
작은 diffusion-style draft model이 여러 토큰을 병렬로 한 번에 제안하고,
큰 target LLM이 검증
```

즉 DFlash는 최종 답변 품질을 작은 diffusion model에 맡기는 방법이 아니다. 최종 출력은 target LLM이 검증해서 통과시킨 토큰으로만 진행한다.

## 2. 왜 speculative decoding이 필요한가

LLM의 일반적인 autoregressive decoding은 이런 구조다.

```text
prompt -> token 1 생성
prompt + token 1 -> token 2 생성
prompt + token 1 + token 2 -> token 3 생성
...
```

각 토큰이 이전 토큰에 의존하므로 순차 실행이 강제된다. GPU는 큰 병렬 계산에 강한데, decode 단계에서는 매번 작은 작업을 반복하게 되어 GPU 활용률이 낮아지기 쉽다.

Speculative decoding은 이 문제를 우회한다.

```text
작은 draft model: "아마 다음 토큰들은 A, B, C, D일 것 같아"
큰 target model: "A, B는 맞고 C부터는 틀렸어. 대신 X를 쓰자"
```

큰 모델이 토큰을 직접 하나씩 생성하는 대신, 작은 모델이 먼저 여러 후보를 내고 큰 모델은 한 번의 병렬 검증으로 여러 위치의 logits를 계산한다. 후보가 많이 맞으면 한 번의 target forward로 여러 토큰을 확정할 수 있다.

## 3. 기존 speculative decoding의 병목

일반적인 draft model도 대개 autoregressive 방식으로 다음 토큰들을 만든다.

```text
draft token 1 생성
draft token 2 생성
draft token 3 생성
draft token 4 생성
...
```

큰 target model보다 작아서 빠르긴 하지만, draft 단계 자체가 여전히 순차적이다. speculation budget을 늘릴수록 draft 비용도 같이 늘어난다.

DFlash 논문의 문제의식은 여기서 시작한다. draft model이 여러 토큰을 순차적으로 만들지 말고, diffusion-style block model처럼 한 블록을 한 번에 만들 수 있으면 draft 비용을 크게 줄일 수 있다.

## 4. DFlash의 핵심 아이디어

DFlash는 작은 block diffusion model을 draft model로 쓴다. 이 draft model은 다음 토큰들을 한 칸씩 순서대로 생성하지 않고, mask로 비워 둔 block 전체를 병렬로 예측한다.

예를 들어 block size가 16이면, SGLang 구현 기준으로 첫 칸은 이미 검증된 현재 토큰이고 나머지 15칸이 draft 후보 위치다.

```text
[현재 확정 토큰, MASK, MASK, MASK, ..., MASK]
```

DFlash draft model은 이 입력을 보고 나머지 칸을 한 번의 forward에서 채우려고 한다.

```text
[현재 확정 토큰, d1, d2, d3, ..., d15]
```

여기서 중요한 보강 장치가 target model의 hidden state다. 작은 draft model 혼자 prompt를 이해해서 미래 토큰을 맞히게 하면 품질이 낮다. DFlash는 target model이 이미 계산한 여러 layer의 hidden state를 가져와 draft model에 조건 정보로 넣는다.

논문 표현으로는 target model의 context feature를 draft model에 condition한다. SGLang 구현에서는 이 feature를 projection한 뒤 draft model layer들의 KV cache에 주입한다.

## 5. DFlash decoding 흐름

전체 흐름을 쉽게 풀면 다음과 같다.

### 5.1 Target prefill

처음에는 target LLM이 prompt를 읽는다. 이때 target model은 다음 토큰 하나를 만들고, 동시에 여러 layer의 hidden state를 남긴다.

```text
prompt
  -> target LLM prefill
  -> 첫 verified token
  -> target hidden states
```

SGLang에서는 target model이 DFlash용 hidden state capture를 지원해야 한다. 예를 들어 Qwen3 계열에는 `set_dflash_layers_to_capture`가 구현되어 있다.

### 5.2 Target hidden feature fusion

DFlash는 target model의 여러 layer hidden state를 그대로 쓰지 않고, concat한 뒤 작은 projection layer로 draft model hidden size에 맞춘다.

```text
여러 target layer hidden states
  -> concat
  -> fc projection
  -> draft context feature
```

SGLang의 `DFlashDraftModel`에는 이 역할을 하는 `fc`와 `hidden_norm`이 있다. 현재 구현의 draft model은 자체 embedding과 lm_head를 갖지 않고 target model의 embedding/lm_head를 재사용한다.

### 5.3 KV injection

DFlash의 큰 차별점은 target context feature를 draft model 입력에 한 번 섞고 끝내는 것이 아니라, draft model 각 layer의 KV cache 쪽에 주입한다는 점이다.

이렇게 하면 draft model의 깊이가 깊어져도 target feature 정보가 중간 layer에서 희석되지 않는다. 논문은 이 구조가 acceptance length를 늘리는 데 중요하다고 설명한다.

SGLang에서는 `dflash_worker.py`의 `_append_target_hidden_to_draft_kv`가 이 일을 한다. 내부적으로 draft model의 `project_target_hidden`을 호출한 뒤, 각 draft layer의 K/V를 materialize해서 draft KV cache에 쓴다.

### 5.4 Parallel draft

이제 draft model은 다음과 같은 block 입력을 받는다.

```text
[verified token, MASK, MASK, MASK, ...]
```

그리고 non-causal attention을 사용한다. 일반 causal attention은 뒤쪽 토큰을 보지 못하지만, DFlash draft block 안에서는 mask token embedding들이 서로를 함께 보며 병렬로 denoise하듯 예측한다.

SGLang의 `DFlashAttention`은 `AttentionType.ENCODER_ONLY`를 사용한다. 코드 주석에도 DFlash가 draft block에 non-causal attention을 쓴다고 되어 있다.

### 5.5 Target verify

Draft model이 후보를 만들면 target model이 검증한다.

예를 들어 DFlash가 이렇게 제안했다고 하자.

```text
c0 = 이미 확정된 현재 토큰
draft block = [c0, d1, d2, d3, d4]
```

Target model은 이 block을 causal 방식으로 보고 각 위치의 다음 토큰 예측을 낸다.

```text
target predictions = [t1, t2, t3, t4, t5]
```

검증 규칙은 앞에서부터 비교하는 것이다.

```text
d1 == t1 이면 d1 accept
d2 == t2 이면 d2 accept
d3 != t3 이면 여기서 stop
그리고 target이 낸 t3를 bonus token으로 append
```

즉 후보가 틀리는 순간부터는 draft를 버리고 target token으로 다시 이어간다.

SGLang의 greedy verify rule은 `compute_dflash_accept_len_and_bonus`에 있다. 코드 기준으로는 `candidates[:, 1:] == target_predict[:, :-1]`가 연속으로 맞는 길이를 accept length로 계산하고, mismatch 위치의 target prediction을 bonus token으로 붙인다.

## 6. 왜 lossless acceleration인가

Lossless라는 말은 draft model이 항상 맞는다는 뜻이 아니다. Draft model은 틀릴 수 있다.

대신 최종으로 append되는 토큰은 다음 둘 중 하나다.

1. target model이 같은 위치에서 예측한 것과 일치한 draft token
2. draft가 틀린 위치에서 target model이 직접 낸 bonus token

그래서 greedy decoding에서는 target model이 혼자 생성했을 때와 같은 결과를 유지할 수 있다. Sampling에서도 올바른 speculative sampling 검증을 쓰면 target 분포를 보존하는 것이 목표다.

주의할 점은 구현별 지원 수준이다. 현재 SGLang 코드에는 non-greedy DFlash verify용 `sgl_kernel` 경로가 있고, 해당 경로를 사용할 수 없으면 warning을 내고 greedy argmax verify로 fallback한다. 따라서 sampling 설정에서의 정확한 동작은 사용 빌드와 디바이스 지원을 확인해야 한다.

## 7. DFlash가 빠른 이유

DFlash가 빠른 이유는 단순히 draft model이 작기 때문만은 아니다.

가장 중요한 차이는 draft 비용의 모양이다.

```text
autoregressive draft:
토큰 1개 생성 비용 x draft 토큰 수

DFlash draft:
block 전체를 한 번의 parallel forward로 생성
```

그래서 block size를 어느 정도 키워도 draft latency가 선형으로 늘지 않는다. 그 덕분에 DFlash는 EAGLE류보다 깊은 draft model을 쓰면서도 낮은 draft latency를 유지할 수 있다.

논문은 Qwen3 계열 등 여러 실험에서 DFlash가 6x 이상 lossless acceleration을 달성하고, EAGLE-3보다 최대 2.5x 높은 speedup을 보였다고 보고한다. 다만 이 수치는 실험 환경, model, backend, batch size, decoding 설정에 따라 달라진다.

## 8. SGLang 코드에서의 구현 위치

현재 repository에서 DFlash를 볼 때 중요한 파일은 아래와 같다.

| 파일 | 역할 |
|---|---|
| `python/sglang/srt/speculative/dflash_worker.py` | target worker와 draft worker를 연결하는 orchestration |
| `python/sglang/srt/speculative/dflash_info.py` | DFlash draft/verify 상태 객체와 verify 후 commit 처리 |
| `python/sglang/srt/speculative/dflash_utils.py` | DFlash config parsing, accept length 계산, sampling verify helper |
| `python/sglang/srt/models/dflash.py` | SGLang용 DFlash draft model 구현 |
| `python/sglang/srt/server_args.py` | `--speculative-algorithm DFLASH` 관련 인자 검증과 제약 |
| `python/sglang/srt/models/qwen3.py` 등 | target model의 DFlash hidden state capture 구현 |

특히 `DFlashDraftModel`은 자체 embedding/lm_head가 없다.

```text
target embedding -> draft input embedding
draft hidden -> target lm_head -> draft token id
```

이 구조는 draft model을 target model에 강하게 맞춘 adapter처럼 만든다. 그래서 HF의 DFlash model card도 draft model을 target model과 함께 사용해야 한다고 설명한다.

## 9. SGLang에서 쓰는 주요 인자

가장 기본 형태는 다음과 같다.

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3-4B-DFlash-b16 \
  --tp-size 1 \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --mem-fraction-static 0.75 \
  --trust-remote-code
```

중요한 인자:

| 인자 | 의미 |
|---|---|
| `--speculative-algorithm DFLASH` | DFlash speculative decoding 사용 |
| `--speculative-draft-model-path` | DFlash draft checkpoint 경로 |
| `--speculative-dflash-block-size` | DFlash verify window length. SGLang에서는 `--speculative-num-draft-tokens`의 alias |
| `--speculative-num-draft-tokens` | DFlash block size로 사용됨 |
| `--speculative-dflash-draft-window-size` | draft model KV cache가 볼 sliding window 크기 |
| `--speculative-draft-attention-backend` | draft worker attention backend 지정 가능 |

주의할 점:

- SGLang DFlash의 block size는 첫 verified token까지 포함하는 window length로 이해하는 것이 안전하다.
- 예를 들어 block size가 16이면, 실제 미래 draft 후보는 보통 15개이고, verify가 모두 맞으면 target bonus token까지 포함해 한 round에 최대 16개 새 토큰이 append될 수 있다.
- DFlash draft worker는 non-causal attention이 필요하므로 모든 attention backend가 맞는 것은 아니다.

### 9.1 왜 DFlash에는 step args가 아니라 draft/block args가 중요한가

SGLang의 speculative decoding 인자 이름을 보면 `--speculative-num-steps`와
`--speculative-num-draft-tokens`가 모두 있다. EAGLE/MTP를 먼저 보면 둘 다 중요해
보이지만, DFlash에서는 의미가 다르다.

핵심 차이는 draft를 만드는 방식이다.

| 방식 | `steps`가 뜻하는 것 | draft token 수가 뜻하는 것 |
|---|---|---|
| EAGLE/MTP 계열 | draft를 몇 번 순차 확장할지 | step/tree 안에서 몇 개 후보를 관리할지 |
| DFlash | 항상 1개 block forward라 사실상 1 | 한 번에 만들고 검증할 block/window 크기 |

EAGLE/MTP 계열은 다음 후보를 여러 step으로 확장한다. 그래서 `num_steps=4`라면
draft model을 step별로 굴리며 후보를 이어 붙이는 구조가 된다.

```text
step 1 -> 후보 확장
step 2 -> 후보 확장
step 3 -> 후보 확장
step 4 -> target verify
```

DFlash는 이 구조가 아니다. DFlash는 block diffusion draft model이 mask로 채워진
block 전체를 한 번의 forward에서 예측한다.

```text
[verified token, MASK, MASK, MASK, ...]
  -> DFlash draft 1 forward
  -> [verified token, d1, d2, d3, ...]
  -> target verify
```

그래서 DFlash에서 사용자가 조절해야 하는 자연스러운 단위는 `step`이 아니라
`block_size`다. SGLang 코드도 이 설계를 그대로 따른다.

- `server_args.py`에서 DFlash는 EAGLE-style `num_steps`와 `topk`를 쓰지 않는다고
  보고, `speculative_num_steps`와 `speculative_eagle_topk`를 1로 강제한다.
- 같은 코드에서 DFlash의 자연스러운 단위는 `block_size`, 즉 verify window
  length라고 주석으로 설명한다.
- `dflash_worker.py`에서는 `[verified_token, MASK, MASK, ...]` 형태의 `block_ids`를
  만들고, block 전체를 한 번에 draft model forward로 통과시킨다.
- 그 뒤 `draft_hidden[:, 1:, :]`만 target LM head에 넣어 실제 미래 draft 후보를
  만든다. 첫 칸은 이미 verified token이기 때문이다.

따라서 DFlash 실행에서 `--speculative-num-steps`를 크게 주는 것은 의미 있는 튜닝이
아니다. SGLang은 값을 1로 되돌린다. 대신 아래 인자들이 실제 DFlash 튜닝 포인트다.

```text
--speculative-draft-model-path
--speculative-num-draft-tokens
--speculative-dflash-block-size
--speculative-dflash-draft-window-size
--speculative-draft-attention-backend
```

실전 감각으로는 이렇게 기억하면 된다.

```text
EAGLE/MTP:
  "몇 step 앞까지 순차적으로 뻗어볼까?"

DFlash:
  "한 번의 병렬 draft에서 몇 칸짜리 block을 채워볼까?"
```

즉 DFlash에서 `draft args만 있다`고 느껴지는 이유는 누락이 아니라 설계 차이다.
step을 늘리는 대신, 한 번의 draft forward가 담당하는 block/window 크기를 조절한다.

## 10. 현재 SGLang 구현상의 제약

현재 코드 기준 주요 제약은 다음과 같다.

- `--speculative-draft-model-path`가 필수다.
- DP attention은 지원하지 않는다.
- pipeline parallelism은 `pp_size == 1`만 지원한다.
- DFlash는 EAGLE-style tree가 아니므로 `speculative_num_steps`와 `speculative_eagle_topk`는 1로 강제된다.
- spec-v1 기준 overlap scheduler는 꺼진다.
- mixed chunked prefill도 DFlash 사용 시 꺼진다.
- target model은 `set_dflash_layers_to_capture`를 구현해야 한다.
- return logprob와 grammar constraint는 현재 scheduler/worker 쪽에서 지원하지 않는 경로로 처리된다.

이 제약들은 `server_args.py`, `scheduler.py`, `dflash_worker.py`에 분산되어 있다. 성능 테스트나 기능 추가를 할 때는 먼저 이 부분을 확인하는 것이 좋다.

## 11. DFlash를 머릿속에 그리는 비유

일반 decoding은 선생님이 답안을 한 글자씩 직접 쓰는 것이다.

일반 speculative decoding은 조교가 답안을 몇 글자 미리 써 오면 선생님이 빠르게 채점하는 것이다. 그런데 조교도 글자를 한 글자씩 쓰면 여전히 시간이 걸린다.

DFlash는 조교가 빈칸 15개짜리 답안지를 한 번에 채워 오는 방식이다. 선생님은 그 답안지를 앞에서부터 확인한다. 맞은 부분은 그대로 쓰고, 틀린 첫 지점에서는 선생님 답으로 고친다.

여기서 조교가 아무 정보 없이 쓰는 것이 아니라, 선생님이 방금 문제를 풀며 만든 중간 풀이 노트를 보고 답안을 채운다. 이 중간 풀이 노트가 target hidden state다.

## 12. 자주 헷갈리는 점

### DFlash는 target LLM을 diffusion model로 바꾸는 것인가

아니다. Target LLM은 그대로 autoregressive LLM이다. DFlash는 draft stage에만 diffusion-style block drafter를 넣는다.

### Draft model이 틀리면 품질이 떨어지는가

검증이 올바르게 구현되어 있으면, 틀린 draft는 accept되지 않는다. 많이 틀리면 속도 이득이 줄어들 뿐이다.

### DFlash draft checkpoint는 아무 target model에나 붙일 수 있는가

보통 그렇지 않다. Draft checkpoint는 특정 target model의 hidden feature, embedding, lm_head와 맞도록 훈련된다. 예를 들어 `z-lab/Qwen3-4B-DFlash-b16`은 `Qwen/Qwen3-4B`와 함께 쓰는 drafter로 설명되어 있다.

### 왜 target hidden state가 중요한가

작은 draft model이 prompt만 보고 미래 토큰을 맞히기는 어렵다. Target hidden state에는 target model이 문맥을 처리하며 만든 풍부한 정보가 들어 있다. DFlash는 이 정보를 draft model의 KV cache에 넣어, 작은 model이 target model의 판단에 가까운 후보를 만들도록 돕는다.

### DFlash의 `block_size`와 vLLM의 `num_speculative_tokens`는 같은가

항상 같은 의미로 보면 위험하다. SGLang 코드에서는 DFlash block size가 verify window length로 쓰이며 첫 verified token을 포함한다. 그래서 `block_size=16`이면 미래 후보는 보통 15개다. 다른 framework는 "speculative tokens"를 미래 후보 개수로 표현할 수 있으므로 문서를 확인해야 한다.

## 13. 더 읽을 자료

- 논문: [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- 프로젝트/코드: [z-lab/dflash](https://github.com/z-lab/dflash)
- HF paper page: [DFlash on Hugging Face Papers](https://huggingface.co/papers/2602.06036)
- HF draft model 예시: [z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16)
- vLLM Speculators 설명: [DFlash algorithm docs](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)
