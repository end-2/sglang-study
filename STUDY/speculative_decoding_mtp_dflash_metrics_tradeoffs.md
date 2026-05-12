# Speculative Decoding MTP/DFlash 지표와 오버헤드 정리

이 문서는 speculative decoding을 MTP/NEXTN 또는 DFlash로 사용할 때 실제 서빙 지표가
어떻게 바뀌는지 정리한 노트다. 핵심 관심사는 `TTFT`, `latency`, `ITL/TPOT`,
`throughput`, GPU compute/memory resource, concurrency 변화다.

검증 기준:

- Speculative Decoding 원 논문: <https://arxiv.org/abs/2211.17192>
- MTP 논문: <https://arxiv.org/abs/2404.19737>
- DFlash 논문: <https://arxiv.org/abs/2602.06036>
- SGLang speculative decoding 문서:
  <https://docs.sglang.io/docs/advanced_features/speculative_decoding>
- SGLang DeepSeek V3 문서:
  <https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/deepseek_v3.md>
- SpecDecode-Bench:
  <https://specdecode-bench.github.io/>
- DeepSeek-V3 Hugging Face model card:
  <https://huggingface.co/deepseek-ai/DeepSeek-V3>
- DFlash Qwen3.6-35B-A3B Hugging Face model card:
  <https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash>
- 현재 repository 코드:
  - `python/sglang/bench_serving.py`
  - `docs/developer_guide/bench_serving.md`
  - `python/sglang/srt/server_args.py`
  - `python/sglang/srt/speculative/spec_info.py`
  - `python/sglang/srt/models/dflash.py`

## 1. 먼저 지표 이름부터 정리

| 지표 | 의미 | speculative decoding에서 기대 효과 |
|---|---|---|
| `TTFT` | Time To First Token. 요청 시작부터 첫 토큰까지 시간 | 보통 큰 개선 대상은 아님. prefill, queue, scheduler 영향이 큼 |
| `E2E latency` | 요청 전체 완료 시간 | 출력 길이가 길고 accept가 높으면 줄어듦 |
| `ITL` | Inter-Token Latency. streaming 중 토큰 사이 간격 | 가장 직접적으로 좋아질 수 있음 |
| `TPOT` | Time Per Output Token. 첫 토큰 이후 평균 토큰 처리 시간 | speculative decoding 효과를 보기 좋은 지표 |
| `output throughput` | 초당 생성 output token 수 | 낮은-중간 concurrency에서 크게 증가 가능 |
| `accept length` | 한 번의 verify에서 확정된 token 수 | 높을수록 이득이 큼 |
| `accept rate` | 제안한 draft token 중 accept된 비율 | 낮으면 verify compute가 낭비됨 |
| concurrency | 동시에 처리하는 요청 수 | 낮을 때 speedup이 크고, 높을수록 줄어드는 경향 |

중요한 감각:

```text
Speculative decoding은 TTFT 가속기라기보다 decode phase 가속기다.

첫 토큰:
  queue + prefill + 첫 decode 영향이 크다.

두 번째 토큰 이후:
  draft가 맞으면 target forward 1번으로 여러 token을 확정한다.
```

그래서 운영 지표를 볼 때는 `TTFT` 하나만 보면 안 되고, 최소한 아래를 같이 봐야 한다.

- `Mean/P99 TTFT`
- `Mean/P99 ITL`
- `TPOT`
- `E2E latency`
- `output tok/s`
- `accept_length` 또는 `spec_accept_rate`
- GPU memory 사용량과 OOM 여부
- concurrency별 변화

## 2. 왜 throughput은 오르고 TTFT는 덜 바뀌나

일반 autoregressive decoding:

```text
target forward -> token 1
target forward -> token 2
target forward -> token 3
...
```

Speculative decoding:

```text
draft side     -> token 후보 여러 개 생성
target forward -> 후보 여러 개를 한 번에 verify
accept         -> 맞은 후보를 여러 개 확정
```

첫 토큰까지는 prompt prefill이 중요하다. speculative decoding은 주로 decode loop에서
"큰 target forward 1번당 확정되는 token 수"를 늘린다. 그래서 짧은 응답보다 긴 응답에서,
single-shot non-streaming보다 streaming decode 지표에서 효과가 더 잘 보인다.

## 3. MTP/NEXTN의 실제 수치

### 3.1 DeepSeek-V3 MTP 공개 수치

SGLang DeepSeek V3 문서 기준:

| 환경 | 설정 | speedup |
|---|---|---|
| DeepSeek-V3, H200 TP8 | batch size 1 | decode speedup 약 `1.8x` |
| DeepSeek-V3, H200 TP8 | batch size 32 | decode speedup 약 `1.5x` |

문서의 기본 DeepSeek MTP 설정:

```bash
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

SGLang에서 `MTP/NEXTN`은 독립 알고리즘이라기보다, MTP/NextN layer를 draft source로 쓰고
EAGLE speculative workflow를 재사용하는 형태로 이해하면 된다.

### 3.2 SpecDecode-Bench에서의 MTP

SpecDecode-Bench는 vLLM 기반으로 여러 speculative decoding 방법을 비교했다. MTP는
GLM-4.5-Air-106B reasoning workload에서 약 `1.3x-1.8x` speedup을 보였다.

단, 이 벤치는 MTP의 잠재력 전체를 대표하는 수치라기보다, 해당 모델의 MTP head와 workload에
대한 결과로 봐야 한다. 벤치 설명에 따르면 MTP의 position-wise acceptance가 뒤쪽 draft token으로
갈수록 급격히 나빠질 수 있고, 그러면 전체 speedup이 제한된다.

### 3.3 MTP의 resource 오버헤드

MTP는 별도 draft LLM을 붙이지 않는 대신, 모델 checkpoint 안에 MTP module/head weight가 추가된다.

DeepSeek-V3 Hugging Face card 기준:

| 항목 | 크기 |
|---|---:|
| Main model weights | `671B` params |
| MTP module weights | `14B` params |
| HF total | `685B` params |

BF16 raw weight로 단순 계산하면 MTP module 14B params는 약 28GB다.
실제 serving에서는 quantization, sharding, tensor parallel 설정에 따라 GPU별 부담이 달라진다.

MTP 오버헤드 요약:

- 추가 MTP weight 로딩 비용
- draft forward 계산 비용
- speculative verify buffer와 CUDA graph 메모리
- acceptance가 낮을 때 wasted verification compute
- `num_steps`를 키울수록 뒤쪽 token reject 가능성 증가

## 4. DFlash의 실제 수치

### 4.1 Qwen3.6-35B-A3B-DFlash 공개 벤치

Hugging Face `z-lab/Qwen3.6-35B-A3B-DFlash` model card 기준:

- Hardware: single NVIDIA B200
- Serving: SGLang
- Thinking enabled
- Max output length: 4096
- Reported metric: end-to-end throughput, prefill 포함
- Target model: `Qwen/Qwen3.6-35B-A3B`
- Draft model: `z-lab/Qwen3.6-35B-A3B-DFlash`

Block size 16:

| Task | Concurrency | AR tok/s | DFlash tok/s | Speedup |
|---|---:|---:|---:|---:|
| Math500 | 1 | 234 | 682 | 2.9x |
| Math500 | 8 | 1266 | 3138 | 2.5x |
| Math500 | 16 | 1954 | 4813 | 2.5x |
| Math500 | 32 | 2755 | 6520 | 2.4x |
| GSM8K | 1 | 235 | 556 | 2.4x |
| GSM8K | 32 | 2699 | 5239 | 1.9x |
| HumanEval | 1 | 238 | 603 | 2.5x |
| HumanEval | 32 | 2767 | 5782 | 2.1x |
| MT-Bench | 1 | 233 | 442 | 1.9x |
| MT-Bench | 32 | 2633 | 4034 | 1.5x |
| Alpaca | 1 | 235 | 393 | 1.7x |
| Alpaca | 32 | 2579 | 3689 | 1.4x |

Block size 8:

| Task | Concurrency | AR tok/s | DFlash tok/s | Speedup |
|---|---:|---:|---:|---:|
| Math500 | 1 | 234 | 617 | 2.6x |
| Math500 | 32 | 2755 | 6614 | 2.4x |
| GSM8K | 1 | 235 | 540 | 2.3x |
| GSM8K | 32 | 2699 | 5713 | 2.1x |
| HumanEval | 1 | 238 | 561 | 2.4x |
| HumanEval | 32 | 2767 | 6059 | 2.2x |
| MT-Bench | 1 | 233 | 438 | 1.9x |
| MT-Bench | 32 | 2633 | 4720 | 1.8x |

### 4.2 DFlash acceptance length

같은 model card의 average acceptance length:

| Task | Block size 8 | Block size 16 |
|---|---:|---:|
| Math500 | 5.56 | 7.35 |
| GSM8K | 5.21 | 6.73 |
| HumanEval | 5.09 | 6.44 |
| MBPP | 4.78 | 5.83 |
| MT-Bench | 4.20 | 5.14 |
| Alpaca | 3.94 | 4.62 |

해석:

- Math/code/reasoning처럼 다음 token이 비교적 예측 가능한 workload는 accept length가 높다.
- Alpaca/MT-Bench처럼 다양하고 자유로운 응답은 accept length가 낮아진다.
- block size를 8에서 16으로 키우면 accept length는 늘지만, 뒤쪽 token이 reject되어 낭비될 수도 있다.

### 4.3 DFlash의 resource 오버헤드

`z-lab/Qwen3.6-35B-A3B-DFlash` draft model card 기준 draft model size는 `0.5B` params, BF16이다.

DFlash에서 추가되는 비용:

- DFlash draft model weight
- draft model forward
- target hidden feature capture
- target hidden projection
- draft KV cache
- target hidden을 draft KV cache에 materialize/inject하는 비용
- verify block buffer
- DFlash 전용 mask token 처리

현재 SGLang 코드 기준 DFlash draft model은 자체 embedding/lm_head가 없다. target model의
embedding/lm_head를 사용하고, target hidden feature를 projection해서 draft block 생성에 쓴다.

## 5. Concurrency와 GPU compute resource 해석

Speculative decoding의 speedup은 concurrency가 커질수록 줄어드는 경향이 있다.

낮은 concurrency:

```text
decode batch가 작음
GPU compute가 덜 차 있음
memory bandwidth / kernel launch overhead가 상대적으로 큼
남는 compute로 draft와 verify를 얹기 쉬움
speedup 큼
```

높은 concurrency:

```text
baseline도 GPU를 꽉 채움
target verify가 compute-bound로 감
draft/verify 추가 작업이 경쟁 비용이 됨
rejected token 검증 비용이 더 아프게 보임
speedup 줄어듦
```

SpecDecode-Bench의 핵심 관찰:

- batch size가 커질수록 speedup은 줄어든다.
- 현실적인 batch에서는 batch 1에서 보던 `3x-4x` 수치를 그대로 기대하면 안 된다.
- target verification이 전체 실행 시간의 약 `42%-95%`를 차지한다.
- rejection sampling 자체는 작고, 문제는 reject된 draft token에 들어간 target verify compute다.

즉 bottleneck은 대체로 이것이다.

```text
draft가 싸게 많이 제안하는가?
제안한 token이 실제로 많이 accept되는가?
target verify가 rejected token에 compute를 낭비하지 않는가?
```

## 6. Latency를 throughput에서 대략 읽는 법

공개 table이 aggregate output throughput만 줄 때, concurrency별 대략적인 per-token decode latency는
아래처럼 근사할 수 있다.

```text
대략 TPOT = concurrency / aggregate_output_tok_s
```

예: DFlash Qwen3.6-35B-A3B, Math500, block size 16, concurrency 32

| 방식 | Aggregate output tok/s | 대략 TPOT |
|---|---:|---:|
| AR baseline | 2755 tok/s | 32 / 2755 = 11.6 ms/token |
| DFlash | 6520 tok/s | 32 / 6520 = 4.9 ms/token |

주의:

- 이 계산은 queueing, request length variance, streaming chunking, prefill을 분리하지 않은 근사다.
- 정확한 latency는 `bench_serving`의 `TTFT`, `ITL`, `TPOT`, `E2E latency`를 봐야 한다.
- output length가 짧으면 TTFT와 scheduler overhead 비중이 커져 speedup 체감이 작을 수 있다.

## 7. MTP vs DFlash 선택 기준

| 기준 | MTP/NEXTN | DFlash |
|---|---|---|
| draft source | target checkpoint 안의 MTP/NextN module | 별도 DFlash draft model |
| 별도 draft checkpoint | 모델에 내장되어 있으면 불필요 | 필요 |
| 대표 장점 | 모델 내장형이라 운영 구성이 단순할 수 있음 | block diffusion draft로 큰 speedup 가능 |
| 대표 오버헤드 | MTP weight와 draft compute | draft model, target hidden capture, draft KV |
| 잘 맞는 상황 | MTP 내장 모델, 낮은-중간 batch, predictable output | matching DFlash checkpoint, 긴 출력, predictable output |
| 조심할 상황 | MTP head acceptance가 낮은 workload | matching checkpoint 없음, VRAM 부족, high concurrency |
| SGLang 실행 감각 | 보통 `EAGLE` 또는 `NEXTN`, `topk=1` | `DFLASH`, block size/window 중심 |

실전 우선순위:

1. 모델에 MTP/NextN이 내장되어 있으면 MTP를 먼저 작게 켜서 측정한다.
2. matching DFlash checkpoint가 있으면 DFlash도 별도로 측정한다.
3. `output tok/s`만 보지 말고 `TTFT`, `P99 ITL`, `E2E latency`, `accept_length`, GPU memory를 같이 본다.
4. concurrency 1, 4, 8, 16, 32처럼 sweep한다.
5. workload를 실제 서비스 prompt/output length와 비슷하게 만든다.

## 8. SGLang에서 직접 측정하는 방법

### 8.1 Baseline server

```bash
python3 -m sglang.launch_server \
  --model-path <TARGET_MODEL> \
  --host 0.0.0.0 \
  --port 30000
```

### 8.2 MTP/NEXTN server 예시

DeepSeek류 MTP:

```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --host 0.0.0.0 \
  --port 30000 \
  --tp 8 \
  --trust-remote-code \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

Qwen NextN 계열은 모델/checkout에 따라 다음처럼 쓸 수 있다.

```bash
python3 -m sglang.launch_server \
  --model-path <QWEN_NEXTN_MODEL> \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

### 8.3 DFlash server 예시

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.6-35B-A3B-DFlash \
  --speculative-num-draft-tokens 16 \
  --attention-backend fa3 \
  --mem-fraction-static 0.75
```

현재 checkout 기준 DFlash 제약:

- `--speculative-draft-model-path` 필요
- `--enable-dp-attention` 미지원
- `pp_size == 1` 필요
- overlap scheduler/spec v2 미지원
- mixed chunked prefill 비활성화
- `speculative_num_steps`는 1로 강제
- `speculative_eagle_topk`는 1로 강제

### 8.4 bench_serving으로 지표 측정

SGLang native endpoint:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model <TARGET_MODEL> \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --random-range-ratio 1.0 \
  --num-prompts 512 \
  --request-rate inf \
  --max-concurrency 32 \
  --output-file bench_spec.jsonl \
  --output-details
```

OpenAI-compatible chat endpoint:

```bash
python3 -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url http://127.0.0.1:30000 \
  --model <TARGET_MODEL> \
  --dataset-name sharegpt \
  --num-prompts 512 \
  --sharegpt-output-len 512 \
  --request-rate inf \
  --max-concurrency 32 \
  --output-file bench_spec_oai.jsonl \
  --output-details
```

Concurrency sweep 예시:

```bash
for c in 1 4 8 16 32 64; do
  python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port 30000 \
    --model <TARGET_MODEL> \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --num-prompts 512 \
    --request-rate inf \
    --max-concurrency "$c" \
    --output-file "bench_c${c}.jsonl" \
    --output-details
done
```

`bench_serving` 출력에서 봐야 할 것:

- Request throughput
- Input token throughput
- Output token throughput
- Total token throughput
- Concurrency
- Mean/Median/P99 E2E latency
- Mean/Median/P99 TTFT
- Mean/Median/P95/P99 ITL
- TPOT
- Accept length, SGLang에서 speculative metadata가 있으면 표시

## 9. 결과 해석 체크리스트

### TTFT

좋은 신호:

- TTFT가 baseline과 비슷하거나 약간만 증가
- P99 TTFT가 크게 튀지 않음

나쁜 신호:

- DFlash hidden capture나 draft worker 초기화/메모리 압박 때문에 TTFT가 크게 증가
- high concurrency에서 queueing으로 TTFT가 폭증

### ITL/TPOT

좋은 신호:

- mean ITL 또는 TPOT가 명확히 감소
- P95/P99 ITL이 함께 감소
- output 길이가 길수록 이득이 커짐

나쁜 신호:

- mean은 좋아졌지만 P99 ITL이 나빠짐
- accept length variance가 커서 일부 요청이 느려짐

### Throughput

좋은 신호:

- output tok/s가 baseline 대비 상승
- concurrency를 올려도 speedup이 유지됨

나쁜 신호:

- concurrency 1에서만 좋고, 8/16/32에서 거의 동일하거나 역전
- high concurrency에서 GPU compute saturation으로 draft overhead가 그대로 비용이 됨

### GPU memory

좋은 신호:

- OOM 없이 cuda graph capture 가능
- KV cache 여유가 있어 max-running-requests를 유지 가능

나쁜 신호:

- speculative enable 후 OOM
- `--cuda-graph-max-bs`, `--mem-fraction-static`, `--max-running-requests`를 크게 낮춰야 함
- draft model 또는 MTP module 때문에 batch capacity가 줄어 throughput 이득을 상쇄

## 10. 짧은 결론

MTP/NEXTN과 DFlash 모두 목표는 같다.

```text
비싼 target model forward 1번으로 확정되는 output token 수를 늘린다.
```

차이는 draft 비용의 모양이다.

- MTP/NEXTN: 모델 안의 MTP module/head가 미래 token을 예측한다.
- DFlash: 별도 block diffusion draft model이 block 전체를 병렬로 만든다.

실전에서 이득은 다음 조건을 만족할 때 커진다.

- output이 충분히 길다.
- draft acceptance가 높다.
- 낮은-중간 concurrency라 GPU compute 여유가 있다.
- draft compute와 verification buffer가 VRAM/compute 병목을 만들지 않는다.
- target/draft tokenizer와 checkpoint가 잘 맞는다.

반대로 다음 상황에서는 느려질 수 있다.

- creative/high-temperature workload라 acceptance가 낮다.
- output이 너무 짧아 TTFT와 scheduler overhead가 대부분이다.
- high concurrency에서 baseline이 이미 compute-bound다.
- draft model 또는 MTP module 때문에 메모리 여유가 줄어 batch capacity가 떨어진다.
- DFlash matching checkpoint가 없거나 hidden capture path가 비효율적이다.
