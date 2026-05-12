# Qwen3.6 35B-A3B MTP SGLang 실행 가이드

이 문서는 SGLang에서 `Qwen/Qwen3.6-35B-A3B` 또는
`Qwen/Qwen3.6-35B-A3B-FP8`을 MTP로 실행할 때 무엇을 켜야 하는지 쉽게
이해하기 위한 메모다.

기준 확인:

- 공식 SGLang Qwen3.6 문서는 Qwen3.6에 `sglang>=0.5.10`이 필요하다고 설명한다.
- Hugging Face 모델카드는 SGLang MTP 실행 예시로 `--speculative-algo NEXTN`을 사용한다.
- 이 repository의 `python/sglang/srt/server_args.py`는 `NEXTN`을 받으면 내부적으로
  `EAGLE`로 바꾼다. 그래서 문서에 따라 `NEXTN` 또는 `EAGLE`이라고 보이더라도 같은
  MTP speculative decoding 경로를 탄다고 이해하면 된다.

## 1. 한 줄 요약

Qwen3.6 35B-A3B의 MTP는 별도 draft model을 지정하지 않는다. 모델 체크포인트 안에
포함된 MTP 가중치를 SGLang의 speculative decoding 경로로 사용하는 방식이다.

가장 무난한 시작 명령:

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

## 2. 왜 FP8부터 추천하는가

`Qwen/Qwen3.6-35B-A3B`는 BF16 원본이고, `Qwen/Qwen3.6-35B-A3B-FP8`은 FP8
버전이다.

대략적인 차이:

| 모델 | 장점 | 주의점 |
|---|---|---|
| `Qwen/Qwen3.6-35B-A3B-FP8` | 가중치 메모리가 작아서 시작하기 쉽다 | FP8 지원이 잘 되는 NVIDIA GPU에서 권장 |
| `Qwen/Qwen3.6-35B-A3B` | 원본 BF16 품질 기준으로 쓰기 좋다 | 가중치만 약 70GB 수준이라 여유 메모리가 필요 |

FP8 모델은 Hugging Face 파일 목록에 `mtp.safetensors`가 따로 있고, BF16 모델은
`model.safetensors.index.json` 안에 `mtp.*` 가중치가 포함되어 있다. 즉 둘 다 MTP
실행에 필요한 가중치가 있다.

BF16 원본으로 실행하려면 위 명령에서 이 부분만 바꾸면 된다.

```bash
--model-path Qwen/Qwen3.6-35B-A3B
```

## 3. MTP가 하는 일

일반 decoding은 토큰을 하나 만들고, 다시 모델을 돌려 다음 토큰을 하나 만드는 식이다.
MTP는 여러 후보 토큰을 미리 초안처럼 만들어 놓고, 큰 모델이 한 번에 검증해서 맞는
토큰들을 받아들이는 방식이다.

비유하면:

1. draft 단계: "다음 몇 단어는 아마 이럴 것" 하고 미리 써 본다.
2. verify 단계: 원래 모델이 그 초안을 확인한다.
3. accept 단계: 맞는 부분은 그대로 채택하고, 틀린 지점부터 다시 생성한다.

이 때문에 interactive 응답에서 latency를 줄이는 데 도움이 된다. 단, 모든 workload에서
무조건 빨라지는 것은 아니다. batch size, prompt 길이, GPU, cache 설정에 따라 이득이
달라진다.

## 4. 옵션을 하나씩 풀어보기

| 옵션 | 쉬운 설명 |
|---|---|
| `SGLANG_ENABLE_SPEC_V2=1` | speculative decoding의 overlap scheduler 경로를 명시적으로 켠다. MTP에서는 켜 두는 편이 안전하다 |
| `--speculative-algorithm NEXTN` | 모델 안의 next-token/MTP 가중치를 draft 모델처럼 사용하겠다는 뜻이다 |
| `--speculative-num-steps 3` | draft를 몇 단계 앞까지 굴릴지 정한다 |
| `--speculative-eagle-topk 1` | 각 단계에서 후보 branch를 몇 개 볼지 정한다. 처음에는 1이 안정적이다 |
| `--speculative-num-draft-tokens 4` | verify할 draft token 수의 상한이다 |
| `--mamba-scheduler-strategy extra_buffer` | Qwen3.6의 Gated Delta Networks 계열 cache를 speculative decoding과 함께 쓰기 위한 설정이다 |
| `--page-size 64` | `extra_buffer` 전략에서 권장되는 paged cache 단위다 |
| `--reasoning-parser qwen3` | `<think>...</think>` 형태의 thinking 내용을 OpenAI-compatible 응답의 `reasoning_content`로 분리한다 |
| `--tool-call-parser qwen3_coder` | Qwen 계열 tool call 포맷을 SGLang이 파싱하게 한다 |
| `--context-length 262144` | Qwen3.6의 기본 long context 길이로 맞춘다 |
| `--mem-fraction-static 0.8` | GPU 메모리 중 SGLang static memory pool로 쓸 비율이다. OOM이면 낮춘다 |

## 5. `NEXTN`과 `EAGLE` 중 무엇을 써야 하나

둘 다 보일 수 있다.

Hugging Face Qwen3.6 모델카드는 MTP 예시에서 `NEXTN`을 쓴다.

```bash
--speculative-algo NEXTN
```

SGLang Qwen3.6 문서는 `EAGLE` 예시를 보여준다.

```bash
--speculative-algorithm EAGLE
```

로컬 코드 기준으로는 `NEXTN`이 들어오면 `EAGLE`로 바뀐다.

```python
if self.speculative_algorithm == "NEXTN":
    self.speculative_algorithm = "EAGLE"
```

따라서 Qwen3.6 native MTP라는 의도를 명확히 남기고 싶으면 `NEXTN`을 쓰고, SGLang
문서 예시와 맞추고 싶으면 `EAGLE`을 써도 된다. 이 학습 노트에서는 Hugging Face
모델카드의 표현에 맞춰 `NEXTN`을 사용한다.

## 6. GPU 수에 따른 조정

공식 문서는 35B-A3B FP8이 약 35GB, BF16이 약 70GB 가중치 메모리를 쓴다고 설명한다.
KV cache, MTP, vision encoder, runtime buffer까지 생각하면 실제 필요 메모리는 더 커진다.

예시:

```bash
# 단일 H100/H200/B200에서 FP8을 먼저 시도
--model-path Qwen/Qwen3.6-35B-A3B-FP8 --tp-size 1

# BF16 원본이 단일 GPU에서 OOM이면 TP를 늘림
--model-path Qwen/Qwen3.6-35B-A3B --tp-size 2

# 8 GPU로 넉넉하게 쪼개서 실행
--model-path Qwen/Qwen3.6-35B-A3B --tp-size 8
```

OOM이 나면 아래 순서로 줄여본다.

1. `--context-length 262144`를 `131072`로 낮춘다.
2. `--mem-fraction-static 0.8`을 `0.75` 또는 `0.7`로 낮춘다.
3. `--tp-size`를 GPU 수에 맞게 늘린다.
4. multimodal 입력을 쓰지 않는다면 text-only 용도로만 운용할 수 있는지 확인한다.

## 7. 서버가 떴는지 확인하기

OpenAI-compatible chat endpoint로 확인한다.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 128
  }'
```

서버 내부 상태에서 speculative acceptance가 궁금하면 `/server_info`도 확인한다.

```bash
curl http://localhost:8000/server_info | jq
```

응답의 `internal_states` 쪽에 speculative 관련 평균 acceptance length가 나오면 MTP가
실제로 돌고 있는지 감을 잡을 수 있다. 필드 이름은 SGLang 버전에 따라 조금 달라질 수 있다.

## 8. 자주 나는 문제

### `unrecognized arguments: --tp`

이 repository의 argparse는 `--tp-size` 또는 `--tensor-parallel-size`를 받는다. 문서나
테스트 일부에는 `--tp`가 보일 수 있지만, 안전하게는 아래처럼 쓴다.

```bash
--tp-size 1
```

### `speculative decoding ... radix cache ... no_buffer` 관련 오류

Qwen3.6은 Gated Delta Networks 기반의 hybrid architecture라 Mamba/Radix cache 설정이
중요하다. MTP와 radix cache를 같이 쓰려면 아래 조합을 우선 사용한다.

```bash
SGLANG_ENABLE_SPEC_V2=1 \
--mamba-scheduler-strategy extra_buffer \
--page-size 64
```

### OOM

긴 context가 가장 큰 원인인 경우가 많다.

```bash
--context-length 131072 --mem-fraction-static 0.75
```

그래도 부족하면 FP8 모델을 쓰거나 `--tp-size`를 늘린다.

### tool call이나 thinking 내용이 이상하게 섞임

서버 실행 때 parser를 켰는지 확인한다.

```bash
--reasoning-parser qwen3 \
--tool-call-parser qwen3_coder
```

## 9. 참고 링크

- SGLang Qwen3.6 cookbook: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6
- Hugging Face BF16 model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Hugging Face FP8 model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8
- SGLang speculative decoding docs: https://docs.sglang.io/advanced_features/speculative_decoding.html
- 로컬 코드 확인 지점: `python/sglang/srt/server_args.py`
