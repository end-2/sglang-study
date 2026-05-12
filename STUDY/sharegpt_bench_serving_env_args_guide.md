# SGLang ShareGPT bench_serving env/args guide

이 문서는 `python/sglang/bench_serving.py`에서 `--dataset-name sharegpt`를 사용할 때의
환경 변수와 CLI 인자를 쉽게 고르는 용도다. 기준 코드는 현재 repository의
`python/sglang/bench_serving.py`, `python/sglang/benchmark/datasets/sharegpt.py`,
`python/sglang/benchmark/utils.py`다.

## 1. 가장 작은 실행 예시

Native SGLang `/generate`:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --dataset-name sharegpt \
  --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenizer /path/or/hf/tokenizer \
  --num-prompts 100 \
  --sharegpt-output-len 128
```

OpenAI chat-compatible `/v1/chat/completions`:

```bash
python3 -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url http://127.0.0.1:30000 \
  --dataset-name sharegpt \
  --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model my-served-model \
  --tokenizer /path/or/hf/tokenizer \
  --num-prompts 100 \
  --sharegpt-output-len 128
```

## 2. ShareGPT 데이터가 처리되는 방식

ShareGPT loader는 JSON list를 읽는다. 각 item은 `conversations` 또는 `conversation`
필드를 가질 수 있고, 최소 2 turn 이상이어야 한다.

처리 순서:

1. 각 대화의 첫 번째 turn을 prompt, 두 번째 turn을 completion으로 사용한다.
2. 전체 dataset을 `random.shuffle`한다. 재현성은 `--seed`로 맞춘다.
3. prompt와 completion을 tokenizer로 encode한다.
4. `prompt_len < 2` 또는 `output_len < 2`인 샘플은 버린다.
5. `--sharegpt-context-len`이 있으면 `prompt_len + output_len`이 그 값을 넘는 샘플을 버린다.
6. `--num-prompts` 개수만큼 모이면 멈춘다.

`--dataset-path`를 비워두면 Hugging Face dataset
`anon8231489123/ShareGPT_Vicuna_unfiltered`의
`ShareGPT_V3_unfiltered_cleaned_split.json`을 다운로드한다. 경로를 지정했는데 JSON이
아니거나 파일이 없으면 자동 다운로드하지 않고 실패한다.

## 3. 백엔드 조합과 실제 endpoint/body

ShareGPT에서 SGLang 계열로 주로 쓰는 조합은 아래 네 가지다.

| `--backend` | endpoint | 용도 | 주요 body shape |
|---|---|---|---|
| `sglang` | `/generate` | SGLang native API | `text`, `sampling_params.max_new_tokens`, `stream`, `return_logprob` |
| `sglang-native` | `/generate` | `sglang`과 같은 request function | `sglang`과 동일 |
| `sglang-oai` | `/v1/completions` | OpenAI completion 호환 API | `model`, `prompt`, `max_tokens`, `stream` |
| `sglang-oai-chat` | `/v1/chat/completions` | OpenAI chat 호환 API | `model`, `messages`, `max_completion_tokens`, `stream` |
| `sglang-embedding` | `/v1/embeddings` | ShareGPT prompt를 embedding input으로 보냄 | `input`, `model` |

기본 port는 SGLang 계열에서 `30000`이다. `--base-url`을 주면 `--host/--port` 조합보다
우선한다.

관찰한 실제 body 예시는
`benchmark/sharegpt_observed_request_bodies.json`에 있다.

argparse의 전체 backend choice는 `sglang`, `sglang-native`, `sglang-oai`,
`sglang-oai-chat`, `sglang-embedding`, `vllm`, `vllm-chat`, `lmdeploy`,
`lmdeploy-chat`, `trt`, `gserver`, `truss`다. 이 문서는 SGLang ShareGPT 사용을
중심으로 설명한다.

## 4. 꼭 자주 쓰는 인자

### 서버 위치

| 인자 | 설명 |
|---|---|
| `--base-url URL` | 이미 알고 있는 base URL. 예: `http://127.0.0.1:30000` |
| `--host HOST` | `--base-url`이 없을 때 사용. 기본값 `0.0.0.0` |
| `--port PORT` | `--base-url`이 없을 때 사용. SGLang 기본값 `30000` |
| `--ready-check-timeout-sec N` | 시작 전 `/v1/models`를 최대 N초 기다림. `0`이면 skip |

### 모델과 tokenizer

| 인자 | 설명 |
|---|---|
| `--model MODEL` | payload에 보낼 model 이름이자, `--tokenizer`가 없을 때 tokenizer id |
| `--served-model-name NAME` | payload의 model 이름만 따로 지정. tokenizer/model path와 serving name이 다를 때 유용 |
| `--tokenizer TOKENIZER` | ShareGPT token length 계산용 tokenizer. 실제 서버 model과 달라도 됨 |

실전에서는 서버가 `/v1/models`로 돌려주는 id가 Hugging Face tokenizer id가 아닐 수 있으므로
`--tokenizer`를 명시하는 편이 안전하다.

#### benchmark client에서 tokenizer가 필요한 이유

ShareGPT 원본은 문자열 대화만 갖고 있고, HTTP serving API는 client에게 prompt가 몇
token인지 미리 알려주지 않는다. 그래서 `bench_serving` client가 tokenizer를 직접
로드해 request를 만들기 전에 token accounting을 한다.

구체적으로 ShareGPT loader는 첫 번째 turn을 prompt, 두 번째 turn을 completion으로
잡은 뒤 둘 다 `tokenizer.encode`해서 `prompt_len`과 `output_len`을 만든다. 이 값은
너무 짧은 샘플 제거, `--sharegpt-context-len` 필터링, `--sharegpt-output-len` 미지정 시
생성 목표 길이 결정에 쓰인다. benchmark 실행 단계에서는 `output_len`이
`max_new_tokens`, `max_tokens`, `max_completion_tokens` 같은 요청 파라미터로 들어가고,
`prompt_len`은 input token throughput과 total token throughput 같은 결과 metric 계산에
들어간다. benchmark 후에는 생성 문자열도 같은 tokenizer로 다시 tokenize해서
retokenized output token metric을 계산한다.

따라서 이 tokenizer는 서버의 추론 실행용이라기보다 benchmark client의 길이 계산과
metric 계산용이다. 실제 서버 model과 다른 tokenizer를 넣어도 요청 자체는 보낼 수 있지만,
정확한 token 수, context filtering, tok/s 비교를 원하면 서버가 쓰는 model/tokenizer와
같은 tokenizer 경로를 주는 것이 맞다.

### ShareGPT dataset

| 인자 | 설명 |
|---|---|
| `--dataset-name sharegpt` | ShareGPT loader 선택 |
| `--dataset-path PATH` | ShareGPT JSON 파일. 비우면 기본 HF dataset 다운로드 |
| `--num-prompts N` | 보낼 request 개수 |
| `--sharegpt-output-len N` | 모든 request의 output token 목표치를 N으로 고정. 지정하지 않으면 completion 길이 사용 |
| `--sharegpt-context-len N` | `prompt_len + output_len <= N`인 샘플만 사용 |
| `--prompt-suffix TEXT` | prompt 끝의 `Assistant:` 앞에 suffix를 삽입 |
| `--apply-chat-template` | tokenizer의 chat template을 prompt 문자열에 적용 |
| `--seed N` | shuffle, LoRA sampling, request 간격 sampling 재현성 |

주의:

- `--sharegpt-output-len`을 지정하면 4 이상이어야 한다.
- `--apply-chat-template`은 prompt 문자열 자체를 템플릿화한다. `sglang-oai-chat`에서는 다시
  `messages=[{"role": "user", "content": prompt}]`로 감싸므로, 모델/토크나이저 의도에 맞는지 확인해야 한다.
- `--tokenize-prompt`는 ShareGPT loader에서 assert로 막혀 있다. ShareGPT와 함께 쓸 수 없다.

### 트래픽 모양

| 인자 | 설명 |
|---|---|
| `--request-rate inf` | 기본값. 모든 request를 거의 동시에 만든다 |
| `--request-rate R` | 초당 평균 R개. Poisson process로 arrival 간격을 샘플링 |
| `--max-concurrency N` | 동시에 실행되는 request 수를 N으로 제한 |
| `--warmup-requests N` | 본 benchmark 전에 첫 샘플로 N개 warmup request 실행. warmup output_len은 최대 32 |

대표 조합:

```bash
# 최대 부하: 한 번에 다 보내기
--request-rate inf

# 실제 트래픽처럼 초당 8개 평균, 동시 32개 제한
--request-rate 8 --max-concurrency 32

# 서버 cache warmup 없이 payload만 관찰
--warmup-requests 0 --num-prompts 2
```

## 5. request body를 바꾸는 인자

| 인자 | 영향 |
|---|---|
| `--disable-stream` | `stream: false`로 보냄 |
| `--disable-ignore-eos` | 기본 `ignore_eos: true`를 끄고 EOS를 존중하게 함 |
| `--extra-request-body JSON` | payload에 JSON을 merge. 같은 key가 있으면 override |
| `--return-logprob` | native `/generate`에는 `return_logprob: true`; completion API에서는 logprobs 관련 옵션과 함께 사용 |
| `--top-logprobs-num N` | native는 `top_logprobs_num`, completion API는 `logprobs`로 전달 |
| `--token-ids-logprob IDS...` | native `/generate`에 `token_ids_logprob` 전달 |
| `--logprob-start-len N` | native `/generate`에 `logprob_start_len` 전달. 기본 `-1` |
| `--return-routed-experts` | native `/generate`에 `return_routed_experts: true` 전달 |
| `--header K=V ...` | 모든 HTTP 요청에 custom header 추가 |

`--extra-request-body` merge는 deep merge가 아니다. 예를 들어 native SGLang에서 sampling params를
바꾸려면 아래처럼 `sampling_params` 전체를 넘긴다.

```bash
--extra-request-body '{"sampling_params":{"temperature":0.7,"top_p":0.9,"max_new_tokens":64,"ignore_eos":true}}'
```

OpenAI completion/chat 계열은 top-level key로 override한다.

```bash
--extra-request-body '{"temperature":0.7,"top_p":0.9}'
```

## 6. LoRA 조합

| 인자 | 설명 |
|---|---|
| `--lora-name A B ...` | 사용할 LoRA 이름/path 목록 |
| `--lora-request-distribution uniform` | 각 request마다 LoRA를 균등 random 선택 |
| `--lora-request-distribution distinct` | LoRA 목록을 순서대로 round-robin |
| `--lora-request-distribution skewed` | Zipf 분포로 일부 LoRA에 더 많이 몰림 |
| `--lora-zipf-alpha X` | skewed 분포 강도. 1보다 커야 함. 기본 1.5 |

payload 차이:

- `sglang`: `lora_path`에 선택된 LoRA가 들어간다.
- `sglang-oai`, `sglang-oai-chat`: `model`과 `lora_path`가 모두 선택된 LoRA로 바뀐다.

`distinct`와 `skewed`는 LoRA가 2개 이상 있어야 한다.

## 7. 출력과 관찰

| 인자 | 설명 |
|---|---|
| `--output-file PATH` | 결과 JSONL 저장 경로. 없으면 `BACKEND_MMDD_NUM_sharegpt.jsonl` |
| `--output-details` | request별 `input_lens`, `output_lens`, `ttfts`, `itls`, `generated_texts`, `errors`도 저장 |
| `--disable-tqdm` | progress bar 제거 |
| `--print-requests` | request 시작/끝을 출력. 현재 `sglang-oai-chat`에서만 허용 |
| `--tag TEXT` | 결과 JSONL의 `tag` 필드에 기록 |
| `--plot-throughput` | throughput/concurrency plot 출력. `termplotlib`와 `gnuplot` 필요 |

`sglang`이 포함된 backend는 benchmark 끝에 `/server_info`를 GET해서 결과 JSONL의 `server_info`에 넣는다.

## 8. profiling/cache/PD 관련 인자

| 인자 | 설명 |
|---|---|
| `--flush-cache` | main run 직전에 `POST /flush_cache` 호출 |
| `--profile` | benchmark 전 `POST /start_profile`, 후 `POST /stop_profile` 호출 |
| `--profile-activities CPU GPU CUDA_PROFILER XPU` | profiler activity 목록 |
| `--profile-start-step N` | profiler 시작 step |
| `--profile-steps N` | profiler step 수. 지정하면 자동 stop 용도 |
| `--profile-num-steps N` | `/start_profile` body의 `num_steps` |
| `--profile-by-stage` | stage별 profiling |
| `--profile-stages ...` | profiling stage 목록 |
| `--profile-output-dir DIR` | profiler output dir |
| `--profile-prefix PREFIX` | profiler trace prefix |
| `--pd-separated` | prefill/decode 분리 서버 benchmark/profile 모드 |
| `--profile-prefill-url URL...` | PD profile start/stop을 prefill worker에 보냄 |
| `--profile-decode-url URL...` | PD profile start/stop을 decode worker에 보냄 |
| `--fake-prefill` | extra body에 `bootstrap_host`, `bootstrap_room`을 넣음 |

`--profile-prefill-url`과 `--profile-decode-url`은 argparse mutual exclusive라 동시에 쓸 수 없다.

## 9. 환경 변수

`bench_serving.py`와 ShareGPT 경로에서 직접 의미가 있는 env는 아래다.

| env | 효과 |
|---|---|
| `OPENAI_API_KEY` | 모든 request/ready check에 `Authorization: Bearer $OPENAI_API_KEY` 추가 |
| `API_KEY` | `OPENAI_API_KEY`가 없을 때 `Authorization: $API_KEY` 추가 |
| `SGLANG_USE_MODELSCOPE=true` | tokenizer/model id를 HF 대신 ModelScope `snapshot_download`로 해석 |
| `SGLANG_TORCH_PROFILER_DIR=/path` | `--profile-output-dir`이 없을 때 profiler output 기본 위치 |
| `SGLANG_IS_IN_CI=true` | backend 이름에 `sglang`이 들어가면 main run 직전 `/flush_cache` 자동 호출 |
| `HF_HUB_OFFLINE=1` | `SGLANG_USE_MODELSCOPE=true`일 때 ModelScope download를 local-only로 제한하는 데 반영 |

Hugging Face/Transformers 표준 env도 tokenizer와 dataset 다운로드에 간접적으로 유용하다.
예: `HF_HOME`, `HF_HUB_CACHE`, `HF_TOKEN`.

인증 우선순위는 `OPENAI_API_KEY`가 `API_KEY`보다 높다.

## 10. ShareGPT에서는 쓰지 않거나 조심할 인자

아래 인자는 argparse에는 있지만 ShareGPT에는 영향이 없거나 다른 dataset 전용이다.

| 인자 | ShareGPT에서의 상태 |
|---|---|
| `--random-input-len`, `--random-output-len`, `--random-range-ratio` | random/image dataset 전용. 결과 JSON에는 값이 기록되지만 ShareGPT sampling에는 영향 없음 |
| `--image-count`, `--image-resolution`, `--random-image-count`, `--image-format`, `--image-content` | image dataset 전용 |
| `--use-trace-timestamps` | mooncake 전용 |
| `--gsp-num-groups`, `--gsp-prompts-per-group`, `--gsp-system-prompt-len`, `--gsp-question-len`, `--gsp-output-len`, `--gsp-range-ratio`, `--gsp-fast-prepare`, `--gsp-send-routing-key`, `--gsp-num-turns`, `--gsp-ordered` | generated-shared-prefix dataset 전용 |
| `--mooncake-slowdown-factor`, `--mooncake-num-rounds`, `--mooncake-workload` | mooncake dataset 전용 |
| `--tokenize-prompt` | ShareGPT에서는 assert로 실패 |

## 11. 조합 cheat sheet

### Native SGLang payload를 보고 싶을 때

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --dataset-name sharegpt \
  --dataset-path /path/to/sharegpt.json \
  --tokenizer /path/to/tokenizer \
  --num-prompts 2 \
  --warmup-requests 0 \
  --sharegpt-output-len 8
```

나가는 body:

```json
{
  "text": "...",
  "sampling_params": {
    "temperature": 0.0,
    "max_new_tokens": 8,
    "ignore_eos": true
  },
  "stream": true,
  "lora_path": null,
  "return_logprob": false,
  "return_routed_experts": false,
  "logprob_start_len": -1
}
```

### OpenAI chat 호환 서버를 재고 싶을 때

```bash
python3 -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url http://127.0.0.1:30000 \
  --dataset-name sharegpt \
  --dataset-path /path/to/sharegpt.json \
  --model served-model \
  --tokenizer /path/to/tokenizer \
  --num-prompts 100 \
  --sharegpt-output-len 128
```

나가는 body:

```json
{
  "model": "served-model",
  "messages": [{"role": "user", "content": "..."}],
  "max_completion_tokens": 128,
  "stream": true,
  "temperature": 0.0,
  "ignore_eos": true
}
```

### 특정 QPS와 concurrency로 재고 싶을 때

```bash
--request-rate 8 --max-concurrency 32 --warmup-requests 5
```

### 서버 cache를 비우고 재고 싶을 때

```bash
--flush-cache
```

또는 CI 환경처럼 자동 flush:

```bash
SGLANG_IS_IN_CI=true python3 -m sglang.bench_serving ...
```

### 인증 header가 필요할 때

```bash
OPENAI_API_KEY=sk-... python3 -m sglang.bench_serving ...
```

또는 custom header:

```bash
--header X-Request-Source=bench X-Experiment=sharegpt-smoke
```

## 12. 검증 방법

이 문서는 아래 방식으로 검증했다.

```bash
.venv/bin/python benchmark/run_bench_serving_with_light_stubs.py --help
.venv/bin/python -m py_compile python/sglang/benchmark/dummy_server.py \
  benchmark/run_bench_serving_with_light_stubs.py
```

또한 `benchmark/sharegpt_observation_sample.json`을 사용해 실제 `bench_serving` 요청을
`python/sglang/benchmark/dummy_server.py`로 받아
`benchmark/sharegpt_observed_request_bodies.json`에 request body를 저장했다.
