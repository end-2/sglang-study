# SGLang MTP / EAGLE Speculative Decoding Guide

이 문서는 SGLang에서 MTP와 EAGLE speculative decoding이 어떻게 연결되는지, 그리고
`--speculative-algorithm EAGLE`만 주었을 때 실제로 어떤 버전과 기본값이 사용되는지 쉽게
정리한 노트입니다.

## 한 줄 요약

- `EAGLE`은 문서상 EAGLE-2를 의미합니다.
- `EAGLE3`를 쓰려면 반드시 `--speculative-algorithm EAGLE3`로 명시해야 합니다.
- `NEXTN`은 내부에서 `EAGLE`로 바뀌는 alias입니다.
- MTP는 별도의 `--speculative-algorithm MTP`가 아니라, EAGLE speculative decoding 경로를
  재사용합니다.
- `--speculative-algorithm EAGLE`만 주고 세부 파라미터를 생략하면 SGLang이
  `speculative_num_steps`, `speculative_eagle_topk`, `speculative_num_draft_tokens`를 자동으로
  채웁니다.

## 먼저 알아야 할 용어

### Speculative Decoding

Speculative decoding은 작은 draft 모델 또는 모델 안의 별도 예측 레이어가 앞으로 나올 토큰 후보를
여러 개 미리 만들고, target 모델이 그 후보를 한 번에 검증하는 방식입니다.

일반 decode는 다음처럼 진행됩니다.

```text
target model -> 1 token
target model -> 1 token
target model -> 1 token
```

speculative decoding은 대략 다음처럼 진행됩니다.

```text
draft side  -> 여러 토큰 후보를 빠르게 생성
target side -> 후보들을 한 번에 검증
accepted    -> 맞은 후보들은 그대로 사용
rejected    -> 틀린 지점부터 target 결과로 보정
```

핵심은 target 모델을 덜 자주 호출하거나, 한 번 호출할 때 더 많은 후보를 검증해서 decode 속도를
올리는 것입니다.

### EAGLE

SGLang 문서에서 EAGLE은 보통 두 버전으로 나뉩니다.

| 이름 | args 값 | 의미 |
| --- | --- | --- |
| EAGLE-2 | `--speculative-algorithm EAGLE` | broad compatibility용 기본 EAGLE 경로 |
| EAGLE-3 | `--speculative-algorithm EAGLE3` | EAGLE3 draft model을 쓰는 경로 |

즉 `EAGLE`이라는 문자열만 보면 EAGLE-3가 아니라 EAGLE-2로 이해하면 됩니다.

### MTP / NextN

MTP는 Multi-Token Prediction의 약자입니다. 어떤 모델은 본체 안에 "다음 토큰 여러 개"를 예측하기
위한 추가 레이어를 갖고 있습니다. SGLang에서는 이런 MTP 레이어를 speculative decoding의 draft
역할로 사용합니다.

코드에서는 MTP를 `NextN`, `nextn`, `num_nextn_predict_layers` 같은 이름으로도 자주 부릅니다.

중요한 점은 다음입니다.

```text
MTP를 켠다
= 별도 MTP algorithm을 선택한다는 뜻이 아님
= EAGLE speculative decoding 경로를 사용한다는 뜻
```

그래서 MTP 예제도 보통 다음처럼 실행합니다.

```bash
python -m sglang.launch_server \
  --model some-mtp-enabled-model \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

## 현재 지원되는 speculative algorithm 값

CLI에서 `--speculative-algorithm`으로 받을 수 있는 값은 다음입니다.

```text
DFLASH
EAGLE
EAGLE3
NEXTN
STANDALONE
NGRAM
```

코드 위치:

- [`server_args.py`](../python/sglang/srt/server_args.py) - `--speculative-algorithm` choices
- [`spec_info.py`](../python/sglang/srt/speculative/spec_info.py) - `SpeculativeAlgorithm` enum

주의할 점:

- CLI의 choices는 대문자 기준입니다. 보통 `EAGLE`처럼 대문자로 줘야 합니다.
- 내부 enum에는 `EAGLE`과 `EAGLE3`가 별도 값으로 존재합니다.
- 다만 `is_eagle()` 체크에서는 `EAGLE`과 `EAGLE3`를 모두 EAGLE family로 취급합니다.

## `NEXTN`은 어떻게 처리되나?

`NEXTN`은 사용자가 넣을 수는 있지만, 서버 args 처리 중 바로 `EAGLE`로 바뀝니다.

```python
if self.speculative_algorithm == "NEXTN":
    self.speculative_algorithm = "EAGLE"
```

그래서 다음 두 설정은 결과적으로 같은 EAGLE 경로를 탑니다.

```bash
--speculative-algorithm NEXTN
```

```bash
--speculative-algorithm EAGLE
```

단, 문서를 읽을 때 `NEXTN`이라고 쓰인 예제가 있으면 보통 "MTP/NextN 레이어를 EAGLE 경로로 쓴다"는
의미로 보면 됩니다.

## `EAGLE`만 주면 어떤 버전이 사용되나?

질문에서 가장 중요한 부분입니다.

```bash
--speculative-algorithm EAGLE
```

이렇게만 주면 SGLang은 **EAGLE-2 계열**로 처리합니다. EAGLE-3로 자동 업그레이드되지 않습니다.

EAGLE-3를 쓰려면 반드시 이렇게 줘야 합니다.

```bash
--speculative-algorithm EAGLE3
```

내부 흐름은 다음과 같습니다.

1. CLI parser가 `EAGLE`을 `speculative_algorithm` 값으로 받습니다.
2. `NEXTN`인 경우에만 `EAGLE`로 변환합니다. 이미 `EAGLE`이면 그대로 둡니다.
3. `SpeculativeAlgorithm.from_string("EAGLE")`이 `SpeculativeAlgorithm.EAGLE`을 반환합니다.
4. `is_eagle()`은 true가 되지만, `is_eagle3()`은 false입니다.
5. 따라서 EAGLE-3 전용 처리에는 들어가지 않습니다.

## `EAGLE`만 주고 세부값을 생략하면 기본값은?

다음 세 값을 생략하면 SGLang이 자동으로 고릅니다.

```text
--speculative-num-steps
--speculative-eagle-topk
--speculative-num-draft-tokens
```

자동 선택 함수는 `auto_choose_speculative_params()`입니다.

대표 기본값은 다음과 같습니다.

| 모델 계열 | 자동 선택값 |
| --- | --- |
| Llama, Grok | `(5, 4, 8)` |
| DeepSeek V2/V3/V3.2, GPT-OSS, GLM MoE, Bailing MoE, MistralLarge3, Pixtral, MiMoV2 등 | `(3, 1, 4)` |
| 그 외 대부분 | `(3, 1, 4)` |

튜플의 의미는 다음 순서입니다.

```text
(speculative_num_steps, speculative_eagle_topk, speculative_num_draft_tokens)
```

예를 들어 기본값이 `(3, 1, 4)`이면 다음과 같습니다.

```bash
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

또한 `speculative_eagle_topk == 1`인데 `speculative_num_draft_tokens`가
`speculative_num_steps + 1`이 아니면, 코드가 `num_draft_tokens`를 `num_steps + 1`로 맞춥니다.

예:

```text
num_steps = 3
topk = 1
num_draft_tokens should be 4
```

## MTP에서 draft model path는 꼭 필요한가?

일반 EAGLE/EAGLE3는 보통 별도의 draft model path가 필요합니다.

```bash
--speculative-draft-model-path some/eagle-draft-model
```

하지만 MTP-enabled 모델은 모델 자체 안의 MTP/NextN 레이어를 draft처럼 사용할 수 있습니다. 그래서 일부
모델은 `--speculative-draft-model-path`를 생략해도 서버 args 처리 중 target model path를 draft path로
사용하도록 맞춥니다.

현재 코드에서 이런 자동 처리가 명시된 target architecture 예시는 다음입니다.

```text
DeepseekV32ForCausalLM
DeepseekV3ForCausalLM
Glm4MoeForCausalLM
Glm4MoeLiteForCausalLM
GlmMoeDsaForCausalLM
BailingMoeForCausalLM
BailingMoeV2ForCausalLM
BailingMoeV2_5ForCausalLM
MistralLarge3ForCausalLM
PixtralForConditionalGeneration
HYV3ForCausalLM
```

이 경우 `--speculative-algorithm EAGLE`만 줘도 draft path가 target model path로 채워질 수 있습니다.

## MTP draft architecture 변환

MTP 모델을 draft worker로 띄울 때는 target architecture를 draft용 architecture로 바꿔서 로드하는
처리가 있습니다. 예를 들면 다음과 같습니다.

| target architecture | draft architecture |
| --- | --- |
| `DeepseekV3ForCausalLM` | `DeepseekV3ForCausalLMNextN` |
| `Glm4MoeForCausalLM` | `Glm4MoeForCausalLMNextN` |
| `Glm4MoeLiteForCausalLM` | `Glm4MoeForCausalLMNextN` |
| `GlmOcrForConditionalGeneration` | `GlmOcrForConditionalGenerationNextN` |
| `LongcatFlashForCausalLM` | `LongcatFlashForCausalLMNextN` |
| `MiMoForCausalLM` | `MiMoMTP` |
| `MiMoV2ForCausalLM` | `MiMoV2MTP` |
| `Step3p5ForCausalLM` | `Step3p5MTP` |
| `BailingMoe*ForCausalLM` | `BailingMoeForCausalLMNextN` |
| `Ernie4_5_MoeForCausalLM` | `Ernie4_5_MoeForCausalLMMTP` |
| `Qwen3NextForCausalLM` | `Qwen3NextForCausalLMMTP` |
| `Qwen3_5*ForConditionalGeneration` | `Qwen3_5ForCausalLMMTP` |
| `ExaoneMoEForCausalLM` | `ExaoneMoEForCausalLMMTP` |
| `NemotronHForCausalLM` | `NemotronHForCausalLMMTP` |
| `HYV3ForCausalLM` | `HYV3ForCausalLMNextN` |

이 변환은 [`model_config.py`](../python/sglang/srt/configs/model_config.py)의
`_config_draft_model()`에서 처리합니다.

## EAGLE worker v1 / v2와 EAGLE-2 / EAGLE-3는 다른 개념

헷갈리기 쉬운 부분입니다.

SGLang 코드에는 다음 두 종류의 "버전" 표현이 함께 나옵니다.

### 1. Algorithm version

이건 논문/알고리즘 쪽 버전입니다.

```text
EAGLE  -> EAGLE-2
EAGLE3 -> EAGLE-3
```

### 2. Worker / scheduler version

이건 SGLang 구현 쪽 버전입니다.

```text
EAGLEWorker   -> 기존 speculative worker
EAGLEWorkerV2 -> overlap scheduler 기반 worker
```

즉 `EAGLEWorkerV2`라고 해서 EAGLE-3를 의미하는 것은 아닙니다.

예를 들어 `--speculative-algorithm EAGLE`로 EAGLE-2를 쓰더라도, overlap scheduler가 켜져 있으면
구현체는 `EAGLEWorkerV2`를 사용할 수 있습니다.

반대로 `--speculative-algorithm EAGLE3`는 알고리즘이 EAGLE-3라는 뜻입니다.

## 자주 쓰는 설정 예시

### DeepSeek V3/V3.2 MTP

DeepSeek 계열 MTP는 보통 다음처럼 사용합니다.

```bash
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3.2-Exp \
  --tp 8 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

세부값을 모두 생략하면 대부분 이 조합에 가까운 `(3, 1, 4)`가 자동 선택됩니다.

### MiMo MTP

문서 예시는 더 작은 speculative 설정을 사용합니다.

```bash
python -m sglang.launch_server \
  --model XiaomiMiMo/MiMo-7B-RL \
  --trust-remote-code \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2
```

MTP는 모델과 batch size에 따라 최적 조합이 달라질 수 있으므로, 실제 성능은 benchmark로 확인하는 것이
좋습니다.

### EAGLE-3

EAGLE-3는 `EAGLE`이 아니라 `EAGLE3`를 명시해야 합니다.

```bash
python -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16
```

## 체크리스트

MTP/EAGLE 설정을 볼 때는 아래 순서로 확인하면 됩니다.

1. `--speculative-algorithm` 값이 무엇인가?
   - `EAGLE`: EAGLE-2 계열
   - `EAGLE3`: EAGLE-3
   - `NEXTN`: 내부에서 `EAGLE`로 변환

2. 모델이 MTP/NextN 레이어를 가진 모델인가?
   - 그렇다면 target model 안의 MTP layer가 draft 역할을 할 수 있습니다.
   - 코드에서 draft architecture 변환이 있는지 확인합니다.

3. 세부 speculative 값이 명시되어 있는가?
   - 없으면 `auto_choose_speculative_params()` 기본값을 봅니다.
   - MTP에서는 흔히 `(3, 1, 4)` 또는 작은 batch 최적화를 위해 `(1, 1, 2)`를 씁니다.

4. EAGLE-3가 필요한가?
   - 필요하면 `EAGLE`이 아니라 `EAGLE3`를 써야 합니다.

5. 별도 draft model이 필요한가?
   - 일반 EAGLE/EAGLE3는 보통 필요합니다.
   - MTP-enabled 모델은 생략 가능한 경우가 있습니다.

## 코드 위치 요약

| 보고 싶은 내용 | 파일 |
| --- | --- |
| CLI args choices | [`server_args.py`](../python/sglang/srt/server_args.py) |
| `NEXTN -> EAGLE` 변환 | [`server_args.py`](../python/sglang/srt/server_args.py) |
| EAGLE/EAGLE3 enum | [`spec_info.py`](../python/sglang/srt/speculative/spec_info.py) |
| EAGLE worker 선택 | [`spec_info.py`](../python/sglang/srt/speculative/spec_info.py) |
| speculative 기본값 자동 선택 | [`server_args.py`](../python/sglang/srt/server_args.py) |
| MTP draft architecture 변환 | [`model_config.py`](../python/sglang/srt/configs/model_config.py) |
| EAGLE 일반 worker | [`eagle_worker.py`](../python/sglang/srt/speculative/eagle_worker.py) |
| EAGLE overlap worker | [`eagle_worker_v2.py`](../python/sglang/srt/speculative/eagle_worker_v2.py) |
| 공식 speculative decoding 설명 | [`speculative_decoding.md`](../docs/advanced_features/speculative_decoding.md) |

## 결론

`--speculative-algorithm EAGLE`만 주면 SGLang은 EAGLE-2 계열의 speculative decoding으로 처리합니다.
MTP 모델에서도 이 EAGLE 경로를 사용합니다. EAGLE-3를 쓰고 싶다면 `EAGLE3`를 명시해야 하며,
`NEXTN`은 이름만 다를 뿐 내부적으로는 `EAGLE`로 바뀝니다.
