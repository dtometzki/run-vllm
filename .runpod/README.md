![vLLM worker banner](https://cpjrphpz3t5wbwfe.public.blob.vercel-storage.com/worker-vllm_banner.jpeg)

Run LLMs using [vLLM](https://docs.vllm.ai) with an OpenAI-compatible API

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-vllm)](https://www.runpod.io/console/hub/runpod-workers/worker-vllm)

---
# vLLM Worker Deployment

This repository provides a high-performance LLM serving solution using [vLLM](https://docs.vllm.ai) with an OpenAI-compatible API layer, optimized for RunPod Serverless environments.

## 🛠 Configuration

Behavior is managed through environment variables. Below are the primary settings used in this deployment:

**Pass any vLLM engine arg** not listed above by setting an env var with the **UPPERCASED** field name (e.g. `MAX_MODEL_LEN=4096`, `ENABLE_CHUNKED_PREFILL=true`). The worker auto-discovers all `AsyncEngineArgs` fields from env. See the [vLLM engine args docs](https://docs.vllm.ai/en/latest/configuration/engine_args) for all available options.

For complete configuration options, see the [full configuration documentation](https://github.com/runpod-workers/worker-vllm/blob/main/docs/configuration.md).

## API Usage

This worker supports two API formats: **RunPod native** and **OpenAI-compatible**.

### RunPod Native API

For testing directly in the RunPod UI, use these examples in your endpoint's request tab.

#### Chat Completions

```json
{
  "input": {
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "What is the capital of France?" }
    ],
    "sampling_params": {
      "max_tokens": 100,
      "temperature": 0.7
    }
  }
}
```

#### Chat Completions (Streaming)

```json
{
  "input": {
    "messages": [
      { "role": "user", "content": "Write a short story about a robot." }
    ],
    "sampling_params": {
      "max_tokens": 500,
      "temperature": 0.8
    },
    "stream": true
  }
}
```

#### Text Generation

For direct text generation without chat format:

```json
{
  "input": {
    "prompt": "The capital of France is",
    "sampling_params": {
      "max_tokens": 64,
      "temperature": 0.0
    }
  }
}
```

#### List Models

```json
{
  "input": {
    "openai_route": "/v1/models"
  }
}
```

---

## 🚀 API Usage

The worker supports both **OpenAI-compatible** and **RunPod Native** API formats.

### OpenAI-Compatible API

Use this for seamless integration with existing SDKs. Set your base URL to:
`https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`

**Python Example:**

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("RUNPOD_API_KEY"),
    base_url="https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
)

response = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    temperature=0.7,
    max_tokens=256
)

print(response.choices[0].message.content)

```

### RunPod Native API

For direct requests via the RunPod UI or specialized integrations:

**Endpoint:** `https://api.runpod.ai/v2/<ENDPOINT_ID>/run`

```json
{
  "input": {
    "messages": [
      { "role": "user", "content": "Explain the benefits of vLLM." }
    ],
    "sampling_params": {
      "max_tokens": 500,
      "temperature": 0.8
    }
  }
}

```

---

## ℹ️ Additional Features

* **Streaming:** Supported on both API paths by setting `"stream": true`.
* **Compatibility:** Supports all architectures recognized by the vLLM engine.
* **Auto-Tool Choice:** Can be enabled via `ENABLE_AUTO_TOOL_CHOICE` if the model supports it.

---
