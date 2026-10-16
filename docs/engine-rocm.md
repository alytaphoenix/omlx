# ROCm (AMD) engine

oMLX runs its models on MLX, and MLX has no ROCm backend: an AMD GPU (Strix
Halo, RDNA/RDNA3, gfx1151 and friends) can never be an MLX rank. What an AMD
box *can* do is serve an OpenAI-compatible API from a ROCm-native engine, and
oMLX can drive that node as a first-class backend — the same product surface
(`/v1/chat/completions`, `/v1/completions`, tools, streaming, the dashboard and
usage stats), with the compute living on the AMD machine.

The model stays in oMLX's model list like any other. The ROCm engine is an
*external engine*, alongside [Splash](Splash.md): selected per model with the
**Use external engine** setting, nothing is loaded in-process.

## When to use it

- You have an AMD GPU host (a Strix Halo mini-PC, a Radeon workstation, a
  ROCm-capable server) running `gufo`, `llama-server` (llama.cpp) or any other
  OpenAI-compatible server built with ROCm/HIP.
- You want oMLX to expose those models through its API, dashboard and clients
  without the model ever touching the Mac's Unified Memory.
- You want to mix Apple-silicon MLX models and AMD ROCm models in one oMLX
  instance, and let each request go to the node that serves it best.

## Setup

1. On the AMD host, run an OpenAI-compatible ROCm server. Any server that
   speaks the OpenAI API works; [gufo](https://github.com/ggml-org/llama.cpp)
   and `llama-server` are the tested ones. Note the base URL and the model id
   it reports on `GET /v1/models`.

2. In oMLX, open the model's **Model Settings → Advanced**, turn on **Use
   external engine**, and pick **ROCm (AMD)**.

3. Give it the endpoint in **External engine options** (the
   `external_engine_config` field; JSON):

   ```json
   {
     "base_url": "http://10.99.0.2:8731",
     "model": "gufo-flash-next",
     "api_key": "",
     "tokenizer": "/path/to/the/model/tokenizer"
   }
   ```

   | Key | Required | Meaning |
   | --- | --- | --- |
   | `base_url` | yes | The ROCm server's base URL. Aliases: `url`, `endpoint`. |
   | `model` | yes | The upstream model id the server advertises. Aliases: `model_name`, `upstream_model`. |
   | `api_key` | no | Sent as `Authorization: Bearer …`. |
   | `tokenizer` | no | A local tokenizer (HF repo id or path) for exact prompt token counts; an estimate is used otherwise. |
   | `label` | no | Display label in stats (for example `valhalla (ROCm)`). |
   | `verify` | no | TLS verification (default `true`; set `false` for a self-signed certificate on a private link). |
   | `extra_headers` | no | Object of extra request headers. |
   | `request_timeout_s` | no | Read timeout (default 3600 s, to outlast a long prefill). |
   | `start_timeout_s` | no | How long `start()` waits for the node to advertise the model (default 60 s). |
   | `supports_tools` | no | Whether to forward a `tools` array (default `true`). |

   An endpoint that is reachable but does not advertise the model fails the
   load with the reason (and the ids it *did* see), rather than silently
   falling back to an MLX engine.

Under the hood the options are stored in the model's `external_engine_config`
model setting, so they persist with the model and participate in profiles.

## What it does

`RocmEngine` proxies an OpenAI-compatible endpoint:

- **Chat** — `stream_chat` / `chat` POST to `/v1/chat/completions`, streaming
  SSE by default and reassembling text, `tool_calls`, `finish_reason` and usage
  (including `prompt_tokens_per_second` / `completion_tokens_per_second` when
  the node reports them).
- **Completions** — `stream_generate` / `generate` POST to `/v1/completions`.
- **Tools** — a `tools` array is forwarded and tool calls are accumulated
  across streamed fragments into structured `GenerationOutput.tool_calls`.
- **Reasoning** — servers that stream chain-of-thought as `reasoning_content`
  (vLLM, llama.cpp/gufo, and Qwen3.8 Flash-Next by default) are wrapped in
  `<think>…</think>` inline, exactly as the Splash engine does, so oMLX's
  shared `ThinkingParser` splits them into the OpenAI `reasoning_content`
  field. `chat_template_kwargs` is forwarded to the endpoint (minus
  `reasoning_effort`), which is how `enable_thinking: false` turns thinking off
  on Qwen templates; `reasoning_effort` is mapped and forwarded separately.
- **Sampling** — `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`,
  `stop`, `seed` and `response_format` are forwarded when set; neutral values
  are omitted so servers that reject unknown llama.cpp-style fields stay happy.

`GET /v1/models` is used at `start()` to confirm the node is reachable and the
model is present, and by the dashboard for live stats. `GET /health` is probed
too when present.

## Memory

The weights live on the AMD node, so a ROCm model is counted as **zero**
resident memory in oMLX. The local footprint is an HTTP client. This is the
point of the feature: a model too large for the Mac runs elsewhere while oMLX
keeps its full memory for MLX models. Loading a ROCm model never evicts an MLX
model, and unloading it starts no local process to stop.

Because the node owns its own KV cache, oMLX's hot-cache budget and the SSD
tier do not apply to ROCm models; `get_cache_stats()` returns `None`.

## Distributed / heterogeneous clusters

This engine is the AMD half of a heterogeneous setup. MLX handles the Apple
silicon side (including the network cluster path, JACCL/RDMA), and ROCm nodes
serve their own models. A request is routed to whichever model the client
asks for; there is no MLX rank on the AMD box and no shared KV. For a
Thunderbolt-attached node, use a static private subnet (for example
`10.99.0.0/24`) rather than link-local `169.254/16`, which macOS may route via
the wrong interface.

## Benchmarks and stats

ROCm models appear in the dashboard's external-engine card with their endpoint,
upstream model id and request count. The admin benchmark and the usage
breakdown work as for any external model; throughput numbers come from the
node's usage payload.

## Limits

- The model must already be running on the node; oMLX does not start or stop a
  remote server. (A future revision may allow a local launcher command.)
- Plain-text completion requires the endpoint to implement `/v1/completions`.
  Servers that only implement chat return a clear error.
- The other acceleration settings (MTP, DFlash, TurboQuant, ANE prefill, and so
  on) configure oMLX's MLX engines and do not apply while the ROCm engine
  serves the model.
