# Nemotron-H prefill kernels on gfx90a

These opt-in paths target
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` on one TP8/EP8 MI250X
node. They require vLLM `0.24.0+lumi_aif_gfx90a_ee0da84` and fail closed
when the model family, device, scheduler bound, tensor contract, or vLLM ABI
does not match the measured setup.

Enable the paths independently:

```bash
export PRIME_ROCM_NEMOTRON_RELU2=1
export PRIME_ROCM_NEMOTRON_MOE_CONFIG=1
export PRIME_ROCM_NEMOTRON_ROUTER_LINEAR=1
```

Keep `max_num_batched_tokens` at or below 32,768. The dispatches accept token
ranges rather than one exact prompt length, so the same chunk kernels can serve
128K and 256K requests.

`PRIME_ROCM_NEMOTRON_RELU2` also requires the narrow vLLM source hook in
`docs/vllm-ee0da84-nemotron-ep-relu2-hook.patch`. The hook exposes the existing
expert assignment to `TritonExperts.activation`; PRIME-RL then runs the local
ReLU-squared kernel only over rows assigned to that EP rank.

`PRIME_ROCM_NEMOTRON_ROUTER_LINEAR` keeps the primary FP32 router weight and
adds a non-persistent BF16 copy made after checkpoint loading. Batches of at
most 512 tokens keep vLLM's FP32 decode path. Larger batches use BF16 MFMA with
FP32 accumulation and output. The extra copy costs 160 MiB per GPU across the
40 router layers.

On a 130,560-token prompt with 32,768-token chunks:

| Stack | Request time |
|---|---:|
| AITER-attention TP8/EP8 baseline | 12.904 s |
| + expert-parallel ReLU-squared | 11.592 s |
| + BF16-to-FP32 prefill router | 10.886 s |

The combined reduction is 15.64%. The ReLU-squared kernel itself predates this
integration; the measured improvement comes from skipping non-local EP rows.

At a 262,144-token model length, the same 32,768-token chunk dispatches
completed a 261,632-token prompt in 24.582 seconds (10,643 prompt tokens/s).
The server retained 19.09 GiB for KV state, or 3.17 full-length requests.
