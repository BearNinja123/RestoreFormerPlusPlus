|Modification|Mem usage (MiB)|Mem usage fraction (relative to baseline)|Iters/sec|Speedup (relative to baseline)|
|------------|---------------|-----------------------------------------|---------|------------------------------|
|Pre-Dec 13 (given code, baseline)|33743|1.0|1.22|1.0|
|16-mixed precision|28057|0.831|1.51|1.238|
|Optimized attention + using Pytorch FlashAttention|28023|0.830|1.51|1.238|
|Optimized vector quantization + replace x*sigmoid(x) -> F.silu|22617|0.670|1.63|1.336|
|optimizer fused=True|22615|0.670|1.74|1.426|
|use only bf16 (except normalization)|14569|0.432|2.12|1.734|
|compile loss function|13969|0.414|2.38|1.951|

Notes:
- batch size 4, 512x512
- VRAM measured from nvidia-smi
- tests run on a single 3g.40gb A100 MIG instance