# OCR Processing Benchmark

## Problem
- 88,671 PDFs (2025-26) need OCR extraction with Docling + Tesseract
- Malayalam + English text, tables, forms
- Phase 1: Thrissur district (9,294 PDFs)

## Benchmark Results (25 sample PDFs)

| Machine | Pipeline | Workers | Avg/PDF | PDFs/hr | Wall time | Thrissur Est |
|---------|----------|---------|---------|---------|-----------|--------------|
| 4-core 16GB | Standard | 4 | 324s | 38 | ~2.2h | 246h (10d) |
| 4-core 16GB | Threaded | 4 | 1615s | 8 | ~2.8h | 1226h |
| **8-core 32GB** | **Standard** | **8** | **282s** | **92** | **16.8m** | **101h (4.2d)** |
| 8-core 32GB + ONNX fix | Standard | 8 | TBD | TBD | TBD | TBD |

**Key findings:**
- `ThreadedPdfPipelineOptions` = 5x slower on 4-core, do not use
- 8-core / 8-worker = **6.99x speedup**, 92 PDFs/hr
- Load settled at 10-12 (1.3-1.5x cores) — Docling's ONNX model spawns ~1-2 background threads per worker regardless of OMP settings
- Memory: 75% peak (23GB / 31GB) — safe headroom

## ONNX Fix (pending test)
Added env vars + programmatic thread pinning to suppress internal model threads:
```python
os.environ["ORT_NUM_INTER_THREADS"] = "1"   # ONNX Runtime
os.environ["ORT_NUM_INTRA_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```
Expected: load drops from ~11 → ~9, ~10-15% faster.

## Per-File Data (8-core / 8-worker run)

| File | Pages | Time |
|------|-------|------|
| Municipali_Kannur_417 | 5 | 135.7s |
| Municipali_Idukki_147 | 5 | 138.5s |
| District_P_Alappuzh_673 | 5 | 158.4s |
| Grama_Panc_Kasargod_442 | 5 | 166.7s |
| Municipali_Kasargod_624 | 13 | 167.4s |
| Corporatio_Ernakula_1125 | 5 | 183.7s |
| Block_Panc_Kasargod_134 | 5 | 189.1s |
| Block_Panc_Idukki_95 | 5 | 190.6s |
| Corporatio_Thiruvan_1174 | 13 | 198.8s |
| District_P_Kannur_225 | 14 | 209.3s |
| District_P_Ernakula_5 | 13 | 217.2s |
| Corporatio_Kozhikod_1371 | 13 | 219.3s |
| Grama_Panc_Idukki_42 | 15 | 234.0s |
| District_P_Kasargod_90 | 15 | 242.2s |
| Block_Panc_Kannur_34 | 15 | 256.2s |
| Municipali_Ernakula_229 | 12 | 285.7s |
| Municipali_Alappuzh_268 | 12 | 379.0s |
| Grama_Panc_Kannur_287 | 12 | 413.4s |
| Block_Panc_Alappuzh_99 | 13 | 446.5s |
| District_P_Idukki_105 | 15 | 465.4s |
| Corporatio_Kannur_423 | 13 | 473.5s |
| Grama_Panc_Ernakula_57 | 12 | 500.6s |
| Block_Panc_Ernakula_83 | 12 | 670.7s |

## Thrissur Plan (9,294 PDFs)

| Option | Setup | Est. Time | Est. Cost |
|--------|-------|-----------|-----------|
| Single machine (this) | 8-core, 8w | 101h (4.2d) | ~$12 |
| 4 machines | 4× n2-standard-8 | ~25h (1d) | ~$40 |
| 10 machines (fast) | 10× n2-standard-8 | ~10h | ~$31 |

## Commands

```bash
# Run ONNX-fixed benchmark
uv run python etc/13_benchmark_comparison.py --workers 8

# Compare all results
uv run python etc/13_benchmark_comparison.py --compare

# Run Thrissur district (after download)
uv run python etc/10_parallel_tesseract_test.py --workers 8
```

## District Breakdown (2025-26)

| District | PDFs | 1-machine Est |
|----------|------|---------------|
| Thrissur | 9,294 | 101h |
| Malappuram | 9,653 | 105h |
| Kannur | 8,531 | 93h |
| Ernakulam | 8,208 | 89h |
| Kottayam | 7,583 | 82h |
| Wayanad | 2,784 | 30h |
| **Total** | **88,671** | **~964h** |
