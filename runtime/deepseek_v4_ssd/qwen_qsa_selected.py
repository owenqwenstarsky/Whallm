"""Default-off QSA Prefill attention that reads only the selected key blocks.

For query position a the key set is the masked path's: the keys of every chosen
complete block (block end before ((a + 1) // 4) * 4) and the causal tail of a's
incomplete block. Scores, softmax and accumulation stay in FP32; only the output
is rounded to BF16. The masked path rounds scores to BF16, so results differ at
that level. Cost does not grow with the context length, unlike masked SDPA.
"""
from functools import lru_cache

import mlx.core as mx

# One threadgroup per (query, KV head). SIMD group s handles query heads
# [s * RS, (s + 1) * RS) of the KV head's group; each lane holds 8 of the 256
# dimensions, so key and value rows load as one coalesced 512-byte read. Keys
# are taken B at a time with one softmax rescale per batch.
SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const int sg = int(simdgroup_index_in_threadgroup);
    const int qi = int(threadgroup_position_in_grid.y);
    const int g = int(threadgroup_position_in_grid.z);
    const int Q = params[0];
    const int C = params[1];
    const int K = params[2];
    const int a = params[3] + qi;
    const int L = ((a + 1) / RATIO) * RATIO;
    const int slots = C * RATIO + RATIO;
    const int d0 = int(lane) * 8;
    const float factor = scale[0];

    float qv[RS][8], o[RS][8], m[RS], l[RS];
    for (int r = 0; r < RS; ++r) {
        const device bfloat16_t* row = q + ((size_t)(g * R + sg * RS + r) * Q + qi) * D + d0;
        for (int i = 0; i < 8; ++i) { qv[r][i] = float(row[i]) * factor; o[r][i] = 0.0f; }
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }
    const device bfloat16_t* kb = k + (size_t)g * K * D + d0;
    const device bfloat16_t* vb = v + (size_t)g * K * D + d0;

    for (int base = 0; base < slots; base += B) {
        int pos[B];
        float s[B][RS];
        for (int j = 0; j < B; ++j) {
            const int slot = base + j;
            int p = -1;
            if (slot < C * RATIO) {
                const int block = chosen[(size_t)qi * C + slot / RATIO];
                if (block * RATIO + RATIO <= L) p = block * RATIO + slot % RATIO;
            } else if (slot < slots) {
                const int t = L + (slot - C * RATIO);
                if (t <= a) p = t;
            }
            pos[j] = p;
            if (p < 0) { for (int r = 0; r < RS; ++r) s[j][r] = -INFINITY; continue; }
            float kv[8];
            const device bfloat16_t* row = kb + (size_t)p * D;
            for (int i = 0; i < 8; ++i) kv[i] = float(row[i]);
            for (int r = 0; r < RS; ++r) {
                float partial = 0.0f;
                for (int i = 0; i < 8; ++i) partial += qv[r][i] * kv[i];
                s[j][r] = simd_sum(partial);
            }
        }
        for (int r = 0; r < RS; ++r) {
            float batch_max = s[0][r];
            for (int j = 1; j < B; ++j) batch_max = max(batch_max, s[j][r]);
            const float m_new = max(m[r], batch_max);
            if (m_new == -INFINITY) continue;
            const float correction = fast::exp(m[r] - m_new);
            l[r] *= correction;
            for (int i = 0; i < 8; ++i) o[r][i] *= correction;
            for (int j = 0; j < B; ++j) {
                const float w = fast::exp(s[j][r] - m_new);
                s[j][r] = w;
                l[r] += w;
            }
            m[r] = m_new;
        }
        for (int j = 0; j < B; ++j) {
            if (pos[j] < 0) continue;
            float vv[8];
            const device bfloat16_t* row = vb + (size_t)pos[j] * D;
            for (int i = 0; i < 8; ++i) vv[i] = float(row[i]);
            for (int r = 0; r < RS; ++r)
                for (int i = 0; i < 8; ++i) o[r][i] += s[j][r] * vv[i];
        }
    }
    for (int r = 0; r < RS; ++r) {
        const float inverse = 1.0f / l[r];
        device bfloat16_t* out = y + ((size_t)(g * R + sg * RS + r) * Q + qi) * D + d0;
        for (int i = 0; i < 8; ++i) out[i] = bfloat16_t(o[r][i] * inverse);
    }
"""

HEAD_DIM = 256
HEADS_PER_SIMDGROUP = 2
KEYS_PER_BATCH = 4


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="whallm_qsa_selected_blocks_v1",
        input_names=["q", "k", "v", "chosen", "params", "scale"],
        output_names=["y"], source=SOURCE, ensure_row_contiguous=True,
    )


def supported(query, key) -> bool:
    heads, kv_heads = query.shape[1], key.shape[1]
    return (mx.metal.is_available() and query.ndim == key.ndim == 4 and query.shape[0] == key.shape[0] == 1
            and query.shape[-1] == key.shape[-1] == HEAD_DIM and heads % kv_heads == 0
            and (heads // kv_heads) % HEADS_PER_SIMDGROUP == 0
            and query.dtype == key.dtype == mx.bfloat16)


def selected_block_attention(query, key, value, chosen, offset: int, ratio: int, scale: float):
    """query [1, Hq, Q, 256]; key, value [1, Hkv, K, 256]; chosen [Q, C] block indices."""
    _, heads, queries, dimension = query.shape
    kv_heads, keys = key.shape[1], key.shape[2]
    per_kv = heads // kv_heads
    groups = per_kv // HEADS_PER_SIMDGROUP
    params = mx.array([queries, chosen.shape[1], keys, offset], dtype=mx.int32)
    (output,) = _kernel()(
        inputs=[query[0], key[0], value[0], chosen.astype(mx.int32), params,
                mx.array([scale], dtype=mx.float32)],
        template=[("D", HEAD_DIM), ("R", per_kv), ("RS", HEADS_PER_SIMDGROUP), ("RATIO", ratio),
                  ("B", KEYS_PER_BATCH)],
        grid=(32 * groups, queries, kv_heads), threadgroup=(32 * groups, 1, 1),
        output_shapes=[(heads, queries, dimension)], output_dtypes=[query.dtype],
    )
    return output[None]
