# VRAM Ledger · 大模型训练显存账本

估算大模型训练时每张卡要用多少显存：把它拆成参数、梯度、优化器状态和激活四笔账，算上 TP / PP / DP、ZeRO 阶段、激活重计算和注意力实现；放不下时，按能省多少列出几种改法。

Estimate per-GPU memory for LLM training, split into parameters, gradients, optimizer states and activations under TP / PP / DP, ZeRO and activation recomputation, with ranked suggestions when it does not fit.

**在线使用：** [ai-loren.github.io/personal-homepage/#projects/vram-ledger](https://ai-loren.github.io/personal-homepage/#projects/vram-ledger)

纯前端、零依赖、不联网：一个 `vram-ledger.js` 文件，计算部分是纯函数。

## 用法

在页面里引入脚本后，工具注册在 `window.SITE_TOOLS['vram-ledger']` 上。

```html
<script src="vram-ledger.js"></script>
<script>
  const ledger = window.SITE_TOOLS['vram-ledger'];
  const result = ledger.estimate({
    model: { layers: 32, hidden: 4096, heads: 32, kvHeads: 8, ffn: 14336, vocab: 128256 }, // Llama 3 8B
    seq: 4096, micro: 1, tp: 1, pp: 1, dp: 8,
    zero: 1, recompute: 'none', attention: 'flash',
    memory: 80, // 每卡显存，GiB
  });
  // result.total / 1024 ** 3 ≈ 60.2，result.level === 'fit'
</script>
```

| 导出 | 说明 |
| --- | --- |
| `estimate(config)` | 返回 `{ params, parts: { weights, grads, optimizer, activations }, total, capacity, level, stage, gpus }`，显存单位是字节；配置不成立时返回 `{ error }`，错误信息写明期望什么、现在是什么、该改成什么 |
| `suggestions(config)` | 最多三种省显存的改法，按省下的量排序，每项是 `{ label, change, total, saved }`；改 TP / PP 的建议会同时减半 DP，总卡数不变 |
| `parameterCount(model)` | 按模型结构算参数量 |
| `mount(element)` | 在元素里渲染一个完整的表单和结果面板；只输出带 `vram-*` 类名的结构，样式由宿主页面提供 |
| `MODELS` / `GPUS` / `DEFAULTS` | 内置的模型预设、显卡型号和表单默认值 |

`config` 的字段：

- `model`：`layers` 层数、`hidden` 隐藏维度、`heads` 注意力头数、`kvHeads` KV 头数、`ffn` FFN 维度、`vocab` 词表大小
- `seq` 序列长度，`micro` micro-batch，`tp` / `pp` / `dp` 并行度
- `zero`：0–3；`recompute`：`'none'` 或 `'full'`；`attention`：`'flash'` 或 `'naive'`
- `memory`：每卡显存（GiB）

`level` 是 `'fit'`（不超过容量的 90%）、`'tight'`（超过 90% 但没超容量）或 `'over'`；`stage` 说明激活按哪一段流水线计：`'only'`、`'first'` 或 `'last'`。

## 怎么算的

- **参数、梯度与优化器状态**：按 bf16 混合精度加 Adam 计，每个参数 2 + 2 + 12 字节，12 字节是 fp32 的主权重、一阶矩和二阶矩。模型先按 TP × PP 切开，ZeRO 再按阶段把优化器状态、梯度、参数依次除以 DP。
- **激活**：按 Llama 式解码层（RMSNorm、GQA、SwiGLU、无 dropout）把反向要用的张量逐个加起来，每层每个 micro-batch 存 `12sbh + 4sb·h_kv + 6sbf` 字节；朴素注意力再加 `2as²b` 的注意力矩阵。TP 大于 1 时按开启序列并行计，全部除以 TP。
- **全量重算**：每层只留输入的 `2sbh/t`，反向时再临时展开一层的全部激活。
- **流水线**按 1F1B：第一段要同时存 PP 个 micro-batch 的激活，所以它的激活和不开流水线时一样多；最后一段只存一个，但要多存输出层的输入和 fp32 logits（`4sbh/t + 4sbV/t`）。结果取两段里更紧的那一段，并假设 micro-batch 数不少于段数。
- **参数量**：词嵌入与输出层按不共享权重计，不计线性层的 bias。Llama 3 8B / 70B 与官方参数量逐位一致；Qwen2.5 少算了 QKV 的 bias（不到十万分之二）。
- **没有算进去的**：CUDA 上下文、通信缓冲、显存碎片和临时张量，以及 MoE、LoRA、FP8 这类配置。所以把容量的 90% 当作安全线。

## 参考

- Rajbhandari et al. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://dl.acm.org/doi/10.5555/3433701.3433727). SC 2020
- Korthikanti et al. [Reducing Activation Recomputation in Large Transformer Models](https://proceedings.mlsys.org/paper_files/paper/2023/hash/80083951326cf5b35e5100260d64ed81-Abstract-mlsys2023.html). MLSys 2023
- Narayanan et al. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://dl.acm.org/doi/10.1145/3458817.3476209). SC 2021

## 测试

```bash
node --test
```

期望值都是按公式手算的字面数字，不从被测代码推导。需要 Node 18 以上，不用安装依赖。

## 和个人主页的关系

这里是正本。[个人主页](https://github.com/ai-loren/personal-homepage)的 `tools/vram-ledger.js` 是逐字节复制件，样式也在主页里；改动先在这里改、测试通过，再复制过去。

## License

[MIT](LICENSE)
