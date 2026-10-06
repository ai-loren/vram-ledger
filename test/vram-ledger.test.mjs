import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { runInNewContext } from 'node:vm';

const context = { window: {} };
runInNewContext(await readFile(new URL('../vram-ledger.js', import.meta.url), 'utf8'), context, { filename: 'vram-ledger.js' });
const ledger = context.window.SITE_TOOLS['vram-ledger'];
const GiB = 1024 ** 3;
const llama8b = { layers: 32, hidden: 4096, heads: 32, kvHeads: 8, ffn: 14336, vocab: 128256 };
const base = { model: llama8b, seq: 4096, micro: 1, tp: 1, pp: 1, dp: 8, zero: 1, recompute: 'none', attention: 'flash', memory: 80 };
const config = (patch) => ({ ...base, ...patch });

test('preset parameter counts match the published model sizes', () => {
  const count = (id) => ledger.parameterCount(ledger.MODELS.find((model) => model.id === id));
  assert.equal(count('llama3-8b'), 8_030_261_248);
  assert.equal(count('llama3-70b'), 70_553_706_496);
  // Qwen2.5 has QKV biases that the ledger leaves out: 28 × (3584 + 2 × 512) and 80 × (8192 + 2 × 1024) parameters.
  assert.equal(count('qwen25-7b'), 7_615_616_512 - 28 * 4608);
  assert.equal(count('qwen25-72b'), 72_706_203_648 - 80 * 10240);
});

test('static memory follows 2 + 2 + 12 bytes per parameter, sharded by model parallelism then ZeRO stage', () => {
  const P = 8_030_261_248;
  const parts = (patch) => ledger.estimate(config(patch)).parts;
  assert.deepEqual([parts({ zero: 0 }).weights, parts({ zero: 0 }).grads, parts({ zero: 0 }).optimizer], [2 * P, 2 * P, 12 * P]);
  assert.deepEqual([parts({ zero: 1 }).weights, parts({ zero: 1 }).grads, parts({ zero: 1 }).optimizer], [2 * P, 2 * P, 12 * P / 8]);
  assert.deepEqual([parts({ zero: 2 }).weights, parts({ zero: 2 }).grads, parts({ zero: 2 }).optimizer], [2 * P, 2 * P / 8, 12 * P / 8]);
  assert.deepEqual([parts({ zero: 3 }).weights, parts({ zero: 3 }).grads, parts({ zero: 3 }).optimizer], [2 * P / 8, 2 * P / 8, 12 * P / 8]);
  const sharded = parts({ zero: 0, tp: 2, pp: 4 });
  assert.equal(sharded.weights, 2 * P / 8);
  assert.equal(sharded.optimizer, 12 * P / 8);
});

test('activations add up the Llama layer tensors, the output layer and the attention matrix when it is stored', () => {
  const layer = 12 * 4096 * 4096 + 4 * 4096 * 1024 + 6 * 4096 * 14336;
  const output = 4 * 4096 * 4096 + 4 * 4096 * 128256;
  assert.equal(layer, 570_425_344);
  assert.equal(ledger.estimate(base).parts.activations, 32 * layer + output);
  assert.equal(ledger.estimate(config({ micro: 2 })).parts.activations, 2 * (32 * layer + output));
  assert.equal(ledger.estimate(config({ attention: 'naive' })).parts.activations, 32 * (layer + 2 * 32 * 4096 * 4096) + output);
  assert.equal(ledger.estimate(config({ tp: 2 })).parts.activations, (32 * layer + output) / 2);
  assert.equal(ledger.estimate(config({ recompute: 'full' })).parts.activations, 32 * 2 * 4096 * 4096 + layer + output);
  assert.equal(ledger.estimate(config({ recompute: 'full', tp: 2 })).parts.activations, (32 * 2 * 4096 * 4096 + layer + output) / 2);
});

test('under 1F1B the first stage holds as many layers of activations as the whole model, the last stage adds the logits', () => {
  const layer = 570_425_344;
  const output = 4 * 4096 * 4096 + 4 * 4096 * 128256;
  const firstHeavy = ledger.estimate(config({ pp: 4 }));
  assert.equal(firstHeavy.stage, 'first');
  assert.equal(firstHeavy.parts.activations, 32 * layer);
  const lastHeavy = ledger.estimate(config({ pp: 4, recompute: 'full', seq: 1024 }));
  const recomputeLayer = (12 * 1024 * 4096 + 4 * 1024 * 1024 + 6 * 1024 * 14336);
  const outputShort = 4 * 1024 * 4096 + 4 * 1024 * 128256;
  assert.equal(lastHeavy.stage, 'last');
  assert.equal(lastHeavy.parts.activations, 8 * 2 * 1024 * 4096 + recomputeLayer + outputShort);
});

test('the default configuration is Llama 3 8B on eight 80 GB cards and fits', () => {
  const result = ledger.estimate({ ...base, model: ledger.MODELS[0] });
  assert.equal((result.total / GiB).toFixed(1), '60.2');
  assert.equal(result.level, 'fit');
  assert.equal(result.gpus, 8);
  const defaults = ledger.DEFAULTS;
  assert.deepEqual([defaults.model, defaults.seq, defaults.micro, defaults.tp, defaults.pp, defaults.dp, defaults.zero, defaults.recompute, defaults.attention, defaults.gpu],
    ['llama3-8b', '4096', '1', '1', '1', '8', '1', 'none', 'flash', 'h100']);
  assert.equal(ledger.GPUS.find((gpu) => gpu.id === 'h100').memory, 80);
});

test('verdict uses 90% of capacity as the safe line', () => {
  const total = ledger.estimate(base).total;
  const at = (memory) => ledger.estimate(config({ memory })).level;
  assert.equal(at(Math.ceil(total / GiB / 0.9) + 1), 'fit');
  assert.equal(at(Math.ceil(total / GiB)), 'tight');
  assert.equal(at(Math.floor(total / GiB / 0.9)), 'tight', 'a card just below total / 0.9 is already past the safe line');
  assert.equal(at(Math.floor(total / GiB)), 'over');
});

test('impossible layouts are rejected with the expected value and the next step', () => {
  const error = (patch) => ledger.estimate(config(patch)).error;
  assert.match(error({ tp: 8, model: { ...llama8b, kvHeads: 4 } }), /TP 要能整除 KV 头数.*4 ÷ 8.*1 \/ 2 \/ 4/);
  assert.match(error({ tp: 4, model: { ...llama8b, heads: 30, kvHeads: 6, hidden: 3840 } }), /TP 要能整除注意力头数.*30 ÷ 4.*1 \/ 2/);
  assert.match(error({ pp: 8, model: { ...llama8b, layers: 28 } }), /PP 要能整除层数.*28 ÷ 8.*1 \/ 2 \/ 4/);
  assert.match(error({ model: { ...llama8b, hidden: 4100 } }), /隐藏维度要能被注意力头数整除/);
  assert.match(error({ model: { ...llama8b, kvHeads: 5 } }), /注意力头数要是 KV 头数的整数倍/);
  assert.equal(error({}), undefined);
});

test('suggestions are real alternatives, sorted by memory saved, and keep the GPU count when they trade DP away', () => {
  const tips = ledger.suggestions(config({ model: ledger.MODELS[1], zero: 0 }));
  assert(tips.length >= 2, `expected several ways to save memory on 70B, got ${JSON.stringify(tips)}`);
  for (let index = 1; index < tips.length; index += 1) assert(tips[index - 1].saved >= tips[index].saved, 'tips are sorted by memory saved');
  for (const tip of tips) {
    const after = ledger.estimate(config({ model: ledger.MODELS[1], zero: 0, ...tip.change }));
    assert.equal(tip.total, after.total, `${tip.label} reports the total it leads to`);
    if (tip.change.dp) assert.equal(after.gpus, 8, `${tip.label} keeps the GPU count`);
  }
  assert.equal(ledger.suggestions(config({ recompute: 'full', zero: 3, micro: 1, tp: 8, pp: 16, dp: 1, model: { ...llama8b, layers: 32 } })).length, 0);
});
