(() => {
  'use strict';

  const GiB = 1024 ** 3;
  const SAFE_RATIO = 0.9;
  const MODELS = [
    { id: 'llama3-8b', name: 'Llama 3 8B', layers: 32, hidden: 4096, heads: 32, kvHeads: 8, ffn: 14336, vocab: 128256 },
    { id: 'llama3-70b', name: 'Llama 3 70B', layers: 80, hidden: 8192, heads: 64, kvHeads: 8, ffn: 28672, vocab: 128256 },
    { id: 'qwen25-7b', name: 'Qwen2.5 7B', layers: 28, hidden: 3584, heads: 28, kvHeads: 4, ffn: 18944, vocab: 152064 },
    { id: 'qwen25-72b', name: 'Qwen2.5 72B', layers: 80, hidden: 8192, heads: 64, kvHeads: 8, ffn: 29568, vocab: 152064 },
  ];
  const GPUS = [
    { id: 'rtx4090', name: 'RTX 4090 · 24 GB', memory: 24 },
    { id: 'a100-40', name: 'A100 · 40 GB', memory: 40 },
    { id: 'h100', name: 'A100 / H100 · 80 GB', memory: 80 },
    { id: 'h20', name: 'H20 · 96 GB', memory: 96 },
    { id: 'h200', name: 'H200 · 141 GB', memory: 141 },
    { id: 'b200', name: 'B200 · 192 GB', memory: 192 },
  ];
  const MODEL_FIELDS = [
    ['layers', '层数', 1024], ['hidden', '隐藏维度', 131072], ['heads', '注意力头数', 1024],
    ['kvHeads', 'KV 头数', 1024], ['ffn', 'FFN 维度', 524288], ['vocab', '词表大小', 4194304],
  ];
  const PARTS = [['weights', '参数'], ['grads', '梯度'], ['optimizer', '优化器状态'], ['activations', '激活']];
  const DEFAULTS = {
    model: 'llama3-8b', ...MODELS[0], seq: '4096', micro: '1', tp: '1', pp: '1', dp: '8',
    zero: '1', recompute: 'none', attention: 'flash', gpu: 'h100',
  };

  // 词嵌入与输出层按不共享权重计（Llama 3、Qwen2.5 都不共享），不计线性层的 bias。
  function parameterCount(model) {
    const kvWidth = model.kvHeads * (model.hidden / model.heads);
    const perLayer = 2 * model.hidden * model.hidden + 2 * model.hidden * kvWidth + 3 * model.hidden * model.ffn + 2 * model.hidden;
    return model.layers * perLayer + 2 * model.vocab * model.hidden + model.hidden;
  }

  const divisorsOf = (n, candidates) => candidates.filter((value) => n % value === 0).join(' / ');

  function validate(config) {
    const { model, tp, pp } = config;
    if (model.hidden % model.heads) return `隐藏维度要能被注意力头数整除，现在是 ${model.hidden} ÷ ${model.heads}。改一下头数或隐藏维度。`;
    if (model.kvHeads > model.heads || model.heads % model.kvHeads) return `注意力头数要是 KV 头数的整数倍，现在是 ${model.heads} 与 ${model.kvHeads}。把 KV 头数改成头数的约数。`;
    if (model.heads % tp) return `TP 要能整除注意力头数，现在是 ${model.heads} ÷ ${tp}。把 TP 改成 ${divisorsOf(model.heads, [1, 2, 4, 8])} 之一。`;
    if (model.kvHeads % tp) return `TP 要能整除 KV 头数（按 KV 组切分），现在是 ${model.kvHeads} ÷ ${tp}。把 TP 改成 ${divisorsOf(model.kvHeads, [1, 2, 4, 8])} 之一。`;
    if (model.layers % pp) return `PP 要能整除层数，否则各段层数不一样；现在是 ${model.layers} ÷ ${pp}。把 PP 改成 ${divisorsOf(model.layers, [1, 2, 4, 8, 16])} 之一。`;
    return '';
  }

  // 返回每张卡上最紧的那一段的显存（字节）。公式与假设见作品详情页「怎么算的」。
  function estimate(config) {
    const error = validate(config);
    if (error) return { error };
    const { model, seq: s, micro: b, tp: t, pp: p, dp: d, zero } = config;
    const h = model.hidden;
    const kvWidth = model.kvHeads * (h / model.heads);
    const params = parameterCount(model);
    const shard = params / (t * p);
    const weights = (zero >= 3 ? 2 / d : 2) * shard;
    const grads = (zero >= 2 ? 2 / d : 2) * shard;
    const optimizer = (zero >= 1 ? 12 / d : 12) * shard;

    const layerFull = (12 * s * b * h + 4 * s * b * kvWidth + 6 * s * b * model.ffn + (config.attention === 'naive' ? 2 * model.heads * s * s * b : 0)) / t;
    const layerStored = config.recompute === 'full' ? (2 * s * b * h) / t : layerFull;
    const replay = config.recompute === 'full' ? layerFull : 0;
    const layersPerStage = model.layers / p;
    const outputLayer = (4 * s * b * h + 4 * s * b * model.vocab) / t;
    const first = layersPerStage * layerStored * p + replay;
    const last = layersPerStage * layerStored + replay + outputLayer;
    const stage = p === 1 ? 'only' : first >= last ? 'first' : 'last';
    const activations = p === 1 ? first + outputLayer : Math.max(first, last);

    const parts = { weights, grads, optimizer, activations };
    const total = weights + grads + optimizer + activations;
    const capacity = config.memory * GiB;
    const level = total <= capacity * SAFE_RATIO ? 'fit' : total <= capacity ? 'tight' : 'over';
    return { params, parts, total, capacity, level, stage, gpus: t * p * d };
  }

  function suggestions(config) {
    const base = estimate(config);
    if (base.error) return [];
    const { model, tp, pp, dp, micro, zero } = config;
    const candidates = [];
    if (config.recompute === 'none') candidates.push(['开全量激活重计算', { recompute: 'full' }]);
    if (config.attention === 'naive') candidates.push(['换成 FlashAttention', { attention: 'flash' }]);
    if (dp > 1 && zero < 3) candidates.push([`ZeRO-${zero} → ZeRO-${zero + 1}`, { zero: zero + 1 }]);
    if (dp % 2 === 0 && tp * 2 <= 8 && model.heads % (tp * 2) === 0 && model.kvHeads % (tp * 2) === 0) {
      candidates.push([`TP ${tp} → ${tp * 2}，DP ${dp} → ${dp / 2}（总卡数不变）`, { tp: tp * 2, dp: dp / 2 }]);
    }
    if (dp % 2 === 0 && pp * 2 <= 16 && model.layers % (pp * 2) === 0) {
      candidates.push([`PP ${pp} → ${pp * 2}，DP ${dp} → ${dp / 2}（总卡数不变）`, { pp: pp * 2, dp: dp / 2 }]);
    }
    if (micro >= 2) candidates.push([`micro-batch ${micro} → ${Math.floor(micro / 2)}（用梯度累积补回全局 batch）`, { micro: Math.floor(micro / 2) }]);
    return candidates
      .map(([label, change]) => {
        const next = { ...config, ...change };
        const result = estimate(next);
        return { label, change, total: result.total, saved: base.total - result.total };
      })
      .filter((option) => option.saved >= 0.1 * GiB)
      .sort((a, b) => b.saved - a.saved)
      .slice(0, 3);
  }

  const gib = (bytes) => (bytes / GiB).toFixed(1);
  const options = (items, selected) => items.map(([value, label]) => `<option value="${value}"${value === selected ? ' selected' : ''}>${label}</option>`).join('');
  const field = (label, control) => `<label class="vram-field"><span>${label}</span>${control}</label>`;
  const number = (name, max) => `<input type="number" inputmode="numeric" name="${name}" min="1" max="${max}" step="1">`;
  const select = (name, items) => `<select name="${name}">${options(items)}</select>`;

  let remembered = { ...DEFAULTS };

  function mount(root) {
    root.innerHTML = `
      <div class="vram">
        <form class="vram-form" novalidate>
          <fieldset><legend>模型</legend>
            ${field('预设', select('model', [...MODELS.map((model) => [model.id, model.name]), ['custom', '自定义']]))}
            <div class="vram-custom">${MODEL_FIELDS.map(([name, label, max]) => field(label, number(name, max))).join('')}</div>
            <p class="vram-params" data-out="params"></p>
          </fieldset>
          <fieldset><legend>序列与批次</legend>
            ${field('序列长度', number('seq', 1048576))}
            ${field('micro-batch', number('micro', 1024))}
          </fieldset>
          <fieldset><legend>并行</legend>
            ${field('TP', select('tp', [1, 2, 4, 8].map((n) => [String(n), String(n)])))}
            ${field('PP', select('pp', [1, 2, 4, 8, 16].map((n) => [String(n), String(n)])))}
            ${field('DP', number('dp', 65536))}
            <p class="vram-gpus" data-out="gpus"></p>
          </fieldset>
          <fieldset><legend>优化器与激活</legend>
            ${field('ZeRO', select('zero', [['0', '0 · 不切分'], ['1', '1 · 切优化器状态'], ['2', '2 · 再切梯度'], ['3', '3 · 再切参数']]))}
            ${field('激活重计算', select('recompute', [['none', '不重算'], ['full', '全量重算']]))}
            ${field('注意力', select('attention', [['flash', 'FlashAttention'], ['naive', '朴素实现（存注意力矩阵）']]))}
          </fieldset>
          <fieldset><legend>显卡</legend>
            ${field('型号', select('gpu', GPUS.map((gpu) => [gpu.id, gpu.name])))}
          </fieldset>
        </form>
        <div class="vram-peek" aria-hidden="true"><span data-out="peek-verdict"></span><b data-out="peek-total"></b></div>
        <section class="vram-result" aria-label="估算结果">
          <div class="vram-headline" aria-live="polite">
            <p class="vram-verdict" data-out="verdict"></p>
            <p class="vram-total"><strong data-out="total"></strong><span data-out="capacity"></span></p>
          </div>
          <div class="vram-bar" aria-hidden="true">
            <div class="vram-fill">${PARTS.map(([key]) => `<i data-part="${key}"></i>`).join('')}</div>
            <span class="vram-mark is-safe"></span><span class="vram-mark is-capacity"></span>
          </div>
          <ul class="vram-legend">${PARTS.map(([key, label]) => `<li data-part="${key}"><span>${label}</span><b data-out="${key}"></b></li>`).join('')}</ul>
          <p class="vram-stage" data-out="stage"></p>
          <div class="vram-tips" data-out="tips"></div>
          <p class="vram-error" role="alert" data-out="error" hidden></p>
        </section>
      </div>`;

    const form = root.querySelector('.vram-form');
    const out = (name) => root.querySelector(`[data-out="${name}"]`);
    const current = () => Object.fromEntries([...form.elements].filter((element) => element.name).map((element) => [element.name, element.value]));
    const write = (values) => {
      for (const element of form.elements) if (element.name && element.name in values) element.value = String(values[element.name]);
      root.querySelector('.vram-custom').hidden = values.model !== 'custom';
    };

    const read = () => {
      const values = current();
      const integer = (name, label, max) => {
        const raw = values[name].trim();
        if (!/^\d+$/.test(raw) || Number(raw) < 1) throw new Error(`${label}需要是正整数，现在是「${raw || '空'}」。填一个大于 0 的整数。`);
        if (Number(raw) > max) throw new Error(`${label}最大支持 ${max}，现在是 ${raw}。填一个小一点的值。`);
        return Number(raw);
      };
      const preset = MODELS.find((model) => model.id === values.model);
      const model = preset || Object.fromEntries(MODEL_FIELDS.map(([name, label, max]) => [name, integer(name, label, max)]));
      return {
        values,
        config: {
          model, seq: integer('seq', '序列长度', 1048576), micro: integer('micro', 'micro-batch', 1024),
          tp: Number(values.tp), pp: Number(values.pp), dp: integer('dp', 'DP', 65536), zero: Number(values.zero),
          recompute: values.recompute, attention: values.attention, memory: GPUS.find((gpu) => gpu.id === values.gpu).memory,
        },
      };
    };

    const showError = (message) => {
      out('error').textContent = message;
      out('error').hidden = false;
      out('peek-verdict').textContent = '配置有误，见下方提示';
      out('peek-total').textContent = '';
      root.querySelector('.vram').dataset.state = 'error';
    };

    const render = () => {
      let state;
      try {
        state = read();
      } catch (error) {
        showError(error.message);
        return;
      }
      remembered = { ...state.values };
      const { config } = state;
      const result = estimate(config);
      if (result.error) {
        showError(result.error);
        return;
      }
      out('error').hidden = true;
      root.querySelector('.vram').dataset.state = result.level;
      out('params').textContent = `参数量约 ${(result.params / 1e9).toFixed(2)}B`;
      out('gpus').textContent = `总卡数 = TP × PP × DP = ${result.gpus}`;
      out('verdict').textContent = { fit: '放得下', tight: '很紧：超过了 90% 安全线', over: '放不下' }[result.level];
      out('total').textContent = gib(result.total);
      out('capacity').textContent = ` / ${config.memory} GiB 每卡`;
      out('peek-verdict').textContent = { fit: '放得下', tight: '很紧', over: '放不下' }[result.level];
      out('peek-total').textContent = `${gib(result.total)} / ${config.memory} GiB`;
      const scale = Math.max(result.total, result.capacity);
      for (const [key] of PARTS) {
        const share = result.parts[key] / scale;
        root.querySelector(`.vram-fill [data-part="${key}"]`).style.setProperty('width', `${(share * 100).toFixed(3)}%`);
        out(key).textContent = `${gib(result.parts[key])} GiB`;
      }
      root.querySelector('.vram-mark.is-safe').style.setProperty('left', `${((result.capacity * SAFE_RATIO) / scale) * 100}%`);
      root.querySelector('.vram-mark.is-capacity').style.setProperty('left', `${(result.capacity / scale) * 100}%`);
      out('stage').textContent = {
        only: '激活按 1 个 micro-batch 计，含输出层的 fp32 logits。',
        first: `激活按流水线第一段计：1F1B 下它同时存着 ${config.pp} 个 micro-batch 的激活。`,
        last: '激活按流水线最后一段计：它要额外存输出层的 fp32 logits。',
      }[result.stage];

      const tips = suggestions(config);
      out('tips').innerHTML = tips.length
        ? `<p class="vram-tips-title">还能这样省</p>${tips.map((tip, index) => `<button type="button" class="vram-tip" data-tip="${index}"><span>${tip.label}</span><b>省 ${gib(tip.saved)} GiB，剩 ${gib(tip.total)}</b></button>`).join('')}`
        : '';
      out('tips').querySelectorAll('[data-tip]').forEach((button) => {
        button.addEventListener('click', () => {
          write({ ...current(), ...tips[Number(button.dataset.tip)].change });
          render();
        });
      });
    };

    form.addEventListener('submit', (event) => event.preventDefault());
    form.addEventListener('input', (event) => {
      if (event.target.name === 'model') {
        const preset = MODELS.find((model) => model.id === event.target.value);
        write(preset ? { ...current(), ...preset, model: preset.id } : current());
      }
      render();
    });
    write(remembered);
    render();
  }

  window.SITE_TOOLS = { ...(window.SITE_TOOLS || {}), 'vram-ledger': { MODELS, GPUS, DEFAULTS, parameterCount, estimate, suggestions, mount } };
})();
