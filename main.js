/* AIP-Speech site — data from acl_latex 1.pdf (Tables 1-5). No invented numbers. */
(function () {
  "use strict";

  /* ------------------------------------------------------------------ */
  /* Data                                                                */
  /* ------------------------------------------------------------------ */

  var MODELS = [
    { id: "qwen2audio", label: "Qwen2-Audio",      color: "var(--m-qwen2audio)" },
    { id: "omni7b",     label: "Qwen2.5-Omni-7B",  color: "var(--m-omni7b)" },
    { id: "omni3b",     label: "Qwen2.5-Omni-3B",  color: "var(--m-omni3b)" },
    { id: "phi4mm",     label: "Phi-4-MM",         color: "var(--m-phi4mm)" },
    { id: "gemma3n",    label: "Gemma 3n",         color: "var(--m-gemma3n)" }
  ];

  // Table 2 — Spearman rank correlation (r_s) between descriptor and degradation.
  // "*" flags Qwen2-Audio KWS values measured under the unsteered baseline protocol.
  var CORR = {
    ASR: {
      "Zero Crossing Rate":        { pooled: 0.645, qwen2audio: 0.711, omni7b: 0.704, omni3b: 0.693, phi4mm: 0.672, gemma3n: 0.469 },
      "Spectral Centroid":         { pooled: 0.629, qwen2audio: 0.693, omni7b: 0.597, omni3b: 0.558, phi4mm: 0.642, gemma3n: 0.451 },
      "Spectral Rolloff":          { pooled: 0.626, qwen2audio: 0.687, omni7b: 0.586, omni3b: 0.544, phi4mm: 0.669, gemma3n: 0.448 },
      "Spectral Bandwidth":        { pooled: 0.550, qwen2audio: 0.621, omni7b: 0.522, omni3b: 0.462, phi4mm: 0.636, gemma3n: 0.337 },
      "Speech Fraction (VAD)":     { pooled: 0.316, qwen2audio: 0.314, omni7b: 0.406, omni3b: 0.397, phi4mm: 0.275, gemma3n: 0.254 },
      "Spectral Flux":             { pooled: 0.217, qwen2audio: 0.248, omni7b: 0.263, omni3b: 0.250, phi4mm: 0.192, gemma3n: 0.107 },
      "Spectral Flatness":         { pooled: 0.080, qwen2audio: 0.150, omni7b: 0.029, omni3b: -0.030, phi4mm: 0.099, gemma3n: 0.177 },
      "Spectral Contrast":         { pooled: -0.005, qwen2audio: 0.053, omni7b: 0.057, omni3b: 0.087, phi4mm: -0.065, gemma3n: -0.068 },
      "RMS Energy Variance":       { pooled: -0.217, qwen2audio: -0.233, omni7b: -0.247, omni3b: -0.179, phi4mm: -0.251, gemma3n: -0.096 }
    },
    KWS: {
      "Zero Crossing Rate":        { pooled: 0.723, qwen2audio: -0.126, omni7b: 0.585, omni3b: 0.647, phi4mm: 0.710, gemma3n: 0.678, star: true },
      "Spectral Centroid":         { pooled: 0.637, qwen2audio: -0.322, omni7b: 0.501, omni3b: 0.488, phi4mm: 0.674, gemma3n: 0.567, star: true },
      "Spectral Rolloff":          { pooled: 0.619, qwen2audio: -0.228, omni7b: 0.444, omni3b: 0.439, phi4mm: 0.656, gemma3n: 0.581, star: true },
      "Spectral Bandwidth":        { pooled: 0.544, qwen2audio: -0.118, omni7b: 0.406, omni3b: 0.402, phi4mm: 0.576, gemma3n: 0.519, star: true },
      "Speech Fraction (VAD)":     { pooled: 0.383, qwen2audio: 0.017, omni7b: 0.457, omni3b: 0.431, phi4mm: 0.362, gemma3n: 0.369, star: true },
      "Spectral Flux":             { pooled: 0.272, qwen2audio: -0.173, omni7b: 0.384, omni3b: 0.278, phi4mm: 0.277, gemma3n: 0.226, star: true },
      "Spectral Flatness":         { pooled: 0.117, qwen2audio: -0.243, omni7b: 0.025, omni3b: -0.029, phi4mm: 0.165, gemma3n: 0.038, star: true },
      "Spectral Contrast":         { pooled: 0.072, qwen2audio: -0.168, omni7b: -0.013, omni3b: 0.075, phi4mm: 0.062, gemma3n: -0.003, star: true },
      "RMS Energy Variance":       { pooled: -0.191, qwen2audio: 0.311, omni7b: -0.156, omni3b: -0.205, phi4mm: -0.217, gemma3n: -0.173, star: true }
    }
  };
  var DESCRIPTOR_ORDER = [
    "Zero Crossing Rate", "Spectral Centroid", "Spectral Rolloff", "Spectral Bandwidth",
    "Speech Fraction (VAD)", "Spectral Flux", "Spectral Flatness", "Spectral Contrast", "RMS Energy Variance"
  ];

  // Table 1 + Table 3 — 20 background recordings, source dataset, descriptor values.
  var BACKGROUNDS = [
    { name: "esc_wind",           source: "ESC-50",    zcr: 0.0226, centroid: 669.76,  bandwidth: 1330.72, rolloff: 1190.62, speech: 0.0000, flux: 0.9022, flatness: 0.0011, contrast: 17.2040, rmsVar: 0.000658 },
    { name: "esc_rooster",        source: "ESC-50",    zcr: 0.0822, centroid: 841.66,  bandwidth: 606.18,  rolloff: 1153.43, speech: 0.0000, flux: 0.6738, flatness: 0.4599, contrast: 27.7246, rmsVar: 0.001060 },
    { name: "esc_dog",            source: "ESC-50",    zcr: 0.0161, centroid: 218.08,  bandwidth: 142.86,  rolloff: 342.65,  speech: 0.0000, flux: 0.3479, flatness: 0.9091, contrast: 16.0911, rmsVar: 0.000359 },
    { name: "esc_clock",          source: "ESC-50",    zcr: 0.0288, centroid: 746.37,  bandwidth: 1351.25, rolloff: 1076.89, speech: 0.0021, flux: 1.7850, flatness: 0.0013, contrast: 19.2353, rmsVar: 0.001402 },
    { name: "snsd_airconditioner",source: "MS-SNSD",   zcr: 0.0333, centroid: 851.80,  bandwidth: 1375.42, rolloff: 1699.16, speech: 0.0000, flux: 0.9540, flatness: 0.0032, contrast: 15.7306, rmsVar: 0.000312 },
    { name: "esc_engine",         source: "ESC-50",    zcr: 0.0488, centroid: 1314.33, bandwidth: 1544.96, rolloff: 2801.37, speech: 0.0000, flux: 0.8921, flatness: 0.0088, contrast: 18.0697, rmsVar: 0.002576 },
    { name: "esc_fire",           source: "ESC-50",    zcr: 0.0912, centroid: 2264.88, bandwidth: 2236.70, rolloff: 5048.61, speech: 0.0000, flux: 2.1282, flatness: 0.0931, contrast: 16.1431, rmsVar: 0.001323 },
    { name: "snsd_square",        source: "MS-SNSD",   zcr: 0.0332, centroid: 1033.03, bandwidth: 1645.28, rolloff: 2351.40, speech: 0.1014, flux: 1.1128, flatness: 0.0049, contrast: 15.2164, rmsVar: 0.004249 },
    { name: "musan_music",        source: "MUSAN",     zcr: 0.1081, centroid: 1701.98, bandwidth: 1668.04, rolloff: 3462.07, speech: 0.3362, flux: 1.3270, flatness: 0.0215, contrast: 20.7004, rmsVar: 0.002288 },
    { name: "esc_keyboard",       source: "ESC-50",    zcr: 0.0848, centroid: 2388.24, bandwidth: 1807.72, rolloff: 4160.08, speech: 0.0000, flux: 3.3657, flatness: 0.1094, contrast: 17.3824, rmsVar: 0.000738 },
    { name: "esc_footsteps",      source: "ESC-50",    zcr: 0.1987, centroid: 2212.21, bandwidth: 1959.91, rolloff: 4548.02, speech: 0.0000, flux: 1.6759, flatness: 0.1182, contrast: 16.8711, rmsVar: 0.001814 },
    { name: "esc_vacuum",         source: "ESC-50",    zcr: 0.5089, centroid: 2755.58, bandwidth: 2109.97, rolloff: 5039.44, speech: 0.0000, flux: 0.9197, flatness: 0.0533, contrast: 19.2603, rmsVar: 0.000078 },
    { name: "esc_helicopter",     source: "ESC-50",    zcr: 0.2609, centroid: 3211.55, bandwidth: 2666.81, rolloff: 6452.88, speech: 0.0000, flux: 0.9847, flatness: 0.0935, contrast: 18.6647, rmsVar: 0.000018 },
    { name: "snsd_airport",       source: "MS-SNSD",   zcr: 0.1516, centroid: 1354.49, bandwidth: 766.39,  rolloff: 2120.14, speech: 0.0000, flux: 1.1973, flatness: 0.0002, contrast: 20.3577, rmsVar: 0.001119 },
    { name: "snsd_cafeteria",     source: "MS-SNSD",   zcr: 0.0744, centroid: 1446.31, bandwidth: 1905.25, rolloff: 3401.42, speech: 0.8367, flux: 1.1776, flatness: 0.0211, contrast: 15.6066, rmsVar: 0.000375 },
    { name: "noisex_babble",      source: "NOISEX-92", zcr: 0.0870, centroid: 1255.53, bandwidth: 1428.23, rolloff: 2535.08, speech: 0.4696, flux: 1.1716, flatness: 0.0114, contrast: 18.5808, rmsVar: 0.000327 },
    { name: "esc_seawave",        source: "ESC-50",    zcr: 0.1446, centroid: 1859.74, bandwidth: 1834.55, rolloff: 3791.06, speech: 0.0000, flux: 0.8943, flatness: 0.0655, contrast: 17.5634, rmsVar: 0.000899 },
    { name: "esc_rain",           source: "ESC-50",    zcr: 0.3420, centroid: 3129.37, bandwidth: 1921.83, rolloff: 5528.74, speech: 0.0000, flux: 0.9045, flatness: 0.2122, contrast: 18.5032, rmsVar: 0.000029 },
    { name: "snsd_restaurant",    source: "MS-SNSD",   zcr: 0.1472, centroid: 2110.20, bandwidth: 2106.19, rolloff: 4668.49, speech: 0.0171, flux: 1.0586, flatness: 0.0909, contrast: 14.9841, rmsVar: 0.000437 },
    { name: "musan_hubbub",       source: "MUSAN",     zcr: 0.1406, centroid: 1904.30, bandwidth: 1705.12, rolloff: 3701.78, speech: 0.6201, flux: 1.9962, flatness: 0.0648, contrast: 19.4705, rmsVar: 0.002083 }
  ];

  // Table 4 — mean ΔWER by accent subgroup, per model.
  var FAIRNESS = [
    { group: "Australian English",  gemma3n: -0.1316, phi4mm: 0.0924,  omni3b: 0.1309,  omni7b: 0.0906,  qwen2audio: 0.0681 },
    { group: "Canadian English",    gemma3n: -0.2173, phi4mm: -0.0915, omni3b: -0.2280, omni7b: 0.0866,  qwen2audio: 0.3350 },
    { group: "England English",     gemma3n: -0.2835, phi4mm: 0.0542,  omni3b: 0.0705,  omni7b: 0.0490,  qwen2audio: 0.0888 },
    { group: "Hong Kong English",   gemma3n: 0.8583,  phi4mm: 0.0583,  omni3b: 0.4083,  omni7b: 0.0625,  qwen2audio: -0.4125 },
    { group: "Indian English",      gemma3n: 0.5370,  phi4mm: 0.1628,  omni3b: 0.2425,  omni7b: 0.1967,  qwen2audio: 0.1713 },
    { group: "Irish English",       gemma3n: 0.2583,  phi4mm: 0.0461,  omni3b: 0.0500,  omni7b: 0.0356,  qwen2audio: 0.2628 },
    { group: "New Zealand English", gemma3n: 0.1071,  phi4mm: 0.0000,  omni3b: 0.0143,  omni7b: 0.0238,  qwen2audio: 0.6881 },
    { group: "Scottish English",    gemma3n: 0.2211,  phi4mm: 0.2164,  omni3b: -0.4461, omni7b: 0.0261,  qwen2audio: 0.1708 },
    { group: "US English",          gemma3n: 0.0854,  phi4mm: 0.0480,  omni3b: 0.0579,  omni7b: 0.0522,  qwen2audio: 0.0490 },
    { group: "Other / Misc.",       gemma3n: 0.2470,  phi4mm: 0.1526,  omni3b: 0.1425,  omni7b: 0.1453,  qwen2audio: 0.1915 }
  ];

  // Table 5 — Residual Effect Ratio (RER) after prompt steering.
  var RER = [
    { model: "Gemma 3n",         asr: 1.664, kws: 0.767 },
    { model: "Phi-4-MM",         asr: 2.423, kws: 0.549 },
    { model: "Qwen2.5-Omni-3B",  asr: 0.812, kws: 2.489 },
    { model: "Qwen2.5-Omni-7B",  asr: 1.064, kws: 1.681 },
    { model: "Qwen2-Audio",      asr: 5.989, kws: 1.000 }
  ];

  /* ------------------------------------------------------------------ */
  /* Small helpers                                                      */
  /* ------------------------------------------------------------------ */

  function el(tag, attrs, parent) {
    var e = document.createElementNS(tag === "svg" || parent === "svg" ? "http://www.w3.org/2000/svg" : null, tag);
    return e;
  }
  function svgEl(tag, attrs) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var k in attrs) { if (attrs.hasOwnProperty(k)) e.setAttribute(k, attrs[k]); }
    return e;
  }
  function fmt3(n) { return (n >= 0 ? "+" : "") + n.toFixed(3); }
  function fmt2(n) { return (n >= 0 ? "+" : "") + n.toFixed(2); }

  function makeTooltip(container) {
    var tip = document.createElement("div");
    tip.className = "viz-tooltip";
    container.style.position = "relative";
    container.appendChild(tip);
    return {
      show: function (anchorEl, html) {
        var cr = container.getBoundingClientRect();
        var ar = anchorEl.getBoundingClientRect();
        tip.innerHTML = html;
        tip.style.left = (ar.left - cr.left + ar.width / 2) + "px";
        tip.style.top = (ar.top - cr.top - 8) + "px";
        tip.classList.add("is-visible");
      },
      hide: function () { tip.classList.remove("is-visible"); }
    };
  }

  /* ------------------------------------------------------------------ */
  /* §01 Pipeline diagram                                                */
  /* ------------------------------------------------------------------ */

  // -- small line-art icons in ink-soft/accent, matching the mono/instrument aesthetic --
  function icoGroup(x, y) {
    return svgEl("g", { transform: "translate(" + x + "," + y + ")" });
  }
  function icoWaveform(x, y, w, seed, color) {
    var g = icoGroup(x, y);
    var n = 7, pts = [];
    for (var i = 0; i <= n; i++) {
      var amp = (Math.sin(i * seed + seed) * 0.5 + 0.5) * 8.5 + 2;
      pts.push((i * (w / n)).toFixed(1) + "," + (i % 2 === 0 ? -amp : amp).toFixed(1));
    }
    g.appendChild(svgEl("polyline", { points: pts.join(" "), fill: "none", stroke: color || "var(--ink-soft)", "stroke-width": 1.6, "stroke-linecap": "round", "stroke-linejoin": "round" }));
    return g;
  }
  function icoMic(x, y) {
    var g = icoGroup(x, y);
    g.appendChild(svgEl("rect", { x: -4, y: -9, width: 8, height: 13, rx: 4, fill: "var(--ink-soft)" }));
    g.appendChild(svgEl("path", { d: "M -8 -1 A 8 8 0 0 0 8 -1", fill: "none", stroke: "var(--ink-soft)", "stroke-width": 1.4 }));
    g.appendChild(svgEl("line", { x1: 0, y1: 7, x2: 0, y2: 11, stroke: "var(--ink-soft)", "stroke-width": 1.4 }));
    g.appendChild(svgEl("line", { x1: -4.5, y1: 11, x2: 4.5, y2: 11, stroke: "var(--ink-soft)", "stroke-width": 1.4, "stroke-linecap": "round" }));
    return g;
  }
  function icoScatter(x, y) {
    var g = icoGroup(x, y);
    [[-8, 5], [-4, 6], [-1, -1], [2, 2], [5, -5], [8, -7]].forEach(function (p) {
      g.appendChild(svgEl("circle", { cx: p[0], cy: p[1], r: 1.9, fill: "var(--accent)" }));
    });
    return g;
  }
  function icoBars(x, y) {
    var g = icoGroup(x, y);
    [4, 8, 6, 11].forEach(function (h, i) {
      g.appendChild(svgEl("rect", { x: -9 + i * 6, y: 8 - h, width: 4, height: h, rx: 0.5, fill: "var(--accent)" }));
    });
    return g;
  }
  function icoPeople(x, y) {
    var g = icoGroup(x, y);
    [-5, 5].forEach(function (dx) {
      g.appendChild(svgEl("circle", { cx: dx, cy: -5.5, r: 2.3, fill: "var(--ink-soft)" }));
      g.appendChild(svgEl("rect", { x: dx - 3.4, y: -3.4, width: 6.8, height: 9.5, rx: 3.4, fill: "var(--ink-soft)" }));
    });
    return g;
  }
  function icoGauge(x, y) {
    var g = icoGroup(x, y);
    g.appendChild(svgEl("path", { d: "M -9 4.5 A 9 9 0 0 1 9 4.5", fill: "none", stroke: "var(--ink-soft)", "stroke-width": 1.5 }));
    g.appendChild(svgEl("line", { x1: 0, y1: 4.5, x2: 5, y2: -4.5, stroke: "var(--accent)", "stroke-width": 1.6, "stroke-linecap": "round" }));
    g.appendChild(svgEl("circle", { cx: 0, cy: 4.5, r: 1.5, fill: "var(--ink-soft)" }));
    return g;
  }

  function renderPipeline() {
    var host = document.getElementById("pipeline-diagram");
    if (!host) return;
    var W = 1060, H = 336;
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "xMinYMin meet" });

    function box(x, y, w, h) {
      svg.appendChild(svgEl("rect", { x: x, y: y, width: w, height: h, rx: 3, fill: "var(--paper)", stroke: "var(--rule-strong)", "stroke-width": 1 }));
    }
    function title(x, y, w, text) {
      var t = svgEl("text", { x: x + w / 2, y: y, "text-anchor": "middle", class: "bar-label", style: "font-weight:600; font-size:10.5px; letter-spacing:0.03em; text-transform:uppercase; fill: var(--ink)" });
      t.textContent = text;
      svg.appendChild(t);
    }
    function line(x, y, text, opts) {
      var t = svgEl("text", Object.assign({ x: x, y: y, class: "axis-label", style: "font-size:10.5px; fill: var(--ink-soft)" }, opts || {}));
      t.textContent = text;
      svg.appendChild(t);
    }
    function arrowH(x1, x2, y) {
      svg.appendChild(svgEl("line", { x1: x1, y1: y, x2: x2 - 6, y2: y, stroke: "var(--ink-soft)", "stroke-width": 1 }));
      svg.appendChild(svgEl("path", { d: "M " + (x2 - 9) + " " + (y - 3.5) + " L " + x2 + " " + y + " L " + (x2 - 9) + " " + (y + 3.5), fill: "none", stroke: "var(--ink-soft)", "stroke-width": 1.2 }));
    }
    function arrowV(x, y1, y2) {
      svg.appendChild(svgEl("line", { x1: x, y1: y1, x2: x, y2: y2 - 6, stroke: "var(--ink-soft)", "stroke-width": 1 }));
      svg.appendChild(svgEl("path", { d: "M " + (x - 3.5) + " " + (y2 - 9) + " L " + x + " " + y2 + " L " + (x + 3.5) + " " + (y2 - 9), fill: "none", stroke: "var(--ink-soft)", "stroke-width": 1.2 }));
    }

    var rowTopY = 20, rowH1 = 184, gap = 16;

    // column 1: two stacked source boxes — icon gets its own row, never sharing the title's line
    var c1x = 16, c1w = 168, subH = (rowH1 - gap) / 2;
    box(c1x, rowTopY, c1w, subH);
    title(c1x, rowTopY + 16, c1w, "Clean Speech");
    svg.appendChild(icoWaveform(c1x + c1w / 2 - 13, rowTopY + 32, 26, 0.9, "var(--ink-soft)"));
    ["LibriSpeech", "Common Voice", "Speech Cmds v2"].forEach(function (t, i) { line(c1x + c1w / 2, rowTopY + 56 + i * 12, t, { "text-anchor": "middle" }); });

    var c1by = rowTopY + subH + gap;
    box(c1x, c1by, c1w, subH);
    title(c1x, c1by + 16, c1w, "Background Audio");
    svg.appendChild(icoWaveform(c1x + c1w / 2 - 13, c1by + 32, 26, 2.1, "var(--accent)"));
    ["ESC-50 · MS-SNSD", "NOISEX-92 · MUSAN"].forEach(function (t, i) { line(c1x + c1w / 2, c1by + 56 + i * 13, t, { "text-anchor": "middle" }); });

    // column 2: controlled mixture (spans both rows, fed from both source boxes)
    var c2x = c1x + c1w + 46, c2w = 158;
    box(c2x, rowTopY, c2w, rowH1);
    title(c2x, rowTopY + 16, c2w, "Controlled Mixture");
    svg.appendChild(icoWaveform(c2x + 18, rowTopY + 58, 20, 0.9, "var(--ink-soft)"));
    line(c2x + 50, rowTopY + 62, "+", { "text-anchor": "middle", "font-weight": "600" });
    svg.appendChild(icoWaveform(c2x + 62, rowTopY + 58, 20, 2.1, "var(--accent)"));
    svg.appendChild(svgEl("line", { x1: c2x + 14, y1: rowTopY + 80, x2: c2x + c2w - 14, y2: rowTopY + 80, class: "grid-line" }));
    svg.appendChild(icoWaveform(c2x + 50, rowTopY + 110, 30, 1.5, "var(--ink)"));
    line(c2x + c2w / 2, rowTopY + 140, "SNR = 10, 5, 0 dB", { "text-anchor": "middle", style: "font-size:10px; fill: var(--ink-soft)" });

    arrowH(c1x + c1w, c2x, rowTopY + subH / 2);
    var midY = rowTopY + subH / 2, lowY = c1by + subH / 2, joinX = c1x + c1w + 20;
    svg.appendChild(svgEl("path", { d: "M " + (c1x + c1w) + " " + lowY + " L " + joinX + " " + lowY + " L " + joinX + " " + (midY + 6), fill: "none", stroke: "var(--ink-soft)", "stroke-width": 1 }));
    svg.appendChild(svgEl("path", { d: "M " + (joinX - 3.5) + " " + (midY - 3) + " L " + joinX + " " + (midY + 6) + " L " + (joinX + 3.5) + " " + (midY - 3), fill: "none", stroke: "var(--ink-soft)", "stroke-width": 1.2 }));

    // column 3: speech LLM evaluation
    var c3x = c2x + c2w + 46, c3w = 148;
    box(c3x, rowTopY, c3w, rowH1);
    svg.appendChild(icoMic(c3x + c3w / 2, rowTopY + 34));
    title(c3x, rowTopY + 60, c3w, "Speech LLMs");
    ["Qwen2-Audio", "Qwen2.5-Omni 3B/7B", "Phi-4-MM", "Gemma 3n"].forEach(function (t, i) { line(c3x + c3w / 2, rowTopY + 82 + i * 15, t, { "text-anchor": "middle" }); });
    arrowH(c2x + c2w, c3x, rowTopY + rowH1 / 2);

    // column 4: paired evaluation
    var c4x = c3x + c3w + 46, c4w = 168;
    box(c4x, rowTopY, c4w, rowH1);
    title(c4x, rowTopY + 16, c4w, "Paired Evaluation");
    line(c4x + c4w / 2, rowTopY + 40, "Clean vs. noisy output", { "text-anchor": "middle" });
    line(c4x + c4w / 2, rowTopY + 66, "Δt = St(noisy) − St(clean)", { "text-anchor": "middle", style: "font-size:10.5px; fill: var(--ink); font-weight:600" });
    svg.appendChild(svgEl("line", { x1: c4x + 14, y1: rowTopY + 80, x2: c4x + c4w - 14, y2: rowTopY + 80, class: "grid-line" }));
    line(c4x + c4w / 2, rowTopY + 104, "ASR → WER", { "text-anchor": "middle" });
    line(c4x + c4w / 2, rowTopY + 122, "KWS → Accuracy", { "text-anchor": "middle" });
    arrowH(c3x + c3w, c4x, rowTopY + rowH1 / 2);

    // column 5: benchmark analyses (tall, 2x2 icon grid)
    var c5x = c4x + c4w + 46, c5w = 196;
    box(c5x, rowTopY, c5w, rowH1);
    title(c5x, rowTopY + 16, c5w, "Benchmark Analyses");
    var qx = [c5x + 46, c5x + c5w - 46], qy = [rowTopY + 64, rowTopY + 134];
    svg.appendChild(icoScatter(qx[0], qy[0])); line(qx[0], qy[0] + 22, "Descriptor rs", { "text-anchor": "middle" });
    svg.appendChild(icoBars(qx[1], qy[0])); line(qx[1], qy[0] + 22, "Bg. ranking", { "text-anchor": "middle" });
    svg.appendChild(icoPeople(qx[0], qy[1])); line(qx[0], qy[1] + 22, "Fairness", { "text-anchor": "middle" });
    svg.appendChild(icoGauge(qx[1], qy[1])); line(qx[1], qy[1] + 22, "RER steering", { "text-anchor": "middle" });
    arrowH(c4x + c4w, c5x, rowTopY + rowH1 / 2);

    // row 2: acoustic descriptor extraction, dashed into Benchmark Analyses
    var r2y = rowTopY + rowH1 + 32, r2h = 80;
    var descX = c1x, descW = c2x + c2w - c1x;
    box(descX, r2y, descW, r2h);
    arrowV(c1x + c1w / 2, c1by + subH, r2y);
    title(descX, r2y + 17, descW, "Acoustic Descriptor Extraction");
    line(descX + descW / 2, r2y + 38, "ZCR · Centroid · Rolloff · Bandwidth", { "text-anchor": "middle" });
    line(descX + descW / 2, r2y + 55, "Flux · Contrast · Flatness · RMS Var.", { "text-anchor": "middle" });

    var dashY = r2y + r2h / 2, dashX2 = c5x + c5w / 2;
    svg.appendChild(svgEl("path", {
      d: "M " + (descX + descW) + " " + dashY + " L " + dashX2 + " " + dashY + " L " + dashX2 + " " + (rowTopY + rowH1),
      fill: "none", stroke: "var(--ink-soft)", "stroke-width": 1, "stroke-dasharray": "3 3"
    }));
    svg.appendChild(svgEl("path", {
      d: "M " + (dashX2 - 3.5) + " " + (rowTopY + rowH1 - 8) + " L " + dashX2 + " " + (rowTopY + rowH1) + " L " + (dashX2 + 3.5) + " " + (rowTopY + rowH1 - 8),
      fill: "none", stroke: "var(--ink-soft)", "stroke-width": 1.2
    }));
    line((descX + descW + dashX2) / 2, dashY - 10, "extracted independently — used only for analysis", { "text-anchor": "middle", style: "font-size:9.5px; fill: var(--muted); font-style: italic" });

    host.innerHTML = "";
    host.appendChild(svg);
  }

  /* ------------------------------------------------------------------ */
  /* §02 Descriptor correlation chart                                    */
  /* ------------------------------------------------------------------ */

  var corrState = { task: "ASR", view: "pooled" };

  function sortedDescriptors(task) {
    return DESCRIPTOR_ORDER.slice().sort(function (a, b) {
      return CORR[task][b].pooled - CORR[task][a].pooled;
    });
  }

  function renderCorrChart() {
    var host = document.getElementById("corr-chart");
    var task = corrState.task;
    var descriptors = sortedDescriptors(task);
    host.innerHTML = "";

    if (corrState.view === "pooled") {
      host.appendChild(diverging1(descriptors, function (d) { return CORR[task][d].pooled; },
        function (d) { return CORR[task][d].star; }, "var(--accent)", 560));
    } else {
      var grid = document.createElement("div");
      grid.style.display = "grid";
      grid.style.gap = "22px";
      grid.style.gridTemplateColumns = window.innerWidth > 700 ? "1fr 1fr" : "1fr";
      MODELS.forEach(function (m) {
        var block = document.createElement("div");
        var label = document.createElement("div");
        label.className = "mono-cap";
        label.style.marginBottom = "6px";
        label.innerHTML = '<span class="legend__swatch" style="background:' + m.color + '"></span> ' + m.label;
        label.style.display = "flex"; label.style.alignItems = "center"; label.style.gap = "6px";
        block.appendChild(label);
        block.appendChild(diverging1(descriptors, function (d) { return CORR[task][d][m.id]; },
          function (d) { return CORR[task][d].star; }, m.color, 280));
        grid.appendChild(block);
      });
      host.appendChild(grid);
    }
  }

  // A single-series diverging horizontal bar chart around 0, given descriptor order + value fn.
  function diverging1(descriptors, valueFn, starFn, color, width) {
    var rowH = 30, padTop = 6, padBottom = 22;
    var H = descriptors.length * rowH + padTop + padBottom;
    var W = width;
    var compact = width < 400;
    var labelW = compact ? 112 : 168, plotX0 = labelW, plotW = W - labelW - (compact ? 38 : 54);
    var maxAbs = 0.8;
    var zeroX = plotX0 + plotW / 2;
    var scale = (plotW / 2) / maxAbs;

    var wrapper = document.createElement("div");
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "xMinYMin meet" });

    // grid + zero line
    var ticks = width < 400 ? [-0.6, 0, 0.6] : [-0.6, -0.3, 0, 0.3, 0.6];
    ticks.forEach(function (t) {
      var x = zeroX + t * scale;
      svg.appendChild(svgEl("line", {
        x1: x, y1: padTop - 2, x2: x, y2: H - padBottom + 2,
        class: t === 0 ? "axis-line" : "grid-line"
      }));
      var lab = svgEl("text", { x: x, y: H - padBottom + 14, "text-anchor": "middle", class: "axis-label" });
      lab.textContent = t.toFixed(1);
      svg.appendChild(lab);
    });

    descriptors.forEach(function (d, i) {
      var y = padTop + i * rowH;
      var val = valueFn(d);
      var barY = y + 6, barH = rowH - 12;
      var x = zeroX + Math.min(val, 0) * scale;
      var w = Math.abs(val) * scale;

      var name = svgEl("text", {
        x: labelW - 10, y: y + rowH / 2 + 4, "text-anchor": "end", class: "bar-label",
        style: compact ? "font-size:9.5px" : ""
      });
      name.textContent = compact ? d.replace(" (VAD)", "") : d;
      svg.appendChild(name);

      var rect = svgEl("rect", {
        x: x, y: barY, width: Math.max(w, 1), height: barH, rx: 2, fill: color, opacity: 0.85
      });
      rect.style.cursor = "pointer";
      svg.appendChild(rect);

      var valX = val >= 0 ? zeroX + val * scale + 6 : zeroX + val * scale - 6;
      var valLab = svgEl("text", {
        x: valX, y: y + rowH / 2 + 4, "text-anchor": val >= 0 ? "start" : "end", class: "bar-label",
        style: compact ? "font-size:9.5px" : ""
      });
      valLab.textContent = fmt3(val) + (starFn(d) ? "*" : "");
      svg.appendChild(valLab);

      var tt = makeTooltip(wrapper);
      rect.addEventListener("mouseenter", function () {
        tt.show(rect, "<strong>" + d + "</strong><br>r<sub>s</sub> = " + fmt3(val) + (starFn(d) ? " *" : ""));
      });
      rect.addEventListener("mouseleave", tt.hide);
    });

    wrapper.appendChild(svg);
    return wrapper;
  }

  function initCorrControls() {
    document.querySelectorAll("#task-toggle button").forEach(function (btn) {
      btn.addEventListener("click", function () {
        document.querySelectorAll("#task-toggle button").forEach(function (b) { b.setAttribute("aria-selected", "false"); });
        btn.setAttribute("aria-selected", "true");
        corrState.task = btn.getAttribute("data-task");
        renderCorrChart();
      });
    });
    document.querySelectorAll("#model-toggle button").forEach(function (btn) {
      btn.addEventListener("click", function () {
        document.querySelectorAll("#model-toggle button").forEach(function (b) { b.setAttribute("aria-selected", "false"); });
        btn.setAttribute("aria-selected", "true");
        corrState.view = btn.getAttribute("data-model");
        renderCorrChart();
      });
    });
  }

  /* ------------------------------------------------------------------ */
  /* §03 Background corpus table                                        */
  /* ------------------------------------------------------------------ */

  var CORPUS_COLS = [
    { key: "name",      label: "Background", type: "text" },
    { key: "source",    label: "Source",     type: "text" },
    { key: "zcr",       label: "ZCR",        type: "num", digits: 4 },
    { key: "centroid",  label: "Centroid (Hz)", type: "num", digits: 1 },
    { key: "bandwidth", label: "Bandwidth (Hz)", type: "num", digits: 1 },
    { key: "rolloff",   label: "Rolloff (Hz)", type: "num", digits: 1 },
    { key: "speech",    label: "Speech Frac.", type: "num", digits: 4 },
    { key: "flux",      label: "Flux",        type: "num", digits: 3 },
    { key: "flatness",  label: "Flatness",    type: "num", digits: 4 },
    { key: "contrast",  label: "Contrast",    type: "num", digits: 2 },
    { key: "rmsVar",    label: "RMS Var.",    type: "num", digits: 6 }
  ];
  var corpusSort = { key: null, dir: 1 };

  function renderCorpusTable() {
    var table = document.getElementById("corpus-table");
    var thead = table.querySelector("thead");
    var tbody = table.querySelector("tbody");

    var headRow = document.createElement("tr");
    CORPUS_COLS.forEach(function (col) {
      var th = document.createElement("th");
      th.textContent = col.label;
      if (corpusSort.key === col.key) {
        th.setAttribute("aria-sort", corpusSort.dir === 1 ? "ascending" : "descending");
        var arrow = document.createElement("span");
        arrow.className = "arrow";
        arrow.textContent = corpusSort.dir === 1 ? "▲" : "▼";
        th.appendChild(arrow);
      }
      th.addEventListener("click", function () {
        if (corpusSort.key === col.key) { corpusSort.dir *= -1; }
        else { corpusSort.key = col.key; corpusSort.dir = 1; }
        renderCorpusTable();
      });
      headRow.appendChild(th);
    });
    thead.innerHTML = "";
    thead.appendChild(headRow);

    var rows = BACKGROUNDS.slice();
    if (corpusSort.key) {
      rows.sort(function (a, b) {
        var av = a[corpusSort.key], bv = b[corpusSort.key];
        if (typeof av === "string") return av.localeCompare(bv) * corpusSort.dir;
        return (av - bv) * corpusSort.dir;
      });
    }

    tbody.innerHTML = "";
    rows.forEach(function (row) {
      var tr = document.createElement("tr");
      CORPUS_COLS.forEach(function (col) {
        var td = document.createElement("td");
        var v = row[col.key];
        td.textContent = col.type === "num" ? v.toFixed(col.digits) : v;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }

  /* ------------------------------------------------------------------ */
  /* §04 Fairness chart                                                  */
  /* ------------------------------------------------------------------ */

  function renderFairLegend() {
    var host = document.getElementById("fair-legend");
    host.innerHTML = "";
    MODELS.forEach(function (m) {
      var item = document.createElement("span");
      item.className = "legend__item";
      item.innerHTML = '<span class="legend__swatch" style="background:' + m.color + '"></span>' + m.label;
      host.appendChild(item);
    });
  }

  function renderFairChart() {
    var host = document.getElementById("fair-chart");
    host.innerHTML = "";

    var groupH = 40, padTop = 10, padBottom = 24;
    var H = FAIRNESS.length * groupH + padTop + padBottom;
    var W = 720;
    var labelW = 150, plotX0 = labelW, plotW = W - labelW - 40;
    var maxAbs = 0.9;
    var zeroX = plotX0 + plotW / 2;
    var scale = (plotW / 2) / maxAbs;

    var wrapper = document.createElement("div");
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "xMinYMin meet" });

    [-0.6, -0.3, 0, 0.3, 0.6].forEach(function (t) {
      var x = zeroX + t * scale;
      svg.appendChild(svgEl("line", { x1: x, y1: padTop - 2, x2: x, y2: H - padBottom + 2, class: t === 0 ? "axis-line" : "grid-line" }));
      var lab = svgEl("text", { x: x, y: H - padBottom + 14, "text-anchor": "middle", class: "axis-label" });
      lab.textContent = t.toFixed(1);
      svg.appendChild(lab);
    });

    var barH = 5, gap = 1.5;
    var order = ["qwen2audio", "omni7b", "omni3b", "phi4mm", "gemma3n"];
    var colorOf = { qwen2audio: "var(--m-qwen2audio)", omni7b: "var(--m-omni7b)", omni3b: "var(--m-omni3b)", phi4mm: "var(--m-phi4mm)", gemma3n: "var(--m-gemma3n)" };

    FAIRNESS.forEach(function (row, gi) {
      var gy = padTop + gi * groupH;
      var name = svgEl("text", { x: labelW - 10, y: gy + groupH / 2 + 4, "text-anchor": "end", class: "bar-label" });
      name.textContent = row.group;
      svg.appendChild(name);

      order.forEach(function (key, si) {
        var val = row[key];
        var barY = gy + 4 + si * (barH + gap);
        var x = zeroX + Math.min(val, 0) * scale;
        var w = Math.max(Math.abs(val) * scale, 1);
        var rect = svgEl("rect", { x: x, y: barY, width: w, height: barH, rx: 1, fill: colorOf[key] });
        svg.appendChild(rect);

        var tt = makeTooltip(wrapper);
        rect.addEventListener("mouseenter", function () {
          var m = MODELS.filter(function (m) { return m.id === key; })[0];
          tt.show(rect, "<strong>" + m.label + "</strong><br>" + row.group + "<br>&Delta;WER = " + fmt3(val));
        });
        rect.addEventListener("mouseleave", tt.hide);
      });
    });

    wrapper.appendChild(svg);
    host.appendChild(wrapper);
  }

  /* ------------------------------------------------------------------ */
  /* §05 RER chart                                                       */
  /* ------------------------------------------------------------------ */

  function renderRerChart() {
    var host = document.getElementById("rer-chart");
    host.innerHTML = "";

    var W = 720, H = 340;
    var padTop = 30, padBottom = 46, padLeft = 46, padRight = 16;
    var plotW = W - padLeft - padRight, plotH = H - padTop - padBottom;
    var maxVal = 6.5;

    var wrapper = document.createElement("div");
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "xMinYMin meet" });

    // legend
    var legendY = 14;
    svg.appendChild(svgEl("rect", { x: padLeft, y: legendY - 8, width: 10, height: 10, fill: "var(--ink)" }));
    var l1 = svgEl("text", { x: padLeft + 16, y: legendY, class: "axis-label" }); l1.textContent = "ASR (WER)"; svg.appendChild(l1);
    svg.appendChild(svgEl("rect", { x: padLeft + 100, y: legendY - 8, width: 10, height: 10, fill: "var(--accent)" }));
    var l2 = svgEl("text", { x: padLeft + 116, y: legendY, class: "axis-label" }); l2.textContent = "KWS (accuracy)"; svg.appendChild(l2);
    svg.appendChild(svgEl("line", { x1: padLeft + 226, y1: legendY - 3, x2: padLeft + 246, y2: legendY - 3, class: "ref-line" }));
    var l3 = svgEl("text", { x: padLeft + 252, y: legendY, class: "axis-label" }); l3.textContent = "RER = 1.0 (no change)"; svg.appendChild(l3);

    function yFor(v) { return padTop + plotH - (v / maxVal) * plotH; }

    [0, 1, 2, 3, 4, 5, 6].forEach(function (t) {
      var y = yFor(t);
      svg.appendChild(svgEl("line", { x1: padLeft, y1: y, x2: W - padRight, y2: y, class: t === 0 ? "axis-line" : "grid-line" }));
      var lab = svgEl("text", { x: padLeft - 8, y: y + 3, "text-anchor": "end", class: "axis-label" });
      lab.textContent = t.toFixed(0);
      svg.appendChild(lab);
    });
    var refY = yFor(1);
    svg.appendChild(svgEl("line", { x1: padLeft, y1: refY, x2: W - padRight, y2: refY, class: "ref-line" }));

    var groupW = plotW / RER.length;
    var barW = 20, gap = 4;

    RER.forEach(function (row, i) {
      var gx = padLeft + i * groupW + groupW / 2;
      [["asr", "var(--ink)", -1], ["kws", "var(--accent)", 1]].forEach(function (pair) {
        var key = pair[0], color = pair[1], dir = pair[2];
        var val = row[key];
        var x = gx + dir * (gap / 2) + (dir < 0 ? -barW : 0);
        var y = yFor(val);
        var rect = svgEl("rect", { x: x, y: y, width: barW, height: yFor(0) - y, fill: color, rx: 1 });
        svg.appendChild(rect);

        var lab = svgEl("text", { x: x + barW / 2, y: y - 5, "text-anchor": "middle", class: "bar-label" });
        lab.textContent = val.toFixed(2);
        svg.appendChild(lab);

        var tt = makeTooltip(wrapper);
        rect.addEventListener("mouseenter", function () {
          tt.show(rect, "<strong>" + row.model + "</strong><br>" + key.toUpperCase() + " RER = " + val.toFixed(3));
        });
        rect.addEventListener("mouseleave", tt.hide);
      });

      var name = svgEl("text", { x: gx, y: H - padBottom + 18, "text-anchor": "middle", class: "axis-label" });
      name.textContent = row.model;
      svg.appendChild(name);
    });

    wrapper.appendChild(svg);
    host.appendChild(wrapper);
  }

  /* ------------------------------------------------------------------ */
  /* Cite: copy button                                                   */
  /* ------------------------------------------------------------------ */

  function initCopyBib() {
    var btn = document.getElementById("copy-bib");
    var pre = document.getElementById("bibtex");
    if (!btn || !pre) return;
    btn.addEventListener("click", function () {
      var text = pre.textContent;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { flagCopied(btn); });
      } else {
        var ta = document.createElement("textarea");
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(ta);
        flagCopied(btn);
      }
    });
  }
  function flagCopied(btn) {
    btn.setAttribute("data-copied", "true");
    setTimeout(function () { btn.removeAttribute("data-copied"); }, 1800);
  }

  /* ------------------------------------------------------------------ */
  /* Boot                                                                */
  /* ------------------------------------------------------------------ */

  function renderAll() {
    renderPipeline();
    renderCorrChart();
    renderCorpusTable();
    renderFairLegend();
    renderFairChart();
    renderRerChart();
  }

  document.addEventListener("DOMContentLoaded", function () {
    initCorrControls();
    initCopyBib();
    renderAll();

    var resizeTimer;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        renderCorrChart();
      }, 200);
    });
  });
})();
