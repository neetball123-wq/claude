// Renders audio through the real Web Audio graph in headless Chromium and
// checks that the processing does what the UI claims.
//
//   node tests/verify-dsp.mjs
//
// Needs Playwright (a local install, or the global one) and Chromium.
import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';
import { writeFileSync, mkdtempSync, readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

async function loadPlaywright() {
  try {
    return await import('playwright');
  } catch {
    const root = execSync('npm root -g').toString().trim();
    return createRequire(join(root, 'noop.js'))('playwright');
  }
}

const here = dirname(fileURLToPath(import.meta.url));
// Served from localhost: a secure context, so the microphone path can be exercised.
const html = readFileSync(join(here, '..', 'index.html'));
const server = createServer((req, res) => {
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  res.end(html);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const pageUrl = `http://127.0.0.1:${server.address().port}/`;

let failures = 0;
function check(name, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
  if (!ok) failures++;
}

const { chromium } = await loadPlaywright();
const browser = await chromium.launch({
  args: ['--autoplay-policy=no-user-gesture-required', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
});
const page = await browser.newPage({ viewport: { width: 390, height: 900 }, deviceScaleFactor: 2 });
// Web fonts are cosmetic; keep the test offline and deterministic.
const FONT_HOSTS = /fonts\.(googleapis|gstatic)\.com/;
await page.route(FONT_HOSTS, (route) => route.abort());
const consoleErrors = [];
page.on('console', (m) => {
  if (m.type() !== 'error') return;
  if (m.text().startsWith('Failed to load resource') && FONT_HOSTS.test(m.location().url || '')) return;
  consoleErrors.push(m.text());
});
page.on('pageerror', (e) => consoleErrors.push(String(e)));
await page.goto(pageUrl);
await page.waitForFunction(() => window.__refiner && window.__refiner.engine);

const r = await page.evaluate(async () => {
  const R = window.__refiner;
  const sr = 48000;

  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < len / 2; k++) {
          const wr = Math.cos(ang * k), wi = Math.sin(ang * k);
          const ar = re[i + k + len / 2], ai = im[i + k + len / 2];
          const xr = ar * wr - ai * wi, xi = ar * wi + ai * wr;
          re[i + k + len / 2] = re[i + k] - xr; im[i + k + len / 2] = im[i + k] - xi;
          re[i + k] += xr; im[i + k] += xi;
        }
      }
    }
  }
  // Welch power spectrum of (L+R)/2 and (L-R)/2 energy
  function analyse(buf, from = 0, to = buf.length) {
    const N = 4096;
    const L = buf.getChannelData(0), Rr = buf.getChannelData(buf.numberOfChannels > 1 ? 1 : 0);
    const psd = new Float64Array(N / 2);
    let frames = 0;
    for (let s = from; s + N <= to; s += N / 2) {
      const re = new Float64Array(N), im = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1));
        re[i] = ((L[s + i] + Rr[s + i]) / 2) * w;
      }
      fft(re, im);
      for (let i = 0; i < N / 2; i++) psd[i] += re[i] * re[i] + im[i] * im[i];
      frames++;
    }
    let peak = 0, sumSq = 0, mid = 0, side = 0, nan = false;
    for (let i = from; i < to; i++) {
      const l = L[i], r = Rr[i];
      if (!Number.isFinite(l) || !Number.isFinite(r)) nan = true;
      peak = Math.max(peak, Math.abs(l), Math.abs(r));
      sumSq += l * l + r * r;
      mid += ((l + r) / 2) ** 2;
      side += ((l - r) / 2) ** 2;
    }
    const band = (f0, f1) => {
      const hz = buf.sampleRate / N;
      let p = 0;
      for (let i = Math.ceil(f0 / hz); i <= Math.floor(f1 / hz) && i < N / 2; i++) p += psd[i];
      return 10 * Math.log10(p / frames + 1e-30);
    };
    return {
      peak, nan,
      rmsDb: 10 * Math.log10(sumSq / (2 * (to - from)) + 1e-30),
      sideShare: side / (mid + side),
      hi: band(16000, 20000), mid: band(1000, 4000), bass: band(40, 120), air: band(10000, 16000),
    };
  }

  const zero = { hf: 0, clarity: 0, bass: 0, space: 0, width: 0, punch: 0 };
  const rec = R.PRESETS[0].s;
  const max = { hf: 1, clarity: 1, bass: 1, space: 1, width: 1, punch: 1 };
  const demo = await R.makeDemo(sr);
  const out = {};
  const span = [sr * 2, demo.length - sr]; // skip the first bars, compare the same region

  out.demo = analyse(demo, ...span);
  const recBuf = await R.renderEnhanced(demo, rec);
  out.rec = analyse(recBuf, ...span);
  const zeroBuf = await R.renderEnhanced(demo, zero);
  out.zero = analyse(zeroBuf, ...span);
  out.hfOnly = analyse(await R.renderEnhanced(demo, { ...zero, hf: 1 }), ...span);
  out.widthOnly = analyse(await R.renderEnhanced(demo, { ...zero, width: 1 }), ...span);
  out.spaceOnly = analyse(await R.renderEnhanced(demo, { ...zero, space: 1 }), ...span);
  out.bassOnly = analyse(await R.renderEnhanced(demo, { ...zero, bass: 1 }), ...span);

  // Raw delay of the processed path, measured by cross-correlation (no compensation).
  async function measureDelay(opts) {
    const off = new OfflineAudioContext(2, sr * 6, sr);
    const src = off.createBufferSource();
    src.buffer = demo;
    const fx = await R.createEnhancer(off, zero, opts);
    src.connect(fx.input);
    fx.output.connect(off.destination);
    src.start(0);
    const outBuf = await off.startRendering();
    const a = demo.getChannelData(0), b = outBuf.getChannelData(0);
    let best = -Infinity, lag = 0;
    for (let l = 0; l <= 1200; l++) {
      let s = 0;
      for (let i = sr * 3; i < sr * 5; i += 2) s += a[i] * b[i + l];
      if (s > best) { best = s; lag = l; }
    }
    return { lag, engine: fx.engine, reported: Math.round(fx.latency * sr) };
  }
  out.delayWorklet = await measureDelay({});
  out.delayNative = await measureDelay({ native: true });
  out.native = analyse(await R.renderEnhanced(demo, rec, null, { native: true }), ...span);

  // Alignment: with everything off the output should line up with the input.
  {
    const a = demo.getChannelData(0), b = zeroBuf.getChannelData(0);
    let best = -Infinity, bestLag = 0;
    for (let lag = -200; lag <= 200; lag++) {
      let s = 0;
      for (let i = sr * 3; i < sr * 5; i++) s += a[i] * b[i + lag];
      if (s > best) { best = s; bestLag = lag; }
    }
    let ab = 0, aa = 0, bb = 0;
    for (let i = sr * 3; i < sr * 5; i++) { ab += a[i] * b[i + bestLag]; aa += a[i] * a[i]; bb += b[i + bestLag] ** 2; }
    out.lag = bestLag;
    out.corr = ab / Math.sqrt(aa * bb);
  }

  // Worst case: a brick-walled, full-scale master at maximum settings.
  {
    const hot = new AudioBuffer({ length: sr * 4, numberOfChannels: 2, sampleRate: sr });
    for (let c = 0; c < 2; c++) {
      const d = hot.getChannelData(c), src = demo.getChannelData(c);
      for (let i = 0; i < d.length; i++) d[i] = Math.max(-1, Math.min(1, src[i + sr * 4] * 3));
    }
    out.hot = analyse(await R.renderEnhanced(hot, max));
  }

  // Mono file
  {
    const mono = new AudioBuffer({ length: sr * 3, numberOfChannels: 1, sampleRate: sr });
    mono.copyToChannel(demo.getChannelData(0).slice(sr * 4, sr * 7), 0);
    const m = await R.renderEnhanced(mono, rec);
    out.mono = { channels: m.numberOfChannels, ...analyse(m) };
  }

  // 44.1 kHz file
  {
    const off = new OfflineAudioContext(2, 44100 * 3, 44100);
    const s = off.createBufferSource();
    s.buffer = demo;
    s.connect(off.destination);
    s.start(0, 4);
    const b441 = await off.startRendering();
    out.sr441 = analyse(await R.renderEnhanced(b441, rec));
  }

  // Preview snapshot used for the still display
  {
    const snap = await R.previewSnapshot(demo, rec, 6);
    const finite = snap.dry.every(Number.isFinite) && snap.wet.every(Number.isFinite);
    out.snapshot = { finite, bins: snap.dry.length, levels: snap.levels };
  }

  // WAV + ZIP
  {
    const short = new AudioBuffer({ length: sr, numberOfChannels: 2, sampleRate: sr });
    short.copyToChannel(recBuf.getChannelData(0).slice(sr * 3, sr * 4), 0);
    short.copyToChannel(recBuf.getChannelData(1).slice(sr * 3, sr * 4), 1);
    const wav = R.encodeWav16(short);
    const zip = R.zipSingle('デモ曲_refined.wav', wav);
    const bytes = new Uint8Array(await zip.arrayBuffer());
    let bin = '';
    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    out.zipB64 = btoa(bin);
  }
  return out;
});

const f = (x) => x.toFixed(1);
console.log('\n-- measured (dB unless noted) --');
for (const k of ['demo', 'zero', 'rec', 'hfOnly', 'widthOnly', 'spaceOnly', 'bassOnly', 'hot', 'sr441']) {
  const m = r[k];
  console.log(`${k.padEnd(10)} peak ${m.peak.toFixed(3)}  rms ${f(m.rmsDb)}  16-20k ${f(m.hi)}  10-16k ${f(m.air)}  1-4k ${f(m.mid)}  40-120 ${f(m.bass)}  side ${(m.sideShare * 100).toFixed(0)}%`);
}
console.log(`alignment lag ${r.lag} samples, correlation ${r.corr.toFixed(4)}`);
console.log(`processing delay: worklet ${r.delayWorklet.lag} samples (${(r.delayWorklet.lag / 48).toFixed(2)} ms), native ${r.delayNative.lag} samples (${(r.delayNative.lag / 48).toFixed(2)} ms)\n`);

check('ゼロ遅延エンジン (AudioWorklet) が使われる', r.delayWorklet.engine === 'worklet');
check('処理による遅れが 0.1ms 未満', r.delayWorklet.lag <= 4, `${r.delayWorklet.lag} samples`);
check('従来エンジンの遅れは申告どおり (約12ms)', Math.abs(r.delayNative.lag - r.delayNative.reported) <= 3, `${r.delayNative.lag} vs ${r.delayNative.reported}`);
check('従来エンジンでも補完・音割れなし', r.native.hi - r.demo.hi > 20 && r.native.peak <= 1 && !r.native.nan, `peak ${r.native.peak.toFixed(4)}`);
check('2つのエンジンの音量差が 1.5dB 以内', Math.abs(r.native.rmsDb - r.rec.rmsDb) < 1.5, `${(r.rec.rmsDb - r.native.rmsDb).toFixed(2)} dB`);

// The demo is band-limited at 15.5 kHz like a 128 kbps MP3; restoration must refill 16-20 kHz.
check('高域補完: 16-20kHz が大きく増える (おすすめ)', r.rec.hi - r.demo.hi > 20, `+${f(r.rec.hi - r.demo.hi)} dB`);
check('高域補完: 補完成分は中域より十分小さい (やり過ぎない)', r.rec.mid - r.rec.hi > 20, `${f(r.rec.mid - r.rec.hi)} dB below 1-4k`);
check('高域補完 だけでも 16-20kHz が増える', r.hfOnly.hi - r.zero.hi > 20, `+${f(r.hfOnly.hi - r.zero.hi)} dB`);
check('すべて0なら 16-20kHz は増えない', r.zero.hi - r.demo.hi < 3, `${f(r.zero.hi - r.demo.hi)} dB`);
check('ステレオ幅: side 成分が増える', r.widthOnly.sideShare > r.zero.sideShare * 1.5, `${(r.zero.sideShare * 100).toFixed(1)}% → ${(r.widthOnly.sideShare * 100).toFixed(1)}%`);
check('空間: side 成分が増える', r.spaceOnly.sideShare > r.zero.sideShare * 1.2, `${(r.zero.sideShare * 100).toFixed(1)}% → ${(r.spaceOnly.sideShare * 100).toFixed(1)}%`);
check('低域: 40-120Hz が増える', r.bassOnly.bass - r.zero.bass > 2, `+${f(r.bassOnly.bass - r.zero.bass)} dB`);
check('すべて0なら原音とほぼ同じ波形', r.corr > 0.95, `corr ${r.corr.toFixed(4)}`);
check('書き出しの時間ずれがない', Math.abs(r.lag) <= 2, `${r.lag} samples`);
for (const k of ['demo', 'zero', 'rec', 'hfOnly', 'widthOnly', 'spaceOnly', 'bassOnly', 'hot', 'mono', 'sr441']) {
  check(`音割れなし・NaNなし (${k})`, !r[k].nan && r[k].peak <= 1.0, `peak ${r[k].peak.toFixed(4)}`);
}
check('最大設定＋爆音素材でも −1dBFS (0.891) を超えない', r.hot.peak <= 0.8914 && !r.hot.nan, `peak ${r.hot.peak.toFixed(4)}`);
check('おすすめ設定の音量変化は ±4dB 以内', Math.abs(r.rec.rmsDb - r.demo.rmsDb) < 4, `${f(r.rec.rmsDb - r.demo.rmsDb)} dB`);
check('モノラル入力 → ステレオ出力', r.mono.channels === 2 && !r.mono.nan);
check('44.1kHz 素材でも動く', r.sr441.hi - r.demo.hi > 15 && !r.sr441.nan, `16-20k ${f(r.sr441.hi)}`);
check('プレビュー解析が有限値', r.snapshot.finite && r.snapshot.levels.dry > 0 && r.snapshot.levels.wet > 0);

// Check the ZIP with the system unzip and the WAV header with Python's wave module.
const dir = mkdtempSync(join(tmpdir(), 'refiner-'));
const zipPath = join(dir, 'out.zip');
writeFileSync(zipPath, Buffer.from(r.zipB64, 'base64'));
let zipOk = false, wavInfo = '';
try {
  execSync(`unzip -t "${zipPath}"`, { stdio: 'pipe' });
  execSync(`unzip -o -q "${zipPath}" -d "${dir}"`);
  wavInfo = execSync(`python3 -c "import wave,sys; w=wave.open(sys.argv[1]); print(w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes())" "${join(dir, 'デモ曲_refined.wav')}"`).toString().trim();
  zipOk = wavInfo === '2 2 48000 48000';
} catch (e) {
  wavInfo = String(e.stderr || e.message);
}
check('ZIP が壊れておらず、中の WAV が 2ch/16bit/48kHz', zipOk, wavInfo);

// UI smoke test: demo loads, still spectrum appears, playback and A/B run without errors.
await page.waitForFunction(() => document.getElementById('screenMsg').hidden, null, { timeout: 15000 });
const idle = await page.evaluate(() => ({
  name: document.getElementById('sourceName').textContent,
  hf: document.getElementById('rHf').textContent,
  width: document.getElementById('rWidth').textContent,
  trim: document.getElementById('rTrim').textContent,
  scrollW: document.documentElement.scrollWidth,
  clientW: document.documentElement.clientWidth,
}));
check('デモ曲が読み込まれる', idle.name.startsWith('デモ曲'), idle.name);
check('再生前から比較の数値が出る', idle.hf !== '—' && idle.width !== '—' && idle.trim !== '—', `${idle.hf} / ${idle.width} / ${idle.trim}`);
check('スマホ幅で横スクロールしない', idle.scrollW <= idle.clientW, `${idle.scrollW} <= ${idle.clientW}`);
await page.screenshot({ path: join(dir, 'idle.png'), fullPage: true });
await page.click('#playBtn');
await page.waitForTimeout(1500);
await page.click('#abDry');
await page.waitForTimeout(400);
await page.click('#abWet');
await page.click('#preset-space');
await page.waitForTimeout(600);
const playing = await page.evaluate(() => document.getElementById('playBtn').getAttribute('aria-label'));
check('再生ボタンで再生状態になる', playing === '一時停止', playing);
await page.screenshot({ path: join(dir, 'playing.png') });

await page.click('#liveBtn');
await page.click('#liveMic');
await page.waitForFunction(() => document.getElementById('sourceName').textContent.startsWith('ライブ入力'), null, { timeout: 5000 });
await page.waitForTimeout(1500);
const live = await page.evaluate(() => ({
  name: document.getElementById('sourceName').textContent,
  meta: document.getElementById('sourceMeta').textContent,
  button: document.getElementById('playBtn').getAttribute('aria-label'),
  latency: document.getElementById('rLatency').textContent,
  msg: document.getElementById('liveMsg').textContent,
  overlayShown: document.getElementById('screenMsg').offsetParent !== null,
  devicesShown: document.getElementById('devices').offsetParent !== null,
  inputs: document.getElementById('inputSelect').options.length,
}));
check('ライブ開始後は「解析中」表示が消える', !live.overlayShown);
check('許可後に入力機器の一覧が出る', live.devicesShown && live.inputs >= 1, `${live.inputs} inputs`);
check('ライブ入力（マイク）が始まる', live.name === 'ライブ入力：マイク・入力機器' && live.button === 'ライブ入力を止める', live.msg || live.name);
check('ライブ中は処理遅延 0 ms と表示', live.meta.includes('処理の遅れ 0 ms'), live.meta);
check('遅延の推定値が出る', /^約 \d+ ms$/.test(live.latency), live.latency);
await page.click('#abDry');
await page.waitForTimeout(200);
await page.click('#abWet');
await page.screenshot({ path: join(dir, 'live.png') });
await page.click('#playBtn');
const after = await page.evaluate(() => ({
  name: document.getElementById('sourceName').textContent,
  button: document.getElementById('playBtn').getAttribute('aria-label'),
}));
check('ライブ入力を止めると元の曲に戻る', after.name.startsWith('デモ曲') && after.button === '再生', after.name);

// No virtual cable on this machine: the game button should explain the setup.
await page.click('#liveGame');
await page.waitForFunction(() => document.getElementById('liveMsg').textContent !== '', null, { timeout: 5000 });
const noCable = await page.evaluate(() => ({
  msg: document.getElementById('liveMsg').textContent,
  howto: document.getElementById('gameHowto').open,
  live: window.__refiner.liveState().live,
}));
check('仮想ケーブルが無いときは準備手順を案内する', noCable.msg.includes('見つかりませんでした') && noCable.howto && !noCable.live, noCable.msg);

const labels = await page.evaluate(() => {
  const { isVirtual, loopRisk } = window.__refiner;
  return {
    virtual: ['CABLE Output (VB-Audio Virtual Cable)', 'BlackHole 2ch', 'VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)', '既定 - CABLE Output (VB-Audio Virtual Cable)'].map(isVirtual),
    real: ['マイク (Realtek(R) Audio)', 'MacBook Pro Microphone', 'Elgato HD60 X', 'USB Audio CODEC'].map(isVirtual),
    risk: [
      loopRisk('CABLE Output (VB-Audio Virtual Cable)', '既定 - CABLE Input (VB-Audio Virtual Cable)', true),
      loopRisk('BlackHole 2ch', 'BlackHole 2ch', true),
      loopRisk('CABLE Output (VB-Audio Virtual Cable)', 'ヘッドホン (Realtek(R) Audio)', true),
      loopRisk('マイク (Realtek(R) Audio)', '既定 - CABLE Input (VB-Audio Virtual Cable)', true),
      loopRisk('BlackHole 2ch', '', false),
    ],
  };
});
check('仮想オーディオの名前を見分ける', labels.virtual.every(Boolean) && !labels.real.some(Boolean), JSON.stringify(labels));
check('音ループの危険を判定する', JSON.stringify(labels.risk) === JSON.stringify([true, true, false, false, true]), JSON.stringify(labels.risk));

// A PC with VB-CABLE where the default output is also the cable (the loop case).
const ctx2 = await browser.newContext({ viewport: { width: 1100, height: 900 } });
await ctx2.route(FONT_HOSTS, (route) => route.abort());
await ctx2.addInitScript(() => {
  const md = navigator.mediaDevices;
  const gum = md.getUserMedia.bind(md);
  // Every request is served by Chromium's fake input, but reports the device
  // that was asked for, as a real browser does.
  md.getUserMedia = async (c) => {
    const wanted = c && c.audio && c.audio.deviceId && c.audio.deviceId.exact;
    const stream = await gum(wanted ? { audio: { ...c.audio, deviceId: undefined } } : c);
    if (wanted) {
      for (const t of stream.getAudioTracks()) {
        const orig = t.getSettings.bind(t);
        t.getSettings = () => ({ ...orig(), deviceId: wanted });
      }
    }
    return stream;
  };
  md.enumerateDevices = async () => [
    { kind: 'audioinput', deviceId: 'default', label: '既定 - マイク (Realtek(R) Audio)', groupId: 'a' },
    { kind: 'audioinput', deviceId: 'mic', label: 'マイク (Realtek(R) Audio)', groupId: 'a' },
    { kind: 'audioinput', deviceId: 'cable', label: 'CABLE Output (VB-Audio Virtual Cable)', groupId: 'b' },
    { kind: 'audiooutput', deviceId: 'default', label: '既定 - CABLE Input (VB-Audio Virtual Cable)', groupId: 'b' },
    { kind: 'audiooutput', deviceId: 'hp', label: 'ヘッドホン (Realtek(R) Audio)', groupId: 'a' },
  ];
  AudioContext.prototype.setSinkId = async function () {};
});
const page2 = await ctx2.newPage();
const errors2 = [];
page2.on('pageerror', (e) => errors2.push(String(e)));
await page2.goto(pageUrl);
await page2.waitForFunction(() => window.__refiner && window.__refiner.engine);
await page2.click('#liveBtn');
await page2.click('#liveGame');
await page2.waitForFunction(() => window.__refiner.liveState().live, null, { timeout: 5000 });
const g1 = await page2.evaluate(() => ({
  st: window.__refiner.liveState(),
  name: document.getElementById('sourceName').textContent,
  input: document.getElementById('inputSelect').value,
  msg: document.getElementById('liveMsg').textContent,
}));
check('ゲームボタンで仮想ケーブルを自動で選ぶ', g1.st.deviceId === 'cable' && g1.input === 'cable' && g1.name === 'ライブ入力：ゲーム・パソコンの音', `${g1.st.deviceId} / ${g1.name}`);
check('出力も仮想ケーブルならループ防止で消音する', g1.st.loopMuted && g1.msg.includes('ループ'), g1.msg);
await page2.selectOption('#outputSelect', 'hp');
await page2.waitForFunction(() => !window.__refiner.liveState().loopMuted, null, { timeout: 3000 }).catch(() => {});
const g2 = await page2.evaluate(() => ({ st: window.__refiner.liveState(), msg: document.getElementById('liveMsg').textContent }));
check('出力をヘッドホンにすると消音が解ける', !g2.st.loopMuted && g2.msg === '', g2.msg);
await page2.selectOption('#inputSelect', 'mic');
await page2.waitForFunction(() => document.getElementById('sourceName').textContent === 'ライブ入力：マイク・入力機器', null, { timeout: 5000 }).catch(() => {});
const g3 = await page2.evaluate(() => ({ st: window.__refiner.liveState(), name: document.getElementById('sourceName').textContent }));
check('入力を切り替えるとライブのまま入れ替わる', g3.st.live && g3.st.deviceId === 'mic' && g3.name === 'ライブ入力：マイク・入力機器', `${g3.st.deviceId} / ${g3.name}`);
await page2.screenshot({ path: join(dir, 'game.png') });
check('ゲーム入力のシナリオでエラーなし', errors2.length === 0, errors2.join(' | '));
await ctx2.close();
check('コンソールエラーなし', consoleErrors.length === 0, consoleErrors.join(' | '));
console.log(`\nscreenshots: ${dir}`);

await browser.close();
server.close();
console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
