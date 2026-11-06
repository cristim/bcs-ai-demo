// Behind-the-scenes panel: one numbered card per stage of the latest exchange.
// Globals from index.html used lazily: lastExchange, revealPanel, broadcastStatus.
// Step control (also via BroadcastChannel {type:'revealStep', step: 1..8 | 'next' | 'prev' | null}):
// null = overview of all cards, otherwise one card shown large.

let revealStep = null; // 0-based index of the focused step, null = overview
let exchangeSeq = 0;
let voiceprint = null;
let voiceprintBusy = false;
const stepSigs = [];

const mk = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
};
const note = (vis, text) => vis.append(mk('p', 'step-note', text));
const quote = (vis, text) => vis.append(mk('p', 'reveal-quote', text));
const secs = (v) => (v == null ? '?' : `${Number(v).toFixed(1)} seconds`);

function audioPlayer(src) {
    const a = mk('audio');
    a.controls = true;
    a.preload = 'none';
    a.src = src;
    return a;
}

async function postJson(url, body) {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    return r.json();
}

async function tokenizeText(text) {
    try {
        const d = await postJson('/tokenize', { text });
        return d.success ? { ids: d.token_ids, tokens: d.tokens } : { error: d.error };
    } catch (err) {
        return { error: 'voice server offline' };
    }
}

// ---- Microphone recording (parallel to SpeechRecognition) ----
let micRec = null;
let micPending = null;
let micText = '';

const decodeAudio = (arrayBuffer) => new OfflineAudioContext(1, 1, 22050).decodeAudioData(arrayBuffer);
const releaseMic = (s) => { if (s.stream) s.stream.getTracks().forEach((t) => t.stop()); };

function startMicRecording() {
    const s = { stopped: false, mr: null, stream: null, resolve: null };
    s.done = new Promise((r) => { s.resolve = r; });
    micRec = s;
    micPending = null;
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
        s.stream = stream;
        if (s.stopped) { releaseMic(s); s.resolve(null); return; }
        const chunks = [];
        s.mr = new MediaRecorder(stream);
        s.mr.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
        s.mr.onstop = async () => {
            releaseMic(s);
            try {
                const blob = new Blob(chunks, { type: s.mr.mimeType });
                const buf = await decodeAudio(await blob.arrayBuffer());
                s.resolve({ url: URL.createObjectURL(blob), duration: buf.duration, buf });
            } catch (err) {
                console.warn('Could not decode the recording:', err);
                s.resolve(null);
            }
        };
        s.mr.start();
    }).catch((err) => { console.warn('Microphone recording unavailable:', err); s.resolve(null); });
}

function stopMicRecording(text) {
    const s = micRec;
    if (!s) return;
    micRec = null;
    s.stopped = true;
    micText = text;
    micPending = s.done;
    if (s.mr && s.mr.state !== 'inactive') s.mr.stop();
}

// The clip belongs to the exchange only if the sent text is what was spoken
function takeMicClip(message) {
    const p = micPending && message === micText ? micPending : null;
    micPending = null;
    return p;
}

// ---- Canvas drawings ----
function drawWave(canvas, buf) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height, d = buf.getChannelData(0);
    const per = Math.max(1, Math.floor(d.length / w));
    let peak = 0.001;
    for (let i = 0; i < d.length; i++) peak = Math.max(peak, Math.abs(d[i]));
    ctx.fillStyle = '#0d0a26';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#7dffb4';
    for (let x = 0; x < w; x++) {
        let lo = 0, hi = 0;
        for (let i = x * per; i < (x + 1) * per && i < d.length; i++) { lo = Math.min(lo, d[i]); hi = Math.max(hi, d[i]); }
        ctx.fillRect(x, (1 - hi / peak) * h / 2, 1, Math.max(1, ((hi - lo) / peak) * h / 2));
    }
}

function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const ang = -2 * Math.PI / len;
        for (let i = 0; i < n; i += len) {
            for (let k = 0; k < len / 2; k++) {
                const c = Math.cos(ang * k), s = Math.sin(ang * k);
                const a = i + k, b = i + k + len / 2;
                const tr = re[b] * c - im[b] * s, ti = re[b] * s + im[b] * c;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr; im[a] += ti;
            }
        }
    }
}

function drawSpectrogram(canvas, buf) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height, N = 512, d = buf.getChannelData(0);
    const hop = Math.max(1, Math.floor((d.length - N) / w));
    const img = ctx.createImageData(w, h);
    for (let x = 0; x < w && x * hop + N <= d.length; x++) {
        const re = new Float32Array(N), im = new Float32Array(N);
        for (let i = 0; i < N; i++) re[i] = d[x * hop + i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / N));
        fft(re, im);
        for (let y = 0; y < h; y++) {
            const bin = Math.floor(((h - 1 - y) / h) * (N / 2));
            const db = 20 * Math.log10(Math.hypot(re[bin], im[bin]) / (N / 4) + 1e-7);
            const t = Math.min(1, Math.max(0, (db + 90) / 80)) * 3;
            const o = (y * w + x) * 4;
            img.data[o] = Math.min(255, t * 255);
            img.data[o + 1] = Math.min(255, Math.max(0, (t - 1) * 255));
            img.data[o + 2] = Math.min(255, Math.max(0, (t - 2) * 255));
            img.data[o + 3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);
}

function drawBarcode(canvas, vals, min, max) {
    const ctx = canvas.getContext('2d');
    const bw = canvas.width / vals.length;
    vals.forEach((v, i) => {
        ctx.fillStyle = `hsl(${240 - 240 * ((v - min) / (max - min || 1))}, 85%, 50%)`;
        ctx.fillRect(i * bw, 0, Math.ceil(bw), canvas.height);
    });
}

const canvasEl = (w, h) => { const c = mk('canvas'); c.width = w; c.height = h; return c; };

function chips(vis, ids, pieces) {
    const wrap = mk('div', 'chips');
    ids.forEach((id, i) => {
        const c = mk('div', 'chip');
        c.append(mk('span', 'chip-t', (pieces && pieces[i] || '').replace('[SPACE]', '␣')), mk('span', 'chip-id', id));
        wrap.append(c);
    });
    vis.append(wrap);
}

// ---- Steps: sig = inputs that decide whether the card must be redrawn ----
const REVEAL_STEPS = [
    {
        title: (ex) => (ex.typed ? 'You typed it' : 'Your voice'),
        kid: (ex) => (ex.typed ? 'This time you typed your question with the keyboard.' : 'A microphone turns your voice into a wiggly line of numbers.'),
        sig: (ex) => [ex.typed, ex.mic === undefined ? 'saving' : ex.mic ? 'ok' : 'none', ex.question],
        build(ex, vis, gu) {
            if (ex.typed) { quote(vis, ex.question); gu.textContent = 'No audio was recorded because the text was typed.'; return; }
            gu.textContent = 'Recorded with getUserMedia + MediaRecorder at the same time as the browser speech recogniser; the waveform is the decoded clip (mono, 22,050 Hz), drawn as min/max per pixel.';
            if (ex.mic === undefined) return note(vis, 'Saving your recording...');
            if (!ex.mic) return note(vis, 'Your voice was heard, but the recording could not be saved (is the microphone blocked?).');
            vis.append(audioPlayer(ex.mic.url), mk('p', 'reveal-time', `Length: ${secs(ex.mic.duration)}`));
            const c = canvasEl(640, 90);
            vis.append(c);
            drawWave(c, ex.mic.buf);
        }
    },
    {
        title: (ex) => (ex.typed ? 'Your words' : 'Sound to words'),
        kid: (ex) => (ex.typed ? 'You typed words, so the computer did not need to listen.' : 'A speech AI listens to the sound and guesses which words you said.'),
        sig: (ex) => [ex.typed, ex.question],
        build(ex, vis, gu) {
            quote(vis, ex.question);
            gu.textContent = ex.typed ? 'Sent to the text AI exactly as typed.' : 'Speech recognition runs in the browser (Web Speech API); the transcript is what was sent to the text AI.';
        }
    },
    {
        title: () => 'Words to numbers',
        kid: () => 'Computers cannot read letters, so every little piece of your question becomes a number code.',
        sig: (ex) => [ex.qTokens],
        build(ex, vis, gu) {
            gu.textContent = 'Gemini uses its own similar number codes; these are the codes the voice AI uses (the XTTS v2 text tokenizer: a byte-pair encoding vocabulary, language tag [en] first, spaces shown as ␣).';
            if (!ex.qTokens) return note(vis, 'Turning words into numbers...');
            if (ex.qTokens.error) return note(vis, `Could not get the codes: ${ex.qTokens.error}`);
            chips(vis, ex.qTokens.ids, ex.qTokens.tokens);
            vis.append(mk('p', 'step-note', 'Gemini uses its own similar number codes; here are the codes the voice AI uses.'));
        }
    },
    {
        title: () => 'Secret instructions',
        kid: () => 'Before your question, the AI is secretly told how to act like SpongeBob.',
        sig: (ex) => [ex.prompt, ex.counts, ex.status],
        build(ex, vis, gu) {
            gu.textContent = 'The personality prompt is placed in front of your message. Token counts come from Gemini count_tokens (input = prompt + question).';
            if (!ex.prompt) return note(vis, ex.status === 'thinking' ? 'Waiting for the text AI...' : 'The secret instructions could not be loaded.');
            const pre = mk('pre', 'reveal-prompt', ex.prompt);
            pre.hidden = true;
            const btn = mk('button', 'reveal-toggle', 'show');
            btn.type = 'button';
            btn.addEventListener('click', () => { pre.hidden = !pre.hidden; btn.textContent = pre.hidden ? 'show' : 'hide'; });
            vis.append(btn, pre);
            if (ex.counts) vis.append(mk('p', 'reveal-time', `Number pieces (tokens) counted by Gemini: ${ex.counts.prompt} going in, ${ex.counts.answer} in the answer`));
        }
    },
    {
        title: () => 'The text brain answers',
        kid: () => 'It guesses the answer one word at a time, like a super-fast guessing game.',
        sig: (ex) => [ex.status, ex.voiceText, ex.gemini],
        build(ex, vis, gu) {
            gu.textContent = 'Generated by Gemini (gemini-2.5-flash); the time is measured on the server around generate_content.';
            if (ex.status === 'thinking') return note(vis, 'Thinking...');
            if (ex.status === 'error') return note(vis, 'The text AI did not answer this time.');
            quote(vis, ex.voiceText);
            vis.append(mk('p', 'reveal-time', `Thinking time: ${secs(ex.gemini)}`));
        }
    },
    {
        title: () => 'Voice fingerprint',
        kid: () => 'The voice AI listened to a short SpongeBob clip and wrote down how he sounds.',
        sig: () => [voiceprint && (voiceprint.error || voiceprint.embedding.length)],
        build(ex, vis, gu) {
            vis.append(audioPlayer('/reference-sample'));
            if (!voiceprint) return note(vis, 'Loading the fingerprint...');
            if (voiceprint.error) return note(vis, `Fingerprint unavailable: ${voiceprint.error}`);
            const c = canvasEl(1024, 60);
            vis.append(c, mk('p', 'reveal-quote', '512 numbers that describe how SpongeBob sounds'),
                mk('p', 'reveal-time', `Smallest ${voiceprint.min}, biggest ${voiceprint.max} (blue = low, red = high)`));
            drawBarcode(c, voiceprint.embedding, voiceprint.min, voiceprint.max);
            gu.textContent = `The real XTTS speaker embedding of the reference clip (${voiceprint.sample_seconds} s), computed once at server start. Also computed: gpt_cond_latent with shape [${voiceprint.gpt_cond_latent_shape.join(', ')}].`;
        }
    },
    {
        title: () => 'Reply to numbers',
        kid: () => 'The answer is also turned into number codes before the voice AI can say it.',
        sig: (ex) => [ex.vd && ex.vd.token_ids, ex.voiceStatus],
        build(ex, vis, gu) {
            gu.textContent = 'XTTS v2 text token ids of the reply (BPE vocabulary, [en] tag first). The voice model predicts audio codes from these ids plus the voice fingerprint.';
            if (ex.voiceStatus === 'error') return note(vis, `Voice AI problem: ${ex.voiceError}`);
            if (!ex.vd || !ex.vd.token_ids) return note(vis, 'Waiting for the voice AI...');
            chips(vis, ex.vd.token_ids, ex.vd.tokens);
        }
    },
    {
        title: () => 'Numbers to sound',
        kid: () => 'The voice AI turns all those numbers into a sound wave, SpongeBob\'s voice saying the answer.',
        sig: (ex) => [ex.voiceStatus, ex.vd && ex.vd.audio_url, !!ex.outBuf, ex.voice],
        build(ex, vis, gu) {
            gu.textContent = 'XTTS v2 output WAV. Waveform and spectrogram (512-point FFT, Hann window, frequency up to 11 kHz) are computed in the browser from the decoded file.';
            if (ex.voiceStatus === 'error') return note(vis, `Voice AI problem: ${ex.voiceError}`);
            if (!ex.vd) return note(vis, 'Making the voice...');
            const d = ex.vd;
            vis.append(audioPlayer(d.audio_url));
            vis.append(mk('p', 'reveal-time', `${d.sample_rate.toLocaleString()} numbers every second, ${d.num_samples.toLocaleString()} numbers in total (${secs(d.duration)} of sound)`));
            vis.append(mk('p', 'reveal-time', `Voice-making time: ${secs(ex.voice)}`));
            if (!ex.outBuf) return note(vis, 'Drawing the sound...');
            const w = canvasEl(640, 90), s = canvasEl(640, 140);
            vis.append(mk('p', 'reveal-label', 'Sound wave'), w, mk('p', 'reveal-label', 'Spectrogram: time goes right, pitch goes up, brighter = louder'), s);
            drawWave(w, ex.outBuf);
            drawSpectrogram(s, ex.outBuf);
        }
    }
];

const stepTitle = (i) => {
    const t = REVEAL_STEPS[i].title;
    return t(lastExchange || {});
};

async function loadVoiceprint() {
    if (voiceprintBusy || (voiceprint && voiceprint.embedding)) return;
    voiceprintBusy = true;
    try {
        const d = await (await fetch('/voiceprint')).json();
        voiceprint = d.success ? d : { error: d.error || 'voice server offline' };
    } catch (err) {
        voiceprint = { error: 'voice server offline' };
    }
    voiceprintBusy = false;
    refreshReveal();
}

async function loadOutputSound(ex) {
    try {
        ex.outBuf = await decodeAudio(await (await fetch(ex.vd.audio_url)).arrayBuffer());
    } catch (err) {
        console.warn('Could not decode the generated sound:', err);
    }
    if (ex === lastExchange) refreshReveal();
}

function buildCards(pipe) {
    REVEAL_STEPS.forEach((s, i) => {
        const card = mk('div', 'step');
        card.tabIndex = 0;
        const head = mk('div', 'step-head');
        head.append(mk('span', 'step-num', i + 1), mk('h3'));
        const det = mk('details', 'gu');
        det.append(mk('summary', null, 'For grown-ups'), mk('div', 'gu-body'));
        card.append(head, mk('p', 'step-kid'), mk('div', 'step-vis'), det);
        card.addEventListener('click', (e) => {
            if (revealStep === null && !e.target.closest('audio, button, summary, details, canvas')) setRevealStep(i + 1);
        });
        card.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && revealStep === null && e.target === card) setRevealStep(i + 1);
        });
        pipe.append(card);
    });
}

function renderReveal() {
    const ex = lastExchange;
    document.getElementById('revealEmpty').hidden = !!ex;
    document.getElementById('revealBody').hidden = !ex;
    if (!ex) return;
    const pipe = document.getElementById('revealPipe');
    if (!pipe.children.length) buildCards(pipe);
    REVEAL_STEPS.forEach((s, i) => {
        const sig = JSON.stringify([ex.id, s.sig(ex), s.title(ex)]);
        if (stepSigs[i] === sig) return;
        stepSigs[i] = sig;
        const card = pipe.children[i];
        const vis = card.querySelector('.step-vis'), gu = card.querySelector('.gu-body');
        card.querySelector('h3').textContent = s.title(ex);
        card.querySelector('.step-kid').textContent = s.kid(ex);
        vis.replaceChildren();
        gu.replaceChildren();
        s.build(ex, vis, gu);
    });
    applyRevealFocus();
}

function applyRevealFocus() {
    const focus = revealStep !== null;
    revealPanel.classList.toggle('is-focus', focus);
    const pipe = document.getElementById('revealPipe');
    [...pipe.children].forEach((c, i) => c.classList.toggle('focus', i === revealStep));
    document.getElementById('revealNav').hidden = !focus;
    if (!focus) return;
    document.getElementById('revealNavTitle').textContent = `Step ${revealStep + 1} of ${REVEAL_STEPS.length}: ${stepTitle(revealStep)}`;
    document.getElementById('revealPrev').disabled = revealStep === 0;
    document.getElementById('revealNext').disabled = revealStep === REVEAL_STEPS.length - 1;
}

const refreshReveal = () => { if (!revealPanel.hidden) renderReveal(); };

// step: 1-based number, 'next', 'prev' or null (overview)
function setRevealStep(step) {
    let next = null;
    if (step === 'next') next = revealStep === null ? 0 : Math.min(REVEAL_STEPS.length - 1, revealStep + 1);
    else if (step === 'prev') next = revealStep === null ? null : Math.max(0, revealStep - 1);
    else if (step != null) next = Math.max(0, Math.min(REVEAL_STEPS.length - 1, Number(step) - 1));
    if (next !== null && !lastExchange) return;
    revealStep = next;
    if (next !== null && revealPanel.hidden) { setReveal(true); return; }
    if (!revealPanel.hidden) renderReveal();
    broadcastStatus();
}

function setReveal(on) {
    revealPanel.hidden = !on;
    if (!on) revealStep = null;
    document.getElementById('revealBtn').setAttribute('aria-pressed', String(on));
    if (on) { renderReveal(); loadVoiceprint(); }
    broadcastStatus();
}

document.getElementById('revealPrev').addEventListener('click', () => setRevealStep('prev'));
document.getElementById('revealNext').addEventListener('click', () => setRevealStep('next'));
document.getElementById('revealOverview').addEventListener('click', () => setRevealStep(null));

document.addEventListener('keydown', (e) => {
    if (revealPanel.hidden || e.ctrlKey || e.metaKey || e.altKey || e.target.closest('input, textarea, select, [contenteditable]')) return;
    if (e.key === 'ArrowRight') { e.preventDefault(); setRevealStep('next'); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); setRevealStep('prev'); }
});
