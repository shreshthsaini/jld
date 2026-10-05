/* Viewers for the recorded examples. Every value is read from data/; nothing is recomputed here. */
'use strict';
const data = window.JLD_DATA;
const videoMaps = window.JLD_VIDEO_MAPS;
const fixed = (value, digits = 3) => Number(value).toFixed(digits);
const $ = selector => document.querySelector(selector);

/* ---------- image comparisons ---------- */
let selectedCase = 0;
let overlay = false;

const METRICS = [
  ['human', 'Human MOS', 2, 'high'],
  ['perjaclens', 'JLD', 3, 'low'],
  ['dists', 'DISTS', 3, 'low'],
  ['lpips_vgg', 'LPIPS-VGG', 3, 'low'],
  ['psnr', 'PSNR (dB)', 2, 'high'],
];

function renderComparison() {
  const item = data.comparisons[selectedCase];
  const dataset = item.dataset === 'tid2013' ? 'TID2013' : 'KADID-10k test';
  const agrees = item.jld_preferred === item.human_preferred;
  $('#case-tabs').innerHTML = data.comparisons.map((entry, index) => `
    <button role="tab" id="case-${index}" aria-controls="comparison" data-case="${index}"
      aria-selected="${index === selectedCase}" tabindex="${index === selectedCase ? 0 : -1}">${index + 1}. ${entry.variants.A.label} / ${entry.variants.B.label}</button>`).join('');

  const better = (key, direction) => {
    const a = item.variants.A.scores[key], b = item.variants.B.scores[key];
    return (direction === 'high' ? a > b : a < b) ? 'A' : 'B';
  };
  const panel = label => {
    const variant = item.variants[label];
    return `<div class="panel">
      <p><b>${label}</b> · ${variant.label}</p>
      <a href="${variant.image}"><img src="${variant.image}" width="504" height="378" alt="Image ${label}: ${variant.label}"></a>
      <a class="map${overlay ? ' overlay' : ''}" href="${variant.map}" style="background-image:url('${variant.image}')"><img src="${variant.map}" width="504" height="378" alt="Per-patch lens map for image ${label}"></a>
    </div>`;
  };
  const view = $('#comparison');
  view.setAttribute('role', 'tabpanel');
  view.setAttribute('aria-labelledby', `case-${selectedCase}`);
  view.innerHTML = `
    <div class="viewer-head">
      <h3>${dataset}, reference ${item.id.split('-').at(-1).toUpperCase()}</h3>
      <span class="verdict${agrees ? '' : ' miss'}">People prefer A. JLD prefers ${item.jld_preferred}${agrees ? '' : ' (a miss)'}.</span>
      <label class="check"><input type="checkbox" id="map-toggle"${overlay ? ' checked' : ''}> Overlay maps on the images</label>
    </div>
    <div class="panels">
      <div class="panel ref">
        <p><b>Reference</b></p>
        <a href="${item.reference}"><img src="${item.reference}" width="504" height="378" alt="Reference image"></a>
      </div>
      ${panel('A')}${panel('B')}
      <div class="panel legend">
        <p>Per-patch lens displacement, one scale for both maps.</p>
        <div class="colorbar"><span>0</span><i aria-hidden="true"></i><span>${fixed(item.map_max)}</span></div>
      </div>
    </div>
    <div class="tablewrap"><table>
      <thead><tr><th>Image</th>${METRICS.map(([, name, , direction]) => `<th>${name} ${direction === 'high' ? '↑' : '↓'}</th>`).join('')}</tr></thead>
      <tbody>${['A', 'B'].map(label => `<tr><td>${label}</td>${METRICS.map(([key, , digits, direction]) =>
        `<td${better(key, direction) === label ? ' class="best"' : ''}>${fixed(item.variants[label].scores[key], digits)}</td>`).join('')}</tr>`).join('')}</tbody>
    </table></div>
    <p class="note">Bold marks the image each measure prefers.</p>`;

  const focusTab = () => $(`#case-${selectedCase}`).focus();
  document.querySelectorAll('#case-tabs button').forEach(button => {
    button.addEventListener('click', () => { selectedCase = Number(button.dataset.case); renderComparison(); focusTab(); });
    button.addEventListener('keydown', event => {
      const count = data.comparisons.length;
      const moves = { ArrowRight: (selectedCase + 1) % count, ArrowLeft: (selectedCase - 1 + count) % count, Home: 0, End: count - 1 };
      if (!(event.key in moves)) return;
      event.preventDefault();
      selectedCase = moves[event.key];
      renderComparison(); focusTab();
    });
  });
  $('#map-toggle').addEventListener('change', event => {
    overlay = event.target.checked;
    renderComparison();
    $('#map-toggle').focus();
  });
}

/* ---------- video maps ---------- */
let selectedScene = 2;
let selectedFrame = 0;
let frameTimer = null;

function stopFramePlayback() {
  clearTimeout(frameTimer);
  frameTimer = null;
  const button = $('#frame-play');
  if (button) { button.textContent = 'Play'; button.setAttribute('aria-pressed', 'false'); }
}

function updateVideoFrame(index) {
  const scene = videoMaps.scenes[selectedScene];
  selectedFrame = Math.max(0, Math.min(index, scene.frames.length - 1));
  const frame = scene.frames[selectedFrame];
  const slider = $('#video-frame');
  slider.value = selectedFrame;
  slider.setAttribute('aria-valuetext', `${fixed(frame.time_seconds, 2)} seconds, frame ${selectedFrame + 1} of ${scene.frames.length}`);
  $('#frame-position').textContent = `${fixed(frame.time_seconds, 2)} s · frame ${selectedFrame + 1} of ${scene.frames.length}`;
  for (const key of ['reference', 'A', 'B', 'mapA', 'mapB']) {
    const image = document.querySelector(`[data-video-panel="${key}"]`);
    image.src = frame[key];
    image.closest('a').href = frame[key];
  }
}

function toggleFramePlayback() {
  if (frameTimer !== null) { stopFramePlayback(); return; }
  document.querySelectorAll('video').forEach(video => video.pause());
  const scene = videoMaps.scenes[selectedScene];
  if (selectedFrame === scene.frames.length - 1) updateVideoFrame(0);
  const button = $('#frame-play');
  button.textContent = 'Pause';
  button.setAttribute('aria-pressed', 'true');
  const next = () => {
    if (selectedFrame >= scene.frames.length - 1) { stopFramePlayback(); return; }
    const delay = 1000 * (scene.frames[selectedFrame + 1].time_seconds - scene.frames[selectedFrame].time_seconds);
    frameTimer = setTimeout(() => { updateVideoFrame(selectedFrame + 1); next(); }, Math.max(30, delay));
  };
  next();
}

function renderVideoMaps() {
  stopFramePlayback();
  const scene = videoMaps.scenes[selectedScene];
  const scores = data.avt.find(item => item.source === scene.id);
  const agrees = scores.comparison.jld_spatial_correct === 1;
  const lower = scores.A.jld_spatial < scores.B.jld_spatial ? 'A' : 'B';
  const higher = scores.A.mos > scores.B.mos ? 'A' : 'B';
  const other = label => (label === 'A' ? 'B' : 'A');
  $('#video-map-tabs').innerHTML = videoMaps.scenes.map((item, index) =>
    `<button data-scene="${index}" aria-pressed="${index === selectedScene}">${item.label}</button>`).join('');
  const image = (key, label) => `<a href="${scene.frames[0][key]}"><img data-video-panel="${key}" src="${scene.frames[0][key]}" width="448" height="252" alt="${scene.label}: ${label}"></a>`;
  $('#video-map-explorer').innerHTML = `
    <div class="viewer-head">
      <h3>${scene.label}</h3>
      <span class="verdict${agrees ? '' : ' miss'}">People prefer ${higher}. Framewise JLD prefers ${lower}${agrees ? '' : ' (a miss)'}.</span>
    </div>
    <div class="frame-controls">
      <button class="control" id="frame-play" aria-pressed="false">Play</button>
      <input id="video-frame" type="range" min="0" max="${scene.frames.length - 1}" value="0" step="1" aria-label="Frame">
      <output id="frame-position" for="video-frame"></output>
    </div>
    <div class="panels">
      <div class="panel ref"><p><b>Reference</b></p>${image('reference', 'reference frame')}</div>
      ${['A', 'B'].map(label => `<div class="panel"><p><b>${label}</b> · ${scores[label].codec}, ${scores[label].resolution}</p>${image(label, `encode ${label}`)}<span class="map">${image('map' + label, `per-patch lens map for encode ${label}`)}</span></div>`).join('')}
      <div class="panel legend">
        <p>Per-patch lens displacement, one scale for both encodes and every frame.</p>
        <div class="colorbar magma"><span>0</span><i aria-hidden="true"></i><span>${fixed(scene.map_scale.max)}</span></div>
      </div>
    </div>
    <div class="tablewrap"><table>
      <thead><tr><th>Encode</th><th>Human MOS ↑</th><th>Framewise JLD ↓</th><th>PSNR (dB) ↑</th></tr></thead>
      <tbody>${['A', 'B'].map(label => `<tr><td>${label}</td>
        <td${higher === label ? ' class="best"' : ''}>${fixed(scores[label].mos, 2)}</td>
        <td${lower === label ? ' class="best"' : ''}>${fixed(scores[label].jld_spatial, 4)}</td>
        <td${scores[label].psnr > scores[other(label)].psnr ? ' class="best"' : ''}>${fixed(scores[label].psnr, 2)}</td></tr>`).join('')}</tbody>
    </table></div>
    <p class="note">Scores describe the whole clip and do not change with the slider. Bold marks the encode each measure prefers.</p>
    <details><summary>Watch all 90 frames as a movie</summary>
      <video id="map-movie" controls muted playsinline preload="none" poster="${scene.video.poster}" aria-label="${scene.label}: frames and lens maps">
        <source src="${scene.video.webm}" type="video/webm"><source src="${scene.video.mp4}" type="video/mp4">
        <a href="${scene.video.mp4}">Download the map movie</a></video>
    </details>`;
  document.querySelectorAll('[data-scene]').forEach(button => button.addEventListener('click', () => {
    selectedScene = Number(button.dataset.scene);
    selectedFrame = 0;
    renderVideoMaps();
    document.querySelector(`[data-scene="${selectedScene}"]`).focus();
  }));
  $('#video-frame').addEventListener('input', event => { stopFramePlayback(); updateVideoFrame(Number(event.target.value)); });
  $('#frame-play').addEventListener('click', toggleFramePlayback);
  $('#map-movie').addEventListener('play', exclusivePlay);
  updateVideoFrame(selectedFrame);
}

/* ---------- Waterloo clips, scored by the video lens ---------- */
function renderVideos() {
  $('#waterloo').innerHTML = data.waterloo.map(item => {
    const [a, b] = item.scores;
    const agrees = a.lens_distance < b.lens_distance;
    const row = (self, rival) => `<tr><td>${self.lane}</td>
      <td${self.mos > rival.mos ? ' class="best"' : ''}>${fixed(self.mos, 2)}</td>
      <td${self.lens_distance < rival.lens_distance ? ' class="best"' : ''}>${fixed(self.lens_distance, 3)}</td>
      <td${self.vmaf > rival.vmaf ? ' class="best"' : ''}>${fixed(self.vmaf, 2)}</td></tr>`;
    return `<article class="video-card">
      <video controls muted playsinline preload="none" poster="assets/videos/waterloo-${item.id}.png" aria-label="${item.label}: reference, A and B side by side">
        <source src="assets/videos/waterloo-${item.id}.mp4" type="video/mp4">
        <a href="assets/videos/waterloo-${item.id}.mp4">Download the ${item.label} clip</a></video>
      <h3>${item.label}<span class="verdict${agrees ? '' : ' miss'}">${agrees ? 'JLD-video agrees with people' : 'JLD-video misses'}</span></h3>
      <table><thead><tr><th>Encode</th><th>Human MOS ↑</th><th>JLD-video ↓</th><th>VMAF ↑</th></tr></thead>
      <tbody>${row(a, b)}${row(b, a)}</tbody></table>
    </article>`;
  }).join('');
}

function exclusivePlay(event) {
  stopFramePlayback();
  document.querySelectorAll('video').forEach(video => { if (video !== event.target) video.pause(); });
}

/* ---------- BibTeX copy ---------- */
function wireCopy() {
  const button = $('#copy-bibtex');
  if (!button) return;
  if (!navigator.clipboard) { button.hidden = true; return; }
  button.addEventListener('click', () => {
    navigator.clipboard.writeText($('#bibtex-text').textContent.trim()).then(() => {
      button.textContent = 'Copied';
      setTimeout(() => { button.textContent = 'Copy'; }, 1500);
    });
  });
}

if (data && $('#comparison')) renderComparison();
if (data && videoMaps && $('#video-map-explorer')) renderVideoMaps();
if (data && $('#waterloo')) {
  renderVideos();
  document.querySelectorAll('#waterloo video').forEach(video => video.addEventListener('play', exclusivePlay));
}
wireCopy();
document.addEventListener('visibilitychange', () => { if (document.hidden) stopFramePlayback(); });
