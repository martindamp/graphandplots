#!/usr/bin/env node
/*
 * make_video.js — renders render.html to european_power_prices.mp4 (1080x1920, 15s, H.264)
 *
 * Prerequisites (one-time):
 *   1. Node.js 18+        https://nodejs.org
 *   2. ffmpeg on PATH     macOS: brew install ffmpeg | Windows: winget install Gyan.FFmpeg | Linux: apt install ffmpeg
 *   3. In this folder:    npm install playwright
 *                         npx playwright install chromium
 *
 * Run:   node make_video.js
 * Output: ./european_power_prices.mp4   (plus a temporary ./frames/ folder)
 */
const { chromium } = require('playwright');
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// ---- settings you can tweak ----
const W = 1080, H = 1920;      // 9:16 Instagram resolution
const FPS = 30;                // frames per second
const DUR = 15.0;              // seconds (match the timeline in render.html)
const SEED = 1337;             // change for a different lightning pattern
const CRF = 18;                // H.264 quality (lower = better/larger, 18 is high quality)
// --------------------------------

const FRAMES = Math.round(FPS * DUR);
const HERE = __dirname;
const HTML = path.join(HERE, 'render.html');
const OUT = path.join(HERE, 'frames');
const MP4 = path.join(HERE, 'european_power_prices.mp4');

(async () => {
  if (!fs.existsSync(HTML)) { console.error('Missing render.html next to this script.'); process.exit(1); }
  fs.rmSync(OUT, { recursive: true, force: true });
  fs.mkdirSync(OUT, { recursive: true });

  console.log(`Rendering ${FRAMES} frames at ${W}x${H}...`);
  const browser = await chromium.launch({ args: ['--no-sandbox', '--force-color-profile=srgb', '--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });

  // deterministic RNG so the lightning is identical every run
  await page.addInitScript((seed) => {
    let s = seed >>> 0;
    Math.random = function () { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  }, SEED);

  await page.goto('file://' + HTML + '#capture', { waitUntil: 'load' });
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  await page.evaluate(({ w, h }) => { window.__setup(w, h); window.__reset(); }, { w: W, h: H });
  await page.evaluate(async () => { if (document.fonts && document.fonts.ready) await document.fonts.ready; });

  for (let i = 0; i < FRAMES; i++) {
    const data = await page.evaluate((t) => { window.__render(t); return document.getElementById('c').toDataURL('image/png'); }, i / FPS);
    fs.writeFileSync(path.join(OUT, 'f' + String(i).padStart(4, '0') + '.png'), Buffer.from(data.split(',')[1], 'base64'));
    if (i % 60 === 0) process.stdout.write(`  frame ${i}/${FRAMES}\r`);
  }
  await browser.close();
  console.log(`\nFrames done. Encoding MP4 with ffmpeg...`);

  const args = [
    '-y', '-framerate', String(FPS), '-i', path.join(OUT, 'f%04d.png'),
    '-c:v', 'libx264', '-profile:v', 'high', '-level', '4.0', '-pix_fmt', 'yuv420p',
    '-r', String(FPS), '-movflags', '+faststart', '-preset', 'slow', '-crf', String(CRF),
    MP4
  ];
  const r = spawnSync('ffmpeg', args, { stdio: 'inherit' });
  if (r.error) {
    console.error('\nffmpeg not found. Install it (see header of this file), then re-run — frames are already in ./frames.');
    process.exit(1);
  }
  // clean up frames
  fs.rmSync(OUT, { recursive: true, force: true });
  console.log(`\nDone  ->  ${MP4}`);
})().catch(e => { console.error(e); process.exit(1); });
