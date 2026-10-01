# EU electricity prices — video generator

Generates `european_power_prices.mp4` (1080×1920, 15 sec, H.264 — klar til Instagram
Reels/Stories) fra den medfølgende `render.html`.

`render.html` er selvstændig: data (Ember / ENTSO-E, 2015–2026), skrifttyper (Anton +
IBM Plex Mono) og al animationskode er indbygget. Ingen internetforbindelse kræves
for at rendere.

## Placering i repo'et

Al koden ligger i mappen `eu-electric-prices/`. Selve GitHub Actions-workflowet ligger
i repo-roden, fordi GitHub kun kører workflows derfra:

```
<repo-rod>/
├── .github/workflows/render-video.yml
└── eu-electric-prices/
    ├── render.html
    ├── make_video.js
    ├── package.json
    └── README.md   (denne fil)
```

## Kør lokalt

### Forudsætninger (én gang)

1. **Node.js 18+** — https://nodejs.org
2. **ffmpeg** på PATH:
   - macOS:   `brew install ffmpeg`
   - Windows: `winget install Gyan.FFmpeg`  (eller https://ffmpeg.org/download.html)
   - Linux:   `sudo apt install ffmpeg`
3. I denne mappe (`eu-electric-prices/`), installer Playwright + Chromium:
   ```
   cd eu-electric-prices
   npm install
   npx playwright install chromium
   ```

### Kør

```
node make_video.js
```

Resultat: `european_power_prices.mp4` i samme mappe. (En midlertidig `frames/`-mappe
oprettes undervejs og slettes automatisk til sidst.)

## Kør i skyen med GitHub Actions

Workflowet `.github/workflows/render-video.yml` renderer videoen på GitHubs servere —
du behøver ikke Node, ffmpeg eller Chromium lokalt. Det bygger inde i
`eu-electric-prices/` automatisk.

Sådan:
1. Push repo'et til GitHub (med både `.github/`- og `eu-electric-prices/`-mapperne).
2. Gå til fanen **Actions** → **Render video** → **Run workflow** (kører manuelt).
   Den kører også automatisk ved push til `main`, når noget i `eu-electric-prices/` ændres.
3. Når jobbet er færdigt, ligger MP4'en under **Artifacts** nederst på kørslen
   (`european_power_prices`) — download zip'en og pak ud.

Vil du have den som en fast fil på en Release: push et tag, fx
```
git tag v1.0 && git push origin v1.0
```
Så vedhæfter workflowet `european_power_prices.mp4` direkte til en GitHub Release.

## Juster

Åbn toppen af `make_video.js`:
- `FPS` — billeder pr. sekund (30 er fint)
- `DUR` — længde i sekunder (skal matche tidslinjen i `render.html`)
- `SEED` — skift tal for et andet lyn-mønster
- `CRF` — kvalitet (lavere = bedre/større fil; 18 = høj kvalitet)

Vil du ændre udseende, farver, tekst eller tempo, ligger alt i `render.html`:
- Tekster: søg på `EUROPEAN`, `POWER PRICES`, `ENERGY CRISIS`, `COVID`, kildelinjen.
- Farver: `:root`-blokken øverst i `<style>` (`--cyan`, `--hot`, `--amber` osv.).
- Tidslinje (intro/optegning/slut): konstanterne `T_INTRO`, `T_DRAW0`, `T_DRAW1`,
  `T_END`, `T_LOOP` i `<script>`. Hvis du ændrer `T_END`, så sæt `DUR` i
  `make_video.js` tilsvarende.

## Forhåndsvis uden at rendere

Du kan åbne `render.html` direkte i en browser for at se animationen loope live
(uden `#capture` i URL'en kører den normalt af sig selv).

## Lyd

MP4'en er uden lyd. Læg torden/musik på i fx Instagram Edits, CapCut eller iMovie —
sigt efter at lyn-droppet lander ~8–9 sek (energikrise-nedslaget).
