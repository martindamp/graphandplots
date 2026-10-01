# Aircraft hijackings — video generator

Generates `aircraft_hijackings.mp4` (1080×1920, 16 sec, H.264 — klar til Instagram
Reels/Stories) fra den medfølgende `render.html`.

`render.html` er selvstændig: data (Aviation Safety Network / US DOT·BTS, 1931–2026,
inkl. forsøg), skrifttyper (Anton + IBM Plex Mono) og al animationskode er indbygget.
Ingen internetforbindelse kræves for at rendere.

Temaet er et mørkt **radar/nattehimmel**-look med jetsilhuetter og en rigtig ildkugle-
eksplosion i toppunktet **1969** (hijacking-tidens "golden age", 86 kapringer).
Opsætningen (render.html ↔ make_video.js-kontrakten, workflow m.m.) følger samme mønster
som `eu-electric-prices/`, så den bygger på præcis samme måde.

## Placering i repo'et

Al koden ligger i mappen `aircraft-hijackings/`. Selve GitHub Actions-workflowet ligger
i repo-roden, fordi GitHub kun kører workflows derfra:

```
<repo-rod>/
├── .github/workflows/render-hijackings.yml
└── aircraft-hijackings/
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
3. I denne mappe (`aircraft-hijackings/`), installer Playwright + Chromium:
   ```
   cd aircraft-hijackings
   npm install
   npx playwright install chromium
   ```

### Kør

```
node make_video.js
```

Resultat: `aircraft_hijackings.mp4` i samme mappe. (En midlertidig `frames/`-mappe
oprettes undervejs og slettes automatisk til sidst.)

## Kør i skyen med GitHub Actions

Workflowet `.github/workflows/render-hijackings.yml` renderer videoen på GitHubs servere —
du behøver ikke Node, ffmpeg eller Chromium lokalt. Det bygger inde i
`aircraft-hijackings/` automatisk.

Sådan:
1. Push repo'et til GitHub (med både `.github/`- og `aircraft-hijackings/`-mapperne).
2. Gå til fanen **Actions** → **Render hijackings video** → **Run workflow** (kører manuelt).
   Den kører også automatisk ved push til `main`, når noget i `aircraft-hijackings/` ændres.
3. Når jobbet er færdigt, ligger MP4'en under **Artifacts** nederst på kørslen
   (`aircraft_hijackings`) — download zip'en og pak ud.

Vil du have den som en fast fil på en Release: push et tag, fx
```
git tag hijack-v1.0 && git push origin hijack-v1.0
```

## Juster

Åbn toppen af `make_video.js`:
- `FPS` — billeder pr. sekund (30 er fint)
- `DUR` — længde i sekunder (skal matche tidslinjen i `render.html`)
- `SEED` — skift tal for et andet eksplosions-/vragstumpe-mønster
- `CRF` — kvalitet (lavere = bedre/større fil; 18 = høj kvalitet)

Vil du ændre udseende, farver, tekst eller tempo, ligger alt i `render.html`:
- Tekster: søg på `AIRCRAFT HIJACKINGS`, `GOLDEN AGE OF SKYJACKING`, `9/11`, `SKYJACKED`, kildelinjen.
- Farver: `:root`-blokken øverst i `<style>` (`--steel`, `--amber`, `--hot`, `--radar` osv.).
- Data: arrayet `V` i `<script>` (én værdi pr. år fra 1931). `YMAX` er toppen af y-aksen.
- Tempo: `SEG_D` i `<script>` er sekunder pr. år i tre afsnit (1931–1960, 1960–2000,
  2000–2026). Midterafsnittet er sat langsommere (`0.20` mod `0.09`) så den dramatiske
  periode 1960–2000 tegnes i ca. halv fart. `T_DRAW1`/`T_END` udregnes automatisk af
  `SEG_D` — hvis du ændrer tempoet, så sæt `DUR` i `make_video.js` til samme værdi som
  `T_END` (udskriv den evt. i konsollen).
- Tidslinje i øvrigt: `T_INTRO`, `T_DRAW0`, slut-hold og `T_LOOP`. Eksplosionen affyres
  automatisk når stregen når 1969 (`tPeak` udregnes ud fra tempoet).

## Forhåndsvis uden at rendere

Du kan åbne `render.html` direkte i en browser for at se animationen loope live
(uden `#capture` i URL'en kører den normalt af sig selv).

## Data & kilder

Kapringer pr. år på verdensplan, **inkl. forsøg** (bedste estimat, da databaser tæller
forskelligt). Hårde tal: US DOT/BTS 1970–2000 samt dokumenterede holdepunkter (1969 ≈ 86,
2001 = 11). Før 1968 og efter 2001 er interpoleret ud fra periode-opsummeringer
(Aviation Safety Network / Our World in Data). Den første registrerede kapring var
21. februar 1931 i Peru.

## Lyd

MP4'en er uden lyd. Læg torden/jetmotor/eksplosion på i fx Instagram Edits, CapCut eller
iMovie — sigt efter at eksplosionen lander ~6 sek (1969-toppunktet).
