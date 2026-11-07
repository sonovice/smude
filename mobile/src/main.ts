import { convertFileSrc, invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import './style.css';
type Imported = { path: string; width: number; height: number };
type Page = { id: string; path: string; width: number; height: number; profile: Record<string, unknown> };
type ResultPages = { seconds: number; pages: Page[]; warnings: string[] };
type Environment = { platform: string; architecture: string; demo: string | null; auto_run: boolean; benchmark_models: boolean };
let busy = false;
let demo: string | null = null;
let benchmarkModels = false;
const root = document.querySelector<HTMLDivElement>('#app')!;
root.innerHTML = `<header><div class="brand"><svg viewBox="0 0 28 28" aria-hidden="true"><path d="M3 8h22M3 13h22M3 18h22M3 23h22M8 4v21"/></svg><span>Smude</span></div><span class="badge">MOBILE TEST</span></header>
<section class="intro"><h1>Aus einem Foto.<br>Eine klare Notenseite.</h1><p>Automatisch entzerren. Direkt auf deinem Gerät.</p></section>
<section class="actions"><button id="pick" class="primary">Bild auswählen <span>↗</span></button><button id="camera">Foto aufnehmen</button><button id="demo" hidden>Testbild verwenden</button></section>
<section class="status" role="status" aria-live="polite"><i></i><span id="status">Bereit</span></section>
<section class="previews"><article><div class="card-head"><span>Original</span><small id="source-size"></small></div><div class="image-box" id="source-box"><span class="placeholder">Dein Notenbild</span><img id="source" hidden alt="Originalbild"></div></article><div id="results"></div></section>
<p class="error" id="error" role="alert" hidden></p><footer><span>GRIDNOTE LABS</span><small id="device"></small></footer>`;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
function status(message: string, running: boolean) {
  $('status').textContent = message; root.classList.toggle('busy', running);
  $('pick').toggleAttribute('disabled', running); $('camera').toggleAttribute('disabled', running); $('demo').toggleAttribute('disabled', running);
}
function error(message: unknown) { $('error').textContent = String(message); $('error').hidden = false; }
async function load(path: string) {
  if (busy) return;
  const started = performance.now();
  busy = true; $('results').replaceChildren(); $('error').hidden = true;
  status('Bild vorbereiten …', true);
  try {
    const original = await invoke<Imported>('import_image', { path });
    const image = $<HTMLImageElement>('source'); image.src = convertFileSrc(original.path); image.hidden = false;
    $('source-box').querySelector('.placeholder')?.remove();
    $('source-size').textContent = `${original.width} × ${original.height}`;
    if (benchmarkModels) {
      benchmarkModels = false; status('Modelllaufzeit prüfen …', true);
      await invoke('benchmark_models', { path: original.path });
    }
    status('Seite erkennen …', true);
    const result = await invoke<ResultPages>('process_image', { path: original.path });
    const previews: HTMLImageElement[] = [];
    for (const page of result.pages) {
      const article = document.createElement('article');
      const head = document.createElement('div'); head.className = 'card-head';
      const title = document.createElement('span'); title.textContent = result.pages.length > 1 ? `Ergebnis · Seite ${page.id}` : 'Ergebnis';
      const save = document.createElement('button'); save.className = 'export'; save.textContent = 'Exportieren ↗';
      save.addEventListener('click', async () => { try { if (await invoke<boolean>('export_page', { path: page.path })) status('Bild exportiert', false); } catch (e) { error(e); } });
      head.append(title, save); const box = document.createElement('div'); box.className = 'image-box';
      const image = new Image(); image.alt = 'Entzerrte Notenseite'; image.src = convertFileSrc(page.path);
      previews.push(image);
      box.append(image); article.append(head, box); $('results').append(article);
    }
    await Promise.all(previews.map(image => image.decode()));
    const visibleSeconds = (performance.now() - started) / 1000;
    status(`${result.pages.length} ${result.pages.length === 1 ? 'Seite' : 'Seiten'} · ${visibleSeconds.toFixed(1)} s`, false);
    $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (result.warnings.length) error(result.warnings.join('\n'));
  } catch (e) { error(e); status('Verarbeitung fehlgeschlagen', false); }
  finally { busy = false; status($('status').textContent!, false); }
}
$('pick').addEventListener('click', async () => {
  if (busy) return;
  try { const path = await open({ multiple: false, directory: false, filters: [{ name: 'Notenbilder', extensions: ['jpg', 'jpeg', 'png', 'heic', 'heif'] }] }); if (typeof path === 'string') await load(path); }
  catch (e) { error(e); }
});
$('camera').addEventListener('click', async () => {
  if (busy) return;
  busy = true; status('Kamera geöffnet …', true); $('error').hidden = true;
  let path: string | null = null;
  try { path = await invoke<string | null>('capture_photo'); }
  catch (e) { error(e); }
  finally { busy = false; status('Bereit', false); }
  if (path) await load(path);
});
$('demo').addEventListener('click', () => { if (demo) void load(demo); });
await listen<{ label: string; page: string }>('processing-stage', event => { if (busy) status(`${event.payload.label} …`, true); }).catch(error);
try {
  const env = await invoke<Environment>('environment'); demo = env.demo; benchmarkModels = env.benchmark_models;
  $('demo').hidden = !demo; $('device').textContent = `${env.platform} · ${env.architecture}`;
  if (env.auto_run && demo) void load(demo);
} catch (e) { error(e); }
