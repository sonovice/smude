import { convertFileSrc, invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import { ModelSession, type DewarpModel } from './model-session';
import './style.css';
type Imported = { path: string; width: number; height: number };
type Page = { id: string; path: string; width: number; height: number; profile: Record<string, unknown> };
type ResultPages = { seconds: number; pages: Page[]; warnings: string[] };
type Environment = { platform: string; architecture: string; demo: string | null; auto_run: boolean; benchmark_models: boolean };
let busy = false;
let activeModel: DewarpModel | null = null;
let demo: string | null = null;
let benchmarkModels = false;
let preferred: DewarpModel = 'mesh';
try { if (localStorage.getItem('dewarp-model') === 'gcs') preferred = 'gcs'; } catch { /* Storage can be unavailable in private webviews. */ }
const session = new ModelSession<Imported, ResultPages>(preferred);
const label = (model: DewarpModel) => model === 'gcs' ? 'GCS' : 'Mesh';
const root = document.querySelector<HTMLDivElement>('#app')!;
root.innerHTML = `<header><div class="brand"><svg viewBox="0 0 28 28" aria-hidden="true"><path d="M3 8h22M3 13h22M3 18h22M3 23h22M8 4v21"/></svg><span>Smude</span></div><span class="badge">MOBILE TEST</span></header>
<section class="intro"><h1>Aus einem Foto.<br>Eine klare Notenseite.</h1><p>Automatisch entzerren. Direkt auf deinem Gerät.</p></section>
<section class="actions"><button id="pick" class="primary">Bild auswählen <span>↗</span></button><button id="camera">Foto aufnehmen</button><button id="demo" hidden>Testbild verwenden</button></section>
<section class="model-picker" aria-label="Entzerrungsverfahren"><span>Entzerrung</span><div class="segments" role="group" aria-label="Verfahren"><button id="model-gcs" aria-pressed="false">GCS</button><button id="model-mesh" aria-pressed="true">Mesh</button></div></section>
<section class="status" role="status" aria-live="polite"><i></i><span id="status">Bereit</span></section>
<section class="previews"><article><div class="card-head"><span>Original</span><small id="source-size"></small></div><div class="image-box" id="source-box"><span class="placeholder">Dein Notenbild</span><img id="source" hidden alt="Originalbild"></div></article><div id="results"></div></section>
<p class="error" id="error" role="alert" hidden></p><footer><span>NOTAVIUM</span><small id="device"></small></footer>`;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
function status(message: string) {
  $('status').textContent = message; root.classList.toggle('busy', busy);
  for (const id of ['pick', 'camera', 'demo']) $(id).toggleAttribute('disabled', busy);
}
function error(message: unknown) { $('error').textContent = String(message); $('error').hidden = false; }
function updateSelection() {
  for (const model of ['gcs', 'mesh'] as const) $('model-' + model).setAttribute('aria-pressed', String(session.model === model));
}
function pending() {
  const article = document.createElement('article');
  const head = document.createElement('div'); head.className = 'card-head'; head.textContent = `Ergebnis · ${label(session.model)}`;
  const box = document.createElement('div'); box.className = 'image-box';
  const hint = document.createElement('span'); hint.className = 'placeholder'; hint.textContent = 'Wird berechnet …';
  box.append(hint); article.append(head, box); $('results').replaceChildren(article);
}
function display(result: ResultPages, model: DewarpModel): HTMLImageElement[] {
  const previews: HTMLImageElement[] = [];
  $('results').replaceChildren();
  for (const page of result.pages) {
    const article = document.createElement('article'); article.dataset.model = model;
    const head = document.createElement('div'); head.className = 'card-head';
    const title = document.createElement('span'); title.textContent = `Ergebnis · ${label(model)}${result.pages.length > 1 ? ` · Seite ${page.id}` : ''}`;
    const save = document.createElement('button'); save.className = 'export'; save.textContent = 'Exportieren ↗';
    save.addEventListener('click', async () => {
      try { if (await invoke<boolean>('export_page', { path: page.path })) status('Bild exportiert'); }
      catch (e) { error(e); }
    });
    head.append(title, save); const box = document.createElement('div'); box.className = 'image-box';
    const image = new Image(); image.alt = `Entzerrte Notenseite · ${label(model)}`; image.src = convertFileSrc(page.path);
    previews.push(image); box.append(image); article.append(head, box); $('results').append(article);
  }
  return previews;
}
function completed(result: ResultPages, seconds?: number) {
  status(`${result.pages.length} ${result.pages.length === 1 ? 'Seite' : 'Seiten'} · ${(seconds ?? result.seconds).toFixed(1)} s · ${label(session.model)}`);
  if (result.warnings.length) error(result.warnings.join('\n'));
}
async function processCurrent(scroll = false) {
  if (busy || !session.input) return;
  const cached = session.result();
  if (cached) { display(cached, session.model); completed(cached); return; }
  busy = true; $('error').hidden = true;
  try {
    // Model changes during a job select the next job. Only one native job runs;
    // a late result is cached for its own model, never shown under another tab.
    for (let request = session.request(); request; request = session.request()) {
      activeModel = request.model;
      const started = performance.now(); pending(); status(`${label(request.model)} · Seite erkennen …`);
      try {
        const result = await invoke<ResultPages>('process_image', { path: request.path, model: request.model });
        if (!session.accept(request, result)) break;
        if (session.model === request.model) {
          const previews = display(result, request.model);
          await Promise.all(previews.map(image => image.decode()));
          if (session.model === request.model) {
            completed(result, (performance.now() - started) / 1000);
            if (scroll) { $('results').scrollIntoView({ behavior: 'smooth', block: 'start' }); scroll = false; }
          }
        }
      } catch (e) {
        if (session.model !== request.model) continue;
        error(e); status('Verarbeitung fehlgeschlagen'); break;
      }
    }
  } finally {
    activeModel = null; busy = false;
    const cached = session.result();
    if (cached) { display(cached, session.model); completed(cached); }
    else status($('status').textContent!);
  }
}
async function load(path: string) {
  if (busy) return;
  busy = true; $('error').hidden = true; status('Bild vorbereiten …');
  let imported = false;
  try {
    const original = await invoke<Imported>('import_image', { path });
    session.setInput(original); imported = true; $('results').replaceChildren();
    const image = $<HTMLImageElement>('source'); image.src = convertFileSrc(original.path); image.hidden = false;
    $('source-box').querySelector('.placeholder')?.remove(); $('source-size').textContent = `${original.width} × ${original.height}`;
    if (benchmarkModels) { benchmarkModels = false; status('Modelllaufzeit prüfen …'); await invoke('benchmark_models', { path: original.path }); }
  } catch (e) { error(e); status('Bild konnte nicht geöffnet werden'); }
  finally { busy = false; status($('status').textContent!); }
  if (imported) await processCurrent(true);
}
for (const model of ['gcs', 'mesh'] as const) $('model-' + model).addEventListener('click', () => {
  if (session.model === model) return;
  session.select(model); updateSelection(); $('error').hidden = true;
  try { localStorage.setItem('dewarp-model', model); } catch { /* Preference is optional. */ }
  const cached = session.result();
  if (cached) { display(cached, model); completed(cached); }
  else if (session.input) {
    pending();
    if (busy) status(`${activeModel ? label(activeModel) + ' wird abgeschlossen' : 'Bild wird vorbereitet'} · ${label(model)} folgt …`);
    else void processCurrent();
  }
});
$('pick').addEventListener('click', async () => {
  if (busy) return;
  try { const path = await open({ multiple: false, directory: false, filters: [{ name: 'Notenbilder', extensions: ['jpg', 'jpeg', 'png', 'heic', 'heif'] }] }); if (typeof path === 'string') await load(path); }
  catch (e) { error(e); }
});
$('camera').addEventListener('click', async () => {
  if (busy) return;
  busy = true; status('Kamera geöffnet …'); $('error').hidden = true;
  let path: string | null = null;
  try { path = await invoke<string | null>('capture_photo'); }
  catch (e) { error(e); }
  finally { busy = false; status('Bereit'); }
  if (path) await load(path);
});
$('demo').addEventListener('click', () => { if (demo) void load(demo); });
updateSelection();
await listen<{ label: string; page: string }>('processing-stage', event => {
  if (busy && activeModel && !session.result()) status(`${label(activeModel)} · ${event.payload.label} …`);
}).catch(error);
try {
  const env = await invoke<Environment>('environment'); demo = env.demo; benchmarkModels = env.benchmark_models;
  $('demo').hidden = !demo; $('device').textContent = `${env.platform} · ${env.architecture}`;
  if (env.auto_run && demo) void load(demo);
} catch (e) { error(e); }
