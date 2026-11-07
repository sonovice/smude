// The Rust dependency normally downloads these itself. Prewarming its cache
// also handles macOS setups where the build-script HTTP resolver fails while
// curl works. Hashes are from ort-sys 2.0.0-rc.15/build/download/dist.tsv.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, openSync, closeSync, readFileSync, renameSync, rmSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = dirname(fileURLToPath(import.meta.url));
const lock = readFileSync(resolve(root, '../Cargo.lock'), 'utf8');
if (!/name = "ort-sys"\nversion = "2\.0\.0-rc\.15"/.test(lock)) throw new Error('Update the mobile runtime hashes for the current ort-sys version.');
const cache = process.platform === 'darwin'
  ? join(homedir(), 'Library/Caches/ort.pyke.io/dfbin')
  : join(process.env.XDG_CACHE_HOME || join(homedir(), '.cache'), 'ort.pyke.io/dfbin');
const builds = [
  ['aarch64-linux-android', 'nnapi', '88909ed4b84a13e818b5ca9c3b752a4f857b6c781c77f6c5e57af83036840d1c'],
  ['aarch64-apple-ios', 'coreml', 'fa5242714c3d26d06ba8f2b92a4d75943dd580f8c78a1f0252498155a1630d31'],
  ['aarch64-apple-ios-sim', 'coreml', 'b16714ab5dd8e39e09677b14d4badcb90388e12d036349cd813f885cabfb8736'],
];
function run(program, args, options = {}) {
  const result = spawnSync(program, args, { stdio: 'inherit', ...options });
  if (result.error || result.status !== 0) throw result.error || new Error(`${program} exited with ${result.status}`);
}
for (const [target, provider, hash] of builds) {
  const destination = join(cache, target, hash);
  if (existsSync(join(destination, 'libonnxruntime.a'))) continue;
  const parent = dirname(destination); mkdirSync(parent, { recursive: true });
  const temporary = mkdtempSync(join(parent, '.smude-download-'));
  try {
    console.log(`Preparing ONNX Runtime 1.31.0 for ${target}`);
    const archive = join(temporary, 'archive.lzma2');
    run('curl', ['--fail', '--location', '--retry', '3', '--max-time', '180', '--output', archive,
      `https://cdn.pyke.io/0/pyke:ort-rs/ms@1.31.0/${target}+${provider}.tar.lzma2`]);
    if (createHash('sha256').update(readFileSync(archive)).digest('hex') !== hash) throw new Error('ONNX Runtime checksum mismatch');
    const tar = join(temporary, 'archive.tar'); const fd = openSync(tar, 'w');
    try { run('xz', ['--decompress', '--stdout', '--format=raw', '--lzma2=dict=64MiB', archive], { stdio: ['ignore', fd, 'inherit'] }); }
    finally { closeSync(fd); }
    const unpacked = join(temporary, 'unpacked'); mkdirSync(unpacked);
    run('tar', ['-xf', tar, '-C', unpacked]);
    if (!existsSync(join(unpacked, 'libonnxruntime.a'))) throw new Error('Missing native runtime library');
    if (!existsSync(destination)) renameSync(unpacked, destination);
  } finally { rmSync(temporary, { recursive: true, force: true }); }
}
