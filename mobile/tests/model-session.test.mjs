import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';

const source = await readFile(new URL('../src/model-session.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } });
const { ModelSession } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);

test('the same photo can be processed with both models and cached independently', () => {
  const session = new ModelSession('gcs');
  const photo = { path: '/original.png', width: 3000, height: 4000 };
  session.setInput(photo);
  const gcs = session.request();
  session.accept(gcs, 'gcs-output');
  session.select('mesh');
  const mesh = session.request();
  assert.equal(mesh.path, gcs.path);
  session.accept(mesh, 'mesh-output');
  assert.equal(session.result(), 'mesh-output');
  session.select('gcs');
  assert.equal(session.result(), 'gcs-output');
  assert.equal(session.request(), null);
  assert.equal(session.input, photo);
});
test('switching while processing never labels a late result as the other model', () => {
  const session = new ModelSession('gcs'); session.setInput({ path: '/a.png' });
  const inFlight = session.request(); session.select('mesh');
  session.accept(inFlight, 'gcs-output');
  assert.equal(session.result(), undefined);
  assert.equal(session.request().model, 'mesh');
  session.select('gcs'); assert.equal(session.result(), 'gcs-output');
});
test('a new photo clears both caches and rejects old in-flight results', () => {
  const session = new ModelSession('gcs'); session.setInput({ path: '/a.png' });
  const old = session.request(); session.setInput({ path: '/b.png' });
  assert.equal(session.accept(old, 'wrong-photo'), false);
  assert.equal(session.result(), undefined); assert.equal(session.request().path, '/b.png');
});
test('the chosen model is also applied to the next photo', () => {
  const session = new ModelSession('gcs'); session.select('mesh');
  assert.equal(session.request(), null);
  session.setInput({ path: '/camera.png' }); assert.equal(session.request().model, 'mesh');
});

test('new sessions default to mesh while explicit GCS remains selectable', () => {
  assert.equal(new ModelSession().model, 'mesh');
  assert.equal(new ModelSession('gcs').model, 'gcs');
});
