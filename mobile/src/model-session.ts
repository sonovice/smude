export type DewarpModel = 'gcs' | 'mesh';
export type ModelRequest = { version: number; path: string; model: DewarpModel };

/** One immutable imported photo, with independent outputs for both models. */
export class ModelSession<Input extends { path: string }, Output> {
  input: Input | null = null;
  model: DewarpModel;
  private version = 0;
  private outputs = new Map<DewarpModel, Output>();

  constructor(model: DewarpModel = 'mesh') { this.model = model; }
  setInput(input: Input) { this.input = input; this.version++; this.outputs.clear(); }
  select(model: DewarpModel) { this.model = model; }
  result(): Output | undefined { return this.outputs.get(this.model); }
  request(): ModelRequest | null {
    return this.input && !this.outputs.has(this.model)
      ? { version: this.version, path: this.input.path, model: this.model } : null;
  }
  accept(request: ModelRequest, output: Output): boolean {
    if (request.version !== this.version || request.path !== this.input?.path) return false;
    this.outputs.set(request.model, output);
    return true;
  }
}
