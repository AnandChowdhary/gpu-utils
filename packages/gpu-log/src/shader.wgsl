// gpu-log compute kernels. Generated/edited by hand; keep in sync with model.ts.
// Convention: one workgroup per sequence, one thread per hidden channel.

struct Params { tokens: u32, hidden: u32, labels: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> features: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) id: vec3<u32>) {
  // TODO: sum sparse embeddings for token id.x into state.
}

@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) id: vec3<u32>) {
  // TODO: project state to logits.
}
