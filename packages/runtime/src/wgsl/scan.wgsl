// Reusable building blocks for gpu-utils shaders. Import as text and concatenate,
// or copy into a package's shader.wgsl.
//
// Associative composition of affine maps h' = a * h + b, used for the parallel
// prefix scan in bidirectional gated-scan taggers (gpu-lexer / gpu-time family).
fn affine_compose(a1: f32, b1: f32, a2: f32, b2: f32) -> vec2<f32> {
  // apply (a1,b1) then (a2,b2): h -> a2*(a1*h + b1) + b2
  return vec2<f32>(a1 * a2, a2 * b1 + b2);
}

fn relu(x: f32) -> f32 { return max(x, 0.0); }
fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
