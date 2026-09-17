import { getDevice } from "./device.ts";

export interface Pass {
  entry: string;
  workgroups: [number, number?, number?];
}

export interface Program {
  device: GPUDevice;
  /** Uploads (or reuses) a storage buffer for the given data. */
  buffer(label: string, data: ArrayBufferView, usage?: GPUBufferUsageFlags): GPUBuffer;
  /** Runs passes in one command buffer and reads back `readback` as Float32Array. */
  run(bindings: GPUBuffer[], passes: Pass[], readback: GPUBuffer): Promise<Float32Array>;
  dispose(): void;
}

/**
 * Compiles one WGSL module into a pipeline per entry point sharing a single
 * auto-derived bind group layout, and manages named buffers. Every gpu-utils
 * model builds on this: upload weights once, then per call upload features,
 * dispatch, and read back logits.
 */
export async function createProgram(shader: string, entries: string[]): Promise<Program> {
  const device = await getDevice();
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: shader });
  const pipelines = new Map<string, GPUComputePipeline>();
  for (const entryPoint of entries) {
    pipelines.set(
      entryPoint,
      device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } }),
    );
  }
  const err = await device.popErrorScope();
  if (err) throw new Error(`WGSL compile failed: ${err.message}`);

  const buffers = new Map<string, GPUBuffer>();
  const layout = pipelines.get(entries[0]!)!.getBindGroupLayout(0);

  return {
    device,
    buffer(label, data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
      const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
      let buf = buffers.get(label);
      if (!buf || buf.size < size) {
        buf?.destroy();
        buf = device.createBuffer({ label, size, usage });
        buffers.set(label, buf);
      }
      device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
      return buf;
    },
    async run(bindings, passes, readback) {
      const bindGroup = device.createBindGroup({
        layout,
        entries: bindings.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const staging = device.createBuffer({
        size: readback.size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      for (const p of passes) {
        pass.setPipeline(pipelines.get(p.entry)!);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(p.workgroups[0], p.workgroups[1] ?? 1, p.workgroups[2] ?? 1);
      }
      pass.end();
      encoder.copyBufferToBuffer(readback, 0, staging, 0, readback.size);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      return out;
    },
    dispose() {
      for (const b of buffers.values()) b.destroy();
      buffers.clear();
    },
  };
}
