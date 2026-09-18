export class WebGPUUnavailableError extends Error {
	override name = "WebGPUUnavailableError";
}

export function hasWebGPU(): boolean {
	return (
		typeof navigator !== "undefined" && "gpu" in navigator && !!navigator.gpu
	);
}

let devicePromise: Promise<GPUDevice> | undefined;

/**
 * Acquires (and caches) a GPUDevice, opting into shader-f16 when available.
 * Rejects with WebGPUUnavailableError when there is no adapter (non-secure context,
 * Linux Chrome without flags, Firefox Android, Node without a polyfill).
 */
export function getDevice(): Promise<GPUDevice> {
	devicePromise ??= (async () => {
		if (!hasWebGPU()) throw new WebGPUUnavailableError("WebGPU unavailable");
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) throw new WebGPUUnavailableError("No WebGPU adapter");
		const requiredFeatures: GPUFeatureName[] = adapter.features.has(
			"shader-f16",
		)
			? ["shader-f16"]
			: [];
		const device = await adapter.requestDevice({ requiredFeatures });
		device.lost.then(() => {
			devicePromise = undefined;
		});
		return device;
	})();
	return devicePromise;
}
