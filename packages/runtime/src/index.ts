export {
  grid,
  type PackedRows,
  packRows,
  type TaggerParams,
  taggerParams,
  tensorOffsets,
  unpackLogits,
} from "./batch.ts";
export { BIO_FORBID, type BioSpan, bioStartMask, bioToSpans, bioTransitions } from "./bio.ts";
export { argmax, viterbi } from "./decode.ts";
export { getDevice, hasWebGPU, WebGPUUnavailableError } from "./device.ts";
export {
  type BatchOutput,
  CONV_TAGGER_MAX_BLOCKS,
  convTaggerEntries,
  convTaggerPasses,
  convTaggerShader,
  runConvTagger,
  runScanTagger,
  SCAN_TAGGER_MAX_LAYERS,
  scanTaggerEntries,
  scanTaggerPasses,
  scanTaggerShader,
} from "./gpu.ts";
export { hashToken } from "./hash.ts";
export {
  affineScan,
  type BiScanWeights,
  biScan,
  concatRows,
  dense,
  depthwiseConv,
  dilatedResidualBlock,
  meanPool,
  relu,
  sigmoid,
  sparseEmbed,
} from "./layers.ts";
export {
  convTaggerForward,
  convTensorNames,
  scanTaggerForward,
  scanTensorNames,
  type TaggerManifest,
  type TaggerModel,
  type TaggerOutput,
  taggerForward,
} from "./models.ts";
export { createProgram, type Pass, type Program } from "./program.ts";
export {
  CharClass,
  Shape,
  type Token,
  tokenize,
} from "./tokenize.ts";
export type { Backend, FeatureRows } from "./types.ts";
export {
  decodeInt6,
  type ModelManifest,
  type TensorEntry,
  tensor,
} from "./weights.ts";
