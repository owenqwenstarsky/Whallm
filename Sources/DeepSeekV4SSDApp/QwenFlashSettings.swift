import Foundation

/// App choices are independent of the effective runtime configuration.
/// Inactive values survive switching backends or temporarily selecting waves.
extension ModelAdvancedSettings {
  var qwenFlashWavesEnabled: Bool { (qwenExpertWaveSlots ?? 0) > 0 }
  var effectiveQwenPackedGDNPrefill: Bool { qwenPackedGDNPrefill == true && mtpEnabled != true }
  /// Sorting only applies to grouped prefill, which waves and a disabled
  /// Prefill acceleration turn off. The saved choice is kept either way.
  var qwenSortedExpertPrefillAvailable: Bool { qwenGroupedExperts == true && !qwenFlashWavesEnabled }
  var effectiveQwenSortedExpertPrefill: Bool {
    qwenSortedExpertPrefill == true && qwenSortedExpertPrefillAvailable
  }
  var qwenWholeLayerExperimentsActive: Bool {
    layerMajorPrefill && batchedExpertPrefill == true && !qwenFlashWavesEnabled
  }

  var effectiveQwenNgramCacheBytes: Int {
    let mib = qwenNgramCacheMiB ?? 0
    guard qwenNgramIO == "pread", (0...512).contains(mib) else { return 0 }
    return mib * 1_048_576
  }

  func validateQwenFlashSettings() throws {
    guard (1...32).contains(qwenPrefillReadExperts ?? 1) else {
      throw ConfigurationError(L10n.string(QwenFlashCopy.invalidReadBatch))
    }
    guard (0...128).contains(qwenPrefillSeedExperts ?? 0) else {
      throw ConfigurationError(L10n.string(QwenFlashCopy.invalidSeeds))
    }
    guard (1...128).contains(qwenQSAQueryChunk ?? 16) else {
      throw ConfigurationError(L10n.string(QwenFlashCopy.invalidQueryChunk))
    }
    guard (0...512).contains(qwenExpertWaveSlots ?? 0) else {
      throw ConfigurationError(L10n.string(QwenFlashCopy.invalidWaves))
    }
    guard ["mmap", "pread"].contains(qwenNgramIO ?? "mmap") else {
      throw ConfigurationError(L10n.string(QwenFlashCopy.invalidBackend))
    }
    guard (0...512).contains(qwenNgramCacheMiB ?? 0) else {
      throw ConfigurationError(L10n.string(QwenFlashCopy.invalidCache))
    }
    guard (0...100).contains(qwen2BitExpertPercent ?? 50) else {
      throw ConfigurationError(L10n.string(QwenFlashCopy.invalidTwoBitShare))
    }
  }
}

/// English localization keys for a Qwen setting, validation and dependency hints.
enum QwenFlashCopy {
  static let wavesConflict = "Expert waves temporarily disable whole-layer batching, next-layer prefetch, and grouped prefill. Their saved choices are restored when waves are disabled."
  static let invalidWaves = "Experts per wave must be an integer from 0 to 512."
  static let invalidBackend = "Choose mmap or pread for the N-gram read backend."
  static let invalidCache = "N-gram row cache must be an integer from 0 to 512 MiB."
  static let invalidQueryChunk = "QSA queries per chunk must be an integer from 1 to 128."
  static let invalidReadBatch = "Experts per prefill read must be an integer from 1 to 32."
  static let invalidSeeds = "Warm decode experts per layer must be an integer from 0 to 128."
  static let commonTensors = "Use 8-bit resident weights"
  static let commonTensorsHint = "Stores the model weights that stay in memory, except experts, in 8 bits instead of BF16. Uses less memory and speeds up generation, especially with MTP. Generated text may change."
  static let twoBitExperts = "Use 2-bit expert weights"
  static let twoBitExpertsHint = "Stores this percentage of routed experts in 2 bits after they are read into memory. The other experts stay in their original format. Installed expert files stay unchanged. A larger share uses less memory. Generated text may change, and output can be slower."
  static let twoBitShare = "Expert weights in 2 bits"
  static let twoBitShareHint = "Percentage of routed experts stored in 2 bits. The default is 50%. 0% keeps every expert in its original format. 100% stores every routed expert in 2 bits."
  static let invalidTwoBitShare = "2-bit expert share must be an integer from 0 to 100."
  static let allKeys = [wavesConflict, invalidWaves, invalidBackend, invalidCache,
    invalidQueryChunk, invalidReadBatch, invalidSeeds, commonTensors, commonTensorsHint,
    twoBitExperts, twoBitExpertsHint, twoBitShare, twoBitShareHint, invalidTwoBitShare]
}
