package com.hsu.baby_cc

import android.content.res.AssetManager

class NativeAnalyzer {
    companion object {
        init {
            System.loadLibrary("baby_cc")
        }
    }

    /**
     * Initializes the ONNX model from assets.
     */
    external fun initModel(assetManager: AssetManager, modelPath: String): Boolean

    /**
     * Sets the FIR kernel for manifold alignment.
     * @param firKernel Array of float containing the 65-tap FIR filter kernel.
     */
    external fun setFIRKernel(firKernel: FloatArray)

    /**
     * Processes audio data and returns wheeze and cough probabilities.
     * @param audioData Raw audio data in [-1.0, 1.0].
     * @return FloatArray where index 0 is wheeze probability, and index 1 is cough probability.
     */
    external fun processAudio(audioData: FloatArray): FloatArray
    
    /**
     * Optional: Process audio and return Mel-spectrogram along with probabilities.
     * Helpful for visualization.
     * @param audioData Raw audio data in [-1.0, 1.0].
     * @return FloatArray containing wheeze prob, cough prob, followed by flattened Mel-spectrogram.
     */
    external fun processAudioWithMel(audioData: FloatArray): FloatArray

    /**
     * Cleans up resources.
     */
    external fun release()
}
