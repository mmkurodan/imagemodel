package com.micklab.imagemodel

import android.content.Context
import java.io.File

/**
 * Resolves SDXL model paths inside app-managed storage.
 */
object SdxlModelLoader {

    private const val TAG = "SdxlModelLoader"
    private const val MODEL_DIRECTORY_NAME = "sdxl"
    private const val MODEL_INDEX_FILE = "model_index.json"

    /**
     * Files that are actually required by the current runtime path.
     * The app only opens the UNet model directly; tokenizer/scheduler assets are not consumed.
     */
    private val REQUIRED_RUNTIME_FILES = listOf("unet/model.onnx")

    @Volatile
    private var appContext: Context? = null

    @Volatile
    private var loggedBasePath: String? = null

    fun initialize(context: Context) {
        val applicationContext = context.applicationContext
        appContext = applicationContext

        val basePath = resolveModelBaseDir(applicationContext).absolutePath
        if (loggedBasePath != basePath) {
            loggedBasePath = basePath
            AppLogStore.i(TAG, "Using SDXL model directory: $basePath")
        }
    }

    fun getModelBaseDir(): File = resolveModelBaseDir(requireContext())

    fun getModelBasePath(): String = getModelBaseDir().absolutePath

    fun getComponentPath(component: String): String {
        return File(getModelBaseDir(), component).absolutePath
    }

    fun getOnnxModelPath(component: String): String {
        return File(getComponentPath(component), "model.onnx").absolutePath
    }

    fun getRequiredRuntimeFiles(): List<String> = REQUIRED_RUNTIME_FILES.toList()

    fun getRequiredRuntimeComponents(): List<String> {
        return REQUIRED_RUNTIME_FILES.map { it.substringBefore('/') }.distinct()
    }

    fun getModelIndexPath(): String {
        return File(getModelBaseDir(), MODEL_INDEX_FILE).absolutePath
    }

    fun isModelAvailable(): Boolean {
        val baseDir = getModelBaseDir()
        if (!baseDir.exists() || !baseDir.isDirectory) {
            AppLogStore.w(TAG, "Base model directory is missing: ${baseDir.absolutePath}")
            return false
        }

        for (relativePath in REQUIRED_RUNTIME_FILES) {
            val requiredFile = File(baseDir, relativePath)
            if (!requiredFile.exists() || !requiredFile.isFile) {
                AppLogStore.w(TAG, "Missing required runtime file: ${requiredFile.absolutePath}")
                return false
            }
        }

        AppLogStore.i(TAG, "All required SDXL runtime assets are available")
        return true
    }

    fun getMissingComponents(): List<String> {
        val missing = mutableListOf<String>()
        val baseDir = getModelBaseDir()

        if (!baseDir.exists() || !baseDir.isDirectory) {
            missing.add("Base directory: ${baseDir.absolutePath}")
            return missing
        }

        for (relativePath in REQUIRED_RUNTIME_FILES) {
            val requiredFile = File(baseDir, relativePath)
            if (!requiredFile.exists() || !requiredFile.isFile) {
                missing.add(relativePath)
            }
        }

        return missing
    }

    private fun requireContext(): Context {
        return appContext ?: throw IllegalStateException("SdxlModelLoader is not initialized")
    }

    private fun resolveModelBaseDir(context: Context): File {
        val storageRoot = context.getExternalFilesDir(null) ?: context.filesDir
        return File(storageRoot, MODEL_DIRECTORY_NAME)
    }
}
