package com.micklab.imagemodel

import android.content.Context
import android.net.Uri
import java.io.BufferedInputStream
import java.io.File
import java.io.IOException
import java.util.zip.ZipInputStream

data class ModelImportResult(
    val extractedFiles: Int,
    val destinationDir: File
)

object SdxlModelArchiveImporter {

    private const val TAG = "SdxlModelArchiveImporter"
    private const val BUFFER_SIZE = 64 * 1024

    fun importFromZip(context: Context, archiveUri: Uri): ModelImportResult {
        SdxlModelLoader.initialize(context)

        val targetDir = SdxlModelLoader.getModelBaseDir()
        val parentDir = targetDir.parentFile
            ?: throw IOException("Model directory parent is unavailable: ${targetDir.absolutePath}")
        if (!parentDir.exists() && !parentDir.mkdirs()) {
            throw IOException("Failed to create model storage root: ${parentDir.absolutePath}")
        }

        val pathPrefix = determineArchivePrefix(context, archiveUri)
        val stagingDir = File(parentDir, "${targetDir.name}_staging_${System.currentTimeMillis()}")
        if (stagingDir.exists() || !stagingDir.mkdirs()) {
            throw IOException("Failed to create staging directory: ${stagingDir.absolutePath}")
        }

        var backupDir: File? = null
        try {
            val extractedFiles = extractArchive(context, archiveUri, pathPrefix, stagingDir)
            if (extractedFiles == 0) {
                throw IllegalStateException("ZIP に展開できるファイルがありません。")
            }

            val missingRequiredFiles = SdxlModelLoader.getRequiredRuntimeFiles()
                .filterNot { File(stagingDir, it).isFile }
            if (missingRequiredFiles.isNotEmpty()) {
                throw IllegalStateException(
                    "ZIP に必要なファイルが含まれていません: ${missingRequiredFiles.joinToString(", ")}"
                )
            }

            backupDir = backupExistingDirectory(targetDir)
            moveDirectory(stagingDir, targetDir)
            backupDir?.deleteRecursively()

            AppLogStore.i(
                TAG,
                "Imported model archive to ${targetDir.absolutePath} (prefix=${pathPrefix.joinToString("/")}, files=$extractedFiles)"
            )
            return ModelImportResult(extractedFiles = extractedFiles, destinationDir = targetDir)
        } catch (e: Exception) {
            AppLogStore.e(TAG, "Failed to import model ZIP from $archiveUri", e)
            restoreBackup(targetDir, backupDir)
            throw e
        } finally {
            if (stagingDir.exists()) {
                stagingDir.deleteRecursively()
            }
            if (backupDir?.exists() == true && targetDir.exists()) {
                backupDir.deleteRecursively()
            }
        }
    }

    private fun determineArchivePrefix(context: Context, archiveUri: Uri): List<String> {
        val entrySegments = mutableListOf<List<String>>()
        openZipInputStream(context, archiveUri).use { zipInput ->
            while (true) {
                val entry = zipInput.nextEntry ?: break
                if (!entry.isDirectory) {
                    entrySegments += normalizeEntrySegments(entry.name)
                }
                zipInput.closeEntry()
            }
        }

        if (entrySegments.isEmpty()) {
            return emptyList()
        }

        val requiredPaths = SdxlModelLoader.getRequiredRuntimeFiles()
            .map { normalizeEntrySegments(it) }
        for (requiredPath in requiredPaths) {
            val match = entrySegments.firstOrNull { it.size >= requiredPath.size && it.takeLast(requiredPath.size) == requiredPath }
            if (match != null) {
                return match.dropLast(requiredPath.size)
            }
        }

        val commonPrefix = entrySegments.drop(1).fold(entrySegments.first()) { prefix, segments ->
            commonPrefix(prefix, segments)
        }
        return if (commonPrefix.size == 1) commonPrefix else emptyList()
    }

    private fun extractArchive(
        context: Context,
        archiveUri: Uri,
        pathPrefix: List<String>,
        stagingDir: File
    ): Int {
        var extractedFiles = 0

        openZipInputStream(context, archiveUri).use { zipInput ->
            while (true) {
                val entry = zipInput.nextEntry ?: break
                val entrySegments = normalizeEntrySegments(entry.name)
                val relativeSegments = stripPrefix(entrySegments, pathPrefix)

                if (relativeSegments != null && relativeSegments.isNotEmpty()) {
                    val destination = resolveDestination(stagingDir, relativeSegments)
                    if (entry.isDirectory) {
                        if (!destination.exists() && !destination.mkdirs()) {
                            throw IOException("Failed to create directory: ${destination.absolutePath}")
                        }
                    } else {
                        val parent = destination.parentFile
                        if (parent != null && !parent.exists() && !parent.mkdirs()) {
                            throw IOException("Failed to create directory: ${parent.absolutePath}")
                        }
                        destination.outputStream().buffered().use { output ->
                            zipInput.copyTo(output, BUFFER_SIZE)
                        }
                        extractedFiles += 1
                    }
                }

                zipInput.closeEntry()
            }
        }

        return extractedFiles
    }

    private fun backupExistingDirectory(targetDir: File): File? {
        if (!targetDir.exists()) {
            return null
        }

        val backupDir = File(
            targetDir.parentFile,
            "${targetDir.name}_backup_${System.currentTimeMillis()}"
        )
        moveDirectory(targetDir, backupDir)
        return backupDir
    }

    private fun restoreBackup(targetDir: File, backupDir: File?) {
        if (backupDir == null || !backupDir.exists()) {
            return
        }

        if (targetDir.exists()) {
            targetDir.deleteRecursively()
        }
        moveDirectory(backupDir, targetDir)
    }

    private fun moveDirectory(sourceDir: File, targetDir: File) {
        if (!sourceDir.exists()) {
            throw IOException("Source directory does not exist: ${sourceDir.absolutePath}")
        }

        val targetParent = targetDir.parentFile
        if (targetParent != null && !targetParent.exists() && !targetParent.mkdirs()) {
            throw IOException("Failed to create directory: ${targetParent.absolutePath}")
        }

        if (targetDir.exists() && !targetDir.deleteRecursively()) {
            throw IOException("Failed to clear directory: ${targetDir.absolutePath}")
        }

        if (sourceDir.renameTo(targetDir)) {
            return
        }

        if (!sourceDir.copyRecursively(targetDir, overwrite = true)) {
            throw IOException("Failed to copy directory to ${targetDir.absolutePath}")
        }
        if (!sourceDir.deleteRecursively()) {
            throw IOException("Failed to delete temporary directory: ${sourceDir.absolutePath}")
        }
    }

    private fun openZipInputStream(context: Context, archiveUri: Uri): ZipInputStream {
        val inputStream = context.contentResolver.openInputStream(archiveUri)
            ?: throw IOException("ZIP ファイルを開けませんでした。")
        return ZipInputStream(BufferedInputStream(inputStream))
    }

    private fun normalizeEntrySegments(entryName: String): List<String> {
        val normalized = entryName.replace('\\', '/')
        return normalized.split('/')
            .filter { it.isNotBlank() && it != "." }
            .also { segments ->
                if (segments.any { it == ".." }) {
                    throw IOException("ZIP 内に不正なパスが含まれています: $entryName")
                }
            }
    }

    private fun stripPrefix(segments: List<String>, prefix: List<String>): List<String>? {
        if (prefix.isEmpty()) {
            return segments
        }
        if (segments.size < prefix.size || segments.subList(0, prefix.size) != prefix) {
            return null
        }
        return segments.drop(prefix.size)
    }

    private fun resolveDestination(baseDir: File, relativeSegments: List<String>): File {
        val destination = relativeSegments.fold(baseDir) { current, segment ->
            File(current, segment)
        }
        val canonicalBase = baseDir.canonicalFile
        val canonicalDestination = destination.canonicalFile
        val allowedPrefix = canonicalBase.path + File.separator
        if (canonicalDestination.path != canonicalBase.path &&
            !canonicalDestination.path.startsWith(allowedPrefix)
        ) {
            throw IOException("ZIP 内に不正な展開先が含まれています: ${destination.path}")
        }
        return canonicalDestination
    }

    private fun commonPrefix(left: List<String>, right: List<String>): List<String> {
        val limit = minOf(left.size, right.size)
        val prefix = ArrayList<String>(limit)
        for (index in 0 until limit) {
            if (left[index] != right[index]) {
                break
            }
            prefix += left[index]
        }
        return prefix
    }
}
