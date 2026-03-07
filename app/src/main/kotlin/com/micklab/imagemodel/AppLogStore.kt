package com.micklab.imagemodel

import android.content.Context
import android.os.Process
import android.util.Log
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.system.exitProcess

/**
 * Persists app logs and logcat snapshots to app-specific external storage.
 */
object AppLogStore {

    private const val ROOT_TAG = "Imagemodel"
    private const val SOURCE_TAG = "AppLogStore"
    private const val EXTERNAL_LOG_DIR = "logs"

    private val fileLock = Any()

    @Volatile
    private var initialized = false

    private var logDirectory: File? = null
    private var sessionLogFile: File? = null

    fun initialize(context: Context) {
        if (initialized) return
        synchronized(fileLock) {
            if (initialized) return

            val externalDir = context.getExternalFilesDir(EXTERNAL_LOG_DIR)
            val targetDir = externalDir ?: File(context.filesDir, EXTERNAL_LOG_DIR)
            if (!targetDir.exists()) {
                targetDir.mkdirs()
            }

            logDirectory = targetDir
            sessionLogFile = File(targetDir, "app_session_${fileTimestamp()}.log")
            initialized = true

            write(Log.INFO, SOURCE_TAG, "Log directory: ${targetDir.absolutePath}")
            installUncaughtExceptionHandler()
            write(Log.INFO, SOURCE_TAG, "App log capture initialized")
        }
    }

    fun d(tag: String, message: String) = write(Log.DEBUG, tag, message)

    fun i(tag: String, message: String) = write(Log.INFO, tag, message)

    fun w(tag: String, message: String) = write(Log.WARN, tag, message)

    fun e(tag: String, message: String, throwable: Throwable? = null) = write(Log.ERROR, tag, message, throwable)



    private fun installUncaughtExceptionHandler() {
        val previousHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            write(Log.ERROR, "UncaughtException", "Fatal crash on thread=${thread.name}", throwable)
            if (previousHandler != null) {
                previousHandler.uncaughtException(thread, throwable)
            } else {
                Process.killProcess(Process.myPid())
                exitProcess(10)
            }
        }
    }

    private fun write(priority: Int, tag: String, message: String, throwable: Throwable? = null) {
        val logMessage = "[$tag] $message"
        Log.println(priority, ROOT_TAG, logMessage)

        val line = buildString {
            append(logTimestamp())
            append(" ")
            append(level(priority))
            append(" [")
            append(tag)
            append("] ")
            append(message)
            if (throwable != null) {
                append('\n')
                append(Log.getStackTraceString(throwable))
            }
            append('\n')
        }

        synchronized(fileLock) {
            sessionLogFile?.appendText(line)
        }
    }

    private fun sanitize(value: String): String {
        return value.replace(Regex("[^a-zA-Z0-9._-]"), "_")
    }

    private fun fileTimestamp(): String {
        return SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
    }

    private fun logTimestamp(): String {
        return SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date())
    }

    private fun level(priority: Int): String = when (priority) {
        Log.DEBUG -> "D"
        Log.INFO -> "I"
        Log.WARN -> "W"
        Log.ERROR -> "E"
        else -> priority.toString()
    }
}
