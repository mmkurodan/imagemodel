package com.micklab.imagemodel

import android.content.Intent
import android.graphics.Typeface
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.Manifest
import android.content.pm.PackageManager
import android.app.Activity
import android.provider.DocumentsContract
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.Executors

/**
 * Error screen displayed when SDXL model is not found.
 * Guides the user through importing a model ZIP into app-managed storage.
 */
class ModelMissingActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "ModelMissingActivity"
    }

    private val importExecutor = Executors.newSingleThreadExecutor()

    private lateinit var importStatusText: TextView
    private lateinit var importProgressBar: ProgressBar
    private lateinit var importButton: Button
    private lateinit var retryButton: Button
    private lateinit var exitButton: Button

    private val importZipLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri == null) {
            setImportStatus("ZIP の選択をキャンセルしました。", isError = false)
            return@registerForActivityResult
        }
        startImport(uri)
    }

    private val requestReadStoragePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { _ ->
        // Whether granted or not, open the picker — ACTION_OPEN_DOCUMENT handles access itself.
        openZipPickerWithInitialUri()
    }

    private val openDocumentWithInitialUriLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val uri = result.data?.data
            if (uri == null) {
                setImportStatus("ZIP の選択をキャンセルしました。", isError = false)
                return@registerForActivityResult
            }
            startImport(uri)
        } else {
            setImportStatus("ZIP の選択をキャンセルしました。", isError = false)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AppLogStore.initialize(this)
        SdxlModelLoader.initialize(this)
        AppLogStore.i(TAG, "ModelMissingActivity onCreate")
        setupUI()
    }

    private fun setupUI() {
        val scrollView = ScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#1a1a2e"))
        }
        
        val mainLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 64, 48, 64)
        }
        
        // Error icon (using text)
        val errorIcon = TextView(this).apply {
            text = "⚠️"
            textSize = 64f
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 32)
        }
        mainLayout.addView(errorIcon)
        
        // Title
        val title = TextView(this).apply {
            text = "モデルがありません"
            textSize = 28f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 24)
        }
        mainLayout.addView(title)
        
        // Main message
        val message = TextView(this).apply {
            text = "UNetランタイムモデルが見つかりませんでした。\nモデルZIPを選択すると、アプリ専用ストレージへ展開します。"
            textSize = 16f
            setTextColor(Color.LTGRAY)
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 24)
        }
        mainLayout.addView(message)
        
        // Path box
        val pathBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#16213e"))
            setPadding(32, 24, 32, 24)
        }
        
        val pathLabel = TextView(this).apply {
            text = "展開先:"
            textSize = 14f
            setTextColor(Color.GRAY)
        }
        pathBox.addView(pathLabel)
        
        val pathText = TextView(this).apply {
            text = "App external files / sdxl"
            textSize = 18f
            setTextColor(Color.parseColor("#4fc3f7"))
            typeface = Typeface.MONOSPACE
            setPadding(0, 8, 0, 0)
        }
        pathBox.addView(pathText)
        
        val fullPathText = TextView(this).apply {
            text = SdxlModelLoader.getModelBasePath()
            textSize = 12f
            setTextColor(Color.GRAY)
            typeface = Typeface.MONOSPACE
            setPadding(0, 4, 0, 0)
        }
        pathBox.addView(fullPathText)
        
        mainLayout.addView(pathBox)
        
        // Required structure
        val structureTitle = TextView(this).apply {
            text = "\n必要なランタイム構成:"
            textSize = 16f
            setTextColor(Color.WHITE)
            setPadding(0, 32, 0, 16)
        }
        mainLayout.addView(structureTitle)
        
        val requiredFolders = buildList {
            addAll(SdxlModelLoader.getRequiredRuntimeComponents().map { "📁 $it/" })
            addAll(SdxlModelLoader.getRequiredRuntimeFiles().map { "📄 $it" })
        }
        
        val structureBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#16213e"))
            setPadding(32, 24, 32, 24)
        }
        
        for (folder in requiredFolders) {
            val folderText = TextView(this).apply {
                text = folder
                textSize = 14f
                setTextColor(Color.LTGRAY)
                typeface = Typeface.MONOSPACE
                setPadding(0, 4, 0, 4)
            }
            structureBox.addView(folderText)
        }
        mainLayout.addView(structureBox)

        val importNote = TextView(this).apply {
            text = "公開 Downloads 直下のモデルは ONNX Runtime から開けない場合があるため、この画面からZIPを取り込んでください。"
            textSize = 13f
            setTextColor(Color.GRAY)
            setPadding(0, 12, 0, 0)
        }
        mainLayout.addView(importNote)

        val optionalNote = TextView(this).apply {
            text = "現在のビルドでは model_index.json と text_encoder / text_encoder_2 / vae_decoder / tokenizer / tokenizer_2 / scheduler は起動条件ではありません。"
            textSize = 13f
            setTextColor(Color.GRAY)
            setPadding(0, 12, 0, 0)
        }
        mainLayout.addView(optionalNote)

        // Show missing components if any
        val missingComponents = SdxlModelLoader.getMissingComponents()
        if (missingComponents.isNotEmpty()) {
            val missingTitle = TextView(this).apply {
                text = "\n❌ 見つからないコンポーネント:"
                textSize = 16f
                setTextColor(Color.parseColor("#e94560"))
                setPadding(0, 24, 0, 8)
            }
            mainLayout.addView(missingTitle)
            
            val missingBox = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setBackgroundColor(Color.parseColor("#2d1f2f"))
                setPadding(32, 24, 32, 24)
            }
            
            for (component in missingComponents) {
                val componentText = TextView(this).apply {
                    text = "• $component"
                    textSize = 14f
                    setTextColor(Color.parseColor("#ff8a80"))
                    setPadding(0, 4, 0, 4)
                }
                missingBox.addView(componentText)
            }
            mainLayout.addView(missingBox)
        }

        importProgressBar = ProgressBar(this).apply {
            visibility = View.GONE
            val params = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            params.gravity = Gravity.CENTER_HORIZONTAL
            params.setMargins(0, 32, 0, 16)
            layoutParams = params
        }
        mainLayout.addView(importProgressBar)

        importStatusText = TextView(this).apply {
            visibility = View.GONE
            textSize = 14f
            setTextColor(Color.parseColor("#81c784"))
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 16)
        }
        mainLayout.addView(importStatusText)

        importButton = Button(this).apply {
            text = "モデルZIPを選択"
            textSize = 16f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#4fc3f7"))
            setPadding(48, 24, 48, 24)
            val params = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            params.setMargins(0, 48, 0, 16)
            layoutParams = params
            setOnClickListener { onImportClicked() }
        }
        mainLayout.addView(importButton)

        retryButton = Button(this).apply {
            text = "再試行"
            textSize = 16f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#e94560"))
            setPadding(48, 24, 48, 24)
            val params = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            params.setMargins(0, 0, 0, 16)
            layoutParams = params
            setOnClickListener { onRetryClicked() }
        }
        mainLayout.addView(retryButton)

        exitButton = Button(this).apply {
            text = "終了"
            textSize = 14f
            setTextColor(Color.LTGRAY)
            setBackgroundColor(Color.parseColor("#333355"))
            val params = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            layoutParams = params
            setOnClickListener { finish() }
        }
        mainLayout.addView(exitButton)
        
        scrollView.addView(mainLayout)
        setContentView(scrollView)
    }

    private fun openZipPickerWithInitialUri() {
        try {
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                putExtra(Intent.EXTRA_ALLOW_MULTIPLE, false)
                putExtra(Intent.EXTRA_MIME_TYPES, arrayOf(
                    "application/zip",
                    "application/x-zip-compressed",
                    "application/octet-stream"
                ))
                type = "*/*"
                flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
            }

            // Use the Downloads document provider URI so the picker opens directly in Downloads.
            val downloadsUri = Uri.parse("content://com.android.providers.downloads.documents/root/downloads")
            intent.putExtra(DocumentsContract.EXTRA_INITIAL_URI, downloadsUri)

            if (intent.resolveActivity(packageManager) != null) {
                openDocumentWithInitialUriLauncher.launch(intent)
            } else {
                importZipLauncher.launch(arrayOf("application/zip", "application/x-zip-compressed", "application/octet-stream"))
            }
        } catch (e: Exception) {
            AppLogStore.w(TAG, "Failed to open picker with initial URI: ${e.message}")
            importZipLauncher.launch(arrayOf("application/zip", "application/x-zip-compressed", "application/octet-stream"))
        }
    }

    private fun onImportClicked() {
        AppLogStore.i(TAG, "Opening model ZIP picker")
        // ACTION_OPEN_DOCUMENT grants access through the picker itself and does not require
        // READ_EXTERNAL_STORAGE. On Android 10–12 (API 29–32), request the permission as a hint
        // so the system can surface the Downloads folder more reliably.
        if (android.os.Build.VERSION.SDK_INT <= android.os.Build.VERSION_CODES.S_V2 &&
            checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
        ) {
            requestReadStoragePermissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
        } else {
            openZipPickerWithInitialUri()
        }
    }

    private fun startImport(uri: Uri) {
        AppLogStore.i(TAG, "Selected model ZIP: $uri")
        try {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        } catch (_: SecurityException) {
            // Some providers do not grant persistable access; immediate extraction still works.
        }

        setBusy(isBusy = true)
        setImportStatus("モデルZIPを展開中...", isError = false)

        importExecutor.execute {
            try {
                val result = SdxlModelArchiveImporter.importFromZip(this, uri)
                runOnUiThread {
                    if (isFinishing || isDestroyed) return@runOnUiThread
                    setBusy(isBusy = false)
                    val message = "モデルを展開しました (${result.extractedFiles} files)。生成画面へ戻ります。"
                    setImportStatus(message, isError = false)
                    AppLogStore.i(TAG, message)
                    setResult(RESULT_OK)
                    finish()
                }
            } catch (e: Exception) {
                val message = "ZIP の展開に失敗しました: ${e.message}"
                AppLogStore.e(TAG, message, e)
                runOnUiThread {
                    if (isFinishing || isDestroyed) return@runOnUiThread
                    setBusy(isBusy = false)
                    setImportStatus(message, isError = true)
                }
            }
        }
    }

    private fun onRetryClicked() {
        if (SdxlModelLoader.isModelAvailable()) {
            AppLogStore.i(TAG, "Retry succeeded; model is now available")
            setResult(RESULT_OK)
            finish()
        } else {
            val message = "必要なファイルがまだ見つかりません。ZIP を取り込むか、展開先を確認してください。"
            AppLogStore.w(TAG, "Retry failed; missing components: ${SdxlModelLoader.getMissingComponents()}")
            setImportStatus(message, isError = true)
        }
    }

    private fun setBusy(isBusy: Boolean) {
        importButton.isEnabled = !isBusy
        retryButton.isEnabled = !isBusy
        exitButton.isEnabled = !isBusy
        importProgressBar.visibility = if (isBusy) View.VISIBLE else View.GONE
    }

    private fun setImportStatus(message: String, isError: Boolean) {
        importStatusText.visibility = if (message.isBlank()) View.GONE else View.VISIBLE
        importStatusText.text = message
        importStatusText.setTextColor(
            if (isError) Color.parseColor("#ff8a80") else Color.parseColor("#81c784")
        )
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        AppLogStore.i(TAG, "Back pressed on model-missing screen")
        finish()
    }

    override fun onDestroy() {
        importExecutor.shutdownNow()
        super.onDestroy()
    }
}
