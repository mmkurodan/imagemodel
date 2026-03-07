package com.micklab.imagemodel

import ai.onnxruntime.OrtException
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.atomic.AtomicBoolean

class GenerationActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "GenerationActivity"
        private val SIZE_OPTIONS = arrayOf("512", "768", "1024")
    }

    private val runner = SdxlGenerationRunner()
    private val generationExecutor = Executors.newSingleThreadExecutor()
    private val cancelRequested = AtomicBoolean(false)

    private var generationTask: Future<*>? = null
    private var selectedSteps: Int = 20
    private var selectedGuidance: Float = 7.5f

    private lateinit var promptInput: EditText
    private lateinit var negativePromptInput: EditText
    private lateinit var widthSpinner: Spinner
    private lateinit var heightSpinner: Spinner
    private lateinit var stepsValue: TextView
    private lateinit var guidanceValue: TextView
    private lateinit var stepsSeekBar: SeekBar
    private lateinit var guidanceSeekBar: SeekBar
    private lateinit var seedInput: EditText
    private lateinit var generateButton: Button
    private lateinit var cancelButton: Button
    private lateinit var progressText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var errorText: TextView
    private lateinit var previewImage: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AppLogStore.initialize(this)
        SdxlModelLoader.initialize(this)
        AppLogStore.i(TAG, "GenerationActivity onCreate")
        setupUi()
    }

    private fun setupUi() {
        val scrollView = ScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#1a1a2e"))
        }

        val mainLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 48, 48, 48)
        }

        val title = TextView(this).apply {
            text = "SDXL 画像生成"
            textSize = 24f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(0, 0, 0, 24)
        }
        mainLayout.addView(title)

        promptInput = EditText(this).apply {
            hint = "プロンプトを入力 (例: cinematic sunset over tokyo skyline)"
            setHintTextColor(Color.GRAY)
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#16213e"))
            setPadding(24, 24, 24, 24)
            minLines = 3
            maxLines = 6
        }
        addLabeledField(mainLayout, "Prompt", promptInput)

        negativePromptInput = EditText(this).apply {
            hint = "ネガティブプロンプト (任意)"
            setHintTextColor(Color.GRAY)
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#16213e"))
            setPadding(24, 24, 24, 24)
            minLines = 2
            maxLines = 4
        }
        addLabeledField(mainLayout, "Negative Prompt", negativePromptInput)

        val optionBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#16213e"))
            setPadding(24, 24, 24, 24)
        }
        mainLayout.addView(optionBox)

        val sizeRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        widthSpinner = createSizeSpinner("1024")
        heightSpinner = createSizeSpinner("1024")
        sizeRow.addView(createOptionGroup("Width", widthSpinner))
        sizeRow.addView(createOptionGroup("Height", heightSpinner))
        optionBox.addView(sizeRow)

        stepsValue = TextView(this).apply {
            setTextColor(Color.LTGRAY)
            textSize = 14f
        }
        stepsSeekBar = SeekBar(this).apply {
            max = 45 // 5..50
            progress = 15 // default 20
        }
        optionBox.addView(createSeekBarGroup("Steps", stepsValue, stepsSeekBar))

        guidanceValue = TextView(this).apply {
            setTextColor(Color.LTGRAY)
            textSize = 14f
        }
        guidanceSeekBar = SeekBar(this).apply {
            max = 190 // 1.0..20.0 => 10..200
            progress = 65 // default 75 (7.5)
        }
        optionBox.addView(createSeekBarGroup("Guidance Scale", guidanceValue, guidanceSeekBar))

        seedInput = EditText(this).apply {
            setTextColor(Color.WHITE)
            setHintTextColor(Color.GRAY)
            setBackgroundColor(Color.parseColor("#0f3460"))
            setPadding(24, 16, 24, 16)
            hint = "Seed (空欄で自動)"
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_SIGNED
        }
        optionBox.addView(createOptionField("Seed", seedInput))

        val buttonRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, 32, 0, 16)
        }

        generateButton = Button(this).apply {
            text = "Generate"
            textSize = 16f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#4caf50"))
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            setOnClickListener { startGeneration() }
        }
        buttonRow.addView(generateButton)

        cancelButton = Button(this).apply {
            text = "Cancel"
            textSize = 16f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#e94560"))
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginStart = 16
            }
            isEnabled = false
            setOnClickListener { cancelGeneration() }
        }
        buttonRow.addView(cancelButton)
        mainLayout.addView(buttonRow)

        progressText = TextView(this).apply {
            text = "待機中"
            textSize = 14f
            setTextColor(Color.LTGRAY)
        }
        mainLayout.addView(progressText)

        progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = 0
            visibility = View.GONE
            setPadding(0, 8, 0, 16)
        }
        mainLayout.addView(progressBar)

        errorText = TextView(this).apply {
            textSize = 14f
            setTextColor(Color.parseColor("#ff8a80"))
            setPadding(0, 0, 0, 16)
        }
        mainLayout.addView(errorText)

        val previewTitle = TextView(this).apply {
            text = "Preview"
            textSize = 18f
            setTextColor(Color.WHITE)
            setPadding(0, 8, 0, 12)
        }
        mainLayout.addView(previewTitle)

        previewImage = ImageView(this).apply {
            setBackgroundColor(Color.parseColor("#0f3460"))
            adjustViewBounds = true
            minimumHeight = 640
            scaleType = ImageView.ScaleType.FIT_CENTER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        mainLayout.addView(previewImage)

        scrollView.addView(mainLayout)
        setContentView(scrollView)

        bindControls()
    }

    private fun bindControls() {
        val spinnerListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                // no-op (values are read at generation time)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                // no-op
            }
        }

        widthSpinner.onItemSelectedListener = spinnerListener
        heightSpinner.onItemSelectedListener = spinnerListener

        stepsSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                selectedSteps = progress + 5
                stepsValue.text = selectedSteps.toString()
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        guidanceSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                selectedGuidance = (progress + 10) / 10f
                guidanceValue.text = String.format("%.1f", selectedGuidance)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        selectedSteps = stepsSeekBar.progress + 5
        selectedGuidance = (guidanceSeekBar.progress + 10) / 10f
        stepsValue.text = selectedSteps.toString()
        guidanceValue.text = String.format("%.1f", selectedGuidance)
    }

    private fun startGeneration() {
        val prompt = promptInput.text.toString().trim()
        if (prompt.isEmpty()) {
            errorText.setTextColor(Color.parseColor("#ff8a80"))
            errorText.text = "プロンプトを入力してください。"
            return
        }

        if (!SdxlModelLoader.isModelAvailable()) {
            navigateToModelMissing()
            return
        }

        val width = widthSpinner.selectedItem.toString().toInt()
        val height = heightSpinner.selectedItem.toString().toInt()
        val seed = parseSeed()

        val request = GenerationRequest(
            prompt = prompt,
            negativePrompt = negativePromptInput.text.toString().trim(),
            width = width,
            height = height,
            steps = selectedSteps,
            guidanceScale = selectedGuidance,
            seed = seed
        )

        cancelRequested.set(false)
        setGeneratingState(true)
        errorText.text = ""
        AppLogStore.i(TAG, "Generation requested: $request")

        generationTask = generationExecutor.submit {
            try {
                val result = runner.generate(
                    request = request,
                    onProgress = { current, total, message ->
                        runOnUiThread { updateProgress(current, total, message) }
                    },
                    isCancelled = { cancelRequested.get() }
                )
                runOnUiThread {
                    previewImage.setImageBitmap(result.bitmap)
                    progressText.text = "完了"
                    if (result.warning != null) {
                        errorText.setTextColor(Color.parseColor("#ffcc80"))
                        errorText.text = result.warning
                    } else {
                        errorText.text = ""
                    }
                    setGeneratingState(false)
                }
            } catch (e: CancellationException) {
                runOnUiThread {
                    progressText.text = "キャンセルしました"
                    setGeneratingState(false)
                    AppLogStore.i(TAG, "Generation cancelled")
                }
            } catch (e: OrtException) {
                showGenerationError("ONNX推論エラー: ${e.message}", e)
            } catch (e: IllegalStateException) {
                showGenerationError("生成状態エラー: ${e.message}", e)
            } catch (e: IllegalArgumentException) {
                showGenerationError("入力エラー: ${e.message}", e)
            }
        }
    }

    private fun cancelGeneration() {
        cancelRequested.set(true)
        generationTask?.cancel(true)
        progressText.text = "キャンセル中..."
        AppLogStore.i(TAG, "Cancellation requested")
    }

    private fun showGenerationError(message: String, throwable: Throwable) {
        AppLogStore.e(TAG, message, throwable)
        runOnUiThread {
            errorText.setTextColor(Color.parseColor("#ff8a80"))
            errorText.text = message
            progressText.text = "エラー"
            setGeneratingState(false)
            if (!SdxlModelLoader.isModelAvailable()) {
                navigateToModelMissing()
            }
        }
    }

    private fun updateProgress(current: Int, total: Int, message: String) {
        progressBar.visibility = View.VISIBLE
        progressBar.max = total
        progressBar.progress = current
        progressText.text = "$message ($current/$total)"
    }

    private fun setGeneratingState(isGenerating: Boolean) {
        generateButton.isEnabled = !isGenerating
        cancelButton.isEnabled = isGenerating
        promptInput.isEnabled = !isGenerating
        negativePromptInput.isEnabled = !isGenerating
        widthSpinner.isEnabled = !isGenerating
        heightSpinner.isEnabled = !isGenerating
        stepsSeekBar.isEnabled = !isGenerating
        guidanceSeekBar.isEnabled = !isGenerating
        seedInput.isEnabled = !isGenerating
        if (!isGenerating) {
            progressBar.visibility = View.GONE
        }
    }

    private fun parseSeed(): Long {
        val raw = seedInput.text.toString().trim()
        val parsed = raw.toLongOrNull() ?: System.currentTimeMillis()
        seedInput.setText(parsed.toString())
        return parsed
    }

    private fun navigateToModelMissing() {
        AppLogStore.w(TAG, "Model became unavailable; navigating to ModelMissingActivity")
        val intent = Intent(this, ModelMissingActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun createSizeSpinner(defaultValue: String): Spinner {
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, SIZE_OPTIONS).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        return Spinner(this).apply {
            this.adapter = adapter
            val defaultIndex = SIZE_OPTIONS.indexOf(defaultValue).coerceAtLeast(0)
            setSelection(defaultIndex)
        }
    }

    private fun createOptionGroup(label: String, spinner: Spinner): LinearLayout {
        val group = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        val labelView = TextView(this).apply {
            text = label
            textSize = 13f
            setTextColor(Color.LTGRAY)
            setPadding(0, 0, 0, 8)
        }
        group.addView(labelView)
        group.addView(spinner)
        return group
    }

    private fun createSeekBarGroup(
        label: String,
        valueView: TextView,
        seekBar: SeekBar
    ): LinearLayout {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 24, 0, 0)
        }
        val labelRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val labelView = TextView(this).apply {
            text = label
            textSize = 13f
            setTextColor(Color.LTGRAY)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        labelRow.addView(labelView)
        labelRow.addView(valueView)
        container.addView(labelRow)
        container.addView(seekBar)
        return container
    }

    private fun createOptionField(label: String, field: EditText): LinearLayout {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 24, 0, 0)
        }
        val labelView = TextView(this).apply {
            text = label
            textSize = 13f
            setTextColor(Color.LTGRAY)
            setPadding(0, 0, 0, 8)
        }
        container.addView(labelView)
        container.addView(field)
        return container
    }

    private fun addLabeledField(parent: LinearLayout, label: String, field: EditText) {
        val title = TextView(this).apply {
            text = label
            textSize = 14f
            setTextColor(Color.LTGRAY)
            setPadding(0, 8, 0, 8)
        }
        parent.addView(title)
        parent.addView(field)
    }

    override fun onDestroy() {
        cancelRequested.set(true)
        generationTask?.cancel(true)
        generationExecutor.shutdownNow()
        AppLogStore.i(TAG, "GenerationActivity onDestroy")
        super.onDestroy()
    }
}
