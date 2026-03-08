# SDXL -> ONNX for Android

このリポジトリの目的は **Hugging Face から SDXL を取得し、Android で利用可能な ONNX 形式に変換すること** のみです。  
ローカル実行を優先し、失敗時は GitHub Actions の `sdxl.yml` を使います。

## 1) ローカルで実行

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r scripts/requirements.txt
python scripts/export_sdxl_onnx.py \
  --model-id stabilityai/stable-diffusion-xl-base-1.0 \
  --output-dir onnx
```

> private/gated モデルの場合は `HF_TOKEN` を設定してください。

## 2) 生成物の確認

`onnx/` 配下に以下が生成されれば成功です。

- `unet/model.onnx`
- `text_encoder/model.onnx`
- `text_encoder_2/model.onnx`
- `vae_decoder/model.onnx`
- `android_onnx_manifest.json`

## 3) ローカル失敗時: GitHub Actions

```bash
gh workflow run sdxl.yml \
  -f model_id=stabilityai/stable-diffusion-xl-base-1.0 \
  -f output_dir=onnx
```

必要なら以下で成果物を取得します。

```bash
gh run list --workflow sdxl.yml
gh run download <run-id> -n sdxl-onnx
```

## 4) 主要ファイル

- `scripts/export_sdxl_onnx.py`: SDXL のダウンロード + ONNX エクスポート
- `scripts/requirements.txt`: 変換に必要な最小依存関係
- `.github/workflows/sdxl.yml`: ローカル失敗時の CI フォールバック
