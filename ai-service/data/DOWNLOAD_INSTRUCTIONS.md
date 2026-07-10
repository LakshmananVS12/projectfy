# RDD2022 Download Instructions (Manual Step Required)

This project expects RDD-style Pascal VOC XML annotations for preprocessing.

## 1) Download dataset manually

Because the agent cannot click through website terms or authenticated downloads, complete these steps manually:

1. Open the official RDD dataset source in your browser (RDD2022 distribution page).
2. Download the image + annotation archives.
3. Extract all files to a local folder, for example:
   - `d:/Project 2/data/raw/RDD2022`

## 2) Verify expected layout

The preprocessor scans recursively, so nested folders are fine, but it needs:

- road images (`.jpg`/`.jpeg`/`.png`)
- annotation files (`.xml`, Pascal VOC format)

## 3) Run preprocessing

From `d:/Project 2/ai-service/data`:

```powershell
python preprocess_rdd2022.py `
  --raw-dir "d:/Project 2/data/raw/RDD2022" `
  --output-dir "d:/Project 2/data/processed/rdd2022" `
  --image-size 640 `
  --min-ravelling-samples 300
```

The script writes:

- `annotations_train.json`
- `annotations_val.json`
- `annotations_test.json`
- `preprocess_report.json` (contains class counts and any dropped classes)

## Notes on class viability

- `ravelling` is automatically dropped when sample count is below the configured threshold.
- If `edge_break` has zero samples in your raw snapshot, the report flags it clearly.
