
# MSc Dissertation Project - Animal Detection and Frame Classification

This project implements an animal detection pipeline using **YOLOv8** models with post-processing mechanisms such as **interpolation** and **low-confidence recovery**. 



## Project Structure

```
MSc2025_Dongjun_Geng.zip
│
├── predictfix_lowC.py             # Low-confidence correction for empty frames
├── prediction_inter_YN.py         # Interpolation fix + Y/N classification
├── prediction_lowC_YN.py          # Low-confidence fix + Y/N classification
├── predictYNBOX.py                # Simple prediction and Y/N box classification
├── predictfixA_inter.py           # Alternative interpolation-based fix
├── train.py                       # training starter
├── README.md                      # This file

```

---

##  How to Run

All scripts are designed to work with **Ultralytics YOLOv8**.

### 1. Install Required Packages

```bash
pip install ultralytics opencv-python pandas
```

### 2. Run a Script (Example)

```bash
python prediction_lowC_YN.py
```

You can also choose:
- `prediction_inter_YN.py`: interpolates false-negative frames.
- `predictfix_lowC.py`: re-runs low-confidence detection if a frame is empty.
- `predictYNBOX.py`: quick detection with Y/N result for each frame.

---

## Dependencies

- Python 3.8+
- `ultralytics` (YOLOv8)
- `opencv-python`
- `pandas`
- `numpy`

Make sure your YOLO model weights (e.g., `best.pt`) are located in the correct path used inside the script.

---

##  Output Format

- Images will be saved to `output_images_Y/` and `output_images_N/`
- A CSV file `frame_classification.csv` will contain per-frame Y/N labels
- Interpolated or fixed frames will be prefixed with `fix_` in the filename

---
\
