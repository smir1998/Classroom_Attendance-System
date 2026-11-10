# Classroom Attendance Studio

A local desktop attendance prototype for webcam sessions and group photographs. OpenCV YuNet detects faces; SFace converts each aligned crop to a numerical embedding. The student dataset is used to build identity profiles, compare group-photo embeddings, mark daily attendance, and export CSV or Excel reports.

## Run on Windows

1. Install Python 3.10 or newer with Tcl/Tk.
2. Open PowerShell in this folder and install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Download the YuNet and SFace model files from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo):

   ```powershell
   python download_models.py
   ```

   This requires internet access once. Model files are saved in `models/` and excluded from Git.

4. Start the app:

   ```powershell
   python app.py
   ```

## Build the student dataset

1. Open **Students & face dataset** and choose **Add student** to capture 24 aligned images, or select a student and choose **Import photos**.
2. For imported photos, use one clear face per image. Group photos are for attendance scans; do not use them as training images because face-to-student labels would be ambiguous.
3. Review images in `data/dataset/<internal-student-id>/`. Capture varied, clear samples under normal lighting and pose.
4. Choose **Train embeddings**. The app runs quality checks, applies horizontal-flip and mild lighting augmentation, removes embedding outliers, and builds up to three pose/appearance prototypes per student. It validates profiles with leave-one-out comparisons and, when there are enough impostor comparisons, calibrates the threshold toward an empirical false-accept rate of at most 1%. The training report shows the observed rates and fallback when the dataset cannot meet that target.
5. Open **Training report** to review how many images were used or filtered and the validation scores. The trained profiles are cached locally in `data/student_profiles.npz`; the app reuses the cache when the dataset is unchanged and retrains after dataset changes.

### Fine-tune the neural network weights

Fine-tuning is optional and needs more disk space and compute than normal attendance. Install the additional PyTorch/FaceNet packages:

```powershell
python -m pip install -r finetune-requirements.txt
```

For an NVIDIA GPU, install the matching CUDA-enabled PyTorch build using the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing `finetune-requirements.txt`. The first run downloads the pretrained VGGFace2 FaceNet weights from [facenet-pytorch](https://github.com/timesler/facenet-pytorch) into the PyTorch cache.

Before starting, enroll at least two students and make sure each has four or more clear face images. The app checks the per-student image counts and displays them if the dataset is too small; it will not start fine-tuning until those minimums are met. Add another student with **Add student**, or select a roster entry and use **Import photos**.

In **Students & face dataset**, choose **Fine-tune weights**. It makes a per-student holdout split (keeping the last captured images for validation to reduce adjacent-frame leakage), uses balanced batches and lighting/flip augmentation, then fine-tunes the final residual blocks and projection layer with early stopping. It saves `data/finetuned_facenet.pt`; the app then rebuilds student profiles with those weights and uses FaceNet embeddings for future attendance scans. On later launches the saved weights load automatically when the optional packages are installed. The training process runs locally and does not upload student images.

The fine-tuned network is initialized from pretrained FaceNet VGGFace2 weights; it is not trained from scratch. Small student datasets can overfit, so review the held-out validation accuracy, capture varied images, and evaluate on separate photos before relying on attendance results. The UI's **Train embeddings** action remains available to rebuild student profiles without changing network weights.

## Take attendance

- **Start camera** scans the live webcam feed.
- **Open group photo** scans a JPG, PNG, BMP, or WebP image, draws a box and match score around each detected face, and lists a match or **Unknown** for review.
- A recognized student is recorded once per local calendar day with a timestamp. Unknown faces are not marked present.
- **Mark present** provides a manual alternative. **Export CSV** and **Export Excel** include every roster student and their present/absent status and timestamp for today.

The displayed cosine similarity is a score from around `0.0` to `1.0`, not a probability. Group photos with small, blurry, occluded, or turned-away faces can be missed or mismatched. A teacher should review results and correct attendance.

## Local data and privacy

`data/attendance.sqlite3` stores the roster and attendance history. Face crops and trained embeddings stay in `data/`; these files are not encrypted. Restrict access to the computer account and folder. **Remove** deletes a student's face crops and attendance history. To erase all app data, close the program and delete `data/`.

Obtain appropriate student/guardian consent and follow school policy and local law before collecting biometric data. Provide non-biometric attendance as an alternative. This prototype has no user accounts, encrypted database, audit trail, liveness detection, or demographic performance evaluation; do not use it as the sole attendance record or for disciplinary decisions.
