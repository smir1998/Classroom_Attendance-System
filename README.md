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

SFace's pretrained neural network is not fine-tuned. Student-specific profiles are learned from its embeddings, a more practical approach for classroom-sized datasets. The validation figures are estimates from the enrolled images and do not guarantee performance on future photos; collect separate evaluation images for a reliable deployment assessment.

## Take attendance

- **Start camera** scans the live webcam feed.
- **Open group photo** scans a JPG, PNG, BMP, or WebP image, draws a box and match score around each detected face, and lists a match or **Unknown** for review.
- A recognized student is recorded once per local calendar day with a timestamp. Unknown faces are not marked present.
- **Mark present** provides a manual alternative. **Export CSV** and **Export Excel** include every roster student and their present/absent status and timestamp for today.

The displayed cosine similarity is a score from around `0.0` to `1.0`, not a probability. Group photos with small, blurry, occluded, or turned-away faces can be missed or mismatched. A teacher should review results and correct attendance.

## Local data and privacy

`data/attendance.sqlite3` stores the roster and attendance history. Face crops and trained embeddings stay in `data/`; these files are not encrypted. Restrict access to the computer account and folder. **Remove** deletes a student's face crops and attendance history. To erase all app data, close the program and delete `data/`.

Obtain appropriate student/guardian consent and follow school policy and local law before collecting biometric data. Provide non-biometric attendance as an alternative. This prototype has no user accounts, encrypted database, audit trail, liveness detection, or demographic performance evaluation; do not use it as the sole attendance record or for disciplinary decisions.
