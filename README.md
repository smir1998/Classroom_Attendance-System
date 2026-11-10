# Classroom Attendance Studio

A local desktop attendance prototype for webcam sessions and group photographs. OpenCV YuNet detects multiple faces, SFace converts each aligned face crop into a numerical embedding, and cosine similarity compares it with each enrolled student's average embedding. The app draws a box and name or **Unknown** with a similarity score, marks identified students present once per day, and exports the class report as CSV or Excel. Camera frames and face images are processed locally.

## Run on Windows

1. Install Python 3.10 or newer with Tcl/Tk.
2. Open PowerShell in this folder and install packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Download the YuNet detector and SFace embedding model from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo):

   ```powershell
   python download_models.py
   ```

   This step needs internet access once. The model files are saved under `models/` and are excluded from Git.

4. Launch the application:

   ```powershell
   python app.py
   ```

## Enroll and build the face dataset

1. Open **Students & face dataset** and choose **Add student**.
2. Enter a unique student ID and name. The camera captures 24 aligned face crops while one face is visible.
3. For existing photos, select that student in the roster and choose **Import photos**. Each imported photo must contain exactly one detectable face.
4. The inspectable face crops are saved to `data/dataset/<internal-student-id>/`. Choose **Train embeddings** after adding or changing dataset images. The app makes one average embedding per student in memory.

Use multiple clear samples for each student with varied but normal head pose and lighting. Do not train using group photos as identity examples; enroll one student per sample so the label is unambiguous.

## Take attendance

- **Start camera** scans the live webcam feed.
- **Open group photo** scans a JPG, PNG, BMP, or WebP photograph. It draws a box around every detected face, lists each match and similarity, then marks recognized students present. Unknown faces are shown for review and are not added to attendance.
- **Mark present** provides a manual option for a selected student.
- A recognized student is recorded once per local calendar day. Each record stores a date and local timestamp.
- **Export CSV** and **Export Excel** save all roster students, including absent students, and their attendance timestamp for today.

Cosine similarity is shown as a score from approximately `0.0` to `1.0`; it is not a probability or confidence percentage. `MATCH_THRESHOLD` in `app.py` defaults to `0.363`. Tune it using consented samples from the actual camera and review false matches before use. Group photographs with small, blurry, occluded, or turned-away faces can be missed or mismatched; have a teacher review results and correct attendance.

The default camera index is `0`; change `CAMERA_INDEX` in `app.py` if necessary.

## Local data and privacy

`data/attendance.sqlite3` stores the roster and attendance history. Face crops are in `data/dataset/`. These files are not encrypted; restrict access to the computer account and folder. **Remove** deletes a student's crops and attendance history. After roster or dataset changes, choose **Train embeddings** before the next scan. To erase app data, close the program and delete the `data/` folder.

Obtain appropriate student/guardian consent and follow school policy and local law before collecting biometric data. Provide non-biometric attendance as an alternative. This prototype has no user accounts, encrypted database, audit trail, liveness detection, or demographic performance evaluation; do not use it as the sole attendance record or for disciplinary decisions.
