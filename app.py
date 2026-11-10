"""Local classroom attendance using YuNet detection and SFace embeddings."""

from __future__ import annotations

import csv
import hashlib
import shutil
import sqlite3
import tkinter as tk
import zipfile
from datetime import date, datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
DB_PATH = DATA_DIR / "attendance.sqlite3"
PROFILE_PATH = DATA_DIR / "student_profiles.npz"
MODEL_DIR = APP_DIR / "models"
DETECTOR_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
EMBEDDING_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
CAMERA_INDEX = 0
FACE_SIZE = (112, 112)
SAMPLES_PER_ENROLLMENT = 24
MATCH_THRESHOLD = 0.363  # Cosine similarity; tune with the local student dataset.
MAX_PROTOTYPES_PER_STUDENT = 3

BG = "#f3f6fa"
INK = "#152238"
MUTED = "#64748b"
BLUE = "#275efe"
GREEN = "#138a61"
RED = "#d74747"
CARD = "#ffffff"


class AttendanceStore:
    """SQLite storage for the roster and timestamped attendance records."""

    def __init__(self, path: Path = DB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_number TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                attendance_date TEXT NOT NULL,
                marked_at TEXT NOT NULL,
                UNIQUE(student_id, attendance_date)
            );
            """
        )
        self.connection.commit()
        self._migrate_legacy_face_samples()
        (DATA_DIR / "lbph_model.yml").unlink(missing_ok=True)

    def _migrate_legacy_face_samples(self) -> None:
        legacy = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'face_samples'"
        ).fetchone()
        if not legacy:
            return
        rows = self.connection.execute(
            "SELECT student_id, image FROM face_samples ORDER BY student_id, id"
        ).fetchall()
        counts: dict[int, int] = {}
        has_dataset: dict[int, bool] = {}
        for row in rows:
            student_id = int(row["student_id"])
            student_dir = DATASET_DIR / str(student_id)
            student_dir.mkdir(parents=True, exist_ok=True)
            if student_id not in has_dataset:
                has_dataset[student_id] = any(student_dir.glob("face_*.jpg"))
            if has_dataset[student_id]:
                continue
            counts[student_id] = counts.get(student_id, 0) + 1
            (student_dir / f"face_{counts[student_id]:03}.jpg").write_bytes(bytes(row["image"]))
        with self.connection:
            self.connection.execute("DROP TABLE face_samples")

    def students(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT id, student_number, name FROM students ORDER BY name COLLATE NOCASE"
        ).fetchall()

    def add_student(self, number: str, name: str, faces: list[bytes]) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO students(student_number, name) VALUES (?, ?)", (number, name)
            )
            student_id = int(cursor.lastrowid)
        try:
            self.add_dataset_images(student_id, faces)
        except OSError:
            self.remove_student(student_id)
            raise
        return student_id

    def add_dataset_images(self, student_id: int, faces: list[bytes]) -> None:
        student_dir = DATASET_DIR / str(student_id)
        student_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(student_dir.glob("face_*.jpg"))
        start = len(existing) + 1
        for index, image in enumerate(faces, start=start):
            (student_dir / f"face_{index:03}.jpg").write_bytes(image)

    def remove_student(self, student_id: int) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM students WHERE id = ?", (student_id,))
        shutil.rmtree(DATASET_DIR / str(student_id), ignore_errors=True)

    def mark_present(self, student_id: int) -> bool:
        now = datetime.now()
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO attendance(student_id, attendance_date, marked_at) "
            "VALUES (?, ?, ?)",
            (student_id, now.date().isoformat(), now.isoformat(timespec="seconds")),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def present_ids(self) -> set[int]:
        rows = self.connection.execute(
            "SELECT student_id FROM attendance WHERE attendance_date = ?", (date.today().isoformat(),)
        ).fetchall()
        return {int(row["student_id"]) for row in rows}

    def marked_at(self, student_id: int, attendance_date: str | None = None) -> str | None:
        attendance_date = attendance_date or date.today().isoformat()
        row = self.connection.execute(
            "SELECT marked_at FROM attendance WHERE student_id = ? AND attendance_date = ?",
            (student_id, attendance_date),
        ).fetchone()
        return str(row["marked_at"]) if row else None

    def report_rows(self, attendance_date: str | None = None) -> list[sqlite3.Row]:
        attendance_date = attendance_date or date.today().isoformat()
        return self.connection.execute(
            "SELECT s.student_number, s.name, a.marked_at FROM students s "
            "LEFT JOIN attendance a ON a.student_id = s.id AND a.attendance_date = ? "
            "ORDER BY s.name COLLATE NOCASE",
            (attendance_date,),
        ).fetchall()

    def close(self) -> None:
        self.connection.close()


class AttendanceApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.store = AttendanceStore()
        self.students_by_id: dict[int, sqlite3.Row] = {}
        self.detector = None
        self.recognizer = None
        self.prototypes: dict[int, list[np.ndarray]] = {}
        self.training_report = ""
        self.match_threshold = MATCH_THRESHOLD
        self.camera: cv2.VideoCapture | None = None
        self.running = False
        self.enrolling: tuple[str, str] | None = None
        self.enrollment_faces: list[bytes] = []
        self.enrollment_frame_counter = 0
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.last_annotated: np.ndarray | None = None
        self.today_present_ids = self.store.present_ids()
        self.current_date = date.today()

        self.root.title("Classroom · Attendance Studio")
        self.root.geometry("1280x820")
        self.root.minsize(1080, 700)
        self.root.configure(bg=BG)
        self._build_ui()
        self._load_engines()
        self._refresh_roster()
        self._load_dataset(silent=True)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=INK, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"), foreground=INK)
        style.configure("Section.TLabel", font=("Segoe UI", 13, "bold"), foreground=INK)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 10), font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", BLUE)])
        style.configure("Treeview", rowheight=30, font=("Segoe UI", 10), background=CARD, fieldbackground=CARD)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), foreground=MUTED)

        shell = ttk.Frame(self.root, padding=(24, 20))
        shell.pack(fill="both", expand=True)
        header = tk.Frame(shell, bg=BG)
        header.pack(fill="x", pady=(0, 18))
        brand = tk.Frame(header, bg=BG)
        brand.pack(side="left")
        tk.Label(brand, text="CLASSROOM", bg=BG, fg=BLUE, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(brand, text="Attendance Studio", bg=BG, fg=INK, font=("Segoe UI", 23, "bold")).pack(anchor="w")
        tk.Label(brand, text="Group photo recognition · Local and private", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 0))
        self.date_badge = tk.Label(
            header, text=date.today().strftime("%a  ·  %d %b %Y"), bg=CARD, fg=INK,
            font=("Segoe UI", 10, "bold"), padx=16, pady=11,
        )
        self.date_badge.pack(side="right", anchor="n", pady=6)

        cards = tk.Frame(shell, bg=BG)
        cards.pack(fill="x", pady=(0, 16))
        self.enrolled_value = self._stat_card(cards, "ENROLLED", "0", "Students in this class")
        self.present_value = self._stat_card(cards, "PRESENT TODAY", "0", "Unique students checked in")
        self.model_value = self._stat_card(cards, "FACE MODEL", "Not trained", "Quality checks · augmentation · validation")

        self.tabs = ttk.Notebook(shell)
        self.tabs.pack(fill="both", expand=True)
        self.session_tab = ttk.Frame(self.tabs, padding=14)
        self.students_tab = ttk.Frame(self.tabs, padding=14)
        self.tabs.add(self.session_tab, text="  Attendance session  ")
        self.tabs.add(self.students_tab, text="  Students & face dataset  ")
        self._build_session_tab()
        self._build_students_tab()

        foot = tk.Frame(shell, bg=BG)
        foot.pack(fill="x", pady=(12, 0))
        tk.Label(foot, text="Biometric data stays on this computer. Provide consent and manual attendance as an alternative.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(side="left")

    def _stat_card(self, parent: tk.Frame, label: str, value: str, detail: str) -> tk.Label:
        card = tk.Frame(parent, bg=CARD, highlightbackground="#e2e8f0", highlightthickness=1, padx=16, pady=12)
        card.pack(side="left", fill="x", expand=True, padx=(0, 12))
        tk.Label(card, text=label, bg=CARD, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w")
        value_label = tk.Label(card, text=value, bg=CARD, fg=INK, font=("Segoe UI", 18, "bold"))
        value_label.pack(anchor="w", pady=(4, 0))
        tk.Label(card, text=detail, bg=CARD, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")
        return value_label

    @staticmethod
    def _button(parent: tk.Widget, text: str, command, primary: bool = False) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command, relief="flat", cursor="hand2",
            bg=BLUE if primary else "#e8eef9", fg="#ffffff" if primary else INK,
            activebackground="#1749e8" if primary else "#dbe5f4",
            activeforeground="#ffffff" if primary else INK,
            font=("Segoe UI", 10, "bold"), padx=14, pady=10, bd=0,
        )

    def _build_session_tab(self) -> None:
        toolbar = tk.Frame(self.session_tab, bg=BG)
        toolbar.pack(fill="x", pady=(0, 12))
        self.camera_button = self._button(toolbar, "Start camera", self._toggle_camera, primary=True)
        self.camera_button.pack(side="left", padx=(0, 8))
        self._button(toolbar, "Open group photo", self._open_photo).pack(side="left", padx=(0, 8))
        self._button(toolbar, "Export CSV", self._export_csv).pack(side="right", padx=(8, 0))
        self._button(toolbar, "Export Excel", self._export_excel).pack(side="right")

        body = tk.Frame(self.session_tab, bg=BG)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=CARD, highlightbackground="#e2e8f0", highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 14))
        right = tk.Frame(body, bg=CARD, width=370, highlightbackground="#e2e8f0", highlightthickness=1)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        preview_head = tk.Frame(left, bg=CARD, padx=16, pady=14)
        preview_head.pack(fill="x")
        tk.Label(preview_head, text="Live preview", bg=CARD, fg=INK, font=("Segoe UI", 13, "bold")).pack(side="left")
        self.source_label = tk.Label(preview_head, text="NO SOURCE", bg="#edf2f7", fg=MUTED,
                                     font=("Segoe UI", 8, "bold"), padx=9, pady=5)
        self.source_label.pack(side="right")
        self.preview = tk.Label(left, text="Start the camera or open a group photograph", bg="#101a2a", fg="#cbd5e1",
                                font=("Segoe UI", 12), compound="center")
        self.preview.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.session_status = tk.Label(left, text="Ready when you are.", bg=CARD, fg=MUTED,
                                       anchor="w", font=("Segoe UI", 9), padx=16, pady=10)
        self.session_status.pack(fill="x")

        result_head = tk.Frame(right, bg=CARD, padx=16, pady=14)
        result_head.pack(fill="x")
        tk.Label(result_head, text="Face matches", bg=CARD, fg=INK, font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(result_head, text="Each face is scored against enrolled embeddings.", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 9), wraplength=330, justify="left").pack(anchor="w", pady=(4, 0))
        self.results = ttk.Treeview(right, columns=("face", "student", "score", "status"), show="headings", height=16)
        for col, label, width in (("face", "FACE", 48), ("student", "STUDENT", 145), ("score", "SIMILARITY", 84), ("status", "STATUS", 75)):
            self.results.heading(col, text=label)
            self.results.column(col, width=width, anchor="w" if col == "student" else "center")
        self.results.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.results.tag_configure("present", foreground=GREEN)
        self.results.tag_configure("unknown", foreground=RED)

    def _build_students_tab(self) -> None:
        head = tk.Frame(self.students_tab, bg=BG)
        head.pack(fill="x", pady=(0, 10))
        tk.Label(head, text="Class roster", bg=BG, fg=INK, font=("Segoe UI", 16, "bold")).pack(side="left")
        self._button(head, "Train embeddings", self._train_model, primary=True).pack(side="right")
        self._button(head, "Training report", self._show_training_report).pack(side="right", padx=(0, 8))
        self._button(head, "Add student", self._begin_enrollment).pack(side="right", padx=(0, 8))
        self._button(head, "Import photos", self._import_student_photos).pack(side="right", padx=(0, 8))
        self._button(head, "Mark present", self._mark_selected_present).pack(side="right", padx=(0, 8))
        self._button(head, "Remove", self._remove_selected).pack(side="right", padx=(0, 8))

        note = tk.Label(
            self.students_tab,
            text="Enroll by camera or select a student and import one or more clear face photos. Face crops are saved under data/dataset/<student-id>/. Train embeddings after making dataset changes.",
            bg=BG, fg=MUTED, justify="left", wraplength=1000, font=("Segoe UI", 9),
        )
        note.pack(fill="x", anchor="w", pady=(0, 12))
        panel = tk.Frame(self.students_tab, bg=CARD, highlightbackground="#e2e8f0", highlightthickness=1)
        panel.pack(fill="both", expand=True)
        self.roster = ttk.Treeview(panel, columns=("number", "name", "attendance", "time"), show="headings")
        for col, label, width in (("number", "STUDENT ID", 180), ("name", "NAME", 360),
                                  ("attendance", "TODAY", 150), ("time", "MARKED AT", 220)):
            self.roster.heading(col, text=label)
            self.roster.column(col, width=width, anchor="w")
        self.roster.pack(fill="both", expand=True, padx=10, pady=10)
        self.roster.tag_configure("present", foreground=GREEN)

    def _load_engines(self) -> None:
        if not DETECTOR_PATH.exists() or not EMBEDDING_PATH.exists():
            self.session_status.configure(text="Face models missing. Run: python download_models.py")
            self.model_value.configure(text="Models missing", fg=RED)
            return
        try:
            self.detector = cv2.FaceDetectorYN.create(
                str(DETECTOR_PATH), "", (320, 320), 0.75, 0.3, 5000
            )
            self.recognizer = cv2.FaceRecognizerSF.create(str(EMBEDDING_PATH), "")
        except (cv2.error, AttributeError) as error:
            self.session_status.configure(text=f"Could not load face models: {error}")
            self.model_value.configure(text="Model error", fg=RED)

    def _refresh_roster(self) -> None:
        if not hasattr(self, "roster"):
            return
        self.roster.delete(*self.roster.get_children())
        students = self.store.students()
        self.students_by_id = {int(row["id"]): row for row in students}
        count = 0
        for row in self.store.report_rows():
            marked_at = row["marked_at"]
            present = marked_at is not None
            count += int(present)
            time_text = str(marked_at)[11:19] if present else "—"
            self.roster.insert(
                "", "end", iid=str(row["student_number"]),
                values=(row["student_number"], row["name"], "Present" if present else "Absent", time_text),
                tags=("present",) if present else (),
            )
        self.enrolled_value.configure(text=str(len(students)))
        self.present_value.configure(text=str(count))

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        vector = vector.astype(np.float32, copy=False)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def _dataset_signature(self) -> str:
        digest = hashlib.sha256()
        if DATASET_DIR.exists():
            for image_path in sorted(DATASET_DIR.rglob("*.jpg")):
                digest.update(str(image_path.relative_to(DATASET_DIR)).encode())
                with image_path.open("rb") as image_file:
                    while chunk := image_file.read(1024 * 1024):
                        digest.update(chunk)
        return digest.hexdigest()

    def _load_profile_cache(self, signature: str) -> bool:
        if not PROFILE_PATH.exists():
            return False
        try:
            with np.load(PROFILE_PATH, allow_pickle=False) as saved:
                if str(saved["signature"].item()) != signature:
                    return False
                labels = saved["labels"].astype(np.int64)
                vectors = saved["vectors"].astype(np.float32)
                if vectors.ndim != 2 or vectors.shape[1] != 128 or len(labels) != len(vectors):
                    return False
                current_ids = set(self.students_by_id)
                if not set(labels.tolist()).issubset(current_ids):
                    return False
                self.prototypes = {}
                for student_id, vector in zip(labels, vectors):
                    self.prototypes.setdefault(int(student_id), []).append(self._normalise(vector))
                self.match_threshold = float(saved["threshold"].item())
                self.training_report = str(saved["report"].item())
                return bool(self.prototypes)
        except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
            return False

    def _save_profile_cache(self, signature: str) -> None:
        labels, vectors = [], []
        for student_id, centers in self.prototypes.items():
            for center in centers:
                labels.append(student_id)
                vectors.append(center)
        if not vectors:
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(
                PROFILE_PATH,
                labels=np.asarray(labels, dtype=np.int64),
                vectors=np.vstack(vectors).astype(np.float32),
                threshold=np.asarray(self.match_threshold, dtype=np.float32),
                signature=np.asarray(signature),
                report=np.asarray(self.training_report),
            )
        except OSError:
            self.session_status.configure(text="Profiles trained; local cache could not be saved.")

    def _invalidate_profiles(self) -> None:
        self.prototypes = {}
        self.training_report = ""
        self.match_threshold = MATCH_THRESHOLD
        self.model_value.configure(text="Retrain required", fg=RED)
        PROFILE_PATH.unlink(missing_ok=True)

    def _embedding(self, image: np.ndarray) -> np.ndarray:
        feature = self.recognizer.feature(image).flatten().astype(np.float32)
        return self._normalise(feature)

    def _augment(self, image: np.ndarray) -> list[np.ndarray]:
        return [
            image,
            cv2.flip(image, 1),
            cv2.convertScaleAbs(image, alpha=0.92, beta=8),
            cv2.convertScaleAbs(image, alpha=1.06, beta=-6),
        ]

    def _student_prototypes(self, vectors: list[np.ndarray]) -> list[np.ndarray]:
        matrix = np.vstack(vectors).astype(np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        similarities = matrix @ matrix.T
        k = min(MAX_PROTOTYPES_PER_STUDENT, max(1, int(np.ceil(np.sqrt(len(matrix) / 8)))))
        centers = [int(np.argmax(similarities.mean(axis=1)))]
        while len(centers) < k:
            nearest = similarities[:, centers].max(axis=1)
            candidate = int(np.argmin(nearest))
            if nearest[candidate] > 0.985:
                break
            centers.append(candidate)
        center_vectors = matrix[centers].copy()
        for _ in range(6):
            assignments = np.argmax(matrix @ center_vectors.T, axis=1)
            updated = []
            for cluster_index, prior in enumerate(center_vectors):
                members = matrix[assignments == cluster_index]
                updated.append(self._normalise(members.mean(axis=0)) if len(members) else prior)
            center_vectors = np.vstack(updated)
        return [self._normalise(center) for center in center_vectors]

    def _validate_embeddings(
        self, originals: dict[int, list[np.ndarray]]
    ) -> tuple[float, float | None, float | None, int, int]:
        genuine_scores: list[float] = []
        impostor_scores: list[float] = []
        for student_id, samples in originals.items():
            if len(samples) < 2:
                continue
            for held_out_index, held_out in enumerate(samples):
                references = samples[:held_out_index] + samples[held_out_index + 1 :]
                genuine_center = self._normalise(np.mean(references, axis=0))
                genuine_scores.append(float(np.dot(held_out, genuine_center)))
                for other_id, other_samples in originals.items():
                    if other_id == student_id:
                        continue
                    other_center = self._normalise(np.mean(other_samples, axis=0))
                    impostor_scores.append(float(np.dot(held_out, other_center)))
        if len(impostor_scores) < 50:
            return MATCH_THRESHOLD, None, None, len(genuine_scores), len(impostor_scores)
        candidates = np.arange(0.20, 0.901, 0.005)
        feasible = [
            float(threshold) for threshold in candidates
            if np.mean(np.asarray(impostor_scores) >= threshold) <= 0.01
        ]
        threshold = min(feasible) if feasible else 0.90
        false_accept = float(np.mean(np.asarray(impostor_scores) >= threshold))
        false_reject = float(np.mean(np.asarray(genuine_scores) < threshold)) if genuine_scores else None
        return threshold, false_accept, false_reject, len(genuine_scores), len(impostor_scores)

    def _load_dataset(self, silent: bool = False, force: bool = False) -> None:
        if self.recognizer is None:
            self.prototypes = {}
            return
        signature = self._dataset_signature()
        if not force and self._load_profile_cache(signature):
            self.model_value.configure(text=f"Ready · {len(self.prototypes)} students", fg=GREEN)
            if not silent:
                self.session_status.configure(text="Loaded the trained student profiles from local cache.")
            return
        if force:
            PROFILE_PATH.unlink(missing_ok=True)

        originals: dict[int, list[np.ndarray]] = {}
        augmented: dict[int, list[np.ndarray]] = {}
        total_images = accepted_images = quality_rejected = outlier_rejected = 0
        image_paths = sorted(DATASET_DIR.rglob("*.jpg")) if DATASET_DIR.exists() else []
        total_paths = len(image_paths)
        for sample_number, image_path in enumerate(image_paths, start=1):
            student_dir = image_path.parent
            if not student_dir.name.isdigit():
                continue
            student_id = int(student_dir.name)
            if student_id not in self.students_by_id:
                continue
            total_images += 1
            if not silent:
                self.session_status.configure(text=f"Training embeddings · sample {sample_number} of {total_paths}…")
                self.root.update_idletasks()
            image = cv2.imread(str(image_path))
            if image is None:
                quality_rejected += 1
                continue
            try:
                faces = self._detect(image)
                if len(faces) > 1:
                    quality_rejected += 1
                    continue
                aligned = self.recognizer.alignCrop(image, faces[0]) if len(faces) == 1 else image
                aligned = cv2.resize(aligned, FACE_SIZE)
                gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness = float(gray.mean())
                if sharpness < 4.0 or brightness < 12.0 or brightness > 245.0:
                    quality_rejected += 1
                    continue
                variants = [self._embedding(item) for item in self._augment(aligned)]
                originals.setdefault(student_id, []).append(variants[0])
                augmented.setdefault(student_id, []).append(variants)
            except cv2.error:
                quality_rejected += 1

        accepted_originals: dict[int, list[np.ndarray]] = {}
        accepted_augmented: dict[int, list[np.ndarray]] = {}
        for student_id, original_vectors in originals.items():
            matrix = np.vstack(original_vectors)
            if len(matrix) >= 5:
                similarity = matrix @ matrix.T
                medoid_index = int(np.argmax(similarity.mean(axis=1)))
                to_medoid = similarity[:, medoid_index]
                median = float(np.median(to_medoid))
                mad = float(np.median(np.abs(to_medoid - median)))
                cutoff = median - max(0.08, 3.0 * mad)
                keep = np.flatnonzero(to_medoid >= cutoff)
                minimum_keep = max(3, int(np.ceil(len(matrix) * 0.60)))
                if len(keep) < minimum_keep:
                    keep = np.argsort(to_medoid)[-minimum_keep:]
                outlier_rejected += len(matrix) - len(keep)
            else:
                keep = np.arange(len(matrix))
            accepted = [int(index) for index in keep]
            accepted_originals[student_id] = [original_vectors[index] for index in accepted]
            accepted_augmented[student_id] = [vector for index in accepted for vector in augmented[student_id][index]]
            accepted_images += len(accepted)

        self.prototypes = {
            student_id: self._student_prototypes(vectors)
            for student_id, vectors in accepted_augmented.items() if vectors
        }
        self.match_threshold, far, frr, genuine_count, impostor_count = self._validate_embeddings(accepted_originals)
        self.training_report = (
            "Advanced embedding training report\n\n"
            f"Dataset images found: {total_images}\n"
            f"Quality filtered: {quality_rejected}\n"
            f"Outliers filtered: {outlier_rejected}\n"
            f"Images used: {accepted_images}\n"
            f"Students profiled: {len(self.prototypes)}\n"
            "Augmentations: horizontal flip and mild lighting variation\n"
            f"Identity prototypes: {sum(len(items) for items in self.prototypes.values())}\n"
        )
        if far is None:
            self.training_report += (
                "Validation: insufficient impostor comparisons for safe threshold calibration; "
                f"using configured threshold {self.match_threshold:.3f}.\n"
            )
        else:
            self.training_report += (
                f"Leave-one-out validation: {genuine_count} genuine and {impostor_count} impostor comparisons\n"
                f"Calibrated threshold: {self.match_threshold:.3f} (empirical false-accept rate {far:.1%}; "
                f"false-reject rate {frr:.1%})\n"
            )
        self.training_report += (
            "SFace's pretrained weights stay frozen; training adapts student identity profiles.\n"
            "Validation uses this enrolled dataset and is an estimate, not a guarantee of real-world accuracy."
        )

        if self.prototypes:
            self.model_value.configure(text=f"Ready · {len(self.prototypes)} students", fg=GREEN)
            self._save_profile_cache(signature)
        else:
            self.model_value.configure(text="No usable faces", fg=RED)
        if not silent:
            if not self.prototypes:
                messagebox.showinfo("No usable face images", "Add clear face photos for students, then train again.")
            else:
                self.session_status.configure(text="Training complete. Review the validation report before attendance.")
                messagebox.showinfo("Training complete", self.training_report)

    def _show_training_report(self) -> None:
        if self.training_report:
            messagebox.showinfo("Student embedding training", self.training_report)
        else:
            messagebox.showinfo("No training report", "Select Train embeddings to build and evaluate student profiles.")

    def _train_model(self) -> None:
        if self.running:
            messagebox.showinfo("Stop camera first", "Stop the live session before rebuilding embeddings.")
            return
        if self.detector is None or self.recognizer is None:
            messagebox.showerror("Face models missing", "Run python download_models.py, then restart the app.")
            return
        self._load_dataset(silent=False, force=True)

    def _begin_enrollment(self) -> None:
        if self.running:
            messagebox.showinfo("Camera in use", "Stop the current camera session before enrolling a student.")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Add student")
        dialog.configure(bg=BG)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        body = tk.Frame(dialog, bg=CARD, padx=22, pady=20)
        body.pack(fill="both", expand=True, padx=12, pady=12)
        tk.Label(body, text="Add a student", bg=CARD, fg=INK, font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )
        tk.Label(body, text="Student ID", bg=CARD, fg=MUTED).grid(row=1, column=0, sticky="w", pady=6)
        tk.Label(body, text="Full name", bg=CARD, fg=MUTED).grid(row=2, column=0, sticky="w", pady=6)
        number_var, name_var = tk.StringVar(), tk.StringVar()
        ttk.Entry(body, textvariable=number_var, width=34).grid(row=1, column=1, pady=6, padx=(14, 0))
        ttk.Entry(body, textvariable=name_var, width=34).grid(row=2, column=1, pady=6, padx=(14, 0))
        tk.Label(body, text=f"Camera enrollment captures {SAMPLES_PER_ENROLLMENT} aligned face crops.",
                 bg=CARD, fg=MUTED, font=("Segoe UI", 9)).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 14))

        def start_capture() -> None:
            number, name = number_var.get().strip(), name_var.get().strip()
            if not number or not name:
                messagebox.showerror("Missing details", "Enter both a student ID and a name.", parent=dialog)
                return
            if any(row["student_number"] == number for row in self.store.students()):
                messagebox.showerror("Duplicate ID", "That student ID is already enrolled.", parent=dialog)
                return
            dialog.destroy()
            self._start_enrollment_capture(number, name)

        self._button(body, "Continue to camera capture", start_capture, primary=True).grid(
            row=4, column=0, columnspan=2, sticky="ew"
        )

    def _start_enrollment_capture(self, number: str, name: str) -> None:
        if self.detector is None:
            messagebox.showerror("Face models missing", "Run python download_models.py before enrolling students.")
            return
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            camera.release()
            messagebox.showerror("Camera unavailable", f"Could not open camera {CAMERA_INDEX}.")
            return
        self.camera = camera
        self.running = True
        self.enrolling = (number, name)
        self.enrollment_faces = []
        self.enrollment_frame_counter = 0
        self.camera_button.configure(text="Stop camera")
        self.source_label.configure(text="ENROLLMENT")
        self.session_status.configure(text="Keep one face in view; look forward, then gently turn left and right.")
        self.tabs.select(self.session_tab)
        self._camera_tick()

    def _toggle_camera(self) -> None:
        if self.running:
            self._stop_camera()
            return
        if self.detector is None or self.recognizer is None:
            messagebox.showerror("Face models missing", "Run python download_models.py before starting attendance.")
            return
        if not self.prototypes:
            messagebox.showinfo("Train embeddings first", "Add student face images, then select Train embeddings.")
            return
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            camera.release()
            messagebox.showerror("Camera unavailable", f"Could not open camera {CAMERA_INDEX}.")
            return
        self.camera = camera
        self.running = True
        self.camera_button.configure(text="Stop camera")
        self.source_label.configure(text="LIVE CAMERA")
        self.session_status.configure(text="Live scan is running. Faces are matched and marked once per day.")
        self._camera_tick()

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        if self.detector is None:
            return np.empty((0, 15), dtype=np.float32)
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        return faces

    def _identify(self, frame: np.ndarray, face: np.ndarray) -> tuple[int | None, float, str]:
        if self.recognizer is None:
            return None, 0.0, "Unknown"
        try:
            aligned = self.recognizer.alignCrop(frame, face)
            feature = self.recognizer.feature(aligned).flatten().astype(np.float32)
            feature /= max(float(np.linalg.norm(feature)), 1e-12)
        except cv2.error:
            return None, 0.0, "Unknown"
        best_id, best_score = None, -1.0
        for student_id, identity_prototypes in self.prototypes.items():
            score = max(float(np.dot(feature, prototype)) for prototype in identity_prototypes)
            if score > best_score:
                best_id, best_score = student_id, score
        if best_id is None or best_score < self.match_threshold:
            return None, max(best_score, 0.0), "Unknown"
        row = self.students_by_id.get(best_id)
        return (best_id, best_score, row["name"]) if row else (None, best_score, "Unknown")

    def _analyze_frame(self, frame: np.ndarray, enroll: bool = False) -> tuple[np.ndarray, list[dict[str, object]]]:
        if date.today() != self.current_date:
            self.current_date = date.today()
            self.today_present_ids = self.store.present_ids()
            self.date_badge.configure(text=self.current_date.strftime("%a  ·  %d %b %Y"))
            self._refresh_roster()
        annotated = frame.copy()
        faces = self._detect(frame)
        records: list[dict[str, object]] = []
        for index, face in enumerate(faces, start=1):
            x, y, width, height = [int(value) for value in face[:4]]
            x, y = max(0, x), max(0, y)
            if enroll:
                self.enrollment_frame_counter += 1
                label = f"Capturing {len(self.enrollment_faces)}/{SAMPLES_PER_ENROLLMENT}"
                color = (50, 190, 250)
                if len(faces) == 1 and self.enrollment_frame_counter % 4 == 0 and len(self.enrollment_faces) < SAMPLES_PER_ENROLLMENT:
                    try:
                        crop = self.recognizer.alignCrop(frame, face)
                        crop = cv2.resize(crop, FACE_SIZE)
                        success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 93])
                        if success:
                            self.enrollment_faces.append(encoded.tobytes())
                    except cv2.error:
                        pass
                records.append({"face": index, "student": "Dataset capture", "score": "—", "status": label})
            else:
                student_id, score, name = self._identify(frame, face)
                is_known = student_id is not None
                color = (64, 184, 113) if is_known else (55, 80, 230)
                label = f"{name}  {score:.2f}" if is_known else f"Unknown  {score:.2f}"
                marked_time = None
                newly_marked = False
                if student_id is not None and student_id not in self.today_present_ids:
                    newly_marked = self.store.mark_present(student_id)
                    self.today_present_ids.add(student_id)
                    marked_time = self.store.marked_at(student_id)
                elif student_id is not None:
                    marked_time = self.store.marked_at(student_id)
                records.append({
                    "face": index, "student": name, "score": f"{score:.3f}",
                    "status": "Present" if is_known else "Unknown", "student_id": student_id,
                    "marked_at": marked_time, "newly_marked": newly_marked,
                })
            cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2, cv2.LINE_AA)
            baseline = max(22, y - 9)
            cv2.putText(annotated, label, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
        return annotated, records

    def _update_results(self, records: list[dict[str, object]]) -> None:
        self.results.delete(*self.results.get_children())
        for item in records:
            known = item.get("student_id") is not None
            self.results.insert(
                "", "end",
                values=(item["face"], item["student"], item["score"], item["status"]),
                tags=("present" if known else "unknown",),
            )
        if not records:
            self.results.insert("", "end", values=("—", "No faces detected", "—", "—"))

    def _show_frame(self, frame: np.ndarray) -> None:
        self.last_annotated = frame.copy()
        height, width = frame.shape[:2]
        max_width, max_height = 840, 560
        scale = min(max_width / width, max_height / height, 1.0)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale))) if scale < 1 else frame
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.preview_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=self.preview_photo, text="")

    def _camera_tick(self) -> None:
        if not self.running or self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.session_status.configure(text="Unable to read from camera.")
            self._stop_camera()
            return
        annotated, records = self._analyze_frame(frame, enroll=self.enrolling is not None)
        self._show_frame(annotated)
        self._update_results(records)
        if self.enrolling is not None and len(self.enrollment_faces) >= SAMPLES_PER_ENROLLMENT:
            number, name = self.enrolling
            try:
                self.store.add_student(number, name, self.enrollment_faces)
                self._invalidate_profiles()
                self.session_status.configure(
                    text=f"Captured {len(self.enrollment_faces)} face crops for {name}. Select Train embeddings."
                )
            except sqlite3.IntegrityError:
                messagebox.showerror("Duplicate ID", "That student ID is already enrolled.")
            except OSError as error:
                messagebox.showerror("Could not save dataset", str(error))
            self.enrolling = None
            self._stop_camera(keep_preview=True)
            self._refresh_roster()
            return
        if any(item.get("newly_marked") for item in records):
            self._refresh_roster()
        if self.running:
            self.root.after(180, self._camera_tick)

    def _open_photo(self) -> None:
        if self.running:
            self._stop_camera()
        if self.detector is None or self.recognizer is None:
            messagebox.showerror("Face models missing", "Run python download_models.py before processing photos.")
            return
        path = filedialog.askopenfilename(
            title="Choose a group photograph",
            filetypes=[("Photographs", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Could not open photo", "This image format could not be read by OpenCV.")
            return
        if not self.prototypes:
            messagebox.showinfo("Train embeddings first", "Add student images and select Train embeddings before scanning.")
            return
        annotated, records = self._analyze_frame(frame)
        self.source_label.configure(text="GROUP PHOTO")
        self._show_frame(annotated)
        self._update_results(records)
        known = sum(item.get("student_id") is not None for item in records)
        unknown = len(records) - known
        self.session_status.configure(
            text=f"Scanned {len(records)} faces from {Path(path).name} · {known} identified and marked present · {unknown} unknown."
        )
        self._refresh_roster()

    def _stop_camera(self, keep_preview: bool = False) -> None:
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.enrolling = None
        self.camera_button.configure(text="Start camera")
        self.source_label.configure(text="PHOTO REVIEW" if keep_preview else "NO SOURCE")
        if not keep_preview:
            self.preview.configure(image="", text="Start the camera or open a group photograph")
            self.preview_photo = None

    def _import_student_photos(self) -> None:
        if self.running:
            messagebox.showinfo("Stop camera first", "Stop the live session before importing face photos.")
            return
        if self.detector is None or self.recognizer is None:
            messagebox.showerror("Face models missing", "Run python download_models.py before importing photos.")
            return
        selected = self.roster.selection()
        if not selected:
            messagebox.showinfo("Select a student", "Select the student whose face dataset you want to add to.")
            return
        student_id = int(next(row["id"] for row in self.store.students() if row["student_number"] == selected[0]))
        paths = filedialog.askopenfilenames(
            title="Import face photos for selected student",
            filetypes=[("Photographs", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not paths:
            return
        crops: list[bytes] = []
        skipped = 0
        for path in paths:
            image = cv2.imread(path)
            if image is None:
                skipped += 1
                continue
            faces = self._detect(image)
            if len(faces) != 1:
                skipped += 1
                continue
            try:
                crop = cv2.resize(self.recognizer.alignCrop(image, faces[0]), FACE_SIZE)
                ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 93])
                if ok:
                    crops.append(encoded.tobytes())
                else:
                    skipped += 1
            except cv2.error:
                skipped += 1
        if crops:
            try:
                self.store.add_dataset_images(student_id, crops)
                self._invalidate_profiles()
            except OSError as error:
                messagebox.showerror("Could not save dataset", str(error))
                return
        self.session_status.configure(text=f"Imported {len(crops)} face crops; skipped {skipped} image(s). Retrain embeddings to apply changes.")
        messagebox.showinfo("Dataset import complete", f"Added {len(crops)} face image(s). Skipped {skipped} image(s).")

    def _mark_selected_present(self) -> None:
        selected = self.roster.selection()
        if not selected:
            messagebox.showinfo("Select a student", "Choose a student in the roster first.")
            return
        number = selected[0]
        student = next(row for row in self.store.students() if row["student_number"] == number)
        self.store.mark_present(int(student["id"]))
        self.today_present_ids.add(int(student["id"]))
        self._refresh_roster()
        marked_at = self.store.marked_at(int(student["id"]))
        self.session_status.configure(text=f"Marked present manually: {student['name']} at {str(marked_at)[11:19]}.")

    def _remove_selected(self) -> None:
        if self.running:
            messagebox.showinfo("Stop camera first", "Stop the live session before removing a student.")
            return
        selected = self.roster.selection()
        if not selected:
            messagebox.showinfo("Select a student", "Choose a student in the roster first.")
            return
        student = next(row for row in self.store.students() if row["student_number"] == selected[0])
        if not messagebox.askyesno("Remove student", f"Delete {student['name']}, their face images, and attendance history?"):
            return
        student_id = int(student["id"])
        self.store.remove_student(student_id)
        self._invalidate_profiles()
        self.today_present_ids.discard(student_id)
        self._refresh_roster()
        self.session_status.configure(text=f"Removed {student['name']} and their local face dataset.")

    def _export_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export attendance report", defaultextension=".csv",
            initialfile=f"attendance-{date.today().isoformat()}.csv",
            filetypes=[("CSV file", "*.csv")],
        )
        if not path:
            return
        rows = self.store.report_rows()
        with open(path, "w", newline="", encoding="utf-8-sig") as output:
            writer = csv.writer(output)
            writer.writerow(["Date", "Student ID", "Name", "Status", "Marked at"])
            for row in rows:
                writer.writerow([date.today().isoformat(), row["student_number"], row["name"],
                                 "Present" if row["marked_at"] else "Absent", row["marked_at"] or ""])
        self.session_status.configure(text=f"CSV report exported · {len(rows)} student records.")

    def _export_excel(self) -> None:
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill
        except ImportError:
            messagebox.showerror("Excel export unavailable", "Install Excel support with: python -m pip install openpyxl")
            return
        path = filedialog.asksaveasfilename(
            title="Export attendance report", defaultextension=".xlsx",
            initialfile=f"attendance-{date.today().isoformat()}.xlsx",
            filetypes=[("Excel workbook", "*.xlsx")],
        )
        if not path:
            return
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Attendance"
        sheet.append(["Date", "Student ID", "Name", "Status", "Marked at"])
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="275EFE")
        for row in self.store.report_rows():
            sheet.append([date.today().isoformat(), row["student_number"], row["name"],
                          "Present" if row["marked_at"] else "Absent", row["marked_at"] or ""])
        for column, width in (("A", 16), ("B", 18), ("C", 30), ("D", 14), ("E", 24)):
            sheet.column_dimensions[column].width = width
        sheet.freeze_panes = "A2"
        workbook.save(path)
        self.session_status.configure(text=f"Excel report exported · {sheet.max_row - 1} student records.")

    def _close(self) -> None:
        self._stop_camera()
        self.store.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        AttendanceApp(root)
    except Exception as error:
        root.destroy()
        messagebox.showerror("Could not start attendance app", str(error))
        return
    root.mainloop()


if __name__ == "__main__":
    main()
