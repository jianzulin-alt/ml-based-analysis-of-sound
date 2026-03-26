# TODO

##  High Priority: Infrastructure & Critical Bug Fixes
- [ ] **Fix Sample Rate Mismatch:** Update `src/preprocessing/audio_io.py` or a standalone script to resample all audio (Train, Test, IRMAS) to a uniform **44,100 Hz**.
- [x] **Solve Label Shuffling Bug:** Ensure `run_train.py` sorts classes alphabetically on fresh runs and locks them during resume.
- [ ] **Verify Preprocessing Consistency:** Double-check that `run_eval.py` is pulling the "Time Capsule" `audio_params` from `run_config.yaml` rather than the global file.
- [x] **Manifest Path Fix:** Ensure `run_train.py` correctly resolves paths to `data/processed/` instead of `tmp_manifests/`.

---

## Dataset Cleaning & Preparation
- [x] **Silent File Cleanup:** Run analysis script on Chinese dataset to identify and delete silent/corrupt clips.
- [x] **Visualise Features:** Complete visualisation of Chinese instrument Mel/CQT features for the report.
- [ ] **The "Open-Set" Mapping:** - [ ] Create a mapping dictionary to group the ~41 film test classes into your 14 training classes.
    - [ ] Map irrelevant/unseen classes (e.g., `banzi_clapper`, `wind_chimes`) to an **"Other/OOD"** category.
- [ ] **Class Expansion:** Decide whether to add `voice` or other high-frequency film elements to the training set.

---

## Phase 2: Handling Class Imbalance
- [ ] **Loss Function Upgrade:** Replace standard Cross-Entropy with **Focal Loss** to prioritise rare instruments like `yangqin`.
- [ ] **Advanced Augmentation:** - [ ] Implement **SpecAugment** (time/frequency masking) in the `FeatureFusionDataset`.
    - [ ] Implement **Mixup** (mathematical blending of samples) for robust feature learning.

---

##  Phase 3: Transfer Learning Strategy (IRMAS ➡️ Chinese)
- [ ] **IRMAS Pre-training:** Train the DenseNet-121 or CNN backbone on the full IRMAS set.
- [ ] **"Freeze & Fine-Tune" Implementation:**
    - [ ] Replace the 11-class IRMAS head with the 14-class Chinese head.
    - [ ] Initial run: Freeze early conv layers, train only the head.
    - [ ] Final run: Unfreeze all layers with a very low learning rate ($1e-5$).

---

##  Phase 4: Domain Adaptation (The "A Touch of Zen" Fix)
- [ ] **Background Noise Injection:** Overlay clean training samples with "silence" (ambient room tone) extracted from the film.
- [ ] **Reverb Augmentation:** Apply artificial impulse responses to training data to match the 1970s cinematic reverb.
- [ ] **Multi-Label Transition:** Update the final layer to **Sigmoid** activation to handle overlapping instruments in film scores.

---

## Evaluation & Honours Reporting
- [ ] **Honours Metrics:** Ensure `run_eval.py` focuses on **Macro F1-Score** and Per-Class Recall.
- [ ] **Confusion Matrix Analysis:** Generate and save high-res matrices for both IRMAS and Film test sets.
- [ ] **Ablation Study:** Compare performance between the standard CNN baseline and the DenseNet-121.

---

### Past bugs
- [ ] Label shuffling corrupts training
- [ ] DSP drift
