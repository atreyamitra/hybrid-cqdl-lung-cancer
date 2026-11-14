# Data protocol

## Implemented input contract

The trainer accepts prepared .npy grayscale crops, not complete DICOM volumes. Each array must be finite, 128×128 and normalized into [0,1]. CSV fields: `path,patient_id,label,split` where split is optional. Preserve original patient identity across slices and nodules. Do not assign a new patient ID to every real image.

The loader trims stray spaces in every field and lower-cases split names, then rejects (with the CSV line number) missing fields, labels other than 0/1, non-.npy or pickled files, wrong shapes, NaN/infinite values, values outside [0,1], repeated paths, exact duplicate float32 crop content and patient IDs that differ only by letter case. The split validator rejects patient overlap and requires both classes in every partition. Duplicate hashes do not detect near-duplicates, and no check can detect a patient who was given two genuinely different IDs: keep the source identifier.

## LIDC-IDRI preparation still required

1. Obtain scans and annotations from the official collection: https://www.cancerimagingarchive.net/collection/lidc-idri/ . Retain the acquisition identifiers and comply with the collection's access/use conditions.
2. Match annotation coordinates to the correct scan series and nodule. Document reader aggregation and excluded scans/nodules.
3. Decide the target. Radiologist malignancy-likelihood ratings are not automatically pathology-confirmed cancer diagnoses. A possible future protocol uses a per-nodule median rating, assigns low scores ≤2 to class 0 and high scores ≥4 to class 1, and excludes intermediate values. This is a proposed rule, not an executed label-generation pipeline. Have the supervisor review the target before training.
4. Resample volumes with recorded voxel spacing, transform annotation coordinates consistently and extract axial nodule-centered crops. Record crop field of view and slice selection. These operations are not implemented in this release.
5. Apply a fixed, recorded CT window and scaling policy, resize to 128×128 and export .npy. Do not perform undocumented contrast operations that change the target appearance. No augmentation is used in this release.
6. Provide reviewed patient-level train/validation/test assignments. The fallback shuffles unique IDs once using `--split-seed` (default 42, independent of the training `--seed`) and allocates about 70/15/15 by patient count. It is not stratified. If it leaves a class absent, the trainer fails and requests a reviewed manifest instead of silently searching for a favorable split.
7. Keep every crop from a patient in one partition. Test labels must not determine epoch selection or thresholds. The trainer fixes threshold 0.5 and selects the lowest validation BCE checkpoint.
8. Train both models with the same manifest and settings. Preserve results with label provenance, preprocessing policy and exclusions.

The demo uses 24 independently generated geometric fixtures with mock IDs. Its 16/4/4 split is chosen for tiny fixtures and is not the proposed 70/15/15 patient study. Current metrics are crop-level. Multiple crops per patient require patient-level aggregation or clustered confidence intervals before patient-level performance claims.
