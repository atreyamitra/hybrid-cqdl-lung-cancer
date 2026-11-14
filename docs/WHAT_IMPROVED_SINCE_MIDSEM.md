# “What improved since midsem?” — viva answer

**About 30 seconds:**

“Since midsem we turned the prototype into something you can use and test fairly. First, there is a local app. You pick a saved model, give it a prepared crop, and it shows the image, the model score, the 0.5 threshold and the predicted class. It rejects photos and DICOM files instead of pretending to handle them, and it labels everything as synthetic and uncalibrated. Second, we added a matched classical comparator. It is identical to the hybrid except that a small Linear(8,8)+tanh block replaces the quantum layer, so the comparison is no longer 8 features against 128. Third, circuit depth is now configurable, and a small experiment runs the matched model and the 1-layer and 3-layer hybrids with three seeds on the same split. The results page reads those saved results directly.”

**If asked “so which model is better?”:**

“We can't say yet. Each run has four synthetic test images, and the spread across seeds is as large as the differences between models. For example, the matched model's accuracy is 0.75 ± 0.25, and the 1-layer hybrid ranges from 0.5 to 1.0 depending on the seed. What we can say is that the 3-layer hybrid predicted every test image positive under all three seeds, and our diagnostic shows its quantum outputs barely vary between images. Real CT data and more test cases are next semester's work.”

**If asked what “midsem” refers to:**

Before this work, GitHub `main` (15 September 2026) already had the validated data pipeline, seeded training, saved evidence, the prediction CLI, 12 tests and a static HTML demo. The supplied files do not record exactly what was shown at the midsem review, so say “the version on GitHub before the end-semester work”.
