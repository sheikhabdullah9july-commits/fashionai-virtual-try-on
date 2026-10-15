# System review — 15 October 2026

## Verified

- Three regression tests pass: page rendering; valid upload/path boundaries; invalid image, missing input, unsupported try-on category, and unknown job handling.
- A real browser upload of a maroon shirt produced four complementary lower garments through the local CLIP model and catalog index.
- All four recommendation crop requests returned HTTP 200.
- The root inference environment used by `app.py` imports PyTorch 2.0.1+cu118, Diffusers 0.24.0, Transformers 4.36.2, and the customized OpenPose wrapper.
- Screenshots in this directory's `screenshots/` folder were captured from the running interface.

## Fixes applied

- Restrict recommendation and try-on source files to the upload directory.
- Verify image contents after upload and reject invalid files; cap requests at 16 MB.
- Serialize inference jobs to protect shared intermediate/output filenames.
- Discard subprocess stdout instead of leaving an unread pipe that can block inference.
- Remove stale pass output before inference so an old image cannot be reported as a new result.
- Disable debug mode by default and correct the sidebar framework label.

## Remaining limits

- A complete new two-pass, 40-step diffusion job was not run. CPU runtime and peak memory remain unmeasured.
- The extra nested `OOTDiffusion/ootd_env` fails to import Diffusers because of an incompatible Hugging Face Hub version. The app uses the root environment, which passed its import check.
- Fresh dependency installation was not reproduced in a clean environment. Existing pins are retained.
- Match percentages are heuristic and can saturate at 100; the interface applies a rank-based spread. Color/style scores do not come from independent classifiers.
- Outer/full-body categories have recommendations but no supported try-on category flow.
- Uploaded original filenames are replaced with UUIDs, so catalog/filename detection usually falls through to image/CLIP detection for browser uploads.
- In-memory jobs, upload retention, authentication, process isolation, and deployment hardening need further work before hosting a multi-user service.

## Repository contents

Source and documentation are included. Virtual environments, weights, datasets, catalog indexes, uploads, crops, personal base-model/product assets, and local machine diagnostics are excluded. OOTDiffusion source is vendored with its license and local modifications; its original nested Git metadata is preserved locally and not published.
