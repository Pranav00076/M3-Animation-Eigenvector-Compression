# The Mathematics Behind Image Compression

*How Eigenvectors and Linear Algebra Help Reduce Image Data*

A cinematic, mathematically rigorous Manim Community Edition documentary powered by real **NumPy**, **SciPy**, and **Pillow** computations.

## Project Structure

```text
image-compression-animation/
  assets/               # Input photograph (automatic procedural fallback if empty)
  src/
    image_processing.py     # Pillow loading, grayscale matrix A, patch extraction
    eigen_decomposition.py  # Symmetric A^T A eigendecomposition via numpy.linalg.eigh
    svd_compression.py      # SVD factorization A = U Sigma V^T & SciPy verification
    reconstruction.py       # Rank-k approximation A_k, storage counts, Frobenius error, PSNR
    pipeline.py             # End-to-end verified computation & asset exporter
  scenes/
    theme.py                # Shared documentary color palette & reusable Manim helpers
    scene01_intro.py        # 0:00-0:35 The mystery & pixel zoom
    scene02_pixels.py       # 0:35-1:15 Grayscale conversion & matrix A construction
    scene03_eigenvectors.py # 1:15-2:05 Symmetric A^T A, eigenvectors, basis change, Lambda
    scene04_svd.py          # 2:05-3:10 A = U Sigma V^T, lambda_i = sigma_i^2, spectrum
    scene05_reconstruction.py # 3:10-4:20 Real k=1,5,10,20,full reconstructions & errors
    scene06_conclusion.py   # 4:20-5:00 Side-by-side comparison, summary, and credits
  tests/
    test_math_pipeline.py   # Pytest mathematical verification suite
  output/               # Computed assets, metrics.json, rendered MP4 clips & final film
  render_all.py         # Orchestrates math verification, scene renders, and FFmpeg assembly
  requirements.txt
```

## Quick Start

1. **Test the mathematical pipeline first:**
   ```bash
   pytest tests/test_math_pipeline.py -v
   python3 -m src.pipeline
   ```

2. **Render Scene 1 prototype (fast 480p preview or 1080p HD):**
   ```bash
   manim -ql scenes/scene01_intro.py Scene01Intro
   manim -qh scenes/scene01_intro.py Scene01Intro
   ```

3. **Render all 6 scenes and assemble the final 1080p MP4:**
   ```bash
   python3 render_all.py --quality h
   ```
   For a rapid preview build across all scenes:
   ```bash
   python3 render_all.py --quality l
   ```
