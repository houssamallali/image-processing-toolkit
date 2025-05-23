# TP7: Image Registration using ICP

This project implements image registration using the Iterative Closest Point (ICP) algorithm and manual point selection.

## Theoretical Background
Rigid image registration seeks a rotation R and translation t that best align two sets of corresponding points {p_i} and {q_i}, by minimizing the least-squares criterion:
\[
 C(R,t) = \sum_i \|q_i - (R\,p_i + t)\|^2.
\]
First, compute the centroids:
\[
 \bar p = \frac1N\sum_i p_i,
 \quad \bar q = \frac1N\sum_i q_i.
\]
Define centered points p'_i = p_i - \bar p,  q'_i = q_i - \bar q.  Form the correlation matrix
\[
 K = \sum_i q'_i\,{p'_i}^T.
\]
Perform Singular Value Decomposition (SVD):
\[
 U\,\Sigma\,V^T = K.
\]
Then the optimal rotation is
\[
 R = U\;\mathrm{diag}(1,\dots,1,\det(UV^T))\;V^T,
\]
and the translation follows:
\[
 t = \bar q - R\,\bar p.
\]
In ICP, one iteratively finds closest point correspondences between two point clouds and re-estimates (R,t) until convergence.

## Files

- `image_registration.py`: Main implementation of registration algorithms
  - Includes ICP implementation
  - Rigid transformation estimation
  - Control point extraction
  - Visualization functions

- `manual_registration_test.py`: Interactive test script
  - Demonstrates manual point selection
  - Shows registration results visually
  - Useful for testing and understanding the registration process

## Usage

1. Run the manual registration test:
```bash
python manual_registration_test.py
```
- Click to select 4 corresponding points in both images
- Points should be selected in the same order
- Results will be saved in `../plots/TP7_Registration/`

## Dependencies

Install dependencies via:
```bash
pip install numpy matplotlib scikit-image opencv-python
```
If you prefer not to use a virtual environment, install them globally using the same command.

## Output

Results are saved in the `../plots/TP7_Registration/` directory:
- `manual_registration_result.png`: Visualization of registration results
- Other plots showing registration process and results 