import cv2
import numpy as np

# Fuse two color images.  Assume that zero indicates an unknown value.
# At pixels where both values are known, the output is the average of the two.
# At pixels where only one is known, the output uses that value.
def fuse_color_images(A, B):
    assert(A.ndim == 3 and B.ndim == 3)
    assert(A.shape == B.shape)

    # Allocate result image.
    C = np.zeros(A.shape, dtype=np.uint8)

    # Create masks for pixels that are not zero.
    A_mask = np.sum(A, axis=2) > 0
    B_mask = np.sum(B, axis=2) > 0

    # Compute regions of overlap.
    A_only = A_mask & ~B_mask
    B_only = B_mask & ~A_mask
    A_and_B = A_mask & B_mask

    C[A_only] = A[A_only]
    C[B_only] = B[B_only]
    C[A_and_B] = 0.5 * A[A_and_B] + 0.5 * B[A_and_B]

    return C

