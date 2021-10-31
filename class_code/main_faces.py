import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

faces_folder = "att_faces"

def main():
    print("Eigenfaces")

    # Read all N images into a list, where each element is size [H,W].
    all_images = read_images(faces_folder)
    N = len(all_images)
    H,W = all_images[0].shape
    print("Read %d images." % N)

    # Randomly choose one face to use as a query image.
    idx = np.random.randint(0, N)
    query_image = all_images[idx]

    # For visualization, display a montage of all images.
    montage_image = montage(all_images, highlight_image_idx = idx)
    cv2.imshow("All faces", montage_image)

    # Remove the query image from the list of input images.
    del all_images[idx]
    N = N - 1

    # Display the query image.
    cv2.imshow("Query", query_image)

    # Put image data into a 2D array of vectors.  Output is size [H*W, N].
    all_vectors = np.array(all_images).reshape(N, H * W).T

    # Calculate the mean (average) vector.
    mean_vector = np.mean(all_vectors, axis=1)

    # Display the mean (average) face.
    mean_face = mean_vector.reshape(H,W)
    cv2.imshow("Mean", mean_face.astype(np.uint8))
    cv2.waitKey(10)

    # Subtract the mean vector from all vectors.
    all_vectors = all_vectors - mean_vector.reshape((H*W,1))

    M = all_vectors.T @ all_vectors     # This is size [N,N]

    # Get eigenvalues and eigenvectors.
    # There are N eigenvalues in w, sorted from highest to lowest.
    # There are N eigenvectors in v, each of length N.
    # Each column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    w, v = np.linalg.eig(M)

    # # Plot the eigenvalues.
    # fig = plt.figure()
    # plt.plot(w)
    # plt.show()

    # Calculate principal components.
    pcs = all_vectors @ v

    # Keep the top K pcs.
    K = 16
    pcs = pcs[:,0:K]

    print("Each PC accounts for this percent of the variance in the input data:")
    variance_explained = []
    for i in w[0:K]:
        variance_explained.append((i/sum(w))*100)
    print(variance_explained)

    # Display the "eigenimages".
    eigenimages = []
    for k in range(K):
        img = pcs[:,k].reshape((H, W))
        img = cv2.normalize(src = img, dst = None, alpha = 0, beta = 255,
                      norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
        eigenimages.append(img)
    montage_image = montage(eigenimages, shrink_factor=1.0)
    cv2.imshow("Eigenimages", montage_image)

    # Here are the input vectors projected onto the principal components.
    signatures = np.zeros((N,K))
    for i in range(N):
        # Each row corresponds to an input vector, and contains the K values resulting
        # from projecting the input vector onto the K principal components.
        signatures[i, :] = all_vectors[:,i].T @ pcs    # Each row is an image signature

    # Prepare the query face, by making it a vector and subtracting the mean.
    query_vector = query_image.reshape(H * W) - mean_vector

    # Project the query vector onto the space of principal components.
    query_projected = query_vector @ pcs      # Result is size (K,)

    print("Here is the signature of the query face.")
    print("These are the coefficients of the query face projected onto the %d PCs:" % K)
    print(query_projected)

    # Ok, compare the query projection vector with all rows in "signatures".
    diff = signatures - query_projected
    euclidean_distances = np.linalg.norm(diff, axis=1)
    idx_sorted = np.argsort(euclidean_distances)
    print("Top matches:", idx_sorted[0:10])

    # Get the images for the top 4 matches.
    matched_images = [all_images[id] for id in idx_sorted[0:4] ]
    montage_image = montage(matched_images, shrink_factor=1.0)
    cv2.imshow("Top matches", montage_image)

    cv2.waitKey(0)

    # # Now, do a brute force calculation of the Euclidean distances.
    # diffs = all_vectors.T - query_vector.T      # This is size (N,H*W)
    # euclidean_distances_brute_force = np.linalg.norm(diffs, axis=1)
    # idx_sorted = np.argsort(euclidean_distances_brute_force)
    # print("Top matches brute force:", idx_sorted[0:10])
    #
    # # Get the images for the top 4 matches.
    # matched_images = [all_images[id] for id in idx_sorted[0:4] ]
    # montage_image = montage(matched_images, shrink_factor=1.0)
    # cv2.imshow("Top matches brute force", montage_image)
    # cv2.waitKey(0)

    print("All done, bye!")


def read_images(faces_folder):
    all_images = []
    for subfolder in os.listdir(faces_folder):
        path_to_folder = os.path.join(faces_folder, subfolder)
        if os.path.isdir(path_to_folder):
            for filename in os.listdir(path_to_folder):
                path_to_file = os.path.join(path_to_folder, filename)
                gray_img = cv2.imread(path_to_file, cv2.COLOR_BGR2GRAY)
                all_images.append(gray_img)
    return all_images

# Create a "montage" image of small images.
def montage(all_images, shrink_factor=0.5, highlight_image_idx=None):
    count = len(all_images)
    shrunken_image = cv2.resize(all_images[0], dsize=None, fx=shrink_factor, fy=shrink_factor)
    m, n = shrunken_image.shape

    mm = int(math.ceil(math.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n), dtype=np.uint8)

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count:
                break
            shrunken_image = cv2.resize(all_images[image_id], dsize=None, fx=shrink_factor, fy=shrink_factor)

            # Black out this image if desired.
            if not highlight_image_idx is None and image_id == highlight_image_idx:
                shrunken_image = 0

            sliceM, sliceN = j * m, k * n
            M[sliceM:sliceM + m, sliceN:sliceN + n] = shrunken_image
            image_id += 1
    return M

if __name__ == "__main__":
    main()
