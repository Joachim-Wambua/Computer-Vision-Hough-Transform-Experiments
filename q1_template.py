"""
Task 1: Line Hough transform.
Template for MA-INF 2201 Computer Vision WS26/27
Exercise 03
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
THETA_MIN = 50
THETA_MAX = 270
RHO_MIN = int(-1 * (2 ** 0.5) * 128)
RHO_MAX = int(+1 * (2 ** 0.5) * 128)


def generate_accumulator(edges: np.ndarray, nd_theta: np.ndarray, nd_rho: np.ndarray) -> np.ndarray:
    """Compute accumulator matrix.

    Args:
        edges (np.ndarray): Centers of points.
        nd_theta (np.ndarray): Range of theta values.
        nd_rho (np.ndarray): Range of rho values.

    Returns:
        np.ndarray: Accumulator matrix.
    """
    # TODO: Compute Accumulator matrix.
    # The implementation here is also will be used in next task. Make implementation generic.
    # Create accumulator matrix filled with zeros - voting grid
    accumulator = np.zeros((len(nd_rho), len(nd_theta)), dtype=np.int32)

    # Distane between consecutive rho bins
    rho_step = nd_rho[1] - nd_rho[0]

    # Loop through image points.
    for point in edges:
        x,y = point

        # Test all possible lines
        for theta_id, theta_angle in enumerate(nd_theta):
            # Convert degrees to radians
            theta_rad = np.deg2rad(theta_angle)

            # Hough Normal Representation Formula
            rho = x * np.cos(theta_rad) + y * np.sin(theta_rad)

            # Convert the calculated rho into an array index
            rho_id = int(
                np.rint((rho - nd_rho[0]) / rho_step)
            )

            # Only count votes within the valid rho range
            if 0 <= rho_id < len(nd_rho):
                accumulator[rho_id, theta_id] += 1

    return accumulator


def main() -> None:
    """Main function."""
    # TODO: Read control image.
    control_image = cv2.imread("data/control.png", cv2.IMREAD_GRAYSCALE)
    if control_image is None:
        raise FileNotFoundError(f"Input Image Not Found!")

    # TODO: Define centers of points and range of theta and rhos.
    binary_mask = (control_image>0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    centers = centroids[1:] # ignore index 0 since it is the background
    nd_theta = np.arange(THETA_MIN,THETA_MAX)
    nd_rho = np.arange(RHO_MIN,RHO_MAX+1)

    accumulator = generate_accumulator(
        centers,
        nd_theta,
        nd_rho,
    )

    # TODO: Perform accumulator thresholding.
    peak_indices = np.argwhere(accumulator >= 3)

    print(f"Point Centers: {centers}")
    print(f"Max Votes: {accumulator.max()}")

    for rho_id, theta_id in peak_indices:
        rho = nd_rho[rho_id]
        theta = nd_theta[theta_id]
        votes = accumulator[rho_id, theta_id]
        print(f"rho={rho}, theta={theta}°, votes={votes}")

    # TODO: Plot lines and hough spaces.
    fig, (ax_image, ax_hough) = plt.subplots(1, 2, figsize=(12, 5))
    height, width = control_image.shape

    ax_image.imshow(control_image, cmap="gray", vmin=0, vmax=255)
    ax_image.set_title("Detected lines on control image")

    heatmap = ax_hough.imshow(
        accumulator,
        origin="lower",
        aspect="auto",
        extent=[
            nd_theta[0] - 0.5, nd_theta[-1] + 0.5,
            nd_rho[0] - 0.5, nd_rho[-1] + 0.5,
        ],
        cmap="gray",
    )
    ax_hough.set_title("Hough accumulator")
    ax_hough.set_xlabel("Theta (degrees)")
    ax_hough.set_ylabel("Rho (pixels)")
    fig.colorbar(heatmap, ax=ax_hough, label="Votes")

    # Draw a long segment for each line and let the axes clip it.
    span = np.hypot(width, height)
    t = np.array([-span, span])

    for rho_id, theta_id in peak_indices:
        rho = nd_rho[rho_id]
        theta = nd_theta[theta_id]
        angle = np.deg2rad(theta)

        ax_hough.scatter(theta, rho, facecolors="none", edgecolors="red")

        x = rho * np.cos(angle) - t * np.sin(angle)
        y = rho * np.sin(angle) + t * np.cos(angle)
        ax_image.plot(x, y, color="red", linewidth=1)

    ax_image.set_xlim(-0.5, width - 0.5)
    ax_image.set_ylim(height - 0.5, -0.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()