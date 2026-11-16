"""
Task 2: Roll correction with Hough Transform.
Template for MA-INF 2201 Computer Vision WS26/27
Exercise 03
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from q1_template import generate_accumulator

# Constants
# Do not change these
THETA_MIN = 75
THETA_MAX = 105

def correct_roll(
        accumulator: np.ndarray,
        theta_deg: np.ndarray,
        nd_rho: np.ndarray
    ) -> tuple[list[tuple[float, float]], float]:
    """Roll correction using topk dominant lines.

    Assume we consider horizontal lines and roll angle is within 15 degrees.
    Args:
        accumulator (np.ndarray): Accumulator cells.
        theta_deg (np.ndarray): Range of thetas in degrees.
        nd_rho (np.ndarray): Range of rhos.

    Returns:
        tuple[list[tuple[float, float]], float]: Tuple containing topk lines and correction angle.
    """
    # TODO: Sort the accumulater cells by votecount in descending order.
    votes = accumulator.ravel() # Flatten voting grid to 1D array
    ranked_indices = np.argsort(-votes) # descending order

    rho_tolerance = 10  # Pixels
    theta_tolerance = 2  # Degrees
    selected_lines = []

    # TODO: Select the top-k dominant distinct lines and calculate the mean theta of the selected lines.
    k = 10
    for flat_index in ranked_indices[:k]:
        vote_count = votes[flat_index]
        if vote_count == 0:
            break

        # Recover bin's rho(row) & theta(col)
        rho_id, theta_id = np.unravel_index(flat_index, accumulator.shape)

        # Use indies to find rho & theta values
        rho = nd_rho[rho_id]
        theta = theta_deg[theta_id]

        # Skip the candidate if both values are close to a line we kept
        is_duplicate = any(
            abs(rho - kept_rho) <= rho_tolerance
            and abs(theta - kept_theta) <= theta_tolerance
            for kept_rho, kept_theta in selected_lines
        )
        if is_duplicate:
            continue

        selected_lines.append((rho, theta))
        print(f"rho={rho} | theta={theta} degrees | votes={vote_count}")

        if len(selected_lines) == k:
            break

    if not selected_lines:
        return [], 0.0

    mean_theta = float(np.mean([theta for i, theta in selected_lines]))

    # TODO: Calculate correction angle and return the top-k lines and correction angle
    # A horizontal line has a normal angle of 90 degrees
    correction_angle = mean_theta - 90.0
    return selected_lines, correction_angle



def main() -> None:
    """Main function."""
    # cv2 read parallel.jpg and convert to grayscale.
    parallel_img = cv2.imread("data/parallel.jpg", cv2.IMREAD_GRAYSCALE)
    if parallel_img is None:
        raise FileNotFoundError("Input Image Not Found!")

    # TODO: Canny edge
    canny_edges = cv2.Canny(parallel_img, 50, 150)
    height, width = canny_edges.shape
    # TODO: Extract coordinates of the edges (similar to previous task)
    rows, cols = np.nonzero(canny_edges)
    edges = np.column_stack((cols, rows))

    theta_deg = np.arange(THETA_MIN, THETA_MAX)
    theta_rad = np.deg2rad(theta_deg)

    rho_max = int(np.ceil(np.hypot(width - 1, height - 1)))
    rho_min = -rho_max
    nd_rho = np.arange(rho_min, rho_max)

    # TODO: Get the accumulator matrix (use the function from previous task)
    accumulator = generate_accumulator(edges, theta_deg, nd_rho)

    # TODO: Get the lines and correction angle from the correct_roll.
    lines, correction_angle = correct_roll(accumulator, theta_deg, nd_rho)
    print(f"Correction angle: {correction_angle:+.2f} degrees")

    # TODO: Compute rotation matrix (cv2.getRotationMatrix2D) and correct image roll (cv2.warpAffine).
    center = ((width - 1) / 2, (height - 1) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
    corrected_img = cv2.warpAffine(parallel_img, rotation_matrix, (width, height))

    # TODO: Plot the canny edge and top-k lines and unwrapped image.
    fig, (ax_edges, ax_corrected, ax_hough) = plt.subplots(
        1, 3, figsize=(15, 5)
    )

    ax_edges.imshow(canny_edges, cmap="gray", vmin=0, vmax=255)
    ax_edges.set_title("Canny edges and dominant lines")

    # Reuse the line drawing formula from Task 1.
    # Start at rho times the unit normal and move along the line by t.
    span = np.hypot(width, height)
    t = np.array([-span, span])

    for rho, theta in lines:
        angle = np.deg2rad(theta)
        x = rho * np.cos(angle) - t * np.sin(angle)
        y = rho * np.sin(angle) + t * np.cos(angle)
        ax_edges.plot(x, y, color="red", linewidth=1)

    # Clip the view to the image and keep row 0 at the top.
    ax_edges.set_xlim(-0.5, width - 0.5)
    ax_edges.set_ylim(height - 0.5, -0.5)

    ax_corrected.imshow(corrected_img, cmap="gray", vmin=0, vmax=255)
    ax_corrected.set_title(f"Corrected image ({correction_angle:+.2f}°)")
    ax_corrected.axis("off")

    # TODO: Plot the hough space image and mark the top-k lines in hough space.
    heatmap = ax_hough.imshow(
        accumulator,
        origin="lower",
        aspect="auto",
        cmap="gray",
        extent=[
            theta_deg[0] - 0.5, theta_deg[-1] + 0.5,
            nd_rho[0] - 0.5, nd_rho[-1] + 0.5,
        ],
    )

    # The accumulator plot uses theta on the x-axis and rho on the y-axis.
    for rho, theta in lines:
        ax_hough.scatter(theta, rho, facecolors="none", edgecolors="red")

    ax_hough.set_title("Hough accumulator")
    ax_hough.set_xlabel("Theta (degrees)")
    ax_hough.set_ylabel("Rho (pixels)")
    fig.colorbar(heatmap, ax=ax_hough, label="Votes")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
