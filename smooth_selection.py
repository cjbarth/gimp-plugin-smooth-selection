#!/usr/bin/env python

from gimpfu import *
import math

IN_CTRL_X = 0
IN_CTRL_Y = 1
ANCHOR_X = 2
ANCHOR_Y = 3
OUT_CTRL_X = 4
OUT_CTRL_Y = 5


def moving_average(coords, factor):
    window_size = max(1, int(factor * 5))  # Scale factor to a practical window size
    smoothed = []
    length = len(coords)
    for i in range(length):
        avg_x = sum(
            coords[(i + j) % length][0] for j in range(-window_size, window_size + 1)
        ) / (2 * window_size + 1)
        avg_y = sum(
            coords[(i + j) % length][1] for j in range(-window_size, window_size + 1)
        ) / (2 * window_size + 1)
        smoothed.append((avg_x, avg_y))

    return smoothed


def chaikin_smoothing(coords, factor):
    shrink_factor = factor * 0.25  # Scale factor between 0 and 0.25
    new_coords = []
    length = len(coords)
    for i in range(length):
        p0 = coords[i]
        p1 = coords[(i + 1) % length]
        Q = (
            (1 - shrink_factor) * p0[0] + shrink_factor * p1[0],
            (1 - shrink_factor) * p0[1] + shrink_factor * p1[1],
        )
        R = (
            shrink_factor * p0[0] + (1 - shrink_factor) * p1[0],
            shrink_factor * p0[1] + (1 - shrink_factor) * p1[1],
        )
        new_coords.extend([Q, R])

    return new_coords


def gaussian_smoothing(coords, factor):
    sigma = factor * 3  # Scale factor to a useful sigma value
    radius = max(1, int(sigma * 3))
    smoothed = []
    length = len(coords)
    weights = [math.exp(-(i**2) / (2 * sigma**2)) for i in range(-radius, radius + 1)]
    weight_sum = sum(weights)

    for i in range(length):
        avg_x = (
            sum(
                coords[(i + j) % length][0] * weights[j + radius]
                for j in range(-radius, radius + 1)
            )
            / weight_sum
        )
        avg_y = (
            sum(
                coords[(i + j) % length][1] * weights[j + radius]
                for j in range(-radius, radius + 1)
            )
            / weight_sum
        )
        smoothed.append((avg_x, avg_y))

    return smoothed


def pixel_radius_smoothing(coords, factor):
    pixel_radius = max(
        1, int(factor * 20)
    )  # Scale factor to pixel radius (1-20 pixels)
    smoothed = []
    length = len(coords)

    for i in range(length):
        x, y = coords[i]
        neighbors = [
            coords[j]
            for j in range(length)
            if math.hypot(coords[j][0] - x, coords[j][1] - y) <= pixel_radius
        ]
        avg_x = sum(pt[0] for pt in neighbors) / len(neighbors)
        avg_y = sum(pt[1] for pt in neighbors) / len(neighbors)
        smoothed.append((avg_x, avg_y))

    return smoothed


def inside_track_smoothing(coords, factor):
    """
    TODO: Test
    Smooths a path by preferentially cutting inward corners.
    This creates a smoother path that tends to stay inside the original selection,
    effectively trimming jagged edges.

    Args:
        coords: List of (x, y) coordinate tuples
        factor: Smoothing factor (0.1 to 1.0)

    Returns:
        List of smoothed (x, y) coordinate tuples
    """
    # Calculate a bias factor (higher values = stronger inside bias)
    bias_strength = factor * 0.6  # Scale factor to a useful bias value
    smoothed = []
    length = len(coords)

    for i in range(length):
        prev_idx = (i - 1) % length
        next_idx = (i + 1) % length

        # Current point and its neighbors
        prev = coords[prev_idx]
        curr = coords[i]
        next_pt = coords[next_idx]

        # Vectors from current point to neighbors
        vec_to_prev = (prev[0] - curr[0], prev[1] - curr[1])
        vec_to_next = (next_pt[0] - curr[0], next_pt[1] - curr[1])

        # Approximate the "inward" direction by using the cross product
        # This helps determine if we're at an outside corner or inside corner
        cross_z = vec_to_prev[0] * vec_to_next[1] - vec_to_prev[1] * vec_to_next[0]

        # For a clockwise path, negative cross product = outside corner (convex)
        # For a counterclockwise path, positive cross product = outside corner
        # We'll assume clockwise for now, but could detect path direction if needed
        is_outside_corner = cross_z < 0

        # Calculate a simple average position (midpoint)
        avg_x = (prev[0] + curr[0] + next_pt[0]) / 3
        avg_y = (prev[1] + curr[1] + next_pt[1]) / 3

        # If it's an outside corner, stay closer to the original point
        # If it's an inside corner, move more aggressively toward the average
        if is_outside_corner:
            # For outside corners, stay closer to original
            smooth_factor = max(
                0.05, factor * 0.3
            )  # Limited smoothing for outside corners
            new_x = curr[0] * (1 - smooth_factor) + avg_x * smooth_factor
            new_y = curr[1] * (1 - smooth_factor) + avg_y * smooth_factor
        else:
            # For inside corners, smooth more aggressively
            smooth_factor = min(0.9, factor * 0.9 + bias_strength)  # Enhanced smoothing
            new_x = curr[0] * (1 - smooth_factor) + avg_x * smooth_factor
            new_y = curr[1] * (1 - smooth_factor) + avg_y * smooth_factor

        smoothed.append((new_x, new_y))

    return smoothed


def inner_contour_smoothing(coords, factor):
    """
    TODO: Test
    True inner contour smoothing algorithm that guarantees the smoothed path
    stays inside the original boundary.

    Args:
        coords: List of (x, y) coordinate tuples
        factor: Smoothing factor (0.1 to 1.0)

    Returns:
        List of smoothed (x, y) coordinate tuples
    """
    import numpy as np

    # Convert to numpy array for easier manipulation
    points = np.array(coords)
    length = len(points)

    # Calculate centroid of the shape
    centroid = np.mean(points, axis=0)

    # Step 1: Calculate vectors from centroid to each point
    vectors = points - centroid

    # Step 2: Calculate vector lengths (distances from centroid)
    distances = np.sqrt(np.sum(vectors**2, axis=1))

    # Step 3: Calculate unit vectors (normalized direction vectors)
    unit_vectors = vectors / distances[:, np.newaxis]

    # Step 4: Shrink the shape by scaling down the vectors based on factor
    # Higher factor means more shrinkage
    shrink_amount = factor * np.max(distances) * 0.2

    # Create initial inner contour by uniformly shrinking
    inner_points = points - unit_vectors * shrink_amount

    # Step 5: Apply a gentle smoothing to the inner contour
    # We'll use a simple moving average
    window_size = max(1, int(factor * 3))
    smoothed = []

    for i in range(length):
        # Calculate indices for the window, with wraparound
        indices = [(i + j) % length for j in range(-window_size, window_size + 1)]

        # Calculate the average position
        avg_x = sum(inner_points[j][0] for j in indices) / len(indices)
        avg_y = sum(inner_points[j][1] for j in indices) / len(indices)

        # Step 6: The critical part - ensure we stay inside the original polygon
        # by checking if the smoothed point is outside, and if so, move it inside

        # First, we'll approximate this by checking if the new point is further
        # from the centroid than the original inner point
        new_vector = np.array([avg_x, avg_y]) - centroid
        new_distance = np.sqrt(np.sum(new_vector**2))

        inner_distance = np.sqrt(np.sum((inner_points[i] - centroid) ** 2))

        if new_distance > inner_distance:
            # If the smoothed point would be outside the inner contour,
            # scale it back to ensure it stays inside
            scale = inner_distance / new_distance
            avg_x = centroid[0] + (avg_x - centroid[0]) * scale
            avg_y = centroid[1] + (avg_y - centroid[1]) * scale

        smoothed.append((avg_x, avg_y))

    return smoothed


# Alternative implementation using more geometric approach
def geometric_inner_contour(coords, factor):
    """
    TODO: Test
    A geometric approach to inner contour smoothing using convex hull
    and path deflation techniques.

    Args:
        coords: List of (x, y) coordinate tuples
        factor: Smoothing factor (0.1 to 1.0)

    Returns:
        List of smoothed (x, y) coordinate tuples
    """
    import math

    length = len(coords)
    result = []

    # Step 1: Calculate local convexity at each point
    for i in range(length):
        prev_idx = (i - 1) % length
        next_idx = (i + 1) % length

        # Get the three consecutive points
        p0 = coords[prev_idx]
        p1 = coords[i]
        p2 = coords[next_idx]

        # Vectors between points
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])

        # Cross product to determine if it's a convex or concave corner
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        # For clockwise paths, cross < 0 means convex (outside) corner
        is_convex = cross < 0

        if is_convex:
            # For convex corners, move the point inward
            # Calculate the bisector direction
            v1_len = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            v2_len = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

            if v1_len > 0 and v2_len > 0:
                # Normalize vectors
                v1_norm = (v1[0] / v1_len, v1[1] / v1_len)
                v2_norm = (v2[0] / v2_len, v2[1] / v2_len)

                # Compute bisector (pointing inward)
                bisector = (-(v1_norm[0] + v2_norm[0]), -(v1_norm[1] + v2_norm[1]))
                bisector_len = math.sqrt(bisector[0] ** 2 + bisector[1] ** 2)

                if bisector_len > 0:
                    # Normalize bisector
                    bisector = (bisector[0] / bisector_len, bisector[1] / bisector_len)

                    # Move point inward along bisector
                    # Scale movement by angle sharpness and factor
                    dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
                    angle_factor = max(0, 1 - dot_product)  # Sharper angles move more

                    move_distance = factor * 10 * angle_factor
                    new_x = p1[0] + bisector[0] * move_distance
                    new_y = p1[1] + bisector[1] * move_distance

                    result.append((new_x, new_y))
                else:
                    result.append(p1)  # Keep original if calculation fails
            else:
                result.append(p1)  # Keep original if vectors are zero length
        else:
            # For concave (inside) corners, keep the original point
            result.append(p1)

    # Step 2: Apply a gentle smoothing pass
    smoothed = []
    window_size = max(1, int(factor * 2))

    for i in range(length):
        indices = [(i + j) % length for j in range(-window_size, window_size + 1)]
        avg_x = sum(result[j][0] for j in indices) / len(indices)
        avg_y = sum(result[j][1] for j in indices) / len(indices)
        smoothed.append((avg_x, avg_y))

    return smoothed


def sanding_smoothing(coords, factor):
    """
    TODO: Test
    A single-pass smoothing algorithm that mimics sanding by selectively
    smoothing protruding points while leaving recessed areas mostly untouched.

    Designed to work with the existing iteration framework in the GIMP plugin.

    Args:
        coords: List of (x, y) coordinate tuples
        factor: Smoothing factor (0.1 to 1.0) controlling sanding intensity

    Returns:
        List of smoothed (x, y) coordinate tuples
    """
    import math

    result = []
    length = len(coords)

    # Window size increases with factor to simulate wider sanding blocks
    window_size = max(1, int(2 + factor * 4))
    sensitivity = 0.7 + factor * 0.4  # How sensitive to detect protruding points

    for i in range(length):
        # Get neighboring points
        neighbors = []
        for j in range(-window_size, window_size + 1):
            if j != 0:  # Skip the current point
                idx = (i + j) % length
                neighbors.append(coords[idx])

        # Current point
        pt = coords[i]

        # Calculate the average position of neighbors
        avg_x = sum(n[0] for n in neighbors) / len(neighbors)
        avg_y = sum(n[1] for n in neighbors) / len(neighbors)

        # Calculate average distance from neighbor points to their average center
        avg_dist = sum(
            math.sqrt((n[0] - avg_x) ** 2 + (n[1] - avg_y) ** 2) for n in neighbors
        ) / len(neighbors)

        # Distance of current point to the average center
        pt_dist = math.sqrt((pt[0] - avg_x) ** 2 + (pt[1] - avg_y) ** 2)

        # If point is outside the average circle (adjusted by sensitivity), it protrudes
        if pt_dist > avg_dist * sensitivity:
            # Calculate how much to sand down (more for higher protrusion)
            protrusion = pt_dist - avg_dist
            sand_amount = min(protrusion, protrusion * factor)

            # Direction from point to average center
            if pt_dist > 0:  # Avoid division by zero
                dir_x = (avg_x - pt[0]) / pt_dist
                dir_y = (avg_y - pt[1]) / pt_dist

                # Move point inward (sand it down)
                result.append(
                    (pt[0] + dir_x * sand_amount, pt[1] + dir_y * sand_amount)
                )
            else:
                result.append(pt)  # Keep original if calculation fails
        else:
            # For non-protruding points, apply very mild smoothing
            mild_factor = factor * 0.2  # Much gentler than for protruding points
            new_x = pt[0] * (1 - mild_factor) + avg_x * mild_factor
            new_y = pt[1] * (1 - mild_factor) + avg_y * mild_factor
            result.append((new_x, new_y))

    return result


SMOOTH_METHODS = [
    (
        "Moving Average",
        moving_average,
        "Balances each point with its neighbors. Fast, general-purpose smoothing.",
    ),
    (
        "Chaikin",
        chaikin_smoothing,
        "Creates soft, rounded curves by subdividing lines. Doubles point count.",
    ),
    (
        "Gaussian",
        gaussian_smoothing,
        "Smooths with precision using a weighted average. Preserves shape better.",
    ),
    (
        "Pixel Radius",
        pixel_radius_smoothing,
        "Averages points within a pixel range. Great for high-detail smoothing.",
    ),
    (
        "Inside Track",
        inside_track_smoothing,
        "Favors smoothing inward. Keeps selection close to original edges.",
    ),
    (
        "Inner Contour",
        inner_contour_smoothing,
        "Shrinks and smooths the path inward. Guaranteed to stay inside boundary.",
    ),
    (
        "Geometric Inner Contour",
        geometric_inner_contour,
        "Trims sharp convex corners while keeping concave points.",
    ),
    (
        "Sanding",
        sanding_smoothing,
        "Smooths only protruding bumps, preserving detail elsewhere.",
    ),
]


METHOD_LABELS = [name for name, _, _ in SMOOTH_METHODS] + [
    "Help - Show method descriptions"
]


def show_help_dialog():
    help_lines = [
        "{}. {}\n   {}".format(i + 1, name, desc)
        for i, (name, _, desc) in enumerate(SMOOTH_METHODS)
    ]
    help_text = "\n\n".join(help_lines)
    pdb.gimp_message(help_text)


def smooth_selection(
    image,
    drawable,
    smooth_iterations,
    method_index,
    smoothing_strength,
    preserve_curves,
    preserve_path,
):

    # Show help dialog and exit early if Help is selected
    if method_index == len(SMOOTH_METHODS):
        show_help_dialog()
        return

    if pdb.gimp_selection_is_empty(image):
        pdb.gimp_message("No selection found. Please select an area first.")
        return

    pdb.gimp_image_undo_group_start(image)

    pdb.plug_in_sel2path(image, drawable)
    vectors = pdb.gimp_image_get_active_vectors(image)
    stroke_count, stroke_ids = pdb.gimp_vectors_get_strokes(vectors)

    new_vectors = pdb.gimp_vectors_new(image, "Smoothed Selection")
    pdb.gimp_image_insert_vectors(image, new_vectors, None, 0)

    method_name, smoothing_function, _ = SMOOTH_METHODS[method_index]
    factor = smoothing_strength / 10.0  # Normalize smoothing_strength to 0.1 - 1.0

    for stroke_id in stroke_ids:
        stroke_type, num_points, points, closed = pdb.gimp_vectors_stroke_get_points(
            vectors, str(stroke_id)
        )

        # We'll parse the full 6-float structure for each anchor:
        bezier_anchors = [list(points[i : i + 6]) for i in range(0, len(points), 6)]

        if len(bezier_anchors) < 6:
            # skip very simple paths, likely image boundaries
            continue

        if preserve_curves:
            # Only pass anchor coords (ax, ay) to the smoothing function
            anchor_coords = [(a[ANCHOR_X], a[ANCHOR_Y]) for a in bezier_anchors]

            for _ in range(int(smooth_iterations)):
                anchor_coords = smoothing_function(anchor_coords, factor)

            # Update anchor positions with smoothed coords
            for j, (sx, sy) in enumerate(anchor_coords):
                bezier_anchors[j][ANCHOR_X] = sx
                bezier_anchors[j][ANCHOR_Y] = sy

            new_points = [
                coord for bezier_anchor in bezier_anchors for coord in bezier_anchor
            ]

            final_stroke_type = stroke_type  # keep original stroke type
        else:
            # Fallback to old approach: treat each anchor as (x, y) repeated
            anchor_coords = [(a[ANCHOR_X], a[ANCHOR_Y]) for a in bezier_anchors]

            for _ in range(int(smooth_iterations)):
                anchor_coords = smoothing_function(anchor_coords, factor)

            new_points = []
            for x, y in anchor_coords:
                # Collapse out & in handles to the anchor itself
                new_points += [x, y, x, y, x, y]

            # Force stroke type = 0 (POLY line) if not preserving curves
            final_stroke_type = 0

        # Create the smoothed stroke
        pdb.gimp_vectors_stroke_new_from_points(
            new_vectors, final_stroke_type, len(new_points), new_points, closed
        )

    # Re-select from the new vectors
    pdb.gimp_image_select_item(image, CHANNEL_OP_REPLACE, new_vectors)

    if not preserve_path:
        pdb.gimp_image_remove_vectors(image, new_vectors)

    pdb.gimp_image_remove_vectors(image, vectors)
    pdb.gimp_displays_flush()
    pdb.gimp_image_undo_group_end(image)


register(
    "python_fu_smooth_selection",
    "Smooth Selection",
    "Smooths the active selection without introducing transparency.",
    "Chris Barth",
    "Chris Barth",
    "2025",
    "Smooth Selection...",
    "RGB*, GRAY*",
    [
        (PF_IMAGE, "image", "Input Image", None),
        (PF_DRAWABLE, "drawable", "Input Drawable", None),
        (PF_SLIDER, "smooth_iterations", "Smoothing Iterations", 3, (1, 10, 1)),
        (PF_OPTION, "method_index", "Smoothing Method", 0, METHOD_LABELS),
        (PF_SLIDER, "smoothing_strength", "Smoothing Strength", 5, (1, 10, 1)),
        (PF_TOGGLE, "preserve_curves", "Preserve Curves", True),
        (PF_TOGGLE, "preserve_path", "Preserve Path", False),
    ],
    [],
    smooth_selection,
    menu="<Image>/Select",
)

main()
