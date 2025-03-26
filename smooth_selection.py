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


# Alternative implementation using more geometric approach
def geometric_inner_contour(coords, factor):
    """
    A geometric approach to inner contour smoothing using convex hull
    and path deflation techniques.

    Args:
        coords: List of (x, y) coordinate tuples
        factor: Smoothing factor (0.1 to 1.0)

    Returns:
        List of smoothed (x, y) coordinate tuples
    """

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
    Smooths only outward-pointing bumps by pulling them toward the line between neighbors.
    `factor` is between 0.1 and 1.0 and controls the fraction of the "bump" to shave off.
    """
    length = len(coords)
    if length < 3:
        return coords  # Not enough points to smooth

    new_coords = [None] * length

    def polygon_area(coords):
        return 0.5 * sum(
            coords[i][0] * coords[(i + 1) % length][1]
            - coords[(i + 1) % length][0] * coords[i][1]
            for i in range(length)
        )

    def triangle_area_sign(a, b, c):
        """Signed area: positive if a->b->c is CCW, negative if CW"""
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def perpendicular_vector_to_line(p, a, c):
        """Returns vector from p to its projection on line AC"""
        acx = c[0] - a[0]
        acy = c[1] - a[1]
        ac_len_sq = acx**2 + acy**2
        if ac_len_sq == 0:
            return 0.0, 0.0  # Degenerate segment

        apx = p[0] - a[0]
        apy = p[1] - a[1]
        t = (apx * acx + apy * acy) / ac_len_sq
        px = a[0] + t * acx
        py = a[1] + t * acy
        return (px - p[0], py - p[1])

    # --- Determine overall winding direction ---
    total_area = polygon_area(coords)
    winding_sign = -1 if total_area < 0 else 1  # CW = -1, CCW = +1

    # --- Sand each point ---
    for i in range(length):
        a = coords[i - 1]
        b = coords[i]
        c = coords[(i + 1) % length]

        signed_area = triangle_area_sign(a, b, c)
        if math.copysign(1, signed_area) != winding_sign:
            # This is an inward dent or flat segment - skip
            new_coords[i] = b
            continue

        # This is an outward bump - shave it
        dx, dy = perpendicular_vector_to_line(b, a, c)
        new_coords[i] = (b[0] + dx * factor, b[1] + dy * factor)

    return new_coords


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
