#!/usr/bin/env python

import imp
import inspect
import math
import os
import sys

from gimpfu import *

# Coordinate indices for Bezier point arrays
BEZIER_IN_CTRL_X = 0
BEZIER_IN_CTRL_Y = 1
BEZIER_ANCHOR_X = 2
BEZIER_ANCHOR_Y = 3
BEZIER_OUT_CTRL_X = 4
BEZIER_OUT_CTRL_Y = 5


def compute_moving_average(points, smoothing_factor, selection_channel):
    window_size = max(1, int(smoothing_factor * 5))
    smoothed_points = []
    point_count = len(points)

    for i in range(point_count):
        avg_x = sum(
            points[(i + j) % point_count][0]
            for j in range(-window_size, window_size + 1)
        ) / (2 * window_size + 1)
        avg_y = sum(
            points[(i + j) % point_count][1]
            for j in range(-window_size, window_size + 1)
        ) / (2 * window_size + 1)
        smoothed_points.append((avg_x, avg_y))

    return smoothed_points


def compute_chaikin_smoothing(points, smoothing_factor, selection_channel):
    shrink_factor = smoothing_factor * 0.25
    smoothed_points = []
    point_count = len(points)

    for i in range(point_count):
        p0 = points[i]
        p1 = points[(i + 1) % point_count]

        q_point = (
            (1 - shrink_factor) * p0[0] + shrink_factor * p1[0],
            (1 - shrink_factor) * p0[1] + shrink_factor * p1[1],
        )
        r_point = (
            shrink_factor * p0[0] + (1 - shrink_factor) * p1[0],
            shrink_factor * p0[1] + (1 - shrink_factor) * p1[1],
        )
        smoothed_points.extend([q_point, r_point])

    return smoothed_points


def compute_gaussian_smoothing(points, smoothing_factor, selection_channel):
    sigma = smoothing_factor * 3
    radius = max(1, int(sigma * 3))
    smoothed_points = []
    point_count = len(points)

    weights = [math.exp(-(i**2) / (2 * sigma**2)) for i in range(-radius, radius + 1)]
    weight_sum = sum(weights)

    for i in range(point_count):
        avg_x = (
            sum(
                points[(i + j) % point_count][0] * weights[j + radius]
                for j in range(-radius, radius + 1)
            )
            / weight_sum
        )
        avg_y = (
            sum(
                points[(i + j) % point_count][1] * weights[j + radius]
                for j in range(-radius, radius + 1)
            )
            / weight_sum
        )
        smoothed_points.append((avg_x, avg_y))

    return smoothed_points


def compute_pixel_radius_smoothing(points, smoothing_factor, selection_channel):
    pixel_radius = max(1, int(smoothing_factor * 20))
    smoothed_points = []
    point_count = len(points)

    for i in range(point_count):
        x, y = points[i]
        neighbors = [
            pt for pt in points if math.hypot(pt[0] - x, pt[1] - y) <= pixel_radius
        ]
        avg_x = sum(pt[0] for pt in neighbors) / len(neighbors)
        avg_y = sum(pt[1] for pt in neighbors) / len(neighbors)
        smoothed_points.append((avg_x, avg_y))

    return smoothed_points


def compute_inward_pixel_radius_smoothing(points, smoothing_factor, selection_channel):
    pixel_radius = max(1, int(smoothing_factor * 20))
    smoothed_points = []
    point_count = len(points)

    def get_point(index):
        return tuple(points[index % len(points)])

    for i in range(point_count):
        base_x, base_y = points[i]
        neighbors = [(base_x, base_y)]  # Always include self

        # Step outward in both directions
        offset = 1
        while offset < point_count and len(neighbors) < 3:
            # Look backward with wraparound
            if i - offset >= 0:
                px, py = get_point(i - offset)
                if math.hypot(px - base_x, py - base_y) <= pixel_radius:
                    neighbors.append((px, py))

            # Look forward with wraparound
            if i + offset < point_count:
                px, py = get_point(i + offset)
                if math.hypot(px - base_x, py - base_y) <= pixel_radius:
                    neighbors.append((px, py))

            offset += 1

        if len(neighbors) == 1:
            # Only the current point was found, skip smoothing
            smoothed_points.append((base_x, base_y))
            continue

        if len(neighbors) == 2:
            # Try to get a 3rd point by checking next closest sequential candidate
            candidates = [get_point(i - offset + 1), get_point(i + offset - 1)]

            # Only consider points not already in neighbors
            unique_candidates = [pt for pt in candidates if pt not in neighbors]

            # Choose the closer of the candidates (if any)
            unique_candidates.sort(
                key=lambda pt: math.hypot(pt[0] - base_x, pt[1] - base_y)
            )
            neighbors.append(unique_candidates[0])

        if len(neighbors) < 3:
            # Still not enough points for meaningful smoothing
            smoothed_points.append((base_x, base_y))
            continue

        # Compute average
        avg_x = sum(pt[0] for pt in neighbors) / len(neighbors)
        avg_y = sum(pt[1] for pt in neighbors) / len(neighbors)

        new_x = base_x + (avg_x - base_x) * smoothing_factor
        new_y = base_y + (avg_y - base_y) * smoothing_factor

        # Check if the new point is still inside the selection
        if is_point_inside_selection(new_x, new_y, selection_channel):
            smoothed_points.append((new_x, new_y))
        else:
            smoothed_points.append((base_x, base_y))

    return smoothed_points


def compute_inside_track_smoothing(points, smoothing_factor, selection_channel):
    """
    Smooths a path by preferentially cutting inward corners.
    This creates a smoother path that tends to stay inside the original selection.
    """
    bias_strength = smoothing_factor * 0.6
    smoothed_points = []
    point_count = len(points)

    for i in range(point_count):
        prev_index = (i - 1) % point_count
        next_index = (i + 1) % point_count

        prev_point = points[prev_index]
        curr_point = points[i]
        next_point = points[next_index]

        vec_to_prev = (prev_point[0] - curr_point[0], prev_point[1] - curr_point[1])
        vec_to_next = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])

        cross_z = vec_to_prev[0] * vec_to_next[1] - vec_to_prev[1] * vec_to_next[0]
        is_outside_corner = cross_z < 0

        avg_x = (prev_point[0] + curr_point[0] + next_point[0]) / 3
        avg_y = (prev_point[1] + curr_point[1] + next_point[1]) / 3

        if is_outside_corner:
            smooth_factor = max(0.05, smoothing_factor * 0.3)
            new_x = curr_point[0] * (1 - smooth_factor) + avg_x * smooth_factor
            new_y = curr_point[1] * (1 - smooth_factor) + avg_y * smooth_factor
        else:
            smooth_factor = min(0.9, smoothing_factor * 0.9 + bias_strength)
            new_x = curr_point[0] * (1 - smooth_factor) + avg_x * smooth_factor
            new_y = curr_point[1] * (1 - smooth_factor) + avg_y * smooth_factor

        smoothed_points.append((new_x, new_y))

    return smoothed_points


def compute_geometric_inner_contour(points, smoothing_factor, selection_channel):
    """
    A geometric approach to inner contour smoothing using convex hull techniques.
    """
    point_count = len(points)
    interim_points = []

    for i in range(point_count):
        prev_index = (i - 1) % point_count
        next_index = (i + 1) % point_count

        prev_point = points[prev_index]
        curr_point = points[i]
        next_point = points[next_index]

        # Vectors between points
        v1 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
        v2 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])

        # Cross product to determine corner type
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        is_convex = cross < 0

        if is_convex:
            v1_len = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            v2_len = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

            if v1_len > 0 and v2_len > 0:
                v1_norm = (v1[0] / v1_len, v1[1] / v1_len)
                v2_norm = (v2[0] / v2_len, v2[1] / v2_len)

                bisector = (-(v1_norm[0] + v2_norm[0]), -(v1_norm[1] + v2_norm[1]))
                bisector_len = math.sqrt(bisector[0] ** 2 + bisector[1] ** 2)

                if bisector_len > 0:
                    bisector = (bisector[0] / bisector_len, bisector[1] / bisector_len)

                    dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
                    angle_factor = max(0, 1 - dot_product)

                    move_distance = smoothing_factor * 10 * angle_factor
                    new_x = curr_point[0] + bisector[0] * move_distance
                    new_y = curr_point[1] + bisector[1] * move_distance

                    interim_points.append((new_x, new_y))
                else:
                    interim_points.append(curr_point)
            else:
                interim_points.append(curr_point)
        else:
            interim_points.append(curr_point)

    # Add a gentle smoothing pass
    smoothed_points = []
    window_size = max(1, int(smoothing_factor * 2))

    for i in range(point_count):
        indices = [(i + j) % point_count for j in range(-window_size, window_size + 1)]
        avg_x = sum(interim_points[j][0] for j in indices) / len(indices)
        avg_y = sum(interim_points[j][1] for j in indices) / len(indices)
        smoothed_points.append((avg_x, avg_y))

    return smoothed_points


def compute_sanding_smoothing(points, smoothing_factor, selection_channel):
    """
    Smooths only outward-pointing bumps by pulling them toward the line between neighbors.
    """
    point_count = len(points)
    if point_count < 3:
        return points

    new_points = [None] * point_count

    def compute_perpendicular_vector_to_line(point, point_a, point_c):
        acx = point_c[0] - point_a[0]
        acy = point_c[1] - point_a[1]
        ac_len_sq = acx**2 + acy**2
        if ac_len_sq == 0:
            return 0.0, 0.0

        apx = point[0] - point_a[0]
        apy = point[1] - point_a[1]
        t = (apx * acx + apy * acy) / ac_len_sq
        px = point_a[0] + t * acx
        py = point_a[1] + t * acy
        return (px - point[0], py - point[1])

    for i in range(point_count):
        point_a = points[i - 1]
        point_b = points[i]
        point_c = points[(i + 1) % point_count]

        if is_corner(point_a, point_b, point_c):
            new_points[i] = point_b
            continue

        dx, dy = compute_perpendicular_vector_to_line(point_b, point_a, point_c)
        proposed_point = (
            point_b[0] + dx * smoothing_factor,
            point_b[1] + dy * smoothing_factor,
        )

        pixel_value = pdb.gimp_drawable_get_pixel(
            selection_channel, int(proposed_point[0]), int(proposed_point[1])
        )[1][0]
        is_inside = pixel_value > 128
        if is_inside:
            new_points[i] = proposed_point
        else:
            new_points[i] = point_b

    return new_points


def show_help_dialog():
    help_lines = [
        "{}. {}\n   {}".format(i + 1, name, desc)
        for i, (name, _, desc) in enumerate(SMOOTH_METHODS)
    ]
    help_text = "\n\n".join(help_lines)
    pdb.gimp_message(help_text)


def is_corner(point_a, point_b, point_c, angle_threshold=135, deviation_threshold=3.0):
    # Vectors AB and BC
    ab_vector = (point_b[0] - point_a[0], point_b[1] - point_a[1])
    bc_vector = (point_c[0] - point_b[0], point_c[1] - point_b[1])

    def normalize_vector(vector):
        mag = math.hypot(*vector)
        return (vector[0] / mag, vector[1] / mag) if mag != 0 else (0, 0)

    ab_normalized = normalize_vector(ab_vector)
    bc_normalized = normalize_vector(bc_vector)

    dot_product = (
        ab_normalized[0] * bc_normalized[0] + ab_normalized[1] * bc_normalized[1]
    )
    dot_product = max(-1.0, min(1.0, dot_product))

    angle = math.acos(dot_product)
    angle_degrees = math.degrees(angle)

    is_sharp_angle = angle_degrees < angle_threshold

    def compute_point_line_distance(point, point_a, point_c):
        acx, acy = point_c[0] - point_a[0], point_c[1] - point_a[1]
        ac_len_sq = acx**2 + acy**2
        if ac_len_sq == 0:
            return math.hypot(point[0] - point_a[0], point[1] - point_a[1])

        apx, apy = point[0] - point_a[0], point[1] - point_a[1]
        t = (apx * acx + apy * acy) / ac_len_sq
        proj_x = point_a[0] + t * acx
        proj_y = point_a[1] + t * acy
        return math.hypot(proj_x - point[0], proj_y - point[1])

    deviation = compute_point_line_distance(point_b, point_a, point_c)
    is_high_deviation = deviation > deviation_threshold

    return is_sharp_angle and is_high_deviation


def is_point_inside_selection(x, y, selection_channel):
    width = pdb.gimp_drawable_width(selection_channel)
    height = pdb.gimp_drawable_height(selection_channel)

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            sample_x = int(x + dx)
            sample_y = int(y + dy)

            if 0 <= sample_x < width and 0 <= sample_y < height:
                pixel_value = pdb.gimp_drawable_get_pixel(
                    selection_channel, sample_x, sample_y
                )[1][0]
                if pixel_value > 128:
                    return True
    return False


SMOOTH_METHODS = [
    (
        "Moving Average",
        compute_moving_average,
        "Balances each point with its neighbors. Fast, general-purpose smoothing.",
    ),
    (
        "Chaikin",
        compute_chaikin_smoothing,
        "Creates soft, rounded curves by subdividing lines. Doubles point count.",
    ),
    (
        "Gaussian",
        compute_gaussian_smoothing,
        "Smooths with precision using a weighted average. Preserves shape better.",
    ),
    (
        "Pixel Radius",
        compute_pixel_radius_smoothing,
        "Averages points within a pixel range. Great for high-detail smoothing.",
    ),
    (
        "Inward Pixel Radius",
        compute_inward_pixel_radius_smoothing,
        "Like Pixel Radius but only pulls in bumps, preserving inward details.",
    ),
    (
        "Inside Track",
        compute_inside_track_smoothing,
        "Favors smoothing inward. Keeps selection close to original edges.",
    ),
    (
        "Geometric Inner Contour",
        compute_geometric_inner_contour,
        "Trims sharp convex corners while keeping concave points.",
    ),
    (
        "Sanding",
        compute_sanding_smoothing,
        "Smooths only protruding bumps, preserves detail elsewhere.",
    ),
]


METHOD_LABELS = [name for name, _, _ in SMOOTH_METHODS] + [
    "Help - Show method descriptions"
]


def bezier_cubic_points_filtered(point_a, point_b, factor):
    factor *= 10

    p0 = (point_a[BEZIER_ANCHOR_X], point_a[BEZIER_ANCHOR_Y])
    p1 = (point_a[BEZIER_OUT_CTRL_X], point_a[BEZIER_OUT_CTRL_Y])
    p2 = (point_b[BEZIER_IN_CTRL_X], point_b[BEZIER_IN_CTRL_Y])
    p3 = (point_b[BEZIER_ANCHOR_X], point_b[BEZIER_ANCHOR_Y])

    def is_flat_enough(p0, p1, p2, p3, threshold):
        # Measure deviation of control points from the baseline (p0-p3)
        def point_line_distance(pt, a, b):
            if a == b:
                return math.hypot(pt[0] - a[0], pt[1] - a[1])
            num = abs(
                (b[1] - a[1]) * pt[0]
                - (b[0] - a[0]) * pt[1]
                + b[0] * a[1]
                - b[1] * a[0]
            )
            den = math.hypot(b[0] - a[0], b[1] - a[1])
            return num / den

        return (
            point_line_distance(p1, p0, p3) < threshold
            and point_line_distance(p2, p0, p3) < threshold
        )

    def subdivide(p0, p1, p2, p3):
        # de Casteljau subdivision
        p01 = midpoint(p0, p1)
        p12 = midpoint(p1, p2)
        p23 = midpoint(p2, p3)
        p012 = midpoint(p01, p12)
        p123 = midpoint(p12, p23)
        p0123 = midpoint(p012, p123)
        return ((p0, p01, p012, p0123), (p0123, p123, p23, p3))

    def midpoint(a, b):
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    def flatten(p0, p1, p2, p3, threshold):
        if is_flat_enough(p0, p1, p2, p3, threshold):
            return [p0, p3]
        else:
            left, right = subdivide(p0, p1, p2, p3)
            return flatten(left[0], left[1], left[2], left[3], threshold)[
                :-1
            ] + flatten(right[0], right[1], right[2], right[3], threshold)

    # Generate raw points
    raw_points = flatten(p0, p1, p2, p3, factor)
    rounded = [tuple(map(int, map(round, pt))) for pt in raw_points]

    # Remove adjacent pixels and collinear middles
    def is_adjacent(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1

    def is_collinear(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0])

    filtered = []
    for pt in rounded:
        if not filtered:
            filtered.append(pt)

    for pt in rounded:
        if not filtered or not is_adjacent(filtered[-1], pt):
            filtered.append(pt)

    i = 1
    while i < len(filtered) - 1:
        if is_collinear(filtered[i - 1], filtered[i], filtered[i + 1]):
            del filtered[i]
        else:
            i += 1

    return filtered


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

    previously_active_layer = pdb.gimp_image_get_active_layer(image)
    pdb.gimp_image_undo_group_start(image)
    selection_channel = pdb.gimp_selection_save(image)

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

        for _ in range(int(smooth_iterations)):
            selection_channel = pdb.gimp_selection_save(image)

            # Parse the full 6-float structure for each anchor
            bezier_anchors = [list(points[i : i + 6]) for i in range(0, len(points), 6)]

            if len(bezier_anchors) < 6:
                # Skip very simple paths, likely image boundaries
                continue

            if preserve_curves:
                # Only pass anchor coords (ax, ay) to the smoothing function
                anchor_coords = [
                    (a[BEZIER_ANCHOR_X], a[BEZIER_ANCHOR_Y]) for a in bezier_anchors
                ]

                anchor_coords = smoothing_function(
                    anchor_coords, factor, selection_channel
                )

                # Update anchor positions with smoothed coords
                for j, (sx, sy) in enumerate(anchor_coords):
                    bezier_anchors[j][BEZIER_ANCHOR_X] = sx
                    bezier_anchors[j][BEZIER_ANCHOR_Y] = sy

                new_points = [
                    coord for bezier_anchor in bezier_anchors for coord in bezier_anchor
                ]

                final_stroke_type = stroke_type  # keep original stroke type
            else:
                anchor_coords = []

                for i in range(len(bezier_anchors) - 1):
                    first = bezier_anchors[i]
                    second = bezier_anchors[i + 1]
                    segment_points = bezier_cubic_points_filtered(first, second, factor)
                    anchor_coords.extend(segment_points)

                if len(bezier_anchors) > 1:
                    first = bezier_anchors[-1]
                    second = bezier_anchors[0]
                    anchor_coords.extend(
                        bezier_cubic_points_filtered(first, second, factor)
                    )

                deduped_anchor_coords = []
                for pt in anchor_coords:
                    if not deduped_anchor_coords or pt != deduped_anchor_coords[-1]:
                        deduped_anchor_coords.append(pt)

                anchor_coords = deduped_anchor_coords

                anchor_coords = smoothing_function(
                    deduped_anchor_coords, factor, selection_channel
                )

                new_points = []
                for x, y in anchor_coords:
                    # Collapse out & in handles to the anchor itself
                    new_points += [x, y, x, y, x, y]

                # Force stroke type = 0 (POLY line) if not preserving curves
                final_stroke_type = 0

            points = new_points

        # Create the smoothed stroke
        pdb.gimp_vectors_stroke_new_from_points(
            new_vectors, final_stroke_type, len(points), points, closed
        )

    # Re-select from the new vectors
    pdb.gimp_image_select_item(image, CHANNEL_OP_REPLACE, new_vectors)

    if not preserve_path:
        pdb.gimp_image_remove_vectors(image, new_vectors)

    pdb.gimp_image_remove_vectors(image, vectors)
    pdb.gimp_image_remove_channel(image, selection_channel)
    pdb.gimp_image_set_active_layer(image, previously_active_layer)
    pdb.gimp_displays_flush()
    pdb.gimp_image_undo_group_end(image)


def _reload():
    py_path = inspect.getsourcefile(sys.modules[__name__])
    pyc_path = py_path + "c"

    if os.path.exists(pyc_path):
        os.remove(pyc_path)

    imp.load_source("smooth_selection", py_path)
    pdb.gimp_message("smooth_selection.py reloaded from source.")


if __name__ == "__main__":
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
            (PF_TOGGLE, "preserve_curves", "Preserve Curves", False),
            (PF_TOGGLE, "preserve_path", "Preserve Path", False),
        ],
        [],
        smooth_selection,
        menu="<Image>/Select",
    )

    main()
