#!/usr/bin/env python

from gimpfu import *
import math


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


SMOOTH_METHODS = [
    ("Moving Average", moving_average),
    ("Chaikin", chaikin_smoothing),
    ("Gaussian", gaussian_smoothing),
    ("Pixel Radius", pixel_radius_smoothing),
]

METHOD_LABELS = [name for name, _ in SMOOTH_METHODS]


def smooth_selection(
    image,
    drawable,
    smooth_iterations,
    method_index,
    smoothing_strength,
    preserve_curves,
    preserve_path,
):
    pdb.gimp_image_undo_group_start(image)

    pdb.plug_in_sel2path(image, drawable)
    vectors = pdb.gimp_image_get_active_vectors(image)
    stroke_count, stroke_ids = pdb.gimp_vectors_get_strokes(vectors)

    if stroke_count == 0:
        pdb.gimp_message("No selection path found!")
        pdb.gimp_image_undo_group_end(image)
        return

    new_vectors = pdb.gimp_vectors_new(image, "Smoothed Selection")
    pdb.gimp_image_insert_vectors(image, new_vectors, None, 0)

    method_name, smoothing_function = SMOOTH_METHODS[method_index]
    factor = smoothing_strength / 10.0  # Normalize smoothing_strength to 0.1 - 1.0

    for stroke_id in stroke_ids:
        stroke_type, num_points, points, closed = pdb.gimp_vectors_stroke_get_points(
            vectors, str(stroke_id)
        )
        coords = [(points[i], points[i + 1]) for i in range(0, len(points), 6)]

        if len(coords) < 6:
            continue  # skip very simple paths, likely image boundaries

        for _ in range(int(smooth_iterations)):
            coords = smoothing_function(coords, factor)

        new_points = []
        for x, y in coords:
            new_points += [x, y, x, y, x, y]

        final_stroke_type = stroke_type if preserve_curves else 0
        pdb.gimp_vectors_stroke_new_from_points(
            new_vectors, final_stroke_type, len(new_points), new_points, closed
        )

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
