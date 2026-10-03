#!/usr/bin/env python3
"""Replace the logo in every frame of an animated GIF, keeping the animation.

For each frame the old logo is segmented from the flat background, its pose
(centre, rotation, scale) is measured from image moments, the old logo is
painted over with the background colour and the new logo is composited with
the same pose. Frame count, per-frame durations, loop count and disposal
are carried over unchanged.

Usage:
    python3 tools/swap_logo.py INPUT.gif NEW_LOGO.(png|webp) OUTPUT.gif
        [--fit 1.0] [--no-rotate] [--bg-tol 40]
"""
import argparse
import math

import cv2
import numpy as np
from PIL import Image, ImageSequence


def background_color(rgb):
    """Most common colour along the frame border."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    colors, counts = np.unique(border.reshape(-1, 3), axis=0, return_counts=True)
    return colors[counts.argmax()].astype(np.int16)


def logo_mask(rgb, bg, tol):
    """Largest connected blob of pixels that differ from the background."""
    diff = np.abs(rgb.astype(np.int16) - bg).sum(axis=2)
    mask = (diff > tol).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    biggest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    blob = (labels == biggest).astype(np.uint8)
    # Fill holes so the outline is solid.
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solid = np.zeros_like(blob)
    cv2.drawContours(solid, contours, -1, 1, cv2.FILLED)
    return solid


def pose(mask, prev_angle=None):
    """Centre, angle (deg, CCW-positive in image coords) and area of a mask."""
    m = cv2.moments(mask, binaryImage=True)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    # minAreaRect gives a stable orientation for a roughly rectangular badge.
    pts = cv2.findNonZero(mask)
    (_, _), (w, h), ang = cv2.minAreaRect(pts)
    if w < h:
        ang += 90
    # Bring into (-90, 90] and keep it continuous across frames.
    ang = ((ang + 90) % 180) - 90
    if prev_angle is not None:
        while ang - prev_angle > 90:
            ang -= 180
        while ang - prev_angle < -90:
            ang += 180
    return cx, cy, -ang, max(w, h), min(w, h), m["m00"]


def render_logo(logo, width, angle):
    """Scale RGBA logo to `width`, rotate by `angle` degrees CCW."""
    scale = width / logo.width
    resized = logo.resize(
        (max(1, round(logo.width * scale)), max(1, round(logo.height * scale))),
        Image.LANCZOS,
    )
    return resized.rotate(angle, resample=Image.BICUBIC, expand=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("logo")
    ap.add_argument("dst")
    ap.add_argument("--fit", type=float, default=1.0,
                    help="new logo width relative to the old logo's width")
    ap.add_argument("--no-rotate", action="store_true",
                    help="keep the new logo upright instead of copying the tilt")
    ap.add_argument("--bg-tol", type=int, default=40)
    args = ap.parse_args()

    logo = Image.open(args.logo).convert("RGBA")
    logo = logo.crop(logo.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox())

    src = Image.open(args.src)
    frames, durations, prev_angle = [], [], None
    for frame in ImageSequence.Iterator(src):
        durations.append(frame.info.get("duration", src.info.get("duration", 100)))
        rgb = np.array(frame.convert("RGB"))
        bg = background_color(rgb)
        mask = logo_mask(rgb, bg, args.bg_tol)
        out = Image.fromarray(rgb)
        if mask is not None:
            cx, cy, ang, long_side, _, _ = pose(mask, prev_angle)
            prev_angle = -ang
            erase = cv2.dilate(mask, np.ones((7, 7), np.uint8))
            rgb[erase.astype(bool)] = bg.astype(np.uint8)
            out = Image.fromarray(rgb)
            new = render_logo(logo, long_side * args.fit, 0 if args.no_rotate else ang)
            out.paste(new, (round(cx - new.width / 2), round(cy - new.height / 2)), new)
        frames.append(out)

    frames[0].save(
        args.dst,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=src.info.get("loop", 0),
        disposal=2,
        optimize=False,
    )
    print(f"{len(frames)} frames -> {args.dst}")


if __name__ == "__main__":
    main()
