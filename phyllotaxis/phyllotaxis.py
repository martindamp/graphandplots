import cv2
import numpy as np
import math
import argparse
import os
import unittest
from typing import Tuple

def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    hex_str = hex_str.lstrip('#')
    lv = len(hex_str)
    rgb = tuple(int(hex_str[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0]) 

def get_phyllotaxis_coords(n, angle_degrees, spread, center):
    phi = np.radians(n * angle_degrees)
    r = spread * np.sqrt(n)
    x = int(center[0] + r * math.cos(phi))
    y = int(center[1] + r * math.sin(phi))
    return x, y, r

def generate_dynamic_filename(args, parser) -> str:
    base_name, extension = os.path.splitext(args.output)
    suffix = ""
    for action in parser._actions:
        arg_name = action.dest
        if arg_name in ['output', 'run_tests', 'fps', 'duration', 'help']:
            continue
        current_val = getattr(args, arg_name)
        if current_val != action.default:
            clean_val = str(current_val).replace('#', '')
            suffix += f"_{arg_name}-{clean_val}"
    return f"{base_name}{suffix}{extension}"

def run_video_generation(args, final_filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(final_filename, fourcc, args.fps, (1920, 1080))
    center = (960, 540)
    total_frames = args.fps * args.duration
    half_frames = total_frames // 2
    max_radius = math.hypot(960, 540)
    
    c_inner = hex_to_bgr(args.color_inner)
    c_outer = hex_to_bgr(args.color_outer)
    c_line = hex_to_bgr(args.color_line)

    print(f"--- 🎬 Rendering Started ---")
    print(f"Target: {final_filename}")
    
    for frame_idx in range(total_frames):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # --- REVERSE LOGIC ---
        if args.reverse:
            if frame_idx <= half_frames:
                # Growing phase (0.0 to 1.0)
                progress = frame_idx / half_frames
            else:
                # Shrinking phase (1.0 back to 0.0)
                progress = 1.0 - ((frame_idx - half_frames) / (total_frames - half_frames - 1))
        else:
            # Standard expansion (0.0 to 1.0)
            progress = frame_idx / total_frames

        # Scale seed count based on progress
        max_seeds_allowed = int(args.duration * args.seeds_per_sec)
        num_seeds = max(1, int(progress * max_seeds_allowed))
        
        # Rotation usually looks better if it keeps spinning forward
        global_rot = np.radians(frame_idx * args.rotation)
        
        coords = []
        for n in range(num_seeds):
            x, y, r_dist = get_phyllotaxis_coords(n, args.angle, args.spread, center)
            dx, dy = x - center[0], y - center[1]
            rx = center[0] + dx * math.cos(global_rot) - dy * math.sin(global_rot)
            ry = center[1] + dx * math.sin(global_rot) + dy * math.cos(global_rot)
            coords.append((int(rx), int(ry), r_dist))

        if args.show_lines:
            for n in range(13, len(coords)):
                for gap in [8, 13]:
                    cv2.line(frame, (coords[n][0], coords[n][1]), 
                             (coords[n-gap][0], coords[n-gap][1]), c_line, 1, cv2.LINE_AA)

        for n in range(len(coords)):
            x, y, r_dist = coords[n]
            dist_norm = min(r_dist / (max_radius * 0.8), 1.0)
            color = [int(c_inner[i]*(1-dist_norm) + c_outer[i]*dist_norm) for i in range(3)]
            radius = int(args.min_size * (1-dist_norm) + args.max_size * dist_norm)
            cv2.circle(frame, (x, y), max(1, radius), color, -1, cv2.LINE_AA)

        out.write(frame)
        if frame_idx % 60 == 0:
            print(f"Progress: {int((frame_idx/total_frames)*100)}% rendered...")

    out.release()
    print(f"\n✅ SUCCESS! File saved at: {os.path.abspath(final_filename)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phyllotaxis engine with Reverse/Looping support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--output", type=str, default="phyllotaxis.mp4", help="Base filename")
    parser.add_argument("--duration", type=int, default=10, help="Total length in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--angle", type=float, default=137.508, help="Divergence angle")
    parser.add_argument("--spread", type=float, default=12.0, help="Compactness")
    parser.add_argument("--seeds_per_sec", type=int, default=200, help="Growth speed")
    parser.add_argument("--rotation", type=float, default=0.2, help="Rotation degrees/frame")
    parser.add_argument("--color_inner", type=str, default="#FF00FF", help="Center color (Hex)")
    parser.add_argument("--color_outer", type=str, default="#00FFFF", help="Edge color (Hex)")
    parser.add_argument("--color_line", type=str, default="#444444", help="Line color (Hex)")
    parser.add_argument("--min_size", type=int, default=2, help="Min seed radius")
    parser.add_argument("--max_size", type=int, default=8, help="Max seed radius")
    parser.add_argument("--show_lines", action="store_true", help="Draw spiral lines")
    parser.add_argument("--reverse", action="store_true", help="Shrink the figure back after half duration")

    args = parser.parse_args()
    final_name = generate_dynamic_filename(args, parser)
    run_video_generation(args, final_name)
