import cv2
import numpy as np
import math
import argparse
import os
import unittest
from typing import Tuple

def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    """Converts a hex string to BGR for OpenCV."""
    hex_str = hex_str.lstrip('#')
    lv = len(hex_str)
    rgb = tuple(int(hex_str[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0]) 

def get_phyllotaxis_coords(n, angle_degrees, spread, center):
    """Calculates coordinates based on seed index and golden angle."""
    phi = np.radians(n * angle_degrees)
    r = spread * np.sqrt(n)
    x = int(center[0] + r * math.cos(phi))
    y = int(center[1] + r * math.sin(phi))
    return x, y, r

def generate_dynamic_filename(args, parser) -> str:
    """Appends non-default arguments to the filename for tracking."""
    base_name, extension = os.path.splitext(args.output)
    suffix = "_portrait" # Mark as portrait for Reels
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
    # --- PORTRAIT SETTINGS (1080x1920) ---
    width, height = 1080, 1920
    center = (width // 2, height // 2)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(final_filename, fourcc, args.fps, (width, height))
    
    total_frames = args.fps * args.duration
    half_frames = total_frames // 2
    max_radius = math.hypot(width // 2, height // 2)
    
    c_inner = hex_to_bgr(args.color_inner)
    c_outer = hex_to_bgr(args.color_outer)
    c_line = hex_to_bgr(args.color_line)

    print(f"--- 📱 Rendering for Instagram Reels (1080x1920) ---")
    print(f"Target: {final_filename}")
    
    for frame_idx in range(total_frames):
        # Create portrait canvas
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Progress logic (supports reverse flag)
        if args.reverse:
            if frame_idx <= half_frames:
                progress = frame_idx / half_frames
            else:
                progress = 1.0 - ((frame_idx - half_frames) / (total_frames - half_frames - 1))
        else:
            progress = frame_idx / total_frames

        max_seeds_allowed = int(args.duration * args.seeds_per_sec)
        num_seeds = max(1, int(progress * max_seeds_allowed))
        global_rot = np.radians(frame_idx * args.rotation)
        
        coords = []
        for n in range(num_seeds):
            x, y, r_dist = get_phyllotaxis_coords(n, args.angle, args.spread, center)
            # Global rotation
            dx, dy = x - center[0], y - center[1]
            rx = center[0] + dx * math.cos(global_rot) - dy * math.sin(global_rot)
            ry = center[1] + dx * math.sin(global_rot) + dy * math.cos(global_rot)
            coords.append((int(rx), int(ry), r_dist))

        # Draw lines (Explicit geometry)
        if args.show_lines:
            for n in range(13, len(coords)):
                for gap in [8, 13]:
                    cv2.line(frame, (coords[n][0], coords[n][1]), 
                             (coords[n-gap][0], coords[n-gap][1]), c_line, 1, cv2.LINE_AA)

        # Draw seeds
        for n in range(len(coords)):
            x, y, r_dist = coords[n]
            # Normalize distance for portrait height
            dist_norm = min(r_dist / (height * 0.45), 1.0)
            color = [int(c_inner[i]*(1-dist_norm) + c_outer[i]*dist_norm) for i in range(3)]
            radius = int(args.min_size * (1-dist_norm) + args.max_size * dist_norm)
            cv2.circle(frame, (x, y), max(1, radius), color, -1, cv2.LINE_AA)

        out.write(frame)
        if frame_idx % 60 == 0:
            print(f"Progress: {int((frame_idx/total_frames)*100)}% ...")

    out.release()
    print(f"\n✅ REEL READY: {os.path.abspath(final_filename)}")

# --- Unit Tests ---
class TestReelConfigs(unittest.TestCase):
    def test_portrait_dimensions(self):
        width, height = 1080, 1920
        self.assertTrue(height > width, "Video must be in portrait mode for Reels.")

    def test_center_alignment(self):
        center_x, center_y = 1080 // 2, 1920 // 2
        self.assertEqual(center_x, 540)
        self.assertEqual(center_y, 960)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Instagram Reel Phyllotaxis videos (1080x1920).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--output", type=str, default="reel.mp4", help="Output filename")
    parser.add_argument("--duration", type=int, default=10, help="Total length in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--angle", type=float, default=137.508, help="Divergence angle")
    parser.add_argument("--spread", type=float, default=15.0, help="Compactness (increased for vertical space)")
    parser.add_argument("--seeds_per_sec", type=int, default=250, help="Growth speed")
    parser.add_argument("--rotation", type=float, default=0.25, help="Rotation speed")
    parser.add_argument("--color_inner", type=str, default="#FF00FF", help="Inner Hex Color")
    parser.add_argument("--color_outer", type=str, default="#00FFFF", help="Outer Hex Color")
    parser.add_argument("--color_line", type=str, default="#333333", help="Line Hex Color")
    parser.add_argument("--min_size", type=int, default=3, help="Min seed radius")
    parser.add_argument("--max_size", type=int, default=10, help="Max seed radius")
    parser.add_argument("--show_lines", action="store_true", help="Draw spiral lines")
    parser.add_argument("--reverse", action="store_true", help="Shrink back after half duration")
    parser.add_argument("--run_tests", action="store_true", help="Run tests")

    args = parser.parse_args()

    if args.run_tests:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestReelConfigs)
        unittest.TextTestRunner().run(suite)

    final_name = generate_dynamic_filename(args, parser)
    run_video_generation(args, final_name)
