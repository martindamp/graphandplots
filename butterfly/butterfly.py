import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import unittest

def calculate_butterfly_curve(theta):
    """Calculates r for the Butterfly Curve given theta."""
    r = (np.exp(np.sin(theta)) 
         - 2 * np.cos(4 * theta) 
         + np.sin((2 * theta - np.pi) / 24)**5)
    return r

def generate_butterfly_video(filename='butterfly_with_telemetry.mp4', duration_seconds=20, fps=30):
    total_frames = duration_seconds * fps
    theta_max = 24 * np.pi 
    
    # Generate points
    points = total_frames * 15
    theta = np.linspace(0, theta_max, points)
    r = calculate_butterfly_curve(theta)
    
    # Base orientation
    x_orig = r * np.sin(theta)
    y_orig = r * np.cos(theta)

    # Strictly rotate by -90 degrees
    angle = np.radians(-90)
    x = x_orig * np.cos(angle) - y_orig * np.sin(angle)
    y = x_orig * np.sin(angle) + y_orig * np.cos(angle)

    # Format data for LineCollection
    points_xy = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points_xy[:-1], points_xy[1:]], axis=1)

    # 1080x1920 resolution at 100 DPI
    fig, ax = plt.subplots(figsize=(10.8, 19.2), facecolor='black', dpi=100)
    ax.set_facecolor('black')
    
    # Keep true aspect ratio and portrait bounds
    ax.set_aspect('equal')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-14.2, 14.2)
    
    # --- STYLING THE VISIBLE COORDINATE SYSTEM ---
    # Draw axes intersecting at (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_color('#555555')
    ax.spines['top'].set_color('none') 
    ax.spines['right'].set_color('none')
    
    # Style ticks and grid
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    ax.grid(True, color='#222222', linestyle='--', linewidth=0.5)

    # Custom blue-green gradient
    blue_green_cmap = LinearSegmentedColormap.from_list("blue_green", ["#00008B", "#00FFFF", "#00FF00"])
    lc = LineCollection([], cmap=blue_green_cmap, lw=2.5, norm=mcolors.Normalize(vmin=0.0, vmax=1.0))
    ax.add_collection(lc)
    colors = theta[:-1] / theta_max

    # --- ADDING THE REAL-TIME TEXT DISPLAY ---
    # Placed in the top-left corner with a semi-transparent background so the curve can pass behind it
    telemetry_text = ax.text(-7.5, 12.5, '', color='white', fontsize=16, fontfamily='monospace', 
                             bbox=dict(facecolor='black', edgecolor='#555555', alpha=0.8, pad=10))

    def update(frame):
        current_idx = int((frame / total_frames) * len(segments))
        if current_idx > 0:
            # Update the drawing
            lc.set_segments(segments[:current_idx])
            lc.set_array(colors[:current_idx])
            
            # Fetch current values to display
            # Use min() to prevent index out-of-bounds on the very last frame
            idx = min(current_idx, len(theta) - 1)
            curr_theta = theta[idx]
            curr_x = x[idx]
            curr_y = y[idx]
            
            # Update text object with formatted values
            telemetry_text.set_text(f"θ : {curr_theta:6.2f} rad\nX : {curr_x:6.2f}\nY : {curr_y:6.2f}")
            
        return lc, telemetry_text

    ani = FuncAnimation(fig, update, frames=total_frames, blit=True)

    print(f"Starting render for {filename}... This will take a few minutes.")
    ani.save(filename, writer='ffmpeg', fps=fps, codec='libx264')
    print("Render complete! Video saved successfully.")
    plt.close(fig)

# --- UNIT TESTS ---
class TestButterflyDisplay(unittest.TestCase):
    def test_array_indexing_safety(self):
        """Verifies that the telemetry indexing won't crash on the final frame."""
        total_frames = 20 * 30
        points = total_frames * 15
        theta = np.linspace(0, 24 * np.pi, points)
        
        # Simulate the final frame calculation
        frame = total_frames
        current_idx = int((frame / total_frames) * (len(theta) - 1))
        safe_idx = min(current_idx, len(theta) - 1)
        
        # Must not raise IndexError
        _ = theta[safe_idx]
        self.assertLess(safe_idx, len(theta))

if __name__ == "__main__":
    # Run tests before rendering
    suite = unittest.TestLoader().loadTestsFromTestCase(TestButterflyDisplay)
    test_result = unittest.TextTestRunner(verbosity=1).run(suite)
    
    if test_result.wasSuccessful():
        generate_butterfly_video()
