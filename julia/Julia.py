import taichi as ti
import numpy as np
import cv2
import unittest
import argparse
import sys

# Initialize Taichi
ti.init(arch=ti.gpu)

# --- TAICHI GPU/CPU KERNEL ---
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(1080, 1920))
MAX_COLORS = 1024
palette_field = ti.Vector.field(3, dtype=ti.f32, shape=MAX_COLORS)

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, 2 * z[0] * z[1]])

@ti.kernel
def compute_julia_taichi(zoom: float, c_real: float, c_imag: float, max_iter: int, width: int, height: int, num_colors: int):
    aspect_ratio = height / width
    for i, j in pixels:
        x = ((i / width) * 2.0 - 1.0) * zoom
        y = ((j / height) * 2.0 * aspect_ratio - aspect_ratio) * zoom
        
        z = ti.Vector([x, y])
        c = ti.Vector([c_real, c_imag])

        iterations = 0
        while z.norm_sqr() <= 4.0 and iterations < max_iter:
            z = complex_sqr(z) + c
            iterations += 1

        if iterations == max_iter:
            pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])
        else:
            color_idx = iterations % num_colors
            pixels[i, j] = palette_field[color_idx]

def load_palette_to_hardware(name, num_colors):
    if num_colors > MAX_COLORS:
        raise ValueError(f"Too many colors requested. Max is {MAX_COLORS}")
        
    anchors = {
        'fire': [[0, 255, 255], [0, 160, 255], [0, 0, 220], [0, 0, 80]],
        'ocean': [[255, 255, 0], [255, 128, 0], [128, 0, 0]],
        'mono': [[255, 255, 255], [10, 10, 10]],
        'ultra': [[128, 0, 0], [255, 255, 255], [0, 165, 255], [0, 0, 50]],
        'rainbow': [[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0], [255, 0, 255], [0, 0, 255]]
    }
    arr = np.array(anchors.get(name, anchors['fire']))
    steps = np.linspace(0, len(arr) - 1, num_colors)
    pal_cpu = np.column_stack([np.interp(steps, np.arange(len(arr)), arr[:, i]) for i in range(3)])
    pal_cpu = pal_cpu.astype(np.float32) / 255.0
    for i in range(num_colors):
        palette_field[i] = pal_cpu[i]

### --- UNIT TESTS --- ###
class TestTaichiJulia(unittest.TestCase):
    def test_taichi_palette_loading(self):
        load_palette_to_hardware('mono', 256)
        test_color = palette_field[0].to_numpy()
        self.assertTrue(np.all(test_color == 1.0)) 

    def test_opencv_memory_layout(self):
        test_array = np.zeros((100, 100, 3), dtype=np.float32)
        transposed = np.transpose(test_array, (1, 0, 2))
        contiguous_array = np.ascontiguousarray(transposed * 255, dtype=np.uint8)
        self.assertTrue(contiguous_array.flags['C_CONTIGUOUS'])

    def test_portrait_dimensions(self):
        self.assertEqual(pixels.shape, (1080, 1920))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Taichi Julia Set Morph Generator")
    parser.add_argument('--colors', type=int, default=256, help="Number of colors in the gradient.")
    parser.add_argument('--palette', type=str, default='fire', choices=['fire', 'ocean', 'mono', 'ultra', 'rainbow'])
    parser.add_argument('--fps', type=int, default=30, help="Frames per second of the output video.")
    parser.add_argument('--duration', type=int, default=10, help="Total duration in seconds for the full loop.")
    parser.add_argument('--real', type=float, default=-0.7, help="Real part of the base C constant.")
    parser.add_argument('--imag', type=float, default=0.27015, help="Imaginary part of the base C constant.")
    parser.add_argument('--iter', type=int, default=300, help="Maximum number of iterations for the Julia set computation.")
    
    args, unknown = parser.parse_known_args()

    # Clear sys.argv for unittests
    sys.argv = [sys.argv[0]]
    test_result = unittest.main(exit=False)
    
    if test_result.result.wasSuccessful():
        print(f"Tests passed! Rendering on: {ti.lang.impl.current_cfg().arch}")
        
        width, height = 1080, 1920
        max_iter = args.iter
        
        # Apply the complex number starting point from user arguments
        base_c = args.real + 1j * args.imag
        
        # Calculate exactly how many frames are needed
        total_frames = args.fps * args.duration
        
        # NATURAL MATH LOOP: We spin the complex rotation exactly one full circle (2*pi).
        # endpoint=False ensures we drop the absolute last frame so it doesn't duplicate frame 0.
        morph_angles = np.linspace(0, 2 * np.pi, total_frames, endpoint=False)
        
        load_palette_to_hardware(args.palette, args.colors)
        
        output_filename = f'julia_{args.palette}_{args.colors}c_{args.fps}fps_{args.iter}iter_{args.duration}s_{args.real}_{args.imag}.mp4'
        
        print(f"\n--- RENDER CONFIGURATION ---")
        print(f"Dimensions: {width}x{height} (Portrait)")
        print(f"Base C: {base_c}")
        print(f"FPS: {args.fps}")
        print(f"Total Output Duration (Continuous Seamless Loop): {args.duration}s")
        print(f"Total Frames: {total_frames}")
        print(f"Maximum Iterations: {max_iter}")
        print(f"----------------------------\n")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, args.fps, (width, height))
        
        for i, angle in enumerate(morph_angles):
            c = base_c * np.exp(1j * angle)
            
            compute_julia_taichi(1.2, c.real, c.imag, max_iter, width, height, args.colors)
            
            img_np = pixels.to_numpy()
            img_np = np.transpose(img_np, (1, 0, 2)) 
            img_cpu = np.ascontiguousarray(img_np * 255, dtype=np.uint8)
            
            sign = "+" if c.imag >= 0 else "-"
            text = f"C = {c.real:.5f} {sign} {abs(c.imag):.5f}i"
            cv2.putText(img_cpu, text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(img_cpu, text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                        
            out.write(img_cpu)
            
            if i % args.fps == 0:
                print(f"Rendered frame {i}/{total_frames} ({(i/total_frames)*100:.1f}%)")
                
        out.release()
        print(f"\nRendering Complete! Video saved successfully as: {output_filename}")
    else:
        print("Unit tests failed. Aborting rendering.")
