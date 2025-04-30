import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from skimage import io, color, img_as_float
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image, ImageTk

# === Fourier Logic from main.py ===

def load_image(path):
    image = img_as_float(io.imread(path))
    if image.ndim == 3:
        image = color.rgb2gray(image)
    return image

def fourier_transform(image):
    return fftshift(fft2(image))

def create_filter(shape, radius, filter_type='low'):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if filter_type == 'low' and dist <= radius:
                mask[i, j] = 1
            elif filter_type == 'high' and dist > radius:
                mask[i, j] = 1
    return mask

def apply_filter(f_shifted, mask):
    return f_shifted * mask

def inverse_fourier(f_filtered):
    return np.abs(ifft2(ifftshift(f_filtered)))

def detect_noise(f_shifted, threshold=0.1):
    magnitude = np.abs(f_shifted)
    noise_map = magnitude > threshold * np.max(magnitude)
    return noise_map

# === GUI Class ===

class UltrasoundEnhancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UltraEnhance Pro – Fourier Image Enhancer")
        self.root.geometry("1600x900")  # Increased width and height
        self.root.configure(bg="#f0f8ff")

        self.image = None
        self.f_shift = None
        self.processed_image = None
        self.low_pass_result = None
        self.high_pass_result = None
        self.noise_map = None

        self.setup_ui()

    def setup_ui(self):
        # Main frame to hold all the content
        main_frame = tk.Frame(self.root, bg="#f0f8ff")
        main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Title Label (Updated)
        title = tk.Label(main_frame, text="Ultra Image Enhancer Pro", font=("Arial", 20, "bold"), bg="#f0f8ff", fg="#2c3e50")
        title.grid(row=0, column=0, columnspan=4, pady=10)

        # Noise Detection Section (Top section)
        noise_frame = tk.Frame(main_frame, bg="#f0f8ff")
        noise_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky="w")

        detect_noise_btn = tk.Button(noise_frame, text="🧠 Detect Noise", command=self.detect_noise, bg="#FF9800", fg="white", font=("Arial", 11))
        detect_noise_btn.grid(row=0, column=0)

        self.auto_noise_toggle = tk.Checkbutton(noise_frame, text="✅ Auto Filter Noise", command=self.auto_filter_noise, bg="#f0f8ff")
        self.auto_noise_toggle.grid(row=0, column=1, padx=10)

        # Upload Button
        upload_btn = tk.Button(main_frame, text="📂 Upload Image", command=self.upload_image, bg="#4CAF50", fg="white", font=("Arial", 12))
        upload_btn.grid(row=2, column=0, pady=5, sticky="w")

        # Filter Controls (Radius Entry, Apply Filters, Auto Enhance)
        control_frame = tk.Frame(main_frame, bg="#f0f8ff")
        control_frame.grid(row=3, column=0, columnspan=4, pady=10, sticky="w")

        tk.Label(control_frame, text="Filter Radius:", font=("Arial", 11), bg="#f0f8ff").grid(row=0, column=0, padx=5)
        self.radius_entry = tk.Entry(control_frame, width=6)
        self.radius_entry.insert(0, "50")
        self.radius_entry.grid(row=0, column=1, padx=5)

        apply_btn = tk.Button(control_frame, text="🎯 Apply Filters", command=self.apply_filters, bg="#2196F3", fg="white", font=("Arial", 11))
        apply_btn.grid(row=0, column=2, padx=10)

        auto_btn = tk.Button(control_frame, text="⚡ Auto Enhance (Low & High-pass)", command=self.auto_enhance, bg="#FF5722", fg="white", font=("Arial", 11))
        auto_btn.grid(row=0, column=3)

        # Image Canvases
        canvas_frame = tk.Frame(main_frame, bg="#f0f8ff")
        canvas_frame.grid(row=4, column=0, columnspan=4, pady=20)

        labels = ["Original", "Fourier", "Enhanced", "Low-pass Output", "High-pass Output", "Noise Map"]
        for i, text in enumerate(labels):
            tk.Label(canvas_frame, text=text, font=("Arial", 10), bg="#f0f8ff").grid(row=0, column=i, padx=5)

        self.canvases = {}
        for i, key in enumerate(["original", "fourier", "enhanced", "low", "high", "noise_map"]):
            canvas = tk.Canvas(canvas_frame, width=250, height=250, bg="white")
            canvas.grid(row=1, column=i, padx=5, pady=5)
            self.canvases[key] = canvas

        # Save Button
        save_btn = tk.Button(main_frame, text="💾 Save Enhanced Image", command=self.save_image, bg="#9C27B0", fg="white", font=("Arial", 12))
        save_btn.grid(row=5, column=0, columnspan=4, pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif")])
        if file_path:
            self.image = load_image(file_path)
            self.f_shift = fourier_transform(self.image)
            self.display_image(self.image, "original")
            self.display_image(np.log1p(np.abs(self.f_shift)) / np.max(np.log1p(np.abs(self.f_shift))), "fourier")

    def display_image(self, img, key):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(img).resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        self.canvases[key].img = img_tk
        self.canvases[key].create_image(0, 0, anchor='nw', image=img_tk)

    def apply_filters(self):
        if self.image is None:
            messagebox.showwarning("No image", "Upload an image first.")
            return

        radius = int(self.radius_entry.get())

        # Fourier domain
        self.f_shift = fourier_transform(self.image)

        # Low-pass filter
        low_mask = create_filter(self.image.shape, radius, 'low')
        low_filtered = apply_filter(self.f_shift, low_mask)
        self.low_pass_result = inverse_fourier(low_filtered)
        self.display_image(self.low_pass_result, "low")

        # High-pass filter
        high_mask = create_filter(self.image.shape, radius, 'high')
        high_filtered = apply_filter(self.f_shift, high_mask)
        self.high_pass_result = inverse_fourier(high_filtered)
        self.display_image(self.high_pass_result, "high")

        # Enhanced = low + high (simple combine)
        self.processed_image = np.clip(self.low_pass_result + self.high_pass_result, 0, 1)
        self.display_image(self.processed_image, "enhanced")

    def auto_enhance(self):
        if self.image is None:
            messagebox.showwarning("No image", "Upload an image first.")
            return

        self.f_shift = fourier_transform(self.image)

        # Auto Enhance: Low-pass + High-pass combined
        low_mask = create_filter(self.image.shape, 40, 'low')
        high_mask = create_filter(self.image.shape, 40, 'high')

        low_filtered = apply_filter(self.f_shift, low_mask)
        high_filtered = apply_filter(self.f_shift, high_mask)

        enhanced_f_shift = low_filtered + high_filtered
        self.processed_image = inverse_fourier(enhanced_f_shift)

        # Display all results
        self.display_image(self.processed_image, "enhanced")
        self.display_image(inverse_fourier(low_filtered), "low")
        self.display_image(inverse_fourier(high_filtered), "high")

    def detect_noise(self):
        if self.image is None:
            messagebox.showwarning("No image", "Upload an image first.")
            return

        self.f_shift = fourier_transform(self.image)
        self.noise_map = detect_noise(self.f_shift)

        self.display_image(self.noise_map, "noise_map")

    def auto_filter_noise(self):
        if self.image is None:
            messagebox.showwarning("No image", "Upload an image first.")
            return

        if self.noise_map is None:
            messagebox.showwarning("No noise detected", "Please detect noise first.")
            return

        # Apply suppression
        self.f_shift = fourier_transform(self.image)
        self.f_shift *= (1 - self.noise_map)  # Suppress noise regions
        self.processed_image = inverse_fourier(self.f_shift)

        self.display_image(self.processed_image, "enhanced")

    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Nothing to save", "Apply enhancement first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            img = np.uint8(np.clip(self.processed_image * 255, 0, 255))
            Image.fromarray(img).save(file_path)
            messagebox.showinfo("Saved", f"Image saved to {file_path}")

# Run
if __name__ == "__main__":
    root = tk.Tk()
    app = UltrasoundEnhancerApp(root)
    root.mainloop()
