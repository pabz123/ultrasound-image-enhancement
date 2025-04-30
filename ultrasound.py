import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
from scipy import ndimage, signal
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class UltrasoundEnhancerApp:
    def __init__(self, root):
        # Window setup
        self.root = root
        self.root.title("🎯 UltraEnhance Pro")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f8ff")

        # Title & subtitle
        tk.Label(root, text="🎯 UltraEnhance Pro", font=("Helvetica", 24, "bold"), bg="#f0f8ff").pack(pady=5)
        tk.Label(root, text="Advanced Ultrasound Image Enhancer", font=("Helvetica", 14), bg="#f0f8ff").pack(pady=5)

        # Image data
        self.image = None
        self.processed_image = None
        self.fourier_image = None

        # Upload frame
        upload_frame = tk.LabelFrame(root, text="Upload Image", padx=10, pady=10, bg="#f0f8ff")
        upload_frame.pack(fill="x", padx=10, pady=5)
        tk.Button(upload_frame, text="📂 Upload", command=self.upload_image,
                  bg="#4CAF50", fg="white", font=("Arial", 12)).pack()

        # Enhancement options frame
        opts = tk.LabelFrame(root, text="Enhancement Options", padx=10, pady=10, bg="#f0f8ff")
        opts.pack(fill="x", padx=10, pady=5)
        tk.Label(opts, text="Manual Filter:", bg="#f0f8ff").grid(row=0, column=0, sticky="w")
        self.filter_combo = ttk.Combobox(opts, values=["Gaussian Low Pass","Median Filter","Bandpass Filter"])
        self.filter_combo.current(0); self.filter_combo.grid(row=0, column=1)
        tk.Label(opts, text="Param1 (σ/size/low):", bg="#f0f8ff").grid(row=1, column=0, sticky="w")
        self.param1 = tk.Entry(opts); self.param1.insert(0,"10"); self.param1.grid(row=1, column=1)
        tk.Label(opts, text="Param2 (high radius):", bg="#f0f8ff").grid(row=2, column=0, sticky="w")
        self.param2 = tk.Entry(opts); self.param2.insert(0,"50"); self.param2.grid(row=2, column=1)
        tk.Button(opts, text="🎯 Apply Manual", command=self.apply_filter,
                  bg="#2196F3", fg="white").grid(row=3, column=0, pady=5)
        tk.Button(opts, text="⚡ Auto Enhance", command=self.auto_enhance,
                  bg="#FF5722", fg="white").grid(row=3, column=1, pady=5)

        # Image canvases
        imgs = tk.Frame(root, bg="#f0f8ff")
        imgs.pack(pady=10)
        tk.Label(imgs, text="Original").grid(row=0, column=0)
        tk.Label(imgs, text="Enhanced").grid(row=0, column=1)
        tk.Label(imgs, text="Fourier Spectrum").grid(row=0, column=2)
        self.c_orig = tk.Canvas(imgs, width=350, height=350, bg="white"); self.c_orig.grid(row=1, column=0, padx=5)
        self.c_enh = tk.Canvas(imgs, width=350, height=350, bg="white"); self.c_enh.grid(row=1, column=1, padx=5)
        self.c_four = tk.Canvas(imgs, width=350, height=350, bg="white"); self.c_four.grid(row=1, column=2, padx=5)

        # Waveform & noise frame
        wf = tk.LabelFrame(root, text="Frequency Wave & Noise Detection", padx=10, pady=10, bg="#f0f8ff")
        wf.pack(fill="x", padx=10, pady=5)
        tk.Button(wf, text="🔍 Show Waveform", command=self.plot_waveform).pack(side="left", padx=5)
        tk.Button(wf, text="🧪 Detect Noise", command=self.detect_noise).pack(side="left", padx=5)
        self.wave_canvas = None

        # Comparison slider
        cmpf = tk.LabelFrame(root, text="Compare Original vs Enhanced", padx=10, pady=10, bg="#f0f8ff")
        cmpf.pack(fill="x", padx=10, pady=5)
        self.compare_slider = tk.Scale(cmpf, from_=0, to=350, orient="horizontal",
                                       label="Reveal Enhanced from pixel", command=self.update_compare)
        self.compare_slider.pack(fill="x")
        self.c_cmp = tk.Canvas(cmpf, width=700, height=350, bg="white"); self.c_cmp.pack()

        # Save button
        tk.Button(root, text="💾 Save Enhanced", command=self.save_image,
                  bg="#9C27B0", fg="white", font=("Arial", 12)).pack(pady=10)

    def upload_image(self):
        fp = filedialog.askopenfilename(filetypes=[("Image Files","*.png;*.jpg;*.jpeg;*.bmp;*.tif")])
        if not fp: return
        self.image = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        self.show_on_canvas(self.image, self.c_orig)
        self.show_fourier()

    def show_on_canvas(self, img, canvas):
        img_r = cv2.resize(img, (350,350))
        tkimg = ImageTk.PhotoImage(Image.fromarray(img_r))
        canvas.img = tkimg; canvas.create_image(0,0,anchor="nw",image=tkimg)

    def show_fourier(self):
        fshift = fftshift(fft2(self.image))
        mag = np.log1p(np.abs(fshift))
        mag = np.uint8(255 * mag/np.max(mag))
        self.fourier_image = mag
        self.show_on_canvas(mag, self.c_four)

    def apply_filter(self):
        if self.image is None: messagebox.showerror("Error","Upload first"); return
        choice = self.filter_combo.get()
        p1, p2 = float(self.param1.get()), float(self.param2.get())
        if choice=="Gaussian Low Pass": img = self.gaussian_low_pass(self.image,p1)
        elif choice=="Median Filter": img = ndimage.median_filter(self.image,size=int(p1))
        else: img = self.bandpass_filter(self.image,p1,p2)
        self.processed_image = img; self.show_on_canvas(img,self.c_enh)

    def gaussian_low_pass(self,img,sigma):
        fshift = fftshift(fft2(img))
        r,c=img.shape; Y,X=np.ogrid[:r,:c]
        c0,c1=r//2,c//2
        mask=np.exp(-((X-c1)**2+(Y-c0)**2)/(2*sigma**2))
        back = np.abs(ifft2(ifftshift(fshift*mask)))
        return back

    def bandpass_filter(self,img,low,high):
        fshift = fftshift(fft2(img))
        r,c=img.shape; Y,X=np.ogrid[:r,:c]
        dist=np.sqrt((X-c//2)**2+(Y-r//2)**2)
        mask=(dist>=low)&(dist<=high)
        back=np.abs(ifft2(ifftshift(fshift*mask)))
        return back

    def auto_enhance(self):
        if self.image is None: messagebox.showerror("Error","Upload first"); return
        img=ndimage.median_filter(self.image,size=3)
        img=cv2.GaussianBlur(img,(5,5),1.5)
        img=cv2.addWeighted(img,1.5,img,-0.5,0)
        self.processed_image=img; self.show_on_canvas(img,self.c_enh)

    def plot_waveform(self):
        if self.fourier_image is None: return
        slice = self.fourier_image[self.image.shape[0]//2,:]
        fig = plt.Figure(figsize=(7,2))
        ax=fig.add_subplot(111); ax.plot(slice); ax.set_title("Freq Waveform"); ax.set_ylim(0,255)
        if self.wave_canvas: self.wave_canvas.get_tk_widget().destroy()
        self.wave_canvas=FigureCanvasTkAgg(fig,master=self.root); self.wave_canvas.draw()
        self.wave_canvas.get_tk_widget().pack(pady=5)

    def detect_noise(self):
        if self.fourier_image is None: return
        slice = self.fourier_image[self.image.shape[0]//2,:]
        thresh=np.mean(slice)+2*np.std(slice)
        peaks,_=signal.find_peaks(slice,height=thresh)
        fig=plt.Figure(figsize=(7,2)); ax=fig.add_subplot(111)
        ax.plot(slice); ax.scatter(peaks,slice[peaks],color='red'); ax.set_title(f"Noise Peaks: {len(peaks)}")
        if self.wave_canvas: self.wave_canvas.get_tk_widget().destroy()
        self.wave_canvas=FigureCanvasTkAgg(fig,master=self.root); self.wave_canvas.draw()
        self.wave_canvas.get_tk_widget().pack(pady=5)

    def update_compare(self,val):
        if self.image is None or self.processed_image is None: return
        o=cv2.resize(self.image,(350,350)); e=cv2.resize(self.processed_image,(350,350))
        thresh=int(val)
        comp=np.zeros_like(o); comp[:,:thresh]=o[:,:thresh]; comp[:,thresh:]=e[:,thresh:]
        tkimg=ImageTk.PhotoImage(Image.fromarray(np.uint8(comp)))
        self.c_cmp.img=tkimg; self.c_cmp.create_image(0,0,anchor='nw',image=tkimg)

    def save_image(self):
        if self.processed_image is None: messagebox.showerror("Error","No image to save"); return
        fp=filedialog.asksaveasfilename(defaultextension='.png')
        if fp:
            Image.fromarray(np.uint8(np.clip(self.processed_image,0,255))).save(fp)
            messagebox.showinfo("Saved","Image saved")

if __name__=='__main__':
    root=tk.Tk(); app=UltrasoundEnhancerApp(root); root.mainloop()
