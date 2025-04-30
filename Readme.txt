# Ultrasound Image Enhancement Using Fourier Transform

This project focuses on improving the quality of ultrasound images using Fourier Transform techniques. It applies frequency-domain filtering to reduce speckle noise and enhance important anatomical structures.

## 📚 Project Summary

- **University:** Mbarara University of Science and Technology  
- **Course:** Integral Calculus  
- **Year:** 1, Semester 2  
- **Group:** 7  
- **Team Members:**  
  - Kyegombe Jumah  
  - Semuli Joseph  
  - Mulungi Precious Pabire  
  - AinemBabazi Daphine  
  - Wasswa Emmanuel  
  - Tushabe Edwig  
  - Tukahirwa Clinton  
  - Ayebare Kevin  
  - Akandwanaho Onesmas

## 🛠 Tools & Libraries Used

- Python
- NumPy
- SciPy
- Matplotlib
- scikit-image

## 🔬 Methodology

1. **Image Loading** – Ultrasound image is loaded and converted to grayscale.
2. **Fourier Transform** – 2D FFT is applied to move to frequency domain.
3. **Filtering** – Low-pass and high-pass filters are used to suppress noise and enhance edges.
4. **Inverse Transform** – Image is converted back to spatial domain.
5. **Visualization** – Results are plotted for analysis.

## 🎯 Objectives

- Reduce speckle noise using low-pass filters
- Enhance structural detail using high-pass filters
- Compare results using SSIM and MSE metrics

## 📈 Results

| Filter Type | Result                                |
|-------------|----------------------------------------|
| Low-pass    | Reduced speckle noise, slightly blurred edges |
| High-pass   | Sharper edges, improved clarity         |

## 🧠 Challenges and Solutions

- **Noise-Detail Tradeoff:** Adaptive filtering and careful radius selection.
- **Artifact Reduction:** Used windowing techniques like Hamming/Gaussian windows.
- **Subjectivity in Enhancement:** Used quantitative metrics (SSIM, MSE).

## 🔍 Future Work

- Explore adaptive and hybrid filters
- Integrate machine learning for intelligent filtering

## 📎 References

- Gonzalez & Woods, *Digital Image Processing (4th ed.)*
- Wikipedia contributors – “Fourier Transforms”
- scikit-image, SciPy, Matplotlib documentation

---

*For more details, see the full report in `project.docx`.*
