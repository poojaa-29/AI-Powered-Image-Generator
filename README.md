# AI-Powered Image Generator

A text-to-image generation system using Stable Diffusion that converts textual descriptions into high-quality images.

## 🎯 Project Overview

This project implements an AI-powered image generator using open-source models (Stable Diffusion) with a user-friendly Streamlit web interface. Users can generate images from text prompts with customizable parameters and styles.

### Architecture
```
┌─────────────────┐
│  User Interface │ (Streamlit)
│   (app.py)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Image Generator │ (Core Logic)
│(image_generator │
│     .py)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Stable Diffusion│ (Hugging Face)
│     Model       │
└─────────────────┘
```

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- 10GB+ free disk space for model files

### Installation Steps

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd ai-image-generator
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

For GPU (CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

For CPU only:
```bash
pip install torch torchvision
pip install -r requirements.txt
```

4. **Download the model:**

The model will automatically download on first run (~5GB). Ensure stable internet connection.

## 💻 Hardware Requirements

### Recommended (GPU):
- NVIDIA GPU with 6GB+ VRAM (GTX 1060/RTX 2060 or better)
- 16GB System RAM
- 15GB free disk space

### Minimum (CPU):
- 16GB System RAM
- 20GB free disk space
- Note: Generation will be significantly slower (2-5 minutes per image)

## 📖 Usage Instructions

### Running the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Generating Images

1. Click "Load Model" in the sidebar (first-time setup)
2. Enter your text prompt (e.g., "a futuristic city at sunset")
3. Adjust parameters:
   - Number of images (1-4)
   - Style (photorealistic, artistic, cartoon, etc.)
   - Advanced settings (guidance scale, steps, resolution)
4. Click "Generate Images"
5. Download generated images

### Example Prompts
```
- "a futuristic city at sunset, cyberpunk style, neon lights"
- "portrait of a robot in Van Gogh style, oil painting"
- "a magical forest with glowing mushrooms, fantasy art"
- "a cozy coffee shop interior, warm lighting, detailed"
- "majestic dragon flying over mountains, epic scale"
```

## 🎨 Prompt Engineering Tips

### Best Practices

1. **Be Specific**: Include details about subject, style, and mood
2. **Use Quality Descriptors**: Add terms like "highly detailed", "4K", "professional"
3. **Reference Styles**: Mention art styles or artists (e.g., "Van Gogh style")
4. **Describe Lighting**: Include lighting details (e.g., "dramatic lighting", "golden hour")
5. **Add Negative Prompts**: Specify what to avoid (e.g., "blurry, distorted")

### Prompt Structure
```
[Subject] + [Style] + [Quality] + [Lighting] + [Additional Details]

Example:
"a cute cat, digital art, highly detailed, soft lighting, sitting on a windowsill"
```

### Style Keywords

- **Photorealistic**: "photograph, realistic, detailed, sharp focus"
- **Artistic**: "artwork, painting, artistic, creative"
- **Anime**: "anime style, manga, cel shaded"
- **3D Render**: "3d render, octane render, ray traced"

## 🛠️ Technology Stack

- **Framework**: PyTorch 2.0+
- **Model**: Stable Diffusion 2.1 (stabilityai)
- **Library**: Hugging Face Diffusers
- **Web Interface**: Streamlit
- **Image Processing**: PIL/Pillow
- **Scheduler**: DPM Solver (faster generation)

## ⚖️ Ethical AI Use

### Content Filtering
- Automatic blocking of inappropriate keywords
- Manual review of generated content recommended
- Report issues for model improvement

### Watermarking
- All generated images include "AI Generated" watermark
- Metadata saved with each image

### Responsible Use Guidelines
- Do not generate harmful or illegal content
- Respect copyright and intellectual property
- Use generated images ethically
- Credit AI generation when sharing

## 🔒 Limitations

### Current Limitations
1. **Generation Time**: 
   - GPU: 10-30 seconds per image
   - CPU: 2-5 minutes per image

2. **Memory Requirements**:
   - 6-8GB VRAM for GPU
   - 16GB+ RAM for CPU

3. **Resolution**: 
   - Maximum 1024x1024 for memory efficiency
   - Higher resolutions may cause OOM errors

4. **Content**: 
   - May struggle with complex scenes
   - Text rendering in images is limited
   - Occasional artifacts in generated images

## 🚧 Future Improvements

1. **Model Enhancements**:
   - Fine-tuning on custom datasets
   - Support for Stable Diffusion XL
   - LoRA model integration

2. **Features**:
   - Image-to-image generation
   - Inpainting and outpainting
   - Style transfer capabilities
   - Batch processing for multiple prompts

3. **Interface**:
   - Image history gallery
   - Prompt templates library
   - Community sharing features

4. **Optimization**:
   - Model quantization for faster inference
   - Streaming generation preview
   - Cloud deployment options

## 📁 Project Structure
```
ai-image-generator/
├── app.py                 # Streamlit web interface
├── image_generator.py     # Core generation logic
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── generated_images/     # Output directory
│   ├── img_*.png        # Generated images
│   └── img_*_metadata.json  # Image metadata
└── samples/              # Sample generated images
```

## 🐛 Troubleshooting

### Out of Memory Error
- Reduce image resolution (512x512)
- Decrease number of inference steps
- Generate fewer images at once
- Enable CPU offloading

### Slow Generation
- Ensure GPU is being used (check sidebar)
- Install xformers for optimization
- Reduce inference steps (trade quality for speed)

### Model Download Issues
- Check internet connection
- Ensure sufficient disk space
- Try manual download from Hugging Face

## 📄 License

This project uses open-source models and libraries:
- Stable Diffusion: CreativeML Open RAIL-M License
- Code: MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your-email]

---

**Note**: This is developed as part of the ML Internship Task Assessment for Talrn.com
```

---

## Additional Files to Include

### **5. `.gitignore`**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Model files
*.ckpt
*.safetensors
models/

# Generated images (optional, you may want to commit samples)
generated_images/*
!generated_images/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/