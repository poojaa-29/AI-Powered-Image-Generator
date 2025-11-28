# 🎨 AI-Powered Image Generator

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Hardware Requirements](#-hardware-requirements)
- [Installation](#-installation)
- [Usage Instructions](#-usage-instructions)
- [Prompt Engineering Tips](#-prompt-engineering-tips)
- [Ethical AI Use](#-ethical-ai-use)
- [Project Structure](#-project-structure)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Troubleshooting](#-troubleshooting)
- [Sample Images](#-sample-images)


---

## 🎯 Project Overview

This project implements an AI-powered image generator using open-source Stable Diffusion models from Hugging Face. Users can generate images from text prompts through an intuitive web interface with full control over generation parameters, styles, and quality settings.

### Key Highlights

- 🚀 **Open-Source**: Uses only free, open-source models and libraries
- 🎨 **Multiple Styles**: 6 built-in artistic styles (photorealistic, artistic, cartoon, anime, oil painting, digital art)
- ⚙️ **Customizable**: Adjustable guidance scale, inference steps, and image dimensions
- 🖼️ **Batch Generation**: Generate multiple images per prompt
- 💾 **Smart Storage**: Automatic saving with metadata and timestamps
- 🔒 **Safe**: Content filtering and AI watermarking included
- 💻 **Flexible**: Supports both GPU (CUDA) and CPU inference

---

## ✨ Features

### Core Functionality

- **Text-to-Image Generation**: Convert any text description into high-quality images
- **Style Presets**: Choose from 6 predefined artistic styles
- **Negative Prompts**: Specify unwanted elements to exclude from generation
- **Adjustable Parameters**: 
  - Number of images per prompt (1-4)
  - Guidance scale (1.0-20.0)
  - Inference steps (20-100)
  - Image resolution (512x512, 768x768)
  - Optional seed for reproducibility

### User Interface

- **Clean Web Interface**: Built with Streamlit for ease of use
- **Real-time Progress**: Live progress bars and status updates
- **Image Preview**: Instant display of generated images
- **Download Options**: Export in PNG or JPEG formats
- **Metadata Display**: View generation parameters for each image

### Safety & Ethics

- **Content Filtering**: Automatic blocking of inappropriate keywords
- **AI Watermarking**: All images marked as "AI Generated"
- **Usage Guidelines**: Clear responsible use instructions
- **Metadata Tracking**: Full transparency of generation parameters

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Web Interface                         │
│                    (Streamlit UI)                         │
│                       app.py                              │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│                  Image Generator Core                     │
│                 (Business Logic)                          │
│                 image_generator.py                        │
│  ┌────────────┬──────────────┬─────────────────────┐    │
│  │  Prompt    │  Generation  │  Post-Processing    │    │
│  │ Enhancement│    Engine    │  & Watermarking     │    │
│  └────────────┴──────────────┴─────────────────────┘    │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│              Stable Diffusion Model                       │
│           (Hugging Face Diffusers)                        │
│                                                           │
│  ┌────────────┬──────────────┬─────────────────────┐    │
│  │   Text     │     U-Net    │        VAE          │    │
│  │  Encoder   │   Denoiser   │      Decoder        │    │
│  └────────────┴──────────────┴─────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Input** → Text prompt enters via Streamlit interface
2. **Prompt Enhancement** → Automatic quality descriptors added based on style
3. **Model Processing** → Stable Diffusion generates image through iterative denoising
4. **Post-Processing** → Watermark applied and metadata recorded
5. **Storage** → Image saved with JSON metadata file
6. **Display** → Preview shown in interface with download option

---

## 🛠️ Technology Stack

### Core Frameworks
- **PyTorch 2.0+** - Deep learning framework
- **Hugging Face Diffusers** - Stable Diffusion pipeline
- **Transformers** - Text encoding models
- **Streamlit** - Web interface framework

### Model Details
- **Base Model**: Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)
- **Scheduler**: DPM Solver Multistep (faster generation)
- **Text Encoder**: CLIP ViT-L/14
- **Image Resolution**: Up to 768x768 pixels
- **Parameters**: ~1 billion

### Supporting Libraries
- **PIL/Pillow** - Image processing and watermarking
- **Accelerate** - Model optimization
- **SafeTensors** - Efficient model loading
- **JSON** - Metadata storage

---

## 💻 Hardware Requirements

### Recommended Configuration (GPU)

| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA GPU with 6GB+ VRAM |
| **Examples** | GTX 1060 6GB, RTX 2060, RTX 3060, or better |
| **CPU** | Any modern processor |
| **RAM** | 16GB system RAM |
| **Storage** | 15GB free space (10GB for model, 5GB for dependencies) |
| **OS** | Windows 10/11, Linux, macOS |
| **Generation Time** | 10-30 seconds per image |

### Minimum Configuration (CPU Fallback)

| Component | Requirement |
|-----------|-------------|
| **CPU** | Multi-core processor (Intel i5/i7 or AMD Ryzen 5/7) |
| **RAM** | 16GB minimum (32GB recommended) |
| **Storage** | 20GB free space |
| **OS** | Windows 10/11, Linux, macOS |
| **Generation Time** | 2-5 minutes per image |


## 📥 Installation

### Prerequisites

Ensure you have the following installed:
- **Python 3.8 or higher** ([Download](https://www.python.org/downloads/))
- **Git** ([Download](https://git-scm.com/downloads))
- **CUDA Toolkit 11.8+** (optional, for GPU support) ([Download](https://developer.nvidia.com/cuda-downloads))

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-image-generator.git
cd ai-image-generator
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your command line.

### Step 3: Install Dependencies

**For GPU (CUDA):**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install diffusers transformers accelerate safetensors streamlit Pillow
```

**For CPU Only:**
```bash
# Install PyTorch CPU version
pip install torch torchvision

# Install other requirements
pip install diffusers transformers accelerate safetensors streamlit Pillow
```

### Step 4: Download Model (First Run)

The Stable Diffusion model (~4GB) will automatically download on first use. This takes 10-20 minutes depending on your internet connection.

**Optional - Pre-download the model:**
```python
from diffusers import StableDiffusionPipeline

# This downloads the model to cache
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
print("Model downloaded successfully!")
```

### Step 5: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.8+

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA availability (GPU only)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check Streamlit
streamlit --version
```

---

## 🚀 Usage Instructions

### Starting the Application

1. **Activate virtual environment** (if not already active):
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

2. **Run Streamlit app**:
```bash
streamlit run app.py
```

3. **Open in browser**: 
   - Automatically opens at `http://localhost:8501`
   - If not, manually navigate to the URL shown in terminal

### Generating Your First Image

#### Step 1: Load the Model
- Click **"Load Model"** button in the left sidebar
- Wait for "Model loaded successfully!" message
- Check hardware info (GPU/CPU) displayed

#### Step 2: Enter a Prompt
Example prompts to try:
```
a futuristic city at sunset, cyberpunk style, neon lights, highly detailed
portrait of a robot in Van Gogh style, oil painting, expressive brushstrokes
a magical forest with glowing mushrooms, fantasy art, ethereal lighting
a cozy coffee shop interior, warm lighting, bokeh, photorealistic
majestic dragon flying over mountains, epic scale, dramatic clouds
```

#### Step 3: Choose Style
Select from dropdown:
- **Photorealistic** - Realistic photography style
- **Artistic** - General artistic/painterly style
- **Cartoon** - Animated, vibrant style
- **Anime** - Japanese manga/anime style
- **Oil Painting** - Classical oil painting style
- **Digital Art** - Modern digital illustration

#### Step 4: Adjust Parameters (Optional)

**Basic Settings:**
- **Number of Images**: 1-4 images per generation
- **Style**: Choose artistic direction

**Advanced Settings** (expand section):
- **Guidance Scale** (7.5 default): Higher = follows prompt more closely
- **Inference Steps** (50 default): More steps = better quality but slower
- **Resolution**: 512x512 (fast) or 768x768 (better quality)
- **Seed**: Enable for reproducible results

#### Step 5: Generate
- Click **"🎨 Generate Images"** button
- Watch progress bar
- View generated images
- Download using the download button

### Using Negative Prompts

Negative prompts help exclude unwanted elements:

**Default negative prompt:**
```
blurry, bad quality, distorted, ugly, low resolution
```

**Additional examples:**
```
# For portraits
extra fingers, deformed hands, blurry face, bad anatomy

# For landscapes
people, humans, text, watermark, signature

# For photorealistic images
cartoon, anime, drawing, painting, illustration
```

---

## 💡 Prompt Engineering Tips

### Best Practices

#### 1. Structure Your Prompts
Use this formula for best results:
```
[Subject] + [Style] + [Quality] + [Lighting] + [Details]
```

**Example:**
```
a red sports car, professional photography, highly detailed, golden hour lighting, shallow depth of field
```

#### 2. Quality Descriptors
Always include quality terms:
- `highly detailed`
- `4K`, `8K`
- `professional photography`
- `sharp focus`
- `trending on artstation`
- `award winning`
- `masterpiece`

#### 3. Style References
Reference artists or art movements:
- `in the style of Van Gogh`
- `Pixar style animation`
- `Studio Ghibli`
- `cyberpunk aesthetic`
- `art nouveau`
- `impressionist painting`

#### 4. Lighting Terms
Specify lighting for better results:
- `golden hour lighting`
- `dramatic lighting`
- `soft diffused light`
- `rim lighting`
- `volumetric lighting`
- `cinematic lighting`

#### 5. Camera/Lens Terms (for photorealistic)
- `shot on Canon 5D`
- `85mm lens`
- `shallow depth of field`
- `bokeh`
- `wide angle`
- `macro photography`

### Example Prompts by Category

#### **Portraits**
```
portrait of an elderly wizard, long white beard, wise eyes, fantasy art, detailed face, dramatic lighting, oil painting style

close-up portrait of a woman with flowers in her hair, soft lighting, bokeh background, professional photography, 85mm lens, shallow depth of field
```

#### **Landscapes**
```
misty mountain landscape at sunrise, dramatic clouds, golden hour, professional landscape photography, highly detailed, 8K

alien planet with two moons, purple sky, strange rock formations, sci-fi concept art, highly detailed, trending on artstation
```

#### **Fantasy/Sci-Fi**
```
futuristic cyberpunk city, neon lights, rain-soaked streets, flying cars, night scene, cinematic, highly detailed, blade runner aesthetic

medieval castle on a cliff, dragons flying in the distance, fantasy art, epic scale, dramatic clouds, detailed architecture
```

#### **Animals**
```
majestic lion with golden mane, sunset savanna, professional wildlife photography, shallow depth of field, highly detailed fur

cute red panda eating bamboo, soft lighting, nature photography, bokeh background, adorable, highly detailed
```

#### **Abstract/Artistic**
```
abstract representation of music, flowing colors, dynamic composition, digital art, vibrant, highly detailed

surreal dreamscape, melting clocks, Salvador Dali style, oil painting, highly detailed, museum quality
```

### Common Mistakes to Avoid

❌ **Too vague**: "a person"
✅ **Better**: "portrait of a young woman with red hair, green eyes, soft lighting, photorealistic"

❌ **Too complex**: "a cat and a dog and a bird and a mouse playing together in a garden with flowers and trees and a house"
✅ **Better**: "a cat and dog playing together in a garden, flowers, warm sunlight, photorealistic"

❌ **Conflicting styles**: "photorealistic anime cartoon oil painting"
✅ **Better**: Choose one style - "anime style" OR "photorealistic" OR "oil painting"

---

## ⚖️ Ethical AI Use

### Responsible Use Guidelines

This tool is designed for ethical and legal image generation. Users must:

✅ **Do:**
- Use for creative and educational purposes
- Generate original artistic content
- Credit AI generation when sharing images
- Respect copyright and intellectual property
- Use appropriate, non-harmful prompts

❌ **Don't:**
- Generate harmful, illegal, or inappropriate content
- Create deepfakes or misleading images of real people
- Violate copyright by copying existing artworks
- Generate content for deceptive purposes
- Bypass content filters

### Content Filtering

The application includes automatic filtering for:
- Explicit or adult content keywords
- Violence and gore
- Hate speech and discrimination
- Harmful or illegal activities

**Note**: Users are responsible for the content they generate. The automated filter is a safety measure but cannot catch all inappropriate use.

### AI Watermarking

All generated images include:
- Visible "AI Generated" watermark in bottom-right corner
- Complete metadata in accompanying JSON files
- Timestamp and generation parameters

**Purpose**: Transparency and prevention of misleading use.

### Data Privacy

- ✅ All processing happens locally on your machine
- ✅ No images or prompts are sent to external servers
- ✅ No data collection or analytics
- ✅ Complete privacy and control

### Legal Considerations

- **License**: Generated images use CreativeML Open RAIL-M License
- **Commercial Use**: Check license terms for commercial applications
- **Copyright**: You are responsible for ensuring legal use
- **Attribution**: Consider crediting the AI model when sharing

---

## 📁 Project Structure

```
ai-image-generator/
│
├── app.py                          # Streamlit web interface
├── image_generator.py              # Core generation logic and model handling
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── generated_images/               # Output directory (auto-created)
│   ├── img_20241128_143022_1.png # Generated image
│   ├── img_20241128_143022_1_metadata.json # Image metadata
│   └── ...
│
├── samples/                        # Sample generated images for showcase
│   ├── cat_photorealistic.png
│   ├── city_cyberpunk.png
│   └── ...
│
└── venv/                          # Virtual environment (not in git)
    └── ...
```

### Key Files Description

#### `app.py`
Streamlit web application providing:
- User interface for prompt input
- Parameter adjustment controls
- Real-time generation progress
- Image display and download
- Content safety filtering

#### `image_generator.py`
Core business logic including:
- Model initialization and loading
- Prompt enhancement based on style
- Image generation pipeline
- Watermarking functionality
- Metadata creation and storage

#### `requirements.txt`
All Python dependencies with minimum versions:
```
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
safetensors>=0.3.0
streamlit>=1.28.0
Pillow>=9.5.0
```

---

## 🚧 Limitations

### Current Limitations

#### 1. Generation Time
- **GPU**: 10-30 seconds per image at 512x512
- **CPU**: 2-5 minutes per image
- Higher resolutions take proportionally longer

#### 2. Memory Requirements
- **GPU**: 6-8GB VRAM minimum
- **CPU**: 16GB+ RAM recommended
- Out-of-memory errors possible with large batches

#### 3. Image Resolution
- Maximum recommended: 768x768 for stability
- Higher resolutions (1024+) may cause OOM errors on lower-end hardware
- Resolution limited by available VRAM/RAM

#### 4. Content Generation
- **Text rendering**: Poor quality for text within images
- **Complex scenes**: May struggle with very detailed multi-object scenes
- **Hands/faces**: Occasional anatomical inaccuracies
- **Fine details**: Small objects may lack detail at lower resolutions

#### 5. Model Limitations
- Knowledge cutoff: Training data up to 2023
- Bias: Inherits biases from training data
- Consistency: Same prompt may produce different results
- Style mixing: Multiple conflicting styles may produce unclear results

#### 6. Technical Constraints
- Internet required for first-time model download
- Disk space: ~15GB needed for full installation
- No image-to-image or inpainting in current version
- Limited to English prompts (best performance)

---

## 🔮 Future Improvements

### Planned Features

#### Phase 1: Enhanced Generation
- [ ] **Image-to-Image**: Modify existing images based on prompts
- [ ] **Inpainting**: Edit specific regions of images
- [ ] **Outpainting**: Extend images beyond their borders
- [ ] **Upscaling**: Increase resolution of generated images
- [ ] **ControlNet**: Precise control over composition

#### Phase 2: Model Enhancements
- [ ] **Stable Diffusion XL**: Upgrade to SDXL for better quality
- [ ] **LoRA Support**: Fine-tune on custom styles
- [ ] **Multiple Model**: Support switching between different models
- [ ] **Custom Training**: Fine-tune on user-provided datasets
- [ ] **Style Transfer**: Transfer style from reference images

#### Phase 3: User Experience
- [ ] **Image History**: Gallery of all generated images
- [ ] **Prompt Library**: Save and reuse favorite prompts
- [ ] **Batch Processing**: Process multiple prompts automatically
- [ ] **Export Options**: Batch download, ZIP archives
- [ ] **Comparison View**: Side-by-side comparison of variations

#### Phase 4: Advanced Features
- [ ] **Animation**: Create short videos from prompts
- [ ] **Interpolation**: Smooth transitions between images
- [ ] **API Endpoint**: RESTful API for programmatic access
- [ ] **Cloud Deployment**: Host on cloud platforms
- [ ] **Mobile App**: Native mobile application

#### Phase 5: Optimization
- [ ] **Model Quantization**: Reduce memory footprint
- [ ] **Streaming Preview**: Show generation in progress
- [ ] **GPU Sharing**: Multi-user support
- [ ] **Caching**: Cache similar prompts for faster generation
- [ ] **Progressive Generation**: Show intermediate steps

#### Phase 6: Community Features
- [ ] **Public Gallery**: Share generations with community
- [ ] **Prompt Sharing**: Community prompt templates
- [ ] **Rating System**: Vote on best generations
- [ ] **Challenges**: Weekly theme challenges
- [ ] **Export to Social**: Direct sharing to social media

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Model Won't Load
**Error**: "Cannot load model stabilityai/stable-diffusion..."

**Solutions**:
```bash
# Solution A: Clear cache and retry
# Windows PowerShell:
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface"

# Mac/Linux:
rm -rf ~/.cache/huggingface

# Solution B: Check internet connection
ping huggingface.co

# Solution C: Use different model
# Edit image_generator.py line 8:
# Change to: model_id="runwayml/stable-diffusion-v1-5"
```

#### Issue 2: Out of Memory (OOM)
**Error**: "CUDA out of memory" or "RuntimeError: out of memory"

**Solutions**:
1. **Reduce image resolution**:
   - Use 512x512 instead of 768x768
2. **Generate fewer images**:
   - Set "Number of Images" to 1
3. **Lower inference steps**:
   - Try 30-40 steps instead of 50
4. **Enable CPU offload** (edit `image_generator.py`):
```python
# Add after line 24:
self.pipe.enable_sequential_cpu_offload()
```

#### Issue 3: Slow Generation (CPU)
**Issue**: Taking 5+ minutes per image

**Solutions**:
1. **Reduce inference steps**: Use 20-30 instead of 50
2. **Lower resolution**: Use 512x512
3. **Use smaller model**:
```python
# Edit image_generator.py line 8:
model_id="segmind/small-sd"  # Only 2GB
```

#### Issue 4: Import Errors
**Error**: "ModuleNotFoundError: No module named 'torch'"

**Solutions**:
```bash
# Ensure venv is activated (look for (venv) in terminal)
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# Reinstall dependencies:
pip install -r requirements.txt
```

#### Issue 5: Streamlit Won't Start
**Error**: "streamlit: command not found"

**Solutions**:
```bash
# Check if streamlit is installed:
pip list | grep streamlit

# If not found, install:
pip install streamlit

# Try running with python -m:
python -m streamlit run app.py
```

#### Issue 6: Generated Images Look Bad
**Issue**: Blurry, distorted, or low-quality images

**Solutions**:
1. **Improve your prompt**:
   - Add quality descriptors: "highly detailed, 4K, sharp focus"
   - Be more specific about subject
2. **Increase inference steps**: Try 75-100 steps
3. **Adjust guidance scale**: Try 7-12 range
4. **Use better negative prompts**:
   ```
   blurry, bad quality, distorted, deformed, ugly, bad anatomy, low resolution, artifacts
   ```

#### Issue 7: Content Filter False Positives
**Issue**: Safe prompts being blocked

**Solutions**:
1. **Rephrase prompt**: Avoid flagged keywords
2. **Edit filter** (if appropriate) in `app.py`:
```python
# Line 15-18: Modify INAPPROPRIATE_KEYWORDS list
# Only remove keywords you're certain are safe for your use case
```

#### Issue 8: Watermark Not Visible
**Issue**: Can't see "AI Generated" watermark

**Solutions**:
1. **Check bright backgrounds**: Watermark is white, may not show on light images
2. **Check metadata file**: Generation info always saved in JSON
3. **Modify watermark color** in `image_generator.py`:
```python
# Line 140: Change watermark color
fill=(0, 0, 0, 200)  # Black instead of white
```

---

## 🖼️ Sample Images

The `samples/` directory contains example images generated with various prompts and styles:

| Image | Prompt | Style |
|-------|--------|-------|
| `cat_photorealistic.png` | "a cute orange cat, professional photography" | Photorealistic |
| `city_cyberpunk.png` | "futuristic city at sunset, neon lights" | Digital Art |
| `robot_vangogh.png` | "portrait of a robot in Van Gogh style" | Oil Painting |
| `forest_fantasy.png` | "magical forest with glowing mushrooms" | Artistic |
| `dragon_epic.png` | "majestic dragon over mountains" | Digital Art |

*Note: Sample images demonstrate the range of styles and quality achievable with this system.*

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Reporting Issues
- Use GitHub Issues to report bugs
- Include error messages and system info
- Provide steps to reproduce

### Suggesting Features
- Open a GitHub Issue with feature request
- Describe use case and expected behavior
- Consider implementation complexity

### Pull Requests
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature-name`
6. Open Pull Request with description

### Code Style
- Follow PEP 8 for Python code
- Add docstrings to functions
- Include comments for complex logic
- Update README for new features

---

## 📄 License

### Project License
This project is licensed under the **MIT License** - see below for details.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Model License
The Stable Diffusion model is licensed under **CreativeML Open RAIL-M License**.

Key points:
- ✅ Free to use for research and commercial purposes
- ✅ Can modify and distribute
- ❌ Must not use for illegal activities
- ❌ Must not generate harmful content

Full license: [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)

---

## 📞 Contact

### Project Developer
- **Name**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)

### Project Links
- **Repository**: [github.com/yourusername/ai-image-generator](https://github.com/yourusername/ai-image-generator)
- **Issues**: [Report a bug](https://github.com/yourusername/ai-image-generator/issues)
- **Discussions**: [Join discussion](https://github.com/yourusername/ai-image-generator/discussions)

### Acknowledgments
- **Talrn.com** - For the internship opportunity and task assessment
- **Stability AI** - For the Stable Diffusion model
- **Hugging Face** - For the Diffusers library and model hosting
- **Streamlit** - For the web framework

---

## 📊 Project Statistics

- **Development Time**: ~40 hours
- **Lines of Code**: ~800
- **Dependencies**: 8 major libraries
- **Model Size**: 4GB
- **Supported Platforms**: Windows, macOS, Linux
- **Python Version**: 3.8+

---

## 🎓 Educational Value

This project demonstrates:
- **Deep Learning**: Implementation of diffusion models
- **Computer Vision**: Image generation and processing
- **Web Development**: Full-stack application with Streamlit
- **Software Engineering**: Clean code, documentation, version control
- **Ethics in AI**: Responsible AI development and usage

---

<div align="center">

**Built with ❤️ for the ML Internship at Talrn.com**

⭐ **Star this repo** if you find it helpful!

[Report Bug](https://github.com/yourusername/ai-image-generator/issues) · [Request Feature](https://github.com/yourusername/ai-image-generator/issues) · [Documentation](https://github.com/yourusername/ai-image-generator/wiki)

</div>

