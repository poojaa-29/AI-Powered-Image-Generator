import streamlit as st
from image_generator import ImageGenerator
import torch
import time

# Page configuration
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Content filter
INAPPROPRIATE_KEYWORDS = [
    "nude", "naked", "nsfw", "explicit", "violence", "gore", 
    "hate", "racist", "sexual"
]

def check_prompt_safety(prompt):
    """Check if prompt contains inappropriate content"""
    prompt_lower = prompt.lower()
    for keyword in INAPPROPRIATE_KEYWORDS:
        if keyword in prompt_lower:
            return False, f"Prompt contains inappropriate content: '{keyword}'"
    return True, "Prompt is safe"

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

# Title and description
st.title("🎨 AI-Powered Image Generator")
st.markdown("""
Generate high-quality images from text descriptions using Stable Diffusion.
This tool uses open-source AI models for creative image generation.

⚠️ **Responsible Use Guidelines:**
- Use respectful and appropriate prompts
- Generated images are watermarked as AI-created
- Do not generate harmful, illegal, or inappropriate content
""")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model loading
    if st.button("Load Model"):
        with st.spinner("Loading AI model... This may take a few minutes on first run."):
            try:
                st.session_state.generator = ImageGenerator()
                st.success("✅ Model loaded successfully!")
                
                device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
                st.info(f"Running on: {device}")
                
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    st.info(f"GPU: {gpu_name}")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    
    st.divider()
    
    st.subheader("Generation Parameters")
    
    num_images = st.slider("Number of Images", 1, 4, 1)
    
    style = st.selectbox(
        "Style",
        ["photorealistic", "artistic", "cartoon", "anime", "oil_painting", "digital_art"]
    )
    
    with st.expander("Advanced Settings"):
        guidance_scale = st.slider(
            "Guidance Scale",
            1.0, 20.0, 7.5, 0.5,
            help="Higher values follow prompt more closely"
        )
        
        num_steps = st.slider(
            "Inference Steps",
            20, 100, 50,
            help="More steps = better quality but slower"
        )
        
        width = st.selectbox("Width", [512, 768], index=0)
        height = st.selectbox("Height", [512, 768], index=0)
        
        seed = st.number_input(
            "Seed (optional)",
            min_value=0,
            max_value=2147483647,
            value=0
        )
        use_seed = st.checkbox("Use seed")
    
    st.divider()
    
    st.subheader("Output Settings")
    output_format = st.selectbox("Image Format", ["PNG", "JPEG"])
    output_dir = st.text_input("Output Directory", "generated_images")

# Main content
col1, col2 = st.columns([2, 3])

with col1:
    st.header("Input")
    
    prompt = st.text_area(
        "Enter your prompt",
        height=100,
        placeholder="Example: a futuristic city at sunset, highly detailed"
    )
    
    negative_prompt = st.text_area(
        "Negative Prompt (optional)",
        height=80,
        placeholder="Elements to avoid",
        value="blurry, bad quality, distorted, ugly, low resolution"
    )
    
    with st.expander("📝 Example Prompts"):
        st.markdown("""
        - "a futuristic city at sunset, cyberpunk style"
        - "portrait of a robot in Van Gogh style"
        - "a magical forest with glowing mushrooms"
        - "a cozy coffee shop interior, warm lighting"
        - "a majestic dragon flying over mountains"
        """)
    
    with st.expander("💡 Prompt Tips"):
        st.markdown("""
        **Good prompts include:**
        - Clear subject description
        - Style references (e.g., "Van Gogh style")
        - Quality descriptors (e.g., "highly detailed", "4K")
        - Lighting/mood (e.g., "dramatic lighting")
        
        **Example structure:**
        `[subject], [style], [quality], [lighting]`
        """)
    
    generate_btn = st.button("🎨 Generate Images", type="primary", use_container_width=True)

with col2:
    st.header("Generated Images")
    
    if generate_btn:
        if st.session_state.generator is None:
            st.error("⚠️ Please load the model first using the sidebar!")
        elif not prompt:
            st.warning("⚠️ Please enter a prompt!")
        else:
            is_safe, safety_msg = check_prompt_safety(prompt)
            if not is_safe:
                st.error(f"⚠️ {safety_msg}")
            else:
                with st.spinner("🎨 Generating images..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(current, total):
                        progress = (current + 1) / total
                        progress_bar.progress(progress)
                        status_text.text(f"Generating image {current + 1} of {total}...")
                    
                    start_time = time.time()
                    
                    try:
                        images, metadata_list = st.session_state.generator.generate_images(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_images=num_images,
                            style=style,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_steps,
                            width=width,
                            height=height,
                            seed=seed if use_seed else None,
                            progress_callback=update_progress
                        )
                        
                        saved_paths = st.session_state.generator.save_images(
                            images,
                            metadata_list,
                            output_dir=output_dir,
                            format=output_format
                        )
                        
                        generation_time = time.time() - start_time
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success(f"✅ Generated {num_images} image(s) in {generation_time:.1f}s")
                        
                        st.session_state.generated_images = list(zip(images, metadata_list, saved_paths))
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()

if st.session_state.generated_images:
    st.divider()
    
    for i, (image, metadata, path) in enumerate(st.session_state.generated_images):
        st.subheader(f"Image {i+1}")
        
        col_img, col_meta = st.columns([2, 1])
        
        with col_img:
            st.image(image, use_container_width=True)
            
            with open(path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Image",
                    data=file,
                    file_name=path.split("/")[-1],
                    mime=f"image/{output_format.lower()}"
                )
        
        with col_meta:
            st.json(metadata)

st.divider()
st.markdown("""
**Hardware Requirements:**
- GPU: NVIDIA GPU with 6GB+ VRAM (recommended)
- CPU: 16GB+ RAM (slower generation)

**Limitations:**
- Generation time: 10-60 seconds per image (GPU) or 2-5 minutes (CPU)
- Memory requirements: 6-8GB VRAM (GPU) or 16GB+ RAM (CPU)

**Future Improvements:**
- Fine-tuning on custom datasets
- Image-to-image generation
- Batch processing capabilities
""")