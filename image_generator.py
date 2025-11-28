import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
from datetime import datetime
import json

class ImageGenerator:
    def __init__(self, model_id="nota-ai/bk-sdm-small", device=None):
        """rmdir /s "%USERPROFILE%\.cache\huggingface"
        Initialize the image generator
        Args:
            model_id: Hugging Face model ID
            device: 'cuda' for GPU, 'cpu' for CPU
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model on {self.device}...")
        
        # Load the model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        
        # Use DPM Solver for faster generation
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
        
        print("Model loaded successfully!")
    
    def enhance_prompt(self, prompt, style="photorealistic"):
        """
        Enhance prompt with quality descriptors
        """
        style_prompts = {
            "photorealistic": "highly detailed, 4K, professional photography, sharp focus, realistic",
            "artistic": "beautiful artwork, trending on artstation, detailed painting, artistic",
            "cartoon": "cartoon style, vibrant colors, animated, illustration",
            "anime": "anime style, manga, highly detailed anime art",
            "oil_painting": "oil painting, classical art, detailed brushstrokes",
            "digital_art": "digital art, concept art, detailed, trending on artstation"
        }
        
        enhancement = style_prompts.get(style, style_prompts["photorealistic"])
        enhanced = f"{prompt}, {enhancement}"
        return enhanced
    
    def generate_images(
        self,
        prompt,
        negative_prompt="",
        num_images=1,
        style="photorealistic",
        guidance_scale=7.5,
        num_inference_steps=50,
        width=512,
        height=512,
        seed=None,
        progress_callback=None
    ):
        """
        Generate images from text prompt
        """
        # Enhance prompt based on style
        enhanced_prompt = self.enhance_prompt(prompt, style)
        
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Default negative prompt
        if not negative_prompt:
            negative_prompt = "blurry, bad quality, distorted, ugly, low resolution, artifacts"
        
        images = []
        metadata_list = []
        
        for i in range(num_images):
            if progress_callback:
                progress_callback(i, num_images)
            
            # Generate image
            with torch.inference_mode():
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                )
            
            image = result.images[0]
            images.append(image)
            
            # Create metadata
            metadata = {
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "style": style,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
                "image_number": i + 1
            }
            metadata_list.append(metadata)
        
        return images, metadata_list
    
    def save_images(self, images, metadata_list, output_dir="generated_images", format="PNG"):
        """
        Save generated images with metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        
        for i, (image, metadata) in enumerate(zip(images, metadata_list)):
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"img_{timestamp}_{i+1}.{format.lower()}"
            filepath = os.path.join(output_dir, filename)
            
            # Add watermark
            image_with_watermark = self.add_watermark(image)
            
            # Save image
            image_with_watermark.save(filepath, format=format)
            
            # Save metadata
            metadata_file = filepath.replace(f".{format.lower()}", "_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            saved_paths.append(filepath)
        
        return saved_paths
    
    def add_watermark(self, image):
        """
        Add AI-generated watermark to image
        """
        from PIL import ImageDraw, ImageFont
        
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        text = "AI Generated"
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        position = (img_copy.width - text_width - 10, img_copy.height - text_height - 10)
        draw.text(position, text, fill=(255, 255, 255, 128), font=font)
        
        return img_copy