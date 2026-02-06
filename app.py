import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import torch.nn as nn

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()        
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(128, 64, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(64, 32, kernel_size=3, stride=1),
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4),
        )

    def forward(self, x):
        return self.model(x)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential()
        current_idx = 0
        
        if upsample:
            self.block.add_module(str(current_idx), nn.Upsample(scale_factor=2, mode='nearest'))
            current_idx += 1
            
        reflection_padding = kernel_size // 2
        self.block.add_module(str(current_idx), nn.ReflectionPad2d(reflection_padding))
        current_idx += 1
        
        self.block.add_module(str(current_idx), nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        current_idx += 1
        
        if normalize:
            self.block.add_module(str(current_idx), nn.InstanceNorm2d(out_channels, affine=False))
            current_idx += 1

        if relu:
            self.block.add_module(str(current_idx), nn.ReLU(inplace=True))
            current_idx += 1

    def forward(self, x):
        return self.block(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),        
            nn.Conv2d(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=False), 
            nn.ReLU(inplace=True),        
            nn.ReflectionPad2d(1),         
            nn.Conv2d(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=False)  
        )

    def forward(self, x):
        return x + self.block(x)

@st.cache_resource
def load_pretrained_model(style_path):
    """Loads YOUR pretrained model from the specific path."""
    with torch.no_grad():
        style_model = TransformerNet()
        try:
            state_dict = torch.load(style_path, map_location='cpu')
            style_model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded model weights from {style_path}")
        except Exception as e:
            print(f"Could not load state_dict, trying full model load: {e}")
            style_model = torch.load(style_path, map_location='cpu')
             
        style_model.eval()
        return style_model

def transform_image(style_model, content_image):
    """Runs the image through the pretrained model to apply style."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess image to be compatible with the model
    
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        style_model.to(device)
        output = style_model(content_image).cpu()
        
    output = output.squeeze(0)
    
    output_np = output.numpy()
    
    p1 = np.percentile(output_np, 1)
    p99 = np.percentile(output_np, 99)
    
    if (p99 - p1) > 1e-5:
        output_np = (output_np - p1) / (p99 - p1)
        output_np = output_np * 255.0
    
    output_np = np.clip(output_np, 0, 255)
    
    output_np = output_np.transpose(1, 2, 0)
    output_np = output_np.astype("uint8")
    
    try:
        pil_img = Image.fromarray(output_np)
        from PIL import ImageEnhance
        converter = ImageEnhance.Color(pil_img)
        pil_img = converter.enhance(1.3) # 30% boost
        return pil_img
    except:
        return Image.fromarray(output_np)

# APP UI

if __name__ == "__main__":
    st.set_page_config(
        page_title="Neural Style Transfer",
        layout="centered"
    )

    st.title("Neural Style Transfer")
    st.caption("### *Deep Learning-based Image Transformation*")
    st.divider()

    tab_setup, tab_info = st.tabs(["Style Workspace", "About this App"])

    with tab_setup:
        col_cfg, col_upload = st.columns([1, 1], gap="large")
        
        with col_cfg:
            st.subheader("1. Style Selection")
            style_choice = st.selectbox(
                "Select a style model:",
                ("Starry Night", "Pointillism")
            )
            
            if style_choice == "Starry Night":
                model_path = "starry.pth"
                st.info("Inspiried by Van Gogh's expressive brushstrokes.")
            else:
                model_path = "pointillism.pth"
                st.info("A technique of painting in which small, distinct dots of color are applied.")
                
        with col_upload:
            st.subheader("2. Image Upload")
            uploaded_file = st.file_uploader("Upload an image file", type=['jpg', 'png', 'jpeg'])
        
        st.divider()

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Full width display for maximum visibility
            st.markdown("### Original Image")
            st.image(image, use_container_width=True)
            
            st.divider()

            if st.button('Apply Transformation', use_container_width=True):
                with st.spinner('Processing image...'):
                    try:
                        model = load_pretrained_model(model_path)
                        output_image = transform_image(model, image)
                        
                        st.markdown("### Stylized Result")
                        st.image(output_image, use_container_width=True)
                        
                        buf = io.BytesIO()
                        output_image.save(buf, format="JPEG")
                        st.download_button(
                                label="Download Processed Image",
                                data=buf.getvalue(),
                                file_name=f"stylized_{style_choice.lower().replace(' ', '_')}.jpg",
                                mime="image/jpeg",
                                use_container_width=True
                            )
                    except FileNotFoundError:
                        st.error(f"Model file '{model_path}' missing.")
                    except Exception as e:
                        st.error(f"Error occurred: {e}")
            else:
                st.info("Click the button to generate your art.")


    with tab_info:
        st.subheader("Professional Image Stylization")
        st.write("""
        TThis application helps you transform your photos using well-known artistic styles. By applying neural style transfer techniques, it reimagines your images with textures, patterns, 
        and brush strokes inspired by classic artworks.

Instead of using basic filters, the system looks at the actual structure and details of your image, then carefully blends them with the chosen artistic style. The result is a stylized image that keeps the original content intact while adding a distinctive artistic touch
        """)
        
        st.divider()
        st.markdown("### Application Features")
        st.write("- **Style Option**: Choose between two styles 'Starry Night' or 'Pointillism'.")
        st.write("- **Transformation**: Your image is combined with artistic textures using a pre-trained deep learning model to maintain balance between content and style.")
        st.write("- **Performance**: The application supports GPU acceleration, allowing faster image processing on compatible systems.")
        
        st.divider()
        st.markdown("### Technical Specification")
        st.json({
            "Engine": "TransformerNet Architecture",
            "Models": ["Starry Night (Van Gogh Style)", "Pointillism (Georges Seurat Style)"],
            "Input": "Single RGB Content Image",
            "Output": "Styled Image"
        })



