# LumiShadDiffusion: Physically-Accurate Shadow Generation via Light-Conditioned Diffusion

Official implementation of **LumiShadDiffusion**, a novel approach for generating physically accurate shadows in synthesized images through explicit lighting control and cross-attention alignment in latent diffusion models.

---

## Overview

LumiShadDiffusion addresses the fundamental challenge of generating physically plausible shadows in text-to-image diffusion models. While existing diffusion models excel at generating diverse visual content, they often produce shadows that lack physical consistency with lighting conditions or suffer from spatial misalignment. Our method introduces explicit shadow control into the diffusion pipeline through specialized token learning and cross-attention alignment, enabling precise manipulation of shadow characteristics based on lighting parameters.

---

## Key Contributions

### 1. Shadow Prompt Learning
We introduce specialized learnable tokens that encode shadow attributes including direction, intensity, and softness. These tokens, denoted as attribute tokens and shape tokens, are integrated directly into the text conditioning mechanism of the diffusion model, enabling fine-grained control over shadow generation without requiring full model retraining.

### 2. Synthetic Ray-Traced Dataset
Our training pipeline leverages Blender's Cycles ray-tracing engine to generate a comprehensive synthetic dataset with ground-truth shadow information. The dataset systematically varies light source parameters including polar and azimuthal angles, light size for shadow softness, and intensity values, providing diverse training examples that capture the physics of light transport and shadow formation through solving the rendering equation.

### 3. Cross-Attention Alignment Loss
We propose a novel cross-attention alignment loss that explicitly guides the model's attention maps to align with ground-truth shadow regions. This loss is computed as the cosine distance between the cross-attention maps corresponding to shadow tokens and the binary shadow masks, ensuring that the model learns to associate shadow-related text tokens with the correct spatial locations in the generated images.

### 4. Bounding Box-Based Segmentation for Shadow Localization
To prevent spatial drift and improve shadow placement accuracy, we introduce a bounding box-based segmentation strategy integrated with the Segment Anything Model (SAM). Instead of processing entire images, we first define axis-aligned bounding boxes around occluding objects and their potential shadow regions, then apply SAM within these constrained areas to generate precise shadow masks. This approach significantly reduces false positives and anchors shadows to their corresponding objects during training.

---

## Method

### Architecture

LumiShadDiffusion builds upon Latent Diffusion Models (LDM) and employs a parameter-efficient fine-tuning strategy called Custom Diffusion. Rather than fine-tuning the entire Stable Diffusion model, we selectively train only the Key (K) and Value (V) projection matrices in the cross-attention layers of the UNet architecture, alongside newly introduced shadow-specific token embeddings in the text encoder. This approach significantly reduces computational requirements while maintaining model expressiveness for shadow-related tasks.

### Light-Source Parameterization

Shadows are controlled through explicit light-source parameters denoted as Ψ = (p_L, θ_L, φ_L, I_L), where p_L represents the three-dimensional light position, θ_L and φ_L specify orientation angles (azimuth and elevation), and I_L controls intensity. These parameters directly influence the incident radiance at each surface point, determining whether the point lies in shadow and the sharpness of shadow boundaries. During inference, users can manipulate these parameters either through natural language descriptions or by directly adjusting numerical values to achieve desired shadow characteristics.

### Training Components

**Frozen Components:**
- Variational Autoencoder (VAE) - both encoder and decoder remain unchanged
- UNet self-attention layers and residual blocks
- Majority of text encoder parameters

**Trainable Components:**
- Cross-attention Key and Value projection matrices in UNet
- Newly introduced shadow attribute tokens and shape tokens
- Optional Query and output projections when full cross-attention training is enabled

### Loss Function

The training objective combines two complementary loss terms:

**Diffusion Loss:** The standard latent diffusion objective that measures the model's ability to predict and remove noise from corrupted latent representations, ensuring overall image quality and coherence.

**Cross-Attention Alignment Loss:** A novel term computed as one minus the cosine similarity between cross-attention maps for shadow tokens and ground-truth shadow masks. This loss explicitly guides the model to focus on correct shadow regions when processing shadow-related text tokens, with bounding box constraints further confining attention weights to plausible shadow areas.

The overall objective is expressed as a weighted combination of these two losses, with a hyperparameter λ controlling the relative importance of attention alignment.

---

## Dataset Generation Pipeline

Our synthetic dataset pipeline consists of five key stages:

1. **Asset Selection:** Curating diverse 3D meshes representing various object geometries, shapes, and materials from professional artists under free-use licenses.

2. **Blender/Cycles Setup:** Configuring the physically-based Cycles ray-tracing engine to accurately simulate light transport, including direct and indirect illumination effects.

3. **Light Sampling:** Systematically randomizing light source parameters across polar angles (0°-90°), azimuthal angles (0°-360°), light sizes (controlling shadow edge softness), and intensity values (modulating shadow darkness).

4. **Shadow-Mask Generation:** Rendering high-resolution RGB images with corresponding binary shadow masks that serve as pixel-perfect ground truth labels.

5. **Quality Assurance:** Validating rendered outputs for physical correctness and geometric accuracy before inclusion in the training set.

This automated pipeline eliminates the need for manual annotation, provides perfect ground-truth labels, and scales to generate datasets of arbitrary size while maintaining professional-grade quality.

---

## Training Procedure

The training process loads a pre-trained Stable Diffusion model and initializes custom attention processors for the cross-attention layers. New modifier tokens representing shadow attributes are added to the tokenizer vocabulary and initialized with embeddings from semantically related words. During training iterations, the model processes batches of synthetic images paired with shadow-attribute-embedded text prompts. At each diffusion timestep, cross-attention maps are extracted from the UNet and compared against ground-truth shadow masks using the alignment loss. Gradients are computed only for the trainable cross-attention parameters and shadow token embeddings, with all other model weights remaining frozen. The optimizer updates these parameters to minimize both reconstruction error and attention misalignment, gradually teaching the model to associate shadow tokens with correct spatial regions while maintaining image generation quality.

---

## Inference and Shadow Control

At inference time, users can generate images with controlled shadows through two mechanisms. First, natural language prompts can include learned shadow tokens along with descriptions of lighting conditions, such as "an object with soft shadow at 45 degrees" where the model interprets shadow-related tokens to guide generation. Second, explicit parameter specification allows direct control by providing numerical values for light position, angles, and intensity, offering precise manipulation of shadow characteristics. The bounding box constraints and learned attention alignment ensure that generated shadows remain spatially consistent with objects and physically plausible under the specified lighting conditions, avoiding common artifacts like floating shadows or incorrect shadow directions.

---

## Technical Specifications

**Base Model:** Stable Diffusion v1.4 or compatible versions

**Fine-Tuning Strategy:** Custom Diffusion (parameter-efficient, trains ~5% of model parameters)

**Training Framework:** PyTorch with Hugging Face Diffusers and Accelerate for distributed training

**Mixed Precision Support:** FP16 and BF16 (BF16 requires Ampere or newer GPUs)

**Memory Optimization:** Optional xformers for memory-efficient attention and 8-bit Adam optimizer

**Rendering Engine:** Blender 3.x with Cycles ray-tracing for dataset generation

**Segmentation Model:** Segment Anything Model (SAM) for refined shadow mask extraction

---

## Model Outputs

The trained model produces:
- High-resolution synthesized images with physically accurate shadows
- Attention map visualizations showing learned associations between text tokens and image regions
- Shadow masks aligned with specified lighting parameters
- Intermediate checkpoints for analysis and ablation studies

---

## Acknowledgments

This work builds upon Stable Diffusion, Custom Diffusion, and the Segment Anything Model. We thank the open-source community for providing 3D assets under free-use licenses and the developers of Blender's Cycles rendering engine for enabling physically-based dataset generation.

---

## License

This project is released under the MIT License. Pre-trained model weights inherit the license of the base Stable Diffusion model (CreativeML Open RAIL-M).

---

## Contact

For questions, issues, or collaboration inquiries, please open an issue in this repository or contact the authors through the information provided in the paper.
