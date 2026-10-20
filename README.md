# ImageSearchTR 🔍

[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Model-MultiEmbedTR-yellow)](https://huggingface.co/utkubascakir/MultiEmbedTR)

A semantic image search engine powered by **[MultiEmbedTR](https://huggingface.co/utkubascakir/MultiEmbedTR)**, a Turkish multimodal embedding model. The project demonstrates how text and image embeddings can be brought together in a shared semantic space, enabling natural language image search in Turkish.

## Features

- 🧠 **Semantic Alignment** - Maps images and text to the same embedding space using MultiEmbedTR
- 🇹🇷 **Turkish Language** - Native support for Turkish queries and captions
- 🔍 **Text-to-Image Search** - Find images using natural language descriptions
- ⚡ **Fast Retrieval** - FAISS indexing for efficient similarity search
- 🎛️ **Hybrid Mode** - Optional caption-based refinement for improved results
- 🎨 **Interactive UI** - Clean Gradio interface for easy experimentation

## How It Works

ImageSearchTR leverages **MultiEmbedTR**, a multimodal embedding model trained to align Turkish text and images in a shared semantic space. This alignment enables semantic search: when you describe an image in Turkish, the model finds visually similar images by comparing embeddings.

**Pipeline:**

1. **Encoding**: Images and their captions are encoded into dense vectors using MultiEmbedTR
   - Images → Visual embeddings (image encoder)
   - Captions → Text embeddings (text encoder)
   - Both map to the same 768-dimensional semantic space

2. **Indexing**: Embeddings are indexed using FAISS for fast similarity search
   - Image embeddings → Primary search index
   - Caption embeddings → Optional auxiliary index for hybrid mode

3. **Search**: User query is encoded and matched against image embeddings
   - Text query → Query embedding (text encoder)
   - FAISS retrieves nearest neighbors by cosine similarity
   - **Hybrid mode** (optional): Combines image and caption similarity scores for refined results

The core idea: semantically similar text and images have similar embeddings, making text-to-image search possible through vector similarity.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ImageSearchTR.git
cd ImageSearchTR

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
faiss-cpu>=1.7.4  # or faiss-gpu for GPU support
gradio>=4.0.0
numpy>=1.24.0
Pillow>=9.0.0
tqdm>=4.65.0
```

## Quick Start

### 1. Download Dataset

Download Turkish image-caption pairs from HuggingFace:

```bash
python download_data.py --num-samples 10000 --output-dir data --seed 42
```

**Arguments:**
- `--num-samples`: Number of image-caption pairs to download (default: 10000)
- `--output-dir`: Directory to save images and captions (default: demo)
- `--seed`: Random seed for reproducibility (default: 14)

**Output structure:**
```
data/
├── captions.jsonl      # Caption metadata
├── 12345.jpeg          # Image files
├── 12346.jpeg
└── ...
```

### 2. Generate Embeddings

Encode images and captions into vector embeddings:

```bash
python encode.py \
    --images_dir data \
    --captions_dir data/captions.jsonl \
    --out_dir artifacts \
    --batch_size_img 32 \
    --batch_size_txt 64
```

**Arguments:**
- `--images_dir`: Directory containing images
- `--captions_dir`: Path to captions.jsonl file
- `--out_dir`: Output directory for embeddings (default: artifacts)
- `--model_name`: HuggingFace model ID (default: utkubascakir/MultiEmbedTR)
- `--batch_size_img`: Batch size for image encoding (default: 32)
- `--batch_size_txt`: Batch size for text encoding (default: 64)
- `--device`: Device to use: auto/cpu/cuda (default: auto)

**Output:**
```
artifacts/
├── image_emb.npy       # Image embeddings (N x D)
├── caption_emb.npy     # Caption embeddings (N x D)
├── paths.json          # Image file paths
└── captions.json       # Caption texts
```

### 3. Build FAISS Indices

Create searchable indices from embeddings:

```bash
python build_index.py \
    --artifacts_dir artifacts \
    --out_img_index artifacts/images.index \
    --out_cap_index artifacts/captions.index
```

**Arguments:**
- `--artifacts_dir`: Directory containing embeddings (default: artifacts)
- `--out_img_index`: Output path for image index (default: artifacts/images.index)
- `--out_cap_index`: Output path for caption index (default: artifacts/captions.index)

### 4. Launch Search Interface

Start the Gradio web interface:

```bash
python app.py
```

The interface will be available at `http://localhost:7860`

## Usage

### Web Interface

1. Enter a description in Turkish (e.g., "karlı dağlar", "deniz kenarında gün batımı")
2. (Optional) Enable **Deep Search** for hybrid image+caption matching
3. Adjust search parameters:
   - **Number of Results**: How many images to return (1-30)
   - **Image-Caption Balance**: 
     - `0.0` = Only caption similarity
     - `1.0` = Only image similarity
     - `0.5-0.7` = Balanced (recommended)
4. Click **Search**

## Project Structure

```
ImageSearchTR/
├── download_data.py        # Dataset downloader
├── encode.py               # Embedding generator
├── build_index.py          # FAISS index builder
├── search.py               # Search backend
├── app.py                  # Gradio web interface
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Downloaded images and captions
│   ├── *.jpeg
│   └── captions.jsonl
└── artifacts/             # Generated embeddings and indices
    ├── image_emb.npy
    ├── caption_emb.npy
    ├── paths.json
    ├── captions.json
    ├── images.index
    └── captions.index
```

## Advanced Configuration

### Custom Dataset

To use your own images and captions:

1. Create a directory with your images
2. Create a `captions.jsonl` file:
```json
{"path": "image1.jpg", "caption": "Açıklama metni"}
{"path": "image2.jpg", "caption": "Başka bir açıklama"}
```
3. Run the encoding pipeline:
```bash
python encode.py --images_dir your_images --captions_dir your_captions.jsonl
python build_index.py
```

### Search Modes

**Image-Only Search** (faster):
```python
results = search_image_only(query_vec, topk=10)
```

**Hybrid Search** (more accurate):
```python
results = search_hybrid(query_vec, topk=10, alpha=0.6)
```

The `alpha` parameter controls the fusion:
- `alpha=1.0`: Pure visual similarity
- `alpha=0.0`: Pure semantic (caption) similarity
- `alpha=0.5-0.7`: Balanced (recommended for most cases)

## Citation

If you use this project, please cite the underlying model:

```bibtex
@misc{multiembedtr2024,
  author = {Utku Başçakır},
  title = {MultiEmbedTR: Turkish Multimodal Embedding Model},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/utkubascakir/MultiEmbedTR}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MultiEmbedTR](https://huggingface.co/utkubascakir/MultiEmbedTR) for the embedding model
- [ITU Perceptron](https://huggingface.co/datasets/ituperceptron/image-captioning-turkish) for the Turkish image captioning dataset
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Gradio](https://gradio.app/) for the web interface
