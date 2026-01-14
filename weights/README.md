# Model Weights Directory

## 📁 Purpose
This directory contains trained model weights for the Maize Disease Alert System.

## 🎯 Required File
Place your trained EfficientNetV2B0 model weights here:
- **Filename:** `maize_effnetv2.h5`
- **Full path:** `weights/maize_effnetv2.h5`

## 📊 Training Specifications
The model weights should be trained with:
- **Architecture:** EfficientNetV2B0 (ImageNet pre-trained)
- **Input shape:** 224×224×3
- **Classes:** 4 (Healthy, Common_Rust, Gray_Leaf_Spot, Northern_Leaf_Blight)
- **Dataset split:** 60% training, 20% validation, 20% testing
- **Fine-tuning:** Top 20% of base model layers trainable

## 🔄 Fallback Behavior
If `maize_effnetv2.h5` is not found:
- The system will use **pseudo-weights** for demonstration purposes
- Predictions will still work but may have lower accuracy
- A warning message will be displayed in the console

## 📝 How to Generate Weights
1. Use the **Dataset Info** tab in the app to split your dataset (60/20/20)
2. Train the EfficientNetV2B0 model using your split data
3. Save the trained model: `model.save_weights('weights/maize_effnetv2.h5')`
4. Place the `.h5` file in this directory

## ⚠️ Important Notes
- The `.h5` file can be large (50-200 MB)
- For GitHub deployment, consider:
  - Using Git LFS for large files
  - Storing weights on cloud storage (AWS S3, Google Cloud Storage)
  - Loading weights from a remote URL at runtime
- Current implementation automatically checks for weights file existence

## 🚀 Production Deployment
For Streamlit Cloud deployment:
```python
# Option 1: Download from cloud storage at startup
import urllib.request
urllib.request.urlretrieve(
    'https://your-storage-url.com/maize_effnetv2.h5',
    'weights/maize_effnetv2.h5'
)

# Option 2: Use Streamlit secrets for secure URL
weights_url = st.secrets["model_weights_url"]
```

## 📧 Contact
For trained weights or training assistance:
- **Email:** nicolenjuguna20@gmail.com
- **GitHub:** [@NicoleNjuguna](https://github.com/NicoleNjuguna)
