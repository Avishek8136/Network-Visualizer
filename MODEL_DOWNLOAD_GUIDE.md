# Model Weights Download Guide

## Current Status
The app now works with **automatic model download** from TensorFlow.js CDN. The MobileNetV2 model (~9MB) downloads automatically when you open the Inference Explorer tab.

## Option 1: Automatic Download (Recommended - Already Implemented)
✅ **No action needed** - The model downloads automatically from:
- TensorFlow.js CDN
- MobileNetV2 with ImageNet weights
- Size: ~9MB
- Cached by browser after first download

## Option 2: Manual Download (If you want local files)

### Download Links for Pre-trained Models:

#### MobileNet (Recommended - Lightweight)
**Correct URLs:**
- **MobileNet V1**: `https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json`
- **MobileNet V2**: `https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json`

**Steps:**
1. Go to: https://github.com/tensorflow/tfjs-models/tree/master/mobilenet
2. Or download directly from TensorFlow.js CDN:
   - Model JSON: `https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json`
   - Weight shards will be downloaded automatically (group1-shard*.bin files)

3. Create a folder: `public/models/mobilenet/`
4. Place the following files there:
   - `model.json`
   - `group1-shard1of4.bin`
   - `group1-shard2of4.bin`
   - `group1-shard3of4.bin`
   - `group1-shard4of4.bin`

#### Alternative Models:

**ResNet50 (Deeper model):**
```
https://storage.googleapis.com/tfjs-models/tfjs/resnet50/model.json
```
Size: ~100MB

**EfficientNet (Best accuracy/size tradeoff):**
```
Available on TensorFlow Hub: https://tfhub.dev/tensorflow/tfjs-model/efficientnet/lite0/feature-vector/2/default/1
```

## How to Use Local Models

If you downloaded models locally, update the code in `InferenceExplorer.tsx`:

```typescript
// Change this line in the useEffect model loading:
const loadedModel = await mobilenet.load({
  version: 2,
  alpha: 1.0,
  modelUrl: 'http://localhost:5173/models/mobilenet/model.json' // Local path
});
```

## Current Implementation

The app currently uses:
- **MobileNetV2** for image classification (automatic download)
- **Custom CNN demonstration** for layer-by-layer visualization
- Both use real pretrained weights from ImageNet

### What Works Now:
✅ Upload or select images  
✅ Get top-3 predictions from MobileNet  
✅ See layer-by-layer processing with actual tensor operations  
✅ Visualize feature maps at each layer  
✅ Step through with animation controls  
✅ All weights download automatically  

### Technical Details:
- Model: MobileNetV2 (1.0 alpha, 224x224 input)
- Weights: Pretrained on ImageNet (1000 classes)
- Backend: WebGL (falls back to CPU if needed)
- Framework: TensorFlow.js 4.x

## No Download Needed! 🎉

The current implementation **automatically downloads and caches** the model when you:
1. Open the "Inference Explorer" tab
2. Wait 3-5 seconds for download to complete
3. Start using the feature immediately

The model is cached by your browser, so subsequent visits are instant!

## Troubleshooting

### If model won't download:
1. **Check internet connection** - Model needs to download ~9MB
2. **Disable browser extensions** - AdBlockers may block CDN
3. **Check browser console** - Look for specific error messages
4. **Try different browser** - Chrome/Edge work best with WebGL
5. **Clear browser cache** - Sometimes helps with corrupted downloads

### Manual download alternative:
If automatic download fails, you can:
1. Download model files from the links above
2. Place in `public/models/` folder
3. Update `modelUrl` parameter in the code
4. Serve from localhost instead of CDN

## Useful Resources

- **TensorFlow.js Models**: https://github.com/tensorflow/tfjs-models
- **TensorFlow Hub**: https://tfhub.dev/
- **MobileNet Paper**: https://arxiv.org/abs/1704.04861
- **TFJS API Docs**: https://js.tensorflow.org/api/latest/

## Model Specifications

### MobileNetV2 (Current)
- **Input**: 224×224×3 RGB image
- **Output**: 1000 ImageNet classes
- **Layers**: ~50 layers (conv, depthwise, batch norm)
- **Parameters**: ~3.5 million
- **Size**: ~9MB
- **Speed**: Fast (optimized for mobile/browser)

### ImageNet Classes
The model can recognize 1000 different categories including:
- Animals (dogs, cats, birds, etc.)
- Vehicles (cars, trucks, airplanes)
- Objects (furniture, electronics)
- And 990+ more categories

## No Action Required! ✅

The app is ready to use. Just:
1. Click "Inference Explorer" tab
2. Wait for model to load (one-time, ~5 seconds)
3. Upload/select an image
4. Click "Run Inference"
5. Explore layers!

Enjoy your Neural Network Inference Explorer! 🧠🚀
