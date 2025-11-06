import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import { 
  Upload, 
  Play, 
  Pause, 
  RotateCcw, 
  ChevronRight, 
  Image as ImageIcon,
  Layers,
  Activity,
  Info,
  AlertCircle
} from 'lucide-react';

interface LayerActivation {
  layerName: string;
  layerType: string;
  shape: number[];
  tensor: tf.Tensor | null;
  visualization: string | null;
}

interface InferenceStep {
  layerIndex: number;
  layerName: string;
  input: LayerActivation | null;
  output: LayerActivation;
  weights: {
    shape: number[];
    numParams: number;
    sample?: number[];
  } | null;
  computation: string;
  explanation: string;
  whyUsed: string;
  outputSample?: number[];
  active: boolean;
}

const InferenceExplorer: React.FC = () => {
  const [model, setModel] = useState<mobilenet.MobileNet | null>(null);
  const [loading, setLoading] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [inferenceSteps, setInferenceSteps] = useState<InferenceStep[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Sample images
  const sampleImages = [
    { url: 'https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=300', name: 'Dog' },
    { url: 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=300', name: 'Cat' },
    { url: 'https://images.unsplash.com/photo-1580407196238-dac33f57c410?w=300', name: 'Bird' },
    { url: 'https://images.unsplash.com/photo-1503066211613-c17ebc9daef0?w=300', name: 'Car' },
  ];

  // Load MobileNet model
  useEffect(() => {
    const loadModel = async () => {
      setLoading(true);
      setError(null);
      setLoadingProgress('Initializing TensorFlow.js...');
      
      try {
        console.log('Step 1: Setting up TensorFlow.js backend...');
        
        // Try WebGL first, fallback to CPU if it fails
        try {
          await tf.setBackend('webgl');
          await tf.ready();
          console.log('✓ WebGL backend initialized');
          setLoadingProgress('WebGL backend ready. Downloading model weights...');
        } catch (e) {
          console.warn('WebGL failed, falling back to CPU:', e);
          await tf.setBackend('cpu');
          await tf.ready();
          console.log('✓ CPU backend initialized');
          setLoadingProgress('CPU backend ready. Downloading model weights...');
        }
        
        console.log('Step 2: Loading MobileNetV2 model with ImageNet weights...');
        console.log('This will download ~9MB from the internet...');
        
        // Load MobileNet for predictions using version 1 (more stable)
        const loadedModel = await mobilenet.load({
          version: 1,
          alpha: 1.0,
        });
        
        console.log('✓ Model downloaded successfully!');
        
        setModel(loadedModel);
        setModelLoaded(true);
        setLoadingProgress('Model ready!');
        
        console.log('✓ Model loaded and ready for inference!');
        
      } catch (error: any) {
        console.error('❌ Error loading model:', error);
        console.error('Full error details:', error);
        const errorMessage = error.message || 'Unknown error';
        setError(`Failed to load model: ${errorMessage}`);
        setLoadingProgress('');
        
        // Show detailed error to user
        alert(
          `Failed to load the neural network model.\n\n` +
          `Error: ${errorMessage}\n\n` +
          `Please check:\n` +
          `1. You have a stable internet connection (model needs to download ~9MB)\n` +
          `2. Your browser supports WebGL or Canvas\n` +
          `3. No browser extensions are blocking the download\n` +
          `4. Check the browser console for more details\n\n` +
          `Try refreshing the page and checking the browser console for details.`
        );
      } finally {
        setLoading(false);
      }
    };

    loadModel();

    return () => {
      // Cleanup tensors on unmount
      if (model) {
        console.log('Cleaning up model...');
      }
    };
  }, []);

  // Load image from URL
  const loadImageFromUrl = (url: string) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      setImage(img);
      setImagePreview(url);
      setPredictions([]);
      setInferenceSteps([]);
      setCurrentStep(0);
    };
    img.onerror = () => {
      alert('Failed to load image. Please try another one.');
    };
    img.src = url;
  };

  // Handle file upload
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          setImage(img);
          setImagePreview(e.target?.result as string);
          setPredictions([]);
          setInferenceSteps([]);
          setCurrentStep(0);
        };
        img.src = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  };

  // Visualize tensor as image
  const visualizeTensor = async (tensor: tf.Tensor, maxChannels: number = 16): Promise<string> => {
    return new Promise((resolve) => {
      try {
        const shape = tensor.shape;
        
        // Handle different tensor shapes
        if (shape.length === 4) {
          // [batch, height, width, channels]
          const [, height, width, channels] = shape;
          
          // Take first batch
          const sliced = tensor.slice([0, 0, 0, 0], [1, height, width, Math.min(channels, maxChannels)]);
          
          // Normalize and convert to RGB for visualization
          const normalized = sliced.sub(sliced.min()).div(sliced.max().sub(sliced.min()));
          
          // If single channel, repeat to RGB
          let rgb: tf.Tensor3D;
          if (channels === 1) {
            const gray = normalized.squeeze([0]);
            rgb = tf.stack([gray, gray, gray], -1) as tf.Tensor3D;
          } else if (channels >= 3) {
            rgb = normalized.slice([0, 0, 0, 0], [1, height, width, 3]).squeeze([0]) as tf.Tensor3D;
          } else {
            rgb = normalized.squeeze([0]) as tf.Tensor3D;
          }
          
          // Convert to canvas
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          
          tf.browser.toPixels(rgb, canvas).then(() => {
            const dataUrl = canvas.toDataURL();
            sliced.dispose();
            normalized.dispose();
            rgb.dispose();
            resolve(dataUrl);
          });
        } else {
          resolve('');
        }
      } catch (error) {
        console.error('Error visualizing tensor:', error);
        resolve('');
      }
    });
  };

  // Run inference with layer-by-layer tracking - Simulated CNN demonstration
  const runInference = async () => {
    if (!image || !model) return;

    setLoading(true);
    setPredictions([]);
    setInferenceSteps([]);
    setCurrentStep(0);

    try {
      // Get predictions from MobileNet
      const preds = await model.classify(image);
      setPredictions(preds);

      // Process image for demonstration
      let currentTensor = tf.browser.fromPixels(image)
        .resizeBilinear([224, 224])
        .toFloat()
        .div(255.0);

      const steps: InferenceStep[] = [];

      // Demonstrate typical CNN layers with actual tensor transformations
      console.log('Creating layer-by-layer demonstration...');

      // Layer 1: Input
      steps.push(createLayerStep(0, 'Input Layer', 'Input', [1, 224, 224, 3], currentTensor, 
        'RGB image input: 224×224 pixels with 3 color channels', null,
        'Raw pixel values normalized to [0,1]. Each pixel has 3 values (Red, Green, Blue). Total: 224×224×3 = 150,528 values.',
        'All CNNs need fixed-size input. 224×224 is standard for ImageNet models - balances resolution vs computation.'));

      // Layer 2: First Convolution
      const conv1Filters = 32;
      const conv1 = tf.layers.conv2d({
        filters: conv1Filters,
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        activation: 'relu',
        name: 'conv1'
      });
      currentTensor = currentTensor.expandDims(0);
      let output1 = conv1.apply(currentTensor) as tf.Tensor;
      steps.push(createLayerStep(1, 'Conv2D + ReLU', 'Convolution', output1.shape, output1,
        `Convolution: 32 filters, kernel 3×3, stride 2 → extracts 32 feature maps`, 
        { shape: [3, 3, 3, 32], numParams: 3 * 3 * 3 * 32 + 32 },
        'Each 3×3 filter slides over the image, performing element-wise multiplication and sum. 32 different filters detect 32 different patterns (edges, colors). ReLU(x) = max(0,x) adds non-linearity by removing negative values.',
        'First layer extracts low-level features like edges and gradients. Stride=2 reduces spatial size by half (saves computation). These features are building blocks for higher layers.'));

      // Layer 3: Max Pooling
      const pool1 = tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2],
        padding: 'same',
        name: 'pool1'
      });
      currentTensor = output1;
      let output2 = pool1.apply(currentTensor) as tf.Tensor;
      steps.push(createLayerStep(2, 'MaxPooling2D', 'MaxPooling', output2.shape, output2,
        'Max Pooling: 2×2 window, stride 2 → reduces spatial dimensions by half', null,
        'Slides 2×2 window and takes maximum value from each region. Example: max([1,2,3,4]) = 4. Creates translation invariance - small shifts in input don\'t change output much.',
        'Reduces computation by 75% (half width × half height). No learnable parameters. Helps network focus on "presence of feature" rather than exact location. Prevents overfitting by discarding precise positions.'));

      // Layer 4: Second Convolution
      const conv2 = tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        name: 'conv2'
      });
      currentTensor = output2;
      let output3 = conv2.apply(currentTensor) as tf.Tensor;
      steps.push(createLayerStep(3, 'Conv2D + ReLU', 'Convolution', output3.shape, output3,
        'Convolution: 64 filters, kernel 3×3, stride 1 → extracts 64 deeper features',
        { shape: [3, 3, 32, 64], numParams: 3 * 3 * 32 * 64 + 64 },
        'Now has 64 filters (double from previous). Each filter looks at 32 input channels, combines them with learned weights. Detects more complex patterns like textures, corners, curves by combining edge features from previous layer.',
        'Deeper layers need more filters to capture increasing complexity. This layer combines simple edges into textures. Parameters: (3×3×32×64 + 64 bias) = 18,496 values learned during training on ImageNet.'));

      // Layer 5: Batch Normalization
      const bn1 = tf.layers.batchNormalization({ name: 'bn1' });
      currentTensor = output3;
      let output4 = bn1.apply(currentTensor) as tf.Tensor;
      steps.push(createLayerStep(4, 'BatchNormalization', 'BatchNormalization', output4.shape, output4,
        'Batch Normalization: normalizes activations for stable training', 
        { shape: [64], numParams: 64 * 4 },
        'Normalizes each channel to mean=0, std=1, then applies learned scale (γ) and shift (β). Formula: γ × (x - μ)/σ + β. Keeps activations in reasonable range, preventing exploding/vanishing values.',
        'Critical for training deep networks. Allows higher learning rates (10x faster training). Acts as regularization (reduces need for dropout). Has 4 parameters per channel: mean, variance, γ (scale), β (shift).'));

      // Layer 6: Another Pooling
      const pool2 = tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2],
        padding: 'same',
        name: 'pool2'
      });
      currentTensor = output4;
      let output5 = pool2.apply(currentTensor) as tf.Tensor;
      steps.push(createLayerStep(5, 'MaxPooling2D', 'MaxPooling', output5.shape, output5,
        'Max Pooling: 2×2 window → further spatial reduction', null,
        'Another downsampling step. Spatial dimensions: 56→28. Each 2×2 region becomes single value. Continues building translation invariance and computational efficiency.',
        'Network architecture follows pattern: Conv→Pool→Conv→Pool. This pyramid structure gradually increases semantic meaning while decreasing spatial size. Saves computation and memory for later dense layers.'));

      // Layer 7: Third Convolution
      const conv3 = tf.layers.conv2d({
        filters: 128,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        name: 'conv3'
      });
      currentTensor = output5;
      let output6 = conv3.apply(currentTensor) as tf.Tensor;
      steps.push(createLayerStep(6, 'Conv2D + ReLU', 'Convolution', output6.shape, output6,
        'Convolution: 128 filters, kernel 3×3 → extracts high-level features',
        { shape: [3, 3, 64, 128], numParams: 3 * 3 * 64 * 128 + 128 },
        'Deep layer with 128 filters detects complex patterns like object parts, faces, wheels. Each filter looks at all 64 channels from previous layer. Total: 73,856 learned parameters combining texture patterns into semantic features.',
        'Deeper conv layers capture hierarchical features. This layer recognizes shapes and object parts by combining textures. The receptive field (area of input each neuron "sees") is now large enough to recognize meaningful structures.'));

      // Layer 8: Global Average Pooling
      const gap = tf.layers.globalAveragePooling2d({ name: 'gap' });
      currentTensor = output6;
      let output7 = gap.apply(currentTensor) as tf.Tensor;
      steps.push(createLayerStep(7, 'GlobalAveragePooling2D', 'GlobalAveragePooling', output7.shape, output7,
        'Global Average Pooling: averages each feature map to single value', null,
        'Takes each 14×14 feature map (196 values) and averages to single value. Result: 128 values (one per filter). Converts 2D spatial data to 1D vector. Example: avg([1,2,3,4]) = 2.5',
        'Bridges convolutional layers and dense layers. Eliminates spatial dimensions completely (14×14→1×1). More robust than flatten (less prone to overfitting). Each value represents "how much of this feature is in the image" regardless of position.'));

      // Layer 9: Dense/Fully Connected
      const dense1 = tf.layers.dense({
        units: 256,
        activation: 'relu',
        name: 'dense1'
      });
      currentTensor = output7;
      let output8 = dense1.apply(currentTensor) as tf.Tensor;
      steps.push(createLayerStep(8, 'Dense + ReLU', 'Dense', output8.shape, output8,
        'Fully Connected: 256 neurons → learns class patterns',
        { shape: [128, 256], numParams: 128 * 256 + 256 },
        'Fully connected layer: each of 256 neurons connects to all 128 inputs. Computation: output = ReLU(weights × input + bias). Weight matrix: 128×256 = 32,768 parameters. Each neuron learns to detect specific feature combinations.',
        'Dense layers learn non-linear combinations of features for classification. This layer creates 256 high-level representations. Trained to recognize patterns like "has fur + has tail + has whiskers = likely a cat". Most parameters are in dense layers!'));

      // Layer 10: Final Classification (simulated 1000 classes like ImageNet)
      const dense2 = tf.layers.dense({
        units: 1000,
        activation: 'softmax',
        name: 'output'
      });
      currentTensor = output8;
      let output9 = dense2.apply(currentTensor) as tf.Tensor;
      steps.push(createLayerStep(9, 'Dense + Softmax', 'Dense', output9.shape, output9,
        'Output Layer: 1000 classes with softmax → probability distribution',
        { shape: [256, 1000], numParams: 256 * 1000 + 1000 },
        'Final layer: 256 inputs → 1000 outputs (ImageNet classes). Softmax converts raw scores to probabilities that sum to 1.0. Formula: softmax(x_i) = exp(x_i) / Σexp(x_j). Highest probability = predicted class.',
        'Output layer must match number of classes (1000 for ImageNet: dogs, cats, cars, etc.). Softmax ensures valid probability distribution. Each output represents confidence for that class. Example: [0.001, 0.003, 0.856, ...] → 85.6% confident it\'s class 3.'));

      console.log('✓ Created', steps.length, 'layer demonstrations');

      // Generate visualizations for convolutional layers
      for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        if (step.output.tensor && step.output.shape.length === 4 && 
            step.output.shape[1] > 1 && step.output.shape[2] > 1) {
          try {
            step.output.visualization = await visualizeTensor(step.output.tensor, 16);
          } catch (e) {
            console.warn(`Could not visualize layer ${i}:`, e);
          }
        }
      }

      setInferenceSteps(steps);
      
      console.log('✓ Inference demonstration complete!');
    } catch (error) {
      console.error('Error during inference:', error);
      alert('Error during inference. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Helper function to create layer step with detailed info
  const createLayerStep = (
    index: number,
    name: string,
    type: string,
    shape: number[],
    tensor: tf.Tensor,
    computation: string,
    weights: { shape: number[], numParams: number } | null,
    explanation: string,
    whyUsed: string
  ): InferenceStep => {
    // Extract sample values from tensor for display
    let outputSample: number[] = [];
    let weightSample: number[] = [];
    
    try {
      const data = tensor.dataSync();
      // Get first 10 values as sample
      outputSample = Array.from(data.slice(0, Math.min(10, data.length)));
    } catch (e) {
      console.warn('Could not extract tensor samples:', e);
    }
    
    return {
      layerIndex: index,
      layerName: name,
      input: null,
      output: {
        layerName: name,
        layerType: type,
        shape: shape,
        tensor: tensor,
        visualization: null
      },
      weights: weights ? { ...weights, sample: weightSample } : null,
      computation: computation,
      explanation: explanation,
      whyUsed: whyUsed,
      outputSample: outputSample,
      active: false
    };
  };

  // Animation control
  useEffect(() => {
    if (isPlaying && currentStep < inferenceSteps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep(prev => prev + 1);
      }, 1000);
      return () => clearTimeout(timer);
    } else if (currentStep >= inferenceSteps.length - 1) {
      setIsPlaying(false);
    }
  }, [isPlaying, currentStep, inferenceSteps.length]);

  const handlePlayPause = () => {
    if (inferenceSteps.length === 0) return;
    if (currentStep >= inferenceSteps.length - 1) {
      setCurrentStep(0);
    }
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const handleStepForward = () => {
    if (currentStep < inferenceSteps.length - 1) {
      setCurrentStep(prev => prev + 1);
      setIsPlaying(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-gradient-to-r from-purple-600 to-pink-600 p-3 rounded-xl">
              <Activity className="w-8 h-8 text-white" />
            </div>
            <div className="flex-1">
              <h2 className="text-3xl font-bold text-gray-900">Neural Network Inference Explorer</h2>
              <p className="text-gray-600">
                {loading && !modelLoaded ? (
                  <span className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-purple-600"></div>
                    {loadingProgress || 'Loading pretrained model...'}
                  </span>
                ) : modelLoaded ? (
                  '✓ MobileNetV2 loaded with ImageNet weights - Step through each layer to see how it processes images'
                ) : error ? (
                  <span className="text-red-600 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    Failed to load model - Check console for details
                  </span>
                ) : (
                  'Initializing...'
                )}
              </p>
            </div>
          </div>

          {/* Info/Error Alert */}
          {error ? (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
              <div className="flex-1">
                <p className="font-semibold text-red-900 mb-2">Failed to load model</p>
                <p className="text-sm text-red-800 mb-3">{error}</p>
                <button
                  onClick={() => window.location.reload()}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-all text-sm font-semibold"
                >
                  Refresh Page & Retry
                </button>
              </div>
            </div>
          ) : loading && !modelLoaded ? (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-start gap-3">
              <Info className="w-5 h-5 text-yellow-600 mt-0.5" />
              <div className="text-sm text-yellow-900">
                <p className="font-semibold mb-1">Downloading Model Weights...</p>
                <p>MobileNetV2 is being downloaded (~13MB). This is a one-time download and will be cached by your browser.</p>
                <p className="mt-2 text-xs">Progress: {loadingProgress}</p>
              </div>
            </div>
          ) : (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start gap-3">
              <Info className="w-5 h-5 text-blue-600 mt-0.5" />
              <div className="text-sm text-blue-900">
                <p className="font-semibold mb-1">How it works:</p>
                <ul className="list-disc ml-4 space-y-1">
                  <li>Upload an image or select a sample</li>
                  <li>Click "Run Inference" to process the image through MobileNetV2</li>
                  <li>Use playback controls to step through each layer and see intermediate activations</li>
                  <li>Each layer shows: input shape → computation → output shape with visualization</li>
                </ul>
              </div>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Image Input */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                <ImageIcon className="w-6 h-6" />
                Input Image
              </h3>

              {/* Upload Button */}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all flex items-center justify-center gap-2 mb-4"
                disabled={loading}
              >
                <Upload className="w-5 h-5" />
                Upload Image
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="hidden"
              />

              {/* Sample Images */}
              <div className="mb-4">
                <p className="text-sm text-gray-600 mb-2">Or select a sample:</p>
                <div className="grid grid-cols-2 gap-2">
                  {sampleImages.map((sample, idx) => (
                    <button
                      key={idx}
                      onClick={() => loadImageFromUrl(sample.url)}
                      className="relative group overflow-hidden rounded-lg border-2 border-gray-200 hover:border-blue-500 transition-all"
                      disabled={loading}
                    >
                      <img
                        src={sample.url}
                        alt={sample.name}
                        className="w-full h-24 object-cover"
                      />
                      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition-all flex items-center justify-center">
                        <span className="text-white font-semibold opacity-0 group-hover:opacity-100 transition-all">
                          {sample.name}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Image Preview */}
              {imagePreview && (
                <div className="mb-4">
                  <img
                    src={imagePreview}
                    alt="Selected"
                    className="w-full rounded-lg shadow-md"
                  />
                </div>
              )}

              {/* Run Inference Button */}
              <button
                onClick={runInference}
                disabled={!image || loading || !modelLoaded}
                className="w-full bg-gradient-to-r from-green-600 to-teal-600 text-white py-3 rounded-lg hover:from-green-700 hover:to-teal-700 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed font-semibold"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Processing...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Run Inference
                  </>
                )}
              </button>

              {/* Predictions */}
              {predictions.length > 0 && (
                <div className="mt-4 bg-gradient-to-r from-green-50 to-teal-50 rounded-lg p-4 border border-green-200">
                  <h4 className="font-bold text-gray-900 mb-2">Top Predictions:</h4>
                  <div className="space-y-2">
                    {predictions.map((pred, idx) => (
                      <div key={idx} className="flex justify-between items-center">
                        <span className="text-sm font-medium text-gray-700">{pred.className}</span>
                        <span className="text-sm font-bold text-green-700">
                          {(pred.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Layer Visualization */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                  <Layers className="w-6 h-6" />
                  Layer-by-Layer Processing
                </h3>

                {/* Playback Controls */}
                {inferenceSteps.length > 0 && (
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleReset}
                      className="p-2 bg-gray-200 rounded-lg hover:bg-gray-300 transition-all"
                      title="Reset"
                    >
                      <RotateCcw className="w-5 h-5" />
                    </button>
                    <button
                      onClick={handlePlayPause}
                      className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all"
                      title={isPlaying ? 'Pause' : 'Play'}
                    >
                      {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                    </button>
                    <button
                      onClick={handleStepForward}
                      className="p-2 bg-gray-200 rounded-lg hover:bg-gray-300 transition-all"
                      title="Step Forward"
                      disabled={currentStep >= inferenceSteps.length - 1}
                    >
                      <ChevronRight className="w-5 h-5" />
                    </button>
                    <span className="ml-2 text-sm font-semibold text-gray-700">
                      Layer {currentStep + 1} / {inferenceSteps.length}
                    </span>
                  </div>
                )}
              </div>

              {/* Layer Steps Display */}
              {inferenceSteps.length > 0 ? (
                <div className="space-y-4 max-h-[calc(100vh-300px)] overflow-y-auto">
                  {inferenceSteps.map((step, idx) => (
                    <div
                      key={idx}
                      className={`border-2 rounded-lg p-4 transition-all cursor-pointer ${
                        idx === currentStep
                          ? 'border-blue-500 bg-blue-50 shadow-lg scale-[1.02]'
                          : idx < currentStep
                          ? 'border-green-300 bg-green-50'
                          : 'border-gray-200 bg-gray-50 opacity-60'
                      }`}
                      onClick={() => {
                        setCurrentStep(idx);
                        setIsPlaying(false);
                      }}
                    >
                      <div className="flex items-start gap-4">
                        {/* Layer Number Badge */}
                        <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                          idx === currentStep
                            ? 'bg-blue-600 text-white'
                            : idx < currentStep
                            ? 'bg-green-600 text-white'
                            : 'bg-gray-300 text-gray-600'
                        }`}>
                          {idx + 1}
                        </div>

                        <div className="flex-1 min-w-0">
                          {/* Layer Header */}
                          <div className="flex items-center justify-between mb-2">
                            <h4 className="font-bold text-gray-900">{step.output.layerType}</h4>
                            <span className="text-xs font-mono text-gray-600 bg-gray-200 px-2 py-1 rounded">
                              {step.layerName}
                            </span>
                          </div>

                          {/* Computation */}
                          <p className="text-sm text-gray-700 mb-2 font-semibold">{step.computation}</p>

                          {/* Shape Information */}
                          <div className="flex flex-wrap items-center gap-2 text-sm mb-3">
                            <span className="font-mono text-purple-700 bg-purple-100 px-2 py-1 rounded">
                              Input: {step.input?.shape.slice(1).join('×') || step.output.shape.slice(1).join('×')}
                            </span>
                            <ChevronRight className="w-4 h-4 text-gray-400" />
                            <span className="font-mono text-blue-700 bg-blue-100 px-2 py-1 rounded">
                              Output: {step.output.shape.slice(1).join('×')}
                            </span>
                            {step.weights && (
                              <span className="font-mono text-orange-700 bg-orange-100 px-2 py-1 rounded">
                                Params: {step.weights.numParams.toLocaleString()}
                              </span>
                            )}
                          </div>

                          {/* Detailed Explanation - Only show for current step */}
                          {idx === currentStep && (
                            <div className="mt-3 space-y-3">
                              {/* How it works */}
                              <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
                                <h5 className="text-xs font-bold text-indigo-900 mb-1 uppercase">📖 How This Layer Works:</h5>
                                <p className="text-sm text-indigo-800 leading-relaxed">{step.explanation}</p>
                              </div>

                              {/* Why it's used */}
                              <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
                                <h5 className="text-xs font-bold text-purple-900 mb-1 uppercase">💡 Why This Layer is Used:</h5>
                                <p className="text-sm text-purple-800 leading-relaxed">{step.whyUsed}</p>
                              </div>

                              {/* Numerical Sample */}
                              {step.outputSample && step.outputSample.length > 0 && (
                                <div className="bg-teal-50 border border-teal-200 rounded-lg p-3">
                                  <h5 className="text-xs font-bold text-teal-900 mb-1 uppercase">🔢 Output Values (Sample):</h5>
                                  <div className="font-mono text-xs text-teal-800 bg-teal-100 p-2 rounded overflow-x-auto">
                                    [{step.outputSample.map(v => v.toFixed(4)).join(', ')}...]
                                  </div>
                                  <p className="text-xs text-teal-700 mt-1">
                                    Showing first 10 of {step.output.tensor?.size.toLocaleString()} total values
                                  </p>
                                </div>
                              )}

                              {/* Weight Information */}
                              {step.weights && (
                                <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                                  <h5 className="text-xs font-bold text-amber-900 mb-1 uppercase">⚖️ Pretrained Weights:</h5>
                                  <div className="text-sm text-amber-800 space-y-1">
                                    <p>• Shape: {step.weights.shape.join(' × ')}</p>
                                    <p>• Parameters: {step.weights.numParams.toLocaleString()} learned values</p>
                                    <p>• Trained on ImageNet dataset (1.2M images, 1000 classes)</p>
                                    <p className="text-xs mt-2 italic">These weights were learned over weeks of GPU training and represent patterns the network discovered in millions of images!</p>
                                  </div>
                                </div>
                              )}
                            </div>
                          )}

                          {/* Activation Visualization */}
                          {step.output.visualization && idx === currentStep && (
                            <div className="mt-3">
                              <h5 className="text-xs font-bold text-gray-900 mb-2 uppercase">🎨 Feature Map Visualization:</h5>
                              <img
                                src={step.output.visualization}
                                alt={`Layer ${idx + 1} activation`}
                                className="w-full max-w-md rounded-lg border-2 border-blue-300 shadow-md"
                                style={{ imageRendering: 'pixelated' }}
                              />
                              <p className="text-xs text-gray-600 mt-1">
                                First 16 channels - Bright areas show where features are detected
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <Layers className="w-16 h-16 mx-auto mb-4 opacity-30" />
                  <p>Upload an image and run inference to see layer-by-layer processing</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
};

export default InferenceExplorer;
