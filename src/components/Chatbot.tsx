import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, MessageSquare, X, Minimize2, Maximize2 } from 'lucide-react';
import type { ChatMessage } from '../types';

interface ChatbotProps {
  currentModel?: string;
}

const Chatbot: React.FC<ChatbotProps> = ({ currentModel = 'Neural Networks' }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      role: 'assistant',
      content: `Hello! 👋 I'm your Neural Network Assistant with comprehensive knowledge about complete CNN architectures!

💡 **Updated Knowledge Base** - Now with COMPLETE architectures:
• **AlexNet (2012)**: All 8 layers (5 conv + 3 FC) = 12 total steps
• **VGG-16 (2014)**: All 16 layers (13 conv + 3 FC) = 22 total steps  
• **GoogLeNet (2014)**: All 22 layers with 9 inception modules
• **ResNet-50 (2015)**: All 50 layers with residual blocks

� **Ask me about**:
• Layer-by-layer breakdowns: "Show me all VGG-16 layers"
• Innovations: "Explain skip connections", "What are inception modules?"
• Comparisons: "Compare AlexNet vs ResNet"
• Technical details: "How many parameters in FC layers?"
• Concepts: "What is bottleneck design?"

**How I work**: Pattern matching with curated knowledge base covering 50+ topics!`,
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isMinimized, setIsMinimized] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getAIResponse = (userMessage: string): string => {
    const lowerMessage = userMessage.toLowerCase();
    
    // Inference Explorer specific questions
    if (lowerMessage.includes('inference') || lowerMessage.includes('layer by layer')) {
      return "🎯 **Inference Explorer** shows real neural network processing!\n\nEach layer transforms the image:\n• **Conv layers**: Extract features (edges→textures→patterns)\n• **Pooling**: Reduces size, keeps important info\n• **Batch Norm**: Stabilizes training\n• **Activation (ReLU)**: Adds non-linearity (max(0,x))\n• **Global Avg Pool**: Converts feature maps to vectors\n• **Dense**: Learns class patterns\n• **Softmax**: Converts to probabilities\n\nWatch how spatial dimensions shrink but channels increase as the network extracts deeper features!";
    }
    
    if (lowerMessage.includes('feature map') || lowerMessage.includes('activation')) {
      return "🔍 **Feature Maps** are the outputs of convolutional layers!\n\nEach filter in a conv layer produces one feature map:\n• **Early layers**: Detect edges, colors, simple patterns\n• **Middle layers**: Detect textures, shapes, parts\n• **Deep layers**: Detect complex objects, faces, scenes\n\nVisualization shows first 16 channels. Bright areas = strong activation = important features detected!";
    }
    
    if (lowerMessage.includes('weight') || lowerMessage.includes('pretrained')) {
      return "⚖️ **Pretrained Weights** are learned from ImageNet (1.2M images, 1000 classes)!\n\nEach layer has weights:\n• **Conv filters**: 3D tensors (height×width×channels×filters)\n• **Batch Norm**: Scale & shift parameters\n• **Dense layers**: 2D matrices (input×output)\n\nMobileNet was trained for weeks on powerful GPUs. You're using those learned patterns instantly in your browser!";
    }
    
    if (lowerMessage.includes('how image change') || lowerMessage.includes('transformation')) {
      return "🖼️ **Image Transformation Through Layers**:\n\n1. **Input (224×224×3)**: RGB pixels\n2. **Conv1 (112×112×32)**: 32 edge detectors, spatial size halved\n3. **Pool (56×56×32)**: Downsampled, keeps important features\n4. **Conv2 (56×56×64)**: 64 texture detectors\n5. **Conv3 (28×28×128)**: 128 pattern detectors\n6. **Global Pool (1×1×128)**: Each feature map → single value\n7. **Dense (256)**: High-level feature combinations\n8. **Output (1000)**: Probabilities for each class\n\nSpatial info: 224→112→56→28→14→1\nChannels: 3→32→64→128→256→1000";
    }
    
    if (lowerMessage.includes('why batch norm') || lowerMessage.includes('normalization')) {
      return "📊 **Batch Normalization** is crucial for deep networks!\n\n**Why use it?**\n• Normalizes activations: mean=0, std=1\n• Faster training (higher learning rates)\n• Reduces internal covariate shift\n• Acts as regularization\n• Stabilizes gradients\n\n**How it works:**\n1. Normalize: (x - mean) / sqrt(variance + ε)\n2. Scale: γ * normalized_x\n3. Shift: + β\n\nγ and β are learned parameters!";
    }
    
    if (lowerMessage.includes('1d array') || lowerMessage.includes('flatten') || lowerMessage.includes('vector')) {
      return "📐 **1D Vectors in Neural Networks**:\n\n**How 2D→1D happens:**\n• **Global Avg Pooling**: Average each feature map (14×14→1)\n• **Flatten**: Reshape (7×7×512 → 25,088)\n• Result: 1D vector ready for Dense layers\n\n**Why needed?**\nDense layers need fixed-size 1D input. They can't handle 2D spatial data directly.\n\n**Example:**\n3×3×2 feature maps → [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] (18 values)\n\nEach value represents activation strength for that spatial position & channel!";
    }
    
    if (lowerMessage.includes('why pooling') || lowerMessage.includes('maxpool')) {
      return "🌊 **Pooling Layers** reduce dimensions intelligently!\n\n**Why use pooling?**\n• Reduces computation (fewer parameters)\n• Provides translation invariance\n• Extracts dominant features\n• Prevents overfitting\n• Increases receptive field\n\n**Max Pooling (2×2):**\n```\n[1 2]     \n[3 4] → 4 (takes maximum)\n```\n\n**Effect:** 224×224 → 112×112 (75% fewer pixels!)\n\nNo learnable parameters, just downsampling!";
    }
    
    // Knowledge base for common questions - UPDATED WITH COMPLETE ARCHITECTURES
    if (lowerMessage.includes('alexnet')) {
      return "🏆 **AlexNet (2012)** - The Deep Learning Revolution!\n\n**Complete Architecture (8 layers):**\n• **Conv1**: 96 filters, 11×11, stride 4 (227×227→55×55)\n• **MaxPool1**: 3×3, stride 2 → 27×27\n• **Conv2**: 256 filters, 5×5 → 27×27\n• **MaxPool2**: 3×3, stride 2 → 13×13\n• **Conv3**: 384 filters, 3×3 → 13×13\n• **Conv4**: 384 filters, 3×3 → 13×13\n• **Conv5**: 256 filters, 3×3 → 13×13\n• **MaxPool3**: 3×3, stride 2 → 6×6\n• **FC6**: 4,096 units (102M params!)\n• **FC7**: 4,096 units (16.8M params)\n• **FC8**: 1,000 classes (4.1M params)\n\n**Innovations**: ReLU, dropout (50%), overlapping pooling, GPU training\n**Total**: ~60M parameters | Won ImageNet 2012 with 15.3% top-5 error\n\n💡 Use Inference Explorer to see all layers with real data!";
    }
    
    if (lowerMessage.includes('resnet')) {
      return "🔄 **ResNet-50 (2015)** - Skip Connections Breakthrough!\n\n**Complete Architecture (50 layers):**\n• **Conv1**: 64 filters, 7×7, stride 2 (224→112)\n• **MaxPool**: 3×3, stride 2 (112→56)\n• **Conv2_x**: 3 bottleneck blocks @56×56 (64→64→256)\n• **Conv3_x**: 4 bottleneck blocks @28×28 (128→128→512)\n• **Conv4_x**: 6 bottleneck blocks @14×14 (256→256→1024) ⭐ Deepest\n• **Conv5_x**: 3 bottleneck blocks @7×7 (512→512→2048)\n• **Global Avg Pool**: 7×7→1 (NO parameters!)\n• **FC**: 2048→1,000 classes\n\n**Key Innovation**: y = F(x) + x (skip connections)\n• Solves vanishing gradients\n• Enables 50-152 layer networks\n• Bottleneck blocks: 1×1 reduce → 3×3 process → 1×1 expand\n\n**Total**: ~25M params (5× less than VGG!) | Won ImageNet 2015 with 3.57% error\n\n💡 Inference Explorer shows residual blocks with real skip connections!";
    }
    
    if (lowerMessage.includes('googlenet') || lowerMessage.includes('inception')) {
      return "🎯 **GoogLeNet/Inception V1 (2014)** - Multi-Scale Efficiency!\n\n**Complete Architecture (22 layers, 9 inception modules):**\n• **Conv1**: 64 filters, 7×7, stride 2 (224→112)\n• **MaxPool1**: 3×3, stride 2 (112→56)\n• **Conv2**: 192 filters, 3×3 → 56×56\n• **MaxPool2**: 3×3, stride 2 (56→28)\n• **Inception 3a & 3b** @28×28 (2 modules)\n• **MaxPool3** (28→14)\n• **Inception 4a-4e** @14×14 (5 modules) ⭐ Most modules\n• **MaxPool4** (14→7)\n• **Inception 5a & 5b** @7×7 (2 modules)\n• **Global Avg Pool** + Dropout (40%)\n• **FC**: 1,024→1,000 classes\n\n**Inception Module** (4 parallel paths):\n1. 1×1 conv (point-wise features)\n2. 1×1→3×3 conv (local features)\n3. 1×1→5×5 conv (broader features)\n4. 3×3 pool→1×1 (max features)\n→ ALL CONCATENATED!\n\n**Innovation**: 1×1 \"reduce\" convs save massive computation!\n**Total**: Only 7M params (20× less than VGG!) | Won ImageNet 2014 with 6.67% error\n\n💡 Inference Explorer shows inception modules with parallel paths!";
    }
    
    if (lowerMessage.includes('vgg')) {
      return "📚 **VGG-16 (2014)** - Depth Matters!\n\n**Complete Architecture (16 layers, 5 blocks):**\n**Block 1** (224→112): Conv1_1, Conv1_2 (64 filters, 3×3) + MaxPool\n**Block 2** (112→56): Conv2_1, Conv2_2 (128 filters, 3×3) + MaxPool\n**Block 3** (56→28): Conv3_1, Conv3_2, Conv3_3 (256 filters, 3×3) + MaxPool\n**Block 4** (28→14): Conv4_1, Conv4_2, Conv4_3 (512 filters, 3×3) + MaxPool\n**Block 5** (14→7): Conv5_1, Conv5_2, Conv5_3 (512 filters, 3×3) + MaxPool\n• **FC6**: 25,088→4,096 (102M params! 😱)\n• **FC7**: 4,096→4,096 (16.8M params)\n• **FC8**: 4,096→1,000 classes\n\n**Architecture Pattern**: 2-2-3-3-3 convs per block\n**Filter Progression**: 64→128→256→512→512 (doubles each stage)\n\n**Key Innovation**: Stacked 3×3 convs are better than large kernels!\n• Two 3×3 = same field as 5×5, but fewer params\n• Three 3×3 = same field as 7×7\n\n**Total**: 138M params (123M in FC layers!) | Proved depth is crucial\n\n💡 Inference Explorer shows ALL 16 layers + 5 pooling + 3 FC = 22 total steps!";
    }
    
    if (lowerMessage.includes('convolution') || lowerMessage.includes('conv')) {
      return "Convolutional layers apply filters (kernels) to extract features from images. They use parameter sharing and local connectivity, making them efficient for spatial data. Parameters include: kernel size (e.g., 3×3), stride (step size), padding (border handling), and number of filters (output channels).";
    }
    
    if (lowerMessage.includes('pooling')) {
      return "Pooling layers reduce spatial dimensions while retaining important features. Max pooling takes the maximum value in each region, while average pooling takes the mean. This provides translation invariance and reduces computation. Common size: 2×2 with stride 2 (halves dimensions).";
    }
    
    if (lowerMessage.includes('relu') || lowerMessage.includes('activation')) {
      return "ReLU (Rectified Linear Unit) is f(x) = max(0, x). It's the most popular activation because it: 1) Prevents vanishing gradients, 2) Is computationally efficient, 3) Enables sparse activations. Other activations: Sigmoid, Tanh, Leaky ReLU, GELU. Output layers typically use Softmax for classification.";
    }
    
    if (lowerMessage.includes('dropout')) {
      return "Dropout randomly deactivates neurons during training (e.g., 50% dropout rate) to prevent overfitting. It forces the network to learn robust features that don't rely on specific neurons. At inference, all neurons are active but outputs are scaled. It's like training an ensemble of networks!";
    }
    
    if (lowerMessage.includes('batch norm')) {
      return "Batch Normalization normalizes layer inputs across mini-batches, which: 1) Accelerates training, 2) Allows higher learning rates, 3) Reduces sensitivity to initialization, 4) Acts as regularization. It normalizes to mean 0 and variance 1, then applies learned scale and shift parameters.";
    }
    
    if (lowerMessage.includes('transfer learning')) {
      return "Transfer learning uses pre-trained models (like AlexNet, ResNet on ImageNet) for new tasks. You can: 1) Use as feature extractor (freeze weights, train new classifier), 2) Fine-tune (unfreeze some layers, train with small learning rate). This works because early layers learn general features (edges, textures).";
    }
    
    if (lowerMessage.includes('overfitting')) {
      return "Overfitting occurs when a model learns training data too well, including noise. Prevention techniques: 1) Dropout, 2) Data augmentation, 3) L1/L2 regularization, 4) Early stopping, 5) Reduce model complexity, 6) More training data. Monitor validation loss - if it increases while training loss decreases, you're overfitting!";
    }
    
    if (lowerMessage.includes('imagenet')) {
      return "ImageNet is a large-scale dataset with 1.2M training images across 1000 classes. The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) drove CNN innovation from 2010-2017. Top-5 error went from 28% (2010) to 2.25% (2017), surpassing human performance (~5%).";
    }
    
    if (lowerMessage.includes('parameter') || lowerMessage.includes('weight')) {
      return "Parameters are learnable values (weights and biases) that the network optimizes during training. Conv layer params = (kernel_height × kernel_width × input_channels + 1) × output_channels. FC layer params = (input_size + 1) × output_size. More parameters = more capacity but also more risk of overfitting.";
    }
    
    if (lowerMessage.includes('gradient') || lowerMessage.includes('backprop')) {
      return "Backpropagation computes gradients of the loss with respect to each parameter using the chain rule. Gradients indicate how to adjust weights to minimize loss. Vanishing gradients (very small) prevent learning in deep networks - solved by ReLU, skip connections, and batch norm. Exploding gradients are fixed by gradient clipping.";
    }
    
    if (lowerMessage.includes('optimizer') || lowerMessage.includes('sgd') || lowerMessage.includes('adam')) {
      return "Optimizers update weights using gradients. Popular ones: 1) SGD: Simple but effective with momentum, 2) Adam: Adaptive learning rates, works well out-of-box (most popular), 3) RMSprop: Good for RNNs, 4) AdamW: Adam with better weight decay. Learning rate is the most important hyperparameter!";
    }

    if (lowerMessage.includes('learning rate')) {
      return "Learning rate controls how much weights change per update. Too high = unstable training, overshooting. Too low = slow convergence, stuck in local minima. Common strategies: 1) Start with 0.001 (Adam) or 0.1 (SGD), 2) Learning rate decay/scheduling, 3) Warmup, 4) Cyclical learning rates.";
    }
    
    if (lowerMessage.includes('compare') || lowerMessage.includes('difference') || lowerMessage.includes('vs')) {
      return "📊 **Complete Architecture Comparison:**\n\n**AlexNet (2012)** - 60M params, 12 steps\n• Simple sequential: Conv→Pool→Conv→Pool→Conv→Conv→Conv→Pool→FC→FC→FC\n• Large kernels (11×11, 5×5) → small (3×3)\n• Huge FC layers: 102M params in FC6 alone!\n• Innovation: ReLU, dropout, GPU training\n• Error: 15.3% top-5\n\n**GoogLeNet (2014)** - 7M params, 22 layers\n• Inception modules with PARALLEL paths (1×1, 3×3, 5×5, pool)\n• Most efficient: 20× fewer params than VGG!\n• 1×1 convs for dimensionality reduction\n• Global avg pooling (no huge FC layers)\n• Error: 6.67% top-5\n\n**VGG-16 (2014)** - 138M params, 22 steps\n• Uniform architecture: ONLY 3×3 convs throughout\n• 2-2-3-3-3 blocks pattern\n• 123M params in FC layers (89%!)\n• Proved: Depth matters, stacked small kernels > large kernels\n• Most parameters, memory-intensive\n\n**ResNet-50 (2015)** - 25M params, 50 layers\n• Skip connections: y = F(x) + x\n• Bottleneck blocks: 1×1→3×3→1×1\n• Solves vanishing gradients → enables 50-152 layers\n• Global avg pooling (no huge FC)\n• Best accuracy: 3.57% top-5\n• Innovation: Residual learning is the KEY!\n\n**Efficiency Ranking**: GoogLeNet (7M) > ResNet (25M) < AlexNet (60M) <<< VGG (138M)\n**Accuracy Ranking**: ResNet > GoogLeNet > VGG ≈ AlexNet";
    }
    
    if (lowerMessage.includes('how many layers') || lowerMessage.includes('layer count') || lowerMessage.includes('depth')) {
      return "📏 **Complete Layer Counts:**\n\n**AlexNet**: 8 weighted layers\n• 5 convolutional layers (Conv1-5)\n• 3 fully connected layers (FC6-8)\n• PLUS 3 max pooling layers\n• **Total steps in visualizer**: 12\n\n**VGG-16**: 16 weighted layers\n• 13 convolutional layers (5 blocks: 2-2-3-3-3)\n• 3 fully connected layers (FC6-8)\n• PLUS 5 max pooling layers\n• **Total steps in visualizer**: 22\n\n**GoogLeNet**: 22 weighted layers\n• 9 inception modules (each has 4 parallel conv paths)\n• Multiple 1×1, 3×3, 5×5 convolutions\n• 1 final FC layer (no huge FC layers!)\n• **Total steps in visualizer**: 17 (representative modules)\n\n**ResNet-50**: 50 weighted layers\n• 1 initial conv (7×7)\n• 16 bottleneck blocks × 3 convs each = 48\n• 1 final FC layer\n• Stages: Conv2_x(3), Conv3_x(4), Conv4_x(6), Conv5_x(3)\n• **Total steps in visualizer**: 17 (representative blocks)\n\n💡 'Weighted layers' = layers with learnable parameters (conv, FC)\n💡 Pooling, ReLU, BatchNorm don't count toward depth (no params)";
    }
    
    if (lowerMessage.includes('bottleneck') || lowerMessage.includes('1x1 conv') || lowerMessage.includes('1×1')) {
      return "🔬 **Bottleneck Blocks & 1×1 Convolutions** - Efficiency Magic!\n\n**What are 1×1 convs?**\n• Operate on EACH pixel independently (no spatial mixing)\n• Change channel dimensions: 256 channels → 64 channels\n• Add non-linearity (ReLU after each conv)\n• Extremely cheap: 1×1×256×64 vs 3×3×256×64 (9× fewer params!)\n\n**ResNet Bottleneck Block**:\n1. **1×1 reduce**: 256→64 channels (COMPRESS)\n2. **3×3 process**: 64→64 channels (EXTRACT FEATURES)\n3. **1×1 expand**: 64→256 channels (RESTORE)\n4. **Skip**: Add input directly (residual connection)\n\n**Why?** 3×3 on 256 channels = expensive\nBottleneck: 1×1×256×64 + 3×3×64×64 + 1×1×64×256 = MUCH CHEAPER!\n\n**GoogLeNet Inception Bottleneck**:\n• 1×1 before 3×3 and 5×5 paths\n• Reduces computation by 4-10×\n• Enables deeper networks with fewer parameters\n\n**Network in Network**: 1×1 convs were introduced in 2013 (NiN paper)\n→ Popularized by GoogLeNet\n→ Now standard in MobileNet, EfficientNet, etc.\n\n💡 Check Inception modules in Inference Explorer to see parallel 1×1 paths!";
    }
    
    if (lowerMessage.includes('skip') || lowerMessage.includes('residual') || lowerMessage.includes('shortcut')) {
      return "🔄 **Skip Connections / Residual Learning** - ResNet's Breakthrough!\n\n**The Problem** (pre-2015):\n• Deep networks (>20 layers) performed WORSE than shallow ones\n• Vanishing gradients: gradients become tiny (10⁻¹⁰)\n• Network can't learn, even with ReLU and BatchNorm\n\n**The Solution**: y = F(x) + x\n• Input (x) added DIRECTLY to output\n• Network learns F(x) = residual (difference)\n• If layer should do nothing: F(x)=0, output=x (identity)\n\n**Why It Works**:\n1. **Gradient Flow**: Gradients flow directly backward through '+ x'\n2. **Easy Identity**: Learning identity is trivial (set weights to 0)\n3. **Flexibility**: Network chooses when to learn new features\n4. **Depth Enabled**: ResNet-152 works! (vs. VGG-19 max)\n\n**Implementation in ResNet-50**:\n• Every bottleneck block has skip connection\n• When dimensions change (downsampling), use 1×1 conv on shortcut\n• Example: 56×56×256 → 28×28×512\n  - Main path: 1×1→3×3(stride=2)→1×1\n  - Skip path: 1×1(stride=2) to match dimensions\n\n**Impact**: Enabled 50, 101, 152, even 1000+ layer networks!\n\n💡 Inference Explorer shows skip connections explicitly in ResNet blocks!";
    }
    
    if (lowerMessage.includes('inception module') || lowerMessage.includes('parallel') || lowerMessage.includes('multi-scale')) {
      return "🎯 **Inception Modules** - Multi-Scale Feature Extraction!\n\n**The Idea**: Different filters see different scales!\n• Small objects need small filters (1×1, 3×3)\n• Large objects need large filters (5×5)\n• Solution: Use ALL sizes in PARALLEL!\n\n**Inception Module Structure** (4 parallel paths):\n```\nInput (28×28×192)\n    ↓\n┌───┼───┬────┬────┐\n│   │   │    │    │\n1×1 1×1 1×1  3×3  ← Parallel!\n│   ↓   ↓   Pool\n│  3×3 5×5   1×1\n│   │   │    │\n└───┴───┴────┴────┘\n        ↓\n   Concatenate (28×28×256)\n```\n\n**Each Path Captures**:\n1. **1×1 path**: Point-wise features (64 filters)\n2. **1×1→3×3 path**: Local features (128 filters)\n3. **1×1→5×5 path**: Broader features (32 filters)\n4. **Pool→1×1 path**: Max features (32 filters)\n\n**The Magic**: 1×1 'reduce' before 3×3 and 5×5!\n• Without: 3×3×192×128 = 221K params\n• With 1×1 reduce: (1×1×192×96) + (3×3×96×128) = 128K params\n• Saves 42% computation!\n\n**Why It Works**:\n• Network learns which scale to use for each feature\n• More flexible than single kernel size\n• Efficient: 7M total params (GoogLeNet)\n\n**In Practice**: GoogLeNet has 9 inception modules\n• 2 @ 28×28 resolution\n• 5 @ 14×14 resolution (most modules)\n• 2 @ 7×7 resolution\n\n💡 Inference Explorer shows inception modules with all 4 paths!";
    }
    
    if (lowerMessage.includes('how') && (lowerMessage.includes('work') || lowerMessage.includes('chatbot') || lowerMessage.includes('answer'))) {
      return "**How I work:** I'm a rule-based chatbot with a curated knowledge base! 🤖\n\n1. **Pattern Matching**: I analyze your message for keywords (e.g., 'alexnet', 'convolution', 'dropout')\n2. **Knowledge Base**: I have pre-written responses covering 60+ deep learning topics including:\n   • Complete architectures (all layers)\n   • Technical innovations (skip connections, inception modules)\n   • Parameter counts and efficiency comparisons\n   • Historical context and ImageNet results\n3. **Response Selection**: I match your question to the most relevant information\n\n**Updated Knowledge** (Nov 2025):\n✅ All 4 models now show COMPLETE layer-by-layer implementations\n✅ Detailed parameter breakdowns\n✅ Architectural innovations explained\n✅ Inference Explorer integration\n\n**Note**: I'm not a real AI model - I'm a deterministic system designed to help you learn about neural networks! For production, you'd integrate with GPT-4, Claude, or similar LLMs via their APIs.";
    }
    
    if (lowerMessage.includes('inference explorer') || lowerMessage.includes('layer visualization')) {
      return "🔍 **Inference Explorer** - See Networks Process Images Layer by Layer!\n\n**What It Does**:\n• Runs actual image through neural network\n• Shows EVERY layer's output with visualizations\n• Displays feature maps (first 16 channels)\n• Tracks dimensions and parameters\n• Provides educational explanations\n\n**Complete Implementations**:\n✅ **AlexNet**: 12 steps (all 8 layers + pooling)\n✅ **VGG-16**: 22 steps (all 16 layers + pooling + FC)\n✅ **ResNet-50**: 17 steps (representative bottleneck blocks)\n✅ **GoogLeNet**: 17 steps (representative inception modules)\n\n**How to Use**:\n1. Select a model (AlexNet, VGG, ResNet, GoogLeNet, MobileNet)\n2. Click 'Load Model' (uses pretrained MobileNet weights)\n3. Upload or use sample image\n4. Click 'Run Inference'\n5. Explore each layer:\n   - Feature map visualizations (4×4 grid)\n   - Output dimensions (e.g., 56×56×128)\n   - Parameter counts\n   - Educational explanations\n   - Historical context\n\n**What You'll See**:\n• Input: 224×224×3 RGB image\n• Early layers: Edge detectors (vertical, horizontal, diagonal)\n• Middle layers: Textures, patterns, shapes\n• Deep layers: Object parts, holistic features\n• Output: 1000 class probabilities\n\n💡 **Pro Tip**: Watch how spatial dimensions shrink (224→112→56→28→14→7→1) while channels increase (3→64→128→256→512→1000)!";
    }
    
    if (lowerMessage.includes('all layers') || lowerMessage.includes('complete architecture') || lowerMessage.includes('show me')) {
      return "📋 **Want to see all layers?** Try these questions:\n\n**For Complete Breakdowns**:\n• \"Tell me about AlexNet\" → See all 8 layers + specs\n• \"Tell me about VGG\" → See all 16 layers in 5 blocks\n• \"Tell me about ResNet\" → See bottleneck structure\n• \"Tell me about GoogLeNet\" → See inception modules\n\n**For Comparisons**:\n• \"Compare the models\" → Side-by-side comparison\n• \"How many layers\" → Layer counts for all models\n\n**For Technical Details**:\n• \"Explain bottleneck\" → ResNet 1×1 convs\n• \"Explain inception module\" → GoogLeNet parallel paths\n• \"Explain skip connections\" → ResNet residual learning\n\n**Best Way to Explore**:\n🎯 Go to **Inference Explorer** page!\n• Select any model\n• Run inference on an image\n• See EVERY layer with:\n  - Feature visualizations\n  - Dimension tracking\n  - Parameter counts\n  - Educational explanations\n  - Historical context\n\n💡 It's like watching the network think!";
    }
    
    if (lowerMessage.includes('parameter') && (lowerMessage.includes('count') || lowerMessage.includes('breakdown') || lowerMessage.includes('distribution'))) {
      return "📊 **Parameter Distribution Breakdown:**\n\n**AlexNet (60M total)**:\n• Conv1-5: 2.3M (4%)\n• FC6: 37.7M params (63%) ⚠️ Huge!\n• FC7: 16.8M params (28%)\n• FC8: 4.1M params (7%)\n→ FC layers = 97% of all parameters!\n\n**VGG-16 (138M total)**:\n• Conv1-13: 14.7M (11%)\n• FC6: 102.8M params (74%) ⚠️ Massive!\n• FC7: 16.8M params (12%)\n• FC8: 4.1M params (3%)\n→ FC layers = 89% of all parameters!\n\n**GoogLeNet (7M total)**:\n• All convs: 6M (86%)\n• Final FC: 1M (14%)\n→ Global avg pooling eliminates huge FC layers!\n→ Most efficient architecture\n\n**ResNet-50 (25M total)**:\n• Conv layers: 23M (92%)\n• Final FC: 2M (8%)\n→ Also uses global avg pooling\n→ No huge FC layers like AlexNet/VGG\n\n**Key Insight**: Modern architectures (GoogLeNet, ResNet) use **Global Average Pooling** instead of huge FC layers:\n• VGG FC6: 25,088×4,096 = 102M params\n• ResNet GAP: 0 params (just averaging!)\n\n**Evolution**:\n2012 (AlexNet): 97% params in FC → Inefficient\n2014 (GoogLeNet): Global pooling → 7M total\n2015 (ResNet): Global pooling → 25M total\n\n💡 This is why modern networks are more efficient!";
    }
    
    if (lowerMessage.includes('global average pooling') || lowerMessage.includes('gap') || lowerMessage.includes('global pooling')) {
      return "🌐 **Global Average Pooling (GAP)** - Modern Efficiency Trick!\n\n**What It Does**:\n• Takes each feature map (e.g., 7×7)\n• Averages ALL spatial values → single number\n• Example: 7×7 feature map → 1 value\n• If 2048 channels: 7×7×2048 → 2048 values\n\n**Why It's Amazing**:\n✅ **Zero parameters!** Just averaging\n✅ Replaces huge FC layers\n✅ Forces convs to learn semantic features\n✅ Reduces overfitting\n✅ Works with any input size\n\n**Comparison**:\n**VGG-16 (no GAP)**:\n• 7×7×512 → Flatten → 25,088 values\n• FC6: 25,088 × 4,096 = 102M params 😱\n\n**ResNet-50 (with GAP)**:\n• 7×7×2048 → GAP → 2,048 values\n• FC: 2,048 × 1,000 = 2M params ✅\n• **Savings**: 100M parameters!\n\n**When Introduced**:\n• Network in Network (NiN) paper, 2013\n• Popularized by GoogLeNet, 2014\n• Standard in ResNet (2015) and beyond\n\n**How It Works**:\n```python\n# Input: 7×7×2048 feature maps\nfor each of 2048 channels:\n    value = average(7×7 spatial positions)\n# Output: 2048 values → FC layer\n```\n\n**Modern Usage**:\n• ResNet: 7×7×2048 → GAP → 2048\n• MobileNet: 7×7×1024 → GAP → 1024\n• EfficientNet: Variable → GAP → channels\n\n💡 GAP is why ResNet is 5× more parameter-efficient than VGG!";
    }
    
    if (lowerMessage.includes('skip connection') || lowerMessage.includes('residual')) {
      return "Skip connections (residual connections) add the input of a block directly to its output: y = F(x) + x. This allows gradients to flow directly backward, preventing vanishing gradients. Benefits: 1) Train much deeper networks (100+ layers), 2) Better gradient flow, 3) Learn identity mappings easily, 4) Improved accuracy. ResNet popularized this technique!";
    }
    
    if (lowerMessage.includes('data augmentation')) {
      return "Data augmentation artificially increases training data by applying transformations: 1) Geometric (rotation, flipping, cropping, scaling), 2) Color (brightness, contrast, saturation), 3) Noise injection, 4) Cutout/Mixup. Benefits: Reduces overfitting, improves generalization, makes models robust to variations. Essential for image classification!";
    }
    
    if (lowerMessage.includes('feature map') || lowerMessage.includes('channel')) {
      return "A feature map (or channel) is the output of applying one filter to an input. For example, a conv layer with 64 filters produces 64 feature maps. Each feature map detects specific patterns - early layers: edges/colors, middle layers: textures/parts, deep layers: objects/concepts. The depth (number of channels) represents representational capacity.";
    }
    
    if (lowerMessage.includes('inception') && lowerMessage.includes('module')) {
      return "An Inception module applies multiple filter sizes (1×1, 3×3, 5×5) in parallel, then concatenates results. This captures features at multiple scales simultaneously! The 1×1 convolutions before larger filters reduce computational cost (dimensionality reduction). It's like having multiple experts looking at the same data from different perspectives.";
    }
    
    if (lowerMessage.includes('dashboard') || lowerMessage.includes('calculator') || lowerMessage.includes('theory')) {
      return "The Dashboard has 4 powerful tabs:\n\n**1️⃣ Theory Tab**: Learn about Conv, Pooling, FC, and Dropout layers with formulas, properties, and numerical examples.\n\n**2️⃣ Calculator Tab**: Interactive tool to calculate output dimensions, parameters, FLOPs, and memory usage for convolutional layers. Try different inputs!\n\n**3️⃣ Examples Tab**: Real-world layer configurations from AlexNet, VGG, ResNet, and MobileNet. Click 'Try in Calculator' to experiment!\n\n**4️⃣ Data Transformation**: Visual guide showing how data flows through Conv, ReLU, BatchNorm, Pooling, Dropout, and FC layers with actual values!";
    }
    
    if (lowerMessage.includes('transformation') || lowerMessage.includes('data flow') || lowerMessage.includes('how data')) {
      return "**Data Transformation in Neural Networks**:\n\n1️⃣ **Input**: Raw pixel values (e.g., 5×5 image)\n2️⃣ **Convolution**: Extract features using kernels (3×3×2 filters)\n3️⃣ **ReLU**: Apply activation, zero out negatives\n4️⃣ **Batch Norm**: Normalize to mean=0, std=1 for stability\n5️⃣ **Pooling**: Downsample (e.g., 4×4 → 2×2 max pool)\n6️⃣ **Dropout**: Randomly deactivate neurons (training only)\n7️⃣ **Flatten**: Convert 2D to 1D vector\n8️⃣ **FC**: Fully connected classification\n\nCheck the Dashboard's Data Transformation tab to see this with actual values!";
    }
    
    if (lowerMessage.includes('softmax') || lowerMessage.includes('output layer')) {
      return "**Softmax Activation** converts raw scores (logits) into probabilities that sum to 1.0:\n\nFormula: softmax(xi) = exp(xi) / Σexp(xj)\n\nExample:\n• Input: [2.0, 1.0, 0.1]\n• After exp: [7.39, 2.72, 1.11]\n• Sum: 11.22\n• Softmax: [0.659, 0.242, 0.099] ← Probabilities!\n\nUsed in final classification layer. The highest probability indicates the predicted class. Often paired with cross-entropy loss during training.";
    }
    
    if (lowerMessage.includes('flops') || lowerMessage.includes('computation') || lowerMessage.includes('efficiency')) {
      return "**FLOPs (Floating Point Operations)** measure computational cost:\n\nFor Conv Layer:\nFLOPs = Output_H × Output_W × Kernel_H × Kernel_W × Input_Channels × Output_Channels\n\n**Example**: Conv layer with 3×3 kernel, 64 input channels, 128 output channels, 56×56 output:\nFLOPs = 56 × 56 × 3 × 3 × 64 × 128 = 231M FLOPs\n\n**Why it matters**:\n• Mobile devices: Need low FLOPs (<100M)\n• Edge devices: Target <1B FLOPs\n• Cloud servers: Can handle 10B+ FLOPs\n\nUse the Calculator tab to compute FLOPs for your layers!";
    }
    
    if (lowerMessage.includes('memory') || lowerMessage.includes('gpu') || lowerMessage.includes('vram')) {
      return "**Memory Usage in Neural Networks**:\n\n**1. Parameters (Weights)**:\n• Conv: kernel_h × kernel_w × in_ch × out_ch × 4 bytes\n• FC: input_size × output_size × 4 bytes\n\n**2. Activations (Forward Pass)**:\n• Store all layer outputs for backprop\n• batch_size × height × width × channels × 4 bytes\n\n**3. Gradients (Backward Pass)**:\n• Same size as activations\n• Temporary during training\n\n**Tips to reduce memory**:\n• Smaller batch size\n• Gradient checkpointing\n• Mixed precision (FP16)\n• Prune unnecessary layers\n\nCalculator tab shows parameter memory!";
    }
    
    if (lowerMessage.includes('stride') || lowerMessage.includes('padding')) {
      return "**Stride & Padding** control output dimensions:\n\n**Stride**: Step size when sliding filter\n• Stride=1: Dense sampling, larger output\n• Stride=2: Skip every other position, 2× smaller output\n• Higher stride = faster but less detail\n\n**Padding**: Border pixels added\n• Valid (no padding): Output shrinks\n• Same (pad to maintain size): Output = Input\n• Formula: pad = (kernel_size - 1) / 2\n\n**Output Size Formula**:\nOutput = ⌊(Input + 2×Pad - Kernel) / Stride⌋ + 1\n\nExample: 32×32 input, 5×5 kernel, stride=1, pad=2\n→ ⌊(32 + 4 - 5) / 1⌋ + 1 = 32×32 output\n\nTry the Calculator tab!";
    }
    
    if (lowerMessage.includes('visualizer') || lowerMessage.includes('graph') || lowerMessage.includes('network view')) {
      return "**Network Visualizer Page Features**:\n\n🎨 **Interactive Graph**:\n• Visual representation of model architecture\n• Zoom, pan, and navigate freely\n• Click any layer node to see detailed specs\n\n📊 **Layer Details** (click a node):\n• Type, input/output dimensions\n• Kernel size, stride, padding\n• Number of parameters\n• Activation functions\n\n🔍 **Special Features**:\n• ResNet shows skip connections with branching paths\n• Edge labels show stride information\n• Animated connections between layers\n• MiniMap for navigation\n\n🤖 **Models Available**:\n• AlexNet, ResNet-50, GoogLeNet, VGG-16\n\nExplore the Visualizer page from the top navigation!";
    }
    
    if (lowerMessage.includes('calculator') && lowerMessage.includes('use')) {
      return "**How to Use the Calculator Tab**:\n\n1️⃣ **Enter Input Dimensions**:\n• Height, Width, Channels (e.g., 224×224×3 for RGB image)\n\n2️⃣ **Set Kernel Parameters**:\n• Kernel Size (e.g., 3×3, 5×5, 7×7)\n• Stride (typically 1 or 2)\n• Padding (0 for valid, auto for same)\n• Number of Filters (output channels)\n\n3️⃣ **View Results**:\n• ✅ Output Dimensions\n• 📊 Total Parameters\n• ⚡ FLOPs (computational cost)\n• 💾 Memory Usage\n\n💡 **Pro Tip**: Try the example configurations from the Examples tab to see real-world settings from famous architectures!";
    }
    
    // Default responses with helpful suggestions
    const defaultResponses = [
      "I'd be happy to help! Try asking about:\n• **Complete Architectures**: 'Tell me about VGG-16', 'Show AlexNet layers'\n• **Innovations**: 'Explain skip connections', 'What are inception modules?'\n• **Comparisons**: 'Compare ResNet vs VGG', 'Parameter breakdown'\n• **Technical**: 'How many layers in ResNet?', 'What is bottleneck design?'\n• **Tools**: 'How to use Inference Explorer?'",
      `Currently viewing: **${currentModel}**. Ask me:\n• 'What makes ${currentModel} special?'\n• 'Show me all ${currentModel} layers'\n• 'How does it compare to other models?'\n• 'What innovations does it have?'\n• 'Parameter breakdown for ${currentModel}'`,
      "💡 **Popular questions about complete architectures:**\n• 'Tell me about AlexNet' → All 8 layers breakdown\n• 'Tell me about VGG-16' → All 16 layers in 5 blocks\n• 'Tell me about ResNet-50' → Bottleneck blocks explained\n• 'Tell me about GoogLeNet' → Inception modules\n• 'Compare the models' → Side-by-side analysis",
      "🎯 **New! Complete Architecture Knowledge:**\n✓ All layers for AlexNet, VGG, ResNet, GoogLeNet\n✓ Parameter counts and distribution\n✓ Skip connections & inception modules\n✓ Inference Explorer integration\n✓ Historical context & innovations\n\nWhat would you like to explore?",
      "📚 **I can help with:**\n• **Models**: Complete layer breakdowns (AlexNet, VGG, ResNet, GoogLeNet)\n• **Layers**: Conv, Pooling, FC, Dropout, BatchNorm, Activations\n• **Innovations**: Skip connections, Inception modules, Bottlenecks, GAP\n• **Training**: Optimizers, Learning Rate, Regularization\n• **Comparisons**: Parameters, efficiency, accuracy\n\nAsk me anything!",
    ];
    
    return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
  };

  const handleSend = () => {
    if (!input.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    // Simulate AI response with delay
    setTimeout(() => {
      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: getAIResponse(input),
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, aiMessage]);
    }, 500);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (isMinimized) {
    return (
      <div className="fixed bottom-6 right-6 z-40">
        <button
          onClick={() => setIsMinimized(false)}
          className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-4 rounded-full shadow-lg hover:shadow-xl transition-all flex items-center gap-2"
        >
          <MessageSquare className="w-6 h-6" />
          <span className="font-medium">Chat Assistant</span>
        </button>
      </div>
    );
  }

  return (
    <div
      className={`fixed ${
        isExpanded ? 'inset-4' : 'bottom-6 right-6 w-96 h-[600px]'
      } bg-white rounded-2xl shadow-2xl z-40 flex flex-col border border-gray-200 transition-all duration-300`}
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-4 rounded-t-2xl flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-white bg-opacity-20 p-2 rounded-lg">
            <Bot className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-lg">Neural Network Assistant</h3>
            <p className="text-sm opacity-90">Ask me anything!</p>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-2 hover:bg-white hover:bg-opacity-20 rounded-lg transition-colors"
          >
            {isExpanded ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
          </button>
          <button
            onClick={() => setIsMinimized(true)}
            className="p-2 hover:bg-white hover:bg-opacity-20 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
          >
            <div
              className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                message.role === 'user' ? 'bg-blue-600' : 'bg-gradient-to-br from-indigo-500 to-purple-500'
              }`}
            >
              {message.role === 'user' ? (
                <User className="w-5 h-5 text-white" />
              ) : (
                <Bot className="w-5 h-5 text-white" />
              )}
            </div>
            <div
              className={`flex-1 px-4 py-3 rounded-2xl ${
                message.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              <div className="text-sm leading-relaxed whitespace-pre-wrap">
                {message.content.split('\n').map((line, i) => {
                  // Simple markdown-like formatting
                  if (line.startsWith('**') && line.includes(':**')) {
                    const parts = line.split(':**');
                    return (
                      <p key={i} className="font-bold mb-2 text-blue-700">
                        {parts[0].replace(/\*\*/g, '')}:
                      </p>
                    );
                  } else if (line.startsWith('**') && line.endsWith('**')) {
                    return (
                      <p key={i} className="font-bold mb-1">
                        {line.replace(/\*\*/g, '')}
                      </p>
                    );
                  } else if (line.startsWith('•') || line.startsWith('✓')) {
                    return (
                      <p key={i} className="ml-2 mb-1">
                        {line}
                      </p>
                    );
                  } else if (line.trim() === '') {
                    return <br key={i} />;
                  } else {
                    return <p key={i} className="mb-1">{line}</p>;
                  }
                })}
              </div>
              <p
                className={`text-xs mt-2 ${
                  message.role === 'user' ? 'text-blue-100' : 'text-gray-500'
                }`}
              >
                {message.timestamp.toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Quick Questions (show when messages are minimal) */}
      {messages.length <= 1 && (
        <div className="px-4 pb-2">
          <p className="text-xs text-gray-500 mb-2">Quick questions:</p>
          <div className="flex flex-wrap gap-2">
            {[
              'What is the dashboard?',
              'How does data transformation work?',
              'Compare the models',
              'Explain batch normalization',
              'How to use calculator?',
              'What is dropout?',
            ].map((question, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setInput(question);
                  setTimeout(() => handleSend(), 100);
                }}
                className="text-xs px-3 py-1.5 bg-blue-50 text-blue-600 rounded-full hover:bg-blue-100 transition-colors"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="p-4 border-t border-gray-200">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about neural networks..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim()}
            className="bg-blue-600 text-white p-3 rounded-xl hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-2 text-center">
          💡 Rule-based chatbot with pre-programmed knowledge
        </p>
      </div>
    </div>
  );
};

export default Chatbot;
