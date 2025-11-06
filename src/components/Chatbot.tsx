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
      content: `Hello! 👋 I'm your Neural Network Assistant with a built-in knowledge base. I can help you understand deep learning architectures, layers, training concepts, and applications!

💡 **Ask me about**:
• Models: AlexNet, ResNet, GoogLeNet, VGG
• Layers: Convolution, Pooling, Dropout, Batch Normalization, Activation Functions
• Training: Optimizers, Learning Rates, Backpropagation
• Concepts: Transfer Learning, Overfitting, Data Augmentation

**How I work**: I use pattern matching to provide relevant information from my comprehensive knowledge base.`,
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
    
    // Knowledge base for common questions
    if (lowerMessage.includes('alexnet')) {
      return "AlexNet (2012) was revolutionary! It's a deep CNN with 8 layers (5 convolutional + 3 fully connected) that won ImageNet 2012. Key innovations: ReLU activation, dropout regularization, overlapping pooling, and GPU training. It has ~60M parameters and achieved 15.3% top-5 error.";
    }
    
    if (lowerMessage.includes('resnet')) {
      return "ResNet introduced skip connections (residual learning) that allow gradients to flow directly through the network. This solved the vanishing gradient problem and enabled training of very deep networks (50, 101, 152 layers). ResNet-50 has ~25.6M parameters and uses bottleneck blocks for efficiency.";
    }
    
    if (lowerMessage.includes('googlenet') || lowerMessage.includes('inception')) {
      return "GoogLeNet (Inception v1, 2014) introduced Inception modules that use parallel convolutions of different sizes (1×1, 3×3, 5×5) to capture multi-scale features. Despite being deep (22 layers), it only has ~6.8M parameters thanks to 1×1 convolutions for dimensionality reduction. It also uses auxiliary classifiers during training.";
    }
    
    if (lowerMessage.includes('vgg')) {
      return "VGG-16 demonstrated that network depth is crucial for performance. It uses a simple, uniform architecture with only 3×3 convolutions stacked together. While very deep (16 layers), it has a huge number of parameters (~138M), mostly in the fully connected layers.";
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
      return "**Model Comparisons:**\n• **AlexNet (60M params)**: First deep CNN success, simple sequential architecture\n• **VGG-16 (138M params)**: Uniform 3×3 convs, very deep but parameter-heavy\n• **GoogLeNet (6.8M params)**: Efficient Inception modules, lowest parameters\n• **ResNet-50 (25.6M params)**: Skip connections enable very deep networks, best accuracy\n\nKey trade-offs: Accuracy vs. Parameters vs. Computation. Modern choice is often ResNet for accuracy or MobileNet/EfficientNet for efficiency.";
    }
    
    if (lowerMessage.includes('how') && (lowerMessage.includes('work') || lowerMessage.includes('chatbot') || lowerMessage.includes('answer'))) {
      return "**How I work:** I'm a rule-based chatbot with a curated knowledge base! 🤖\n\n1. **Pattern Matching**: I analyze your message for keywords (e.g., 'alexnet', 'convolution', 'dropout')\n2. **Knowledge Base**: I have pre-written responses covering 50+ deep learning topics\n3. **Response Selection**: I match your question to the most relevant information\n\n**Note**: I'm not a real AI model - I'm a deterministic system designed to help you learn about neural networks! For production, you'd integrate with GPT-4, Claude, or similar LLMs via their APIs.";
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
      "I'd be happy to help! Try asking about:\n• Specific models: 'Tell me about AlexNet', 'How does ResNet work?'\n• Layers: 'What is convolution?', 'Explain pooling'\n• Training: 'What is dropout?', 'How does backpropagation work?'\n• Concepts: 'Compare the models', 'What is transfer learning?'",
      `Currently viewing: **${currentModel}**. Ask me:\n• 'What makes ${currentModel} special?'\n• 'How does it compare to other models?'\n• 'What are the key layers?'\n• Or ask about any deep learning concept!`,
      "💡 **Popular questions:**\n• 'Explain skip connections'\n• 'What is the difference between models?'\n• 'How does dropout prevent overfitting?'\n• 'What is data augmentation?'\n• 'How do I choose a learning rate?'",
      "I have knowledge about:\n✓ Models: AlexNet, ResNet, GoogLeNet, VGG\n✓ Layers: Conv, Pooling, FC, Dropout, BatchNorm\n✓ Training: Optimizers, Learning Rate, Regularization\n✓ Concepts: Transfer Learning, Overfitting, Gradients\n\nWhat would you like to explore?",
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
