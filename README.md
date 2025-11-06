# Neural Network Visualizer

A modern, interactive React web application for visualizing and exploring popular deep learning architectures including AlexNet, ResNet-50, GoogLeNet (Inception v1), and VGG-16.

## Features

### 🎯 Interactive Model Visualization
- **Graph-based architecture display** using React Flow
- Zoom and pan capabilities for detailed exploration
- Color-coded layer types for easy identification
- MiniMap for quick navigation through complex architectures

### 📊 Model Information Panel
- Comprehensive details about each neural network
- Layer-by-layer breakdown with parameters
- Key features and innovations
- Performance metrics and use cases
- Links to original research papers

### 🤖 AI Chatbot Assistant
- Interactive chatbot to answer questions about neural networks
- Built-in knowledge base covering:
  - Architecture details (AlexNet, ResNet, GoogLeNet, VGG)
  - Layer types (Convolution, Pooling, Dropout, etc.)
  - Training concepts (Backpropagation, Optimizers, Learning Rate)
  - Common issues (Overfitting, Vanishing Gradients)
  - Best practices and applications
- Minimizable and expandable interface

### 🎨 Modern UI/UX
- Clean, responsive design with Tailwind CSS
- Gradient accents and smooth animations
- Intuitive navigation and controls
- Professional color-coded components

## Supported Models

1. **AlexNet (2012)**
   - 60M parameters
   - 8 layers
   - First successful deep CNN for ImageNet

2. **ResNet-50 (2015)**
   - 25.6M parameters
   - 50 layers with skip connections
   - Winner of ILSVRC 2015

3. **GoogLeNet/Inception v1 (2014)**
   - 6.8M parameters
   - Inception modules with parallel convolutions
   - Winner of ILSVRC 2014

4. **VGG-16 (2014)**
   - 138M parameters
   - 16 layers with uniform 3×3 convolutions
   - Simple and deep architecture

## Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## Technologies Used

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **React Flow (@xyflow/react)** - Interactive graph visualization
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icon set

## Project Structure

```
src/
├── components/
│   ├── ModelSelector.tsx       # Dropdown for model selection
│   ├── NetworkVisualizer.tsx   # React Flow graph visualization
│   ├── ModelInfoPanel.tsx      # Sliding info panel
│   └── Chatbot.tsx            # AI assistant chatbot
├── data/
│   └── models.ts              # Neural network model data
├── types/
│   └── index.ts               # TypeScript interfaces
├── App.tsx                    # Main application component
└── main.tsx                   # Entry point
```

## Usage

1. **Select a Model**: Use the dropdown at the top to choose from AlexNet, ResNet-50, GoogLeNet, or VGG-16

2. **Explore the Architecture**: 
   - Scroll through the layers in the visualization
   - Zoom in/out using the controls
   - Click and drag to pan around
   - Use the minimap for quick navigation

3. **View Model Details**: Click the "Model Info" button to open a detailed panel showing:
   - Model description and history
   - Key features and innovations
   - Architecture specifications
   - Layer-by-layer breakdown

4. **Ask Questions**: Click the chat button to interact with the AI assistant:
   - Ask about specific models
   - Learn about layer types
   - Understand training concepts
   - Get recommendations

## Key Features Explained

### React Flow Visualization
Each layer is rendered as a custom node with:
- Layer name and type
- Output shape (dimensions)
- Parameter count
- Color coding by layer type

### Chatbot Knowledge Base
The chatbot can answer questions about:
- **Architectures**: AlexNet, ResNet, GoogLeNet, VGG
- **Layers**: Convolution, Pooling, Dropout, Batch Norm
- **Training**: Backpropagation, Optimizers (SGD, Adam), Learning Rate
- **Concepts**: Transfer Learning, Overfitting, Gradients, Activation Functions

## Future Enhancements

- [ ] Add more models (MobileNet, EfficientNet, Vision Transformers)
- [ ] Interactive layer editing and custom model builder
- [ ] Export visualizations as images
- [ ] Integration with real neural network libraries (TensorFlow.js, ONNX.js)
- [ ] Performance comparison tools
- [ ] Training simulation and visualization
- [ ] 3D visualization option

## License

MIT License

## Credits

Built with ❤️ using modern web technologies for educational purposes.

---

**Note**: This is an educational tool for understanding neural network architectures. The models shown are simplified representations of the actual implementations
