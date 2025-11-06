import React, { useState } from 'react';
import { Calculator, BookOpen, Layers, TrendingUp, Settings, PlayCircle, Image, ArrowRight } from 'lucide-react';

interface LayerCalculation {
  inputHeight: number;
  inputWidth: number;
  inputChannels: number;
  kernelSize: number;
  stride: number;
  padding: number;
  numFilters: number;
}

const Dashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'theory' | 'calculator' | 'examples' | 'transformation'>('theory');
  const [selectedLayer, setSelectedLayer] = useState<'conv' | 'pool' | 'fc' | 'dropout'>('conv');
  
  // Calculator state
  const [calc, setCalc] = useState<LayerCalculation>({
    inputHeight: 224,
    inputWidth: 224,
    inputChannels: 3,
    kernelSize: 3,
    stride: 1,
    padding: 1,
    numFilters: 64,
  });

  const [results, setResults] = useState({
    outputHeight: 0,
    outputWidth: 0,
    outputChannels: 0,
    parameters: 0,
    flops: 0,
  });

  const calculateConvOutput = () => {
    const outputHeight = Math.floor(
      (calc.inputHeight + 2 * calc.padding - calc.kernelSize) / calc.stride + 1
    );
    const outputWidth = Math.floor(
      (calc.inputWidth + 2 * calc.padding - calc.kernelSize) / calc.stride + 1
    );
    const outputChannels = calc.numFilters;
    const parameters = 
      calc.kernelSize * calc.kernelSize * calc.inputChannels * calc.numFilters + calc.numFilters;
    const flops = 
      outputHeight * outputWidth * calc.kernelSize * calc.kernelSize * calc.inputChannels * calc.numFilters;

    setResults({
      outputHeight,
      outputWidth,
      outputChannels,
      parameters,
      flops,
    });
  };

  const layerTheory = {
    conv: {
      title: 'Convolutional Layer',
      description: 'Applies learnable filters to extract spatial features from input data.',
      formula: 'Output Size = ⌊(Input + 2×Padding - Kernel) / Stride⌋ + 1',
      latex: '$$H_{out} = \\left\\lfloor \\frac{H_{in} + 2P - K}{S} \\right\\rfloor + 1$$',
      parameters: 'Parameters = (K × K × C_in × C_out) + C_out',
      purpose: 'Feature extraction through local connectivity and weight sharing',
      properties: [
        'Translation equivariance',
        'Parameter sharing reduces memory',
        'Captures spatial hierarchies',
        'Local receptive fields',
      ],
      example: {
        input: '224×224×3 RGB image',
        kernel: '3×3 filter, 64 filters',
        stride: '1 pixel',
        padding: '1 pixel (same)',
        output: '224×224×64',
        params: '(3×3×3×64) + 64 = 1,792 parameters',
      },
    },
    pool: {
      title: 'Pooling Layer',
      description: 'Downsamples spatial dimensions while retaining important features.',
      formula: 'Output Size = ⌊(Input - Kernel) / Stride⌋ + 1',
      latex: '$$H_{out} = \\left\\lfloor \\frac{H_{in} - K}{S} \\right\\rfloor + 1$$',
      parameters: 'Parameters = 0 (no learnable weights)',
      purpose: 'Dimensionality reduction and translation invariance',
      properties: [
        'No learnable parameters',
        'Reduces spatial dimensions',
        'Provides translation invariance',
        'Controls overfitting',
      ],
      example: {
        input: '224×224×64 feature map',
        kernel: '2×2 window',
        stride: '2 pixels',
        padding: '0',
        output: '112×112×64',
        params: '0 parameters (no learning)',
      },
    },
    fc: {
      title: 'Fully Connected (Dense) Layer',
      description: 'Connects every neuron to all neurons in the previous layer.',
      formula: 'Output = Input × Weight + Bias',
      latex: '$$y = Wx + b$$',
      parameters: 'Parameters = (Input × Output) + Output',
      purpose: 'High-level reasoning and classification',
      properties: [
        'Global connectivity',
        'High parameter count',
        'Learns complex patterns',
        'Final classification/regression',
      ],
      example: {
        input: '4096 features (flattened)',
        output: '1000 classes (ImageNet)',
        weights: '4096 × 1000 matrix',
        bias: '1000 values',
        params: '(4096 × 1000) + 1000 = 4,097,000 parameters',
        computation: 'Matrix multiplication + bias addition',
      },
    },
    dropout: {
      title: 'Dropout Layer',
      description: 'Randomly deactivates neurons during training to prevent overfitting.',
      formula: 'Output = Input × Mask (training) or Input × (1-p) (testing)',
      latex: '$$y = \\begin{cases} x \\odot m & \\text{training} \\\\ x \\cdot (1-p) & \\text{testing} \\end{cases}$$',
      parameters: 'Parameters = 0 (no learnable weights)',
      purpose: 'Regularization to prevent co-adaptation of neurons',
      properties: [
        'No learnable parameters',
        'Probability p (e.g., 0.5)',
        'Only active during training',
        'Ensemble effect',
      ],
      example: {
        input: '4096 neurons',
        dropoutRate: '0.5 (50% dropped)',
        training: 'Each neuron has 50% chance of being set to 0',
        testing: 'All neurons active, scaled by 0.5',
        output: '4096 neurons (same size)',
        params: '0 parameters',
      },
    },
  };

  const commonExamples = [
    {
      name: 'AlexNet Conv1',
      input: '227×227×3',
      kernel: '11×11',
      stride: 2,
      padding: 0,
      filters: 96,
      output: '55×55×96',
      params: '(11×11×3×96) + 96 = 34,944',
    },
    {
      name: 'VGG Block',
      input: '224×224×64',
      kernel: '3×3',
      stride: 1,
      padding: 1,
      filters: 128,
      output: '224×224×128',
      params: '(3×3×64×128) + 128 = 73,856',
    },
    {
      name: 'ResNet Identity',
      input: '56×56×64',
      kernel: '3×3',
      stride: 1,
      padding: 1,
      filters: 64,
      output: '56×56×64',
      params: '(3×3×64×64) + 64 = 36,928',
    },
    {
      name: 'MaxPool Standard',
      input: '224×224×64',
      kernel: '2×2',
      stride: 2,
      padding: 0,
      filters: 64,
      output: '112×112×64',
      params: '0 (no parameters)',
    },
  ];

  return (
    <div className="w-full h-full bg-gradient-to-br from-gray-50 to-blue-50 overflow-auto">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-6 shadow-lg">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Layers className="w-8 h-8" />
            Neural Network Layer Explorer
          </h1>
          <p className="text-blue-100 text-sm">
            Understand layer operations through theory, mathematics, and interactive calculations
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-8 py-8">
        {/* Tab Navigation */}
        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setActiveTab('theory')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'theory'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            <BookOpen className="w-5 h-5" />
            Theory & Math
          </button>
          <button
            onClick={() => setActiveTab('calculator')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'calculator'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            <Calculator className="w-5 h-5" />
            Interactive Calculator
          </button>
          <button
            onClick={() => setActiveTab('examples')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'examples'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            <TrendingUp className="w-5 h-5" />
            Real Examples
          </button>
          <button
            onClick={() => setActiveTab('transformation')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'transformation'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            <Image className="w-5 h-5" />
            Data Transformation
          </button>
        </div>

        {/* Theory Tab */}
        {activeTab === 'theory' && (
          <div className="space-y-6">
            {/* Layer Type Selector */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Select Layer Type</h2>
              <div className="grid grid-cols-4 gap-4">
                {(['conv', 'pool', 'fc', 'dropout'] as const).map((type) => (
                  <button
                    key={type}
                    onClick={() => setSelectedLayer(type)}
                    className={`p-4 rounded-lg border-2 transition-all ${
                      selectedLayer === type
                        ? 'border-blue-600 bg-blue-50 shadow-md'
                        : 'border-gray-200 hover:border-blue-300'
                    }`}
                  >
                    <div className="font-semibold text-gray-800">
                      {layerTheory[type].title}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Theory Content */}
            <div className="bg-white rounded-lg shadow-md p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <Settings className="w-6 h-6 text-blue-600" />
                {layerTheory[selectedLayer].title}
              </h2>
              
              <p className="text-gray-600 text-lg mb-6">
                {layerTheory[selectedLayer].description}
              </p>

              <div className="grid md:grid-cols-2 gap-6">
                {/* Mathematical Formulas */}
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-gray-800 mb-4">📐 Mathematical Formula</h3>
                  <div className="bg-white rounded-lg p-4 mb-4 border-2 border-blue-200">
                    <div className="font-mono text-sm text-gray-700 mb-2">
                      {layerTheory[selectedLayer].formula}
                    </div>
                    <div className="text-xs text-gray-500 mt-2">
                      {layerTheory[selectedLayer].latex}
                    </div>
                  </div>
                  <div className="bg-white rounded-lg p-4 border-2 border-green-200">
                    <div className="font-semibold text-gray-700 mb-2">Parameters:</div>
                    <div className="font-mono text-sm text-gray-600">
                      {layerTheory[selectedLayer].parameters}
                    </div>
                  </div>
                </div>

                {/* Properties */}
                <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-gray-800 mb-4">🎯 Key Properties</h3>
                  <div className="space-y-3">
                    {layerTheory[selectedLayer].properties.map((prop, idx) => (
                      <div key={idx} className="flex items-start gap-2">
                        <div className="w-2 h-2 rounded-full bg-purple-500 mt-2"></div>
                        <div className="text-gray-700">{prop}</div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-4 bg-white rounded-lg p-4 border-2 border-purple-200">
                    <div className="font-semibold text-gray-700 mb-2">Purpose:</div>
                    <div className="text-sm text-gray-600">
                      {layerTheory[selectedLayer].purpose}
                    </div>
                  </div>
                </div>
              </div>

              {/* Numerical Example */}
              <div className="mt-6 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">🔢 Numerical Example</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  {Object.entries(layerTheory[selectedLayer].example).map(([key, value]) => (
                    <div key={key} className="bg-white rounded-lg p-3 border border-orange-200">
                      <div className="text-xs font-semibold text-orange-600 uppercase mb-1">
                        {key.replace(/([A-Z])/g, ' $1').trim()}
                      </div>
                      <div className="text-sm font-mono text-gray-800">{value}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Calculator Tab */}
        {activeTab === 'calculator' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <Calculator className="w-6 h-6 text-blue-600" />
                Convolutional Layer Calculator
              </h2>

              <div className="grid md:grid-cols-2 gap-8">
                {/* Input Parameters */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-700 mb-4">Input Parameters</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Input Height (pixels)
                      </label>
                      <input
                        type="number"
                        value={calc.inputHeight}
                        onChange={(e) => setCalc({ ...calc, inputHeight: parseInt(e.target.value) || 0 })}
                        className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Input Width (pixels)
                      </label>
                      <input
                        type="number"
                        value={calc.inputWidth}
                        onChange={(e) => setCalc({ ...calc, inputWidth: parseInt(e.target.value) || 0 })}
                        className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Input Channels
                      </label>
                      <input
                        type="number"
                        value={calc.inputChannels}
                        onChange={(e) => setCalc({ ...calc, inputChannels: parseInt(e.target.value) || 0 })}
                        className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                  </div>

                  <h3 className="text-lg font-semibold text-gray-700 mb-4 mt-6">Layer Configuration</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Kernel Size
                      </label>
                      <input
                        type="number"
                        value={calc.kernelSize}
                        onChange={(e) => setCalc({ ...calc, kernelSize: parseInt(e.target.value) || 0 })}
                        className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Stride
                      </label>
                      <input
                        type="number"
                        value={calc.stride}
                        onChange={(e) => setCalc({ ...calc, stride: parseInt(e.target.value) || 0 })}
                        className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Padding
                      </label>
                      <input
                        type="number"
                        value={calc.padding}
                        onChange={(e) => setCalc({ ...calc, padding: parseInt(e.target.value) || 0 })}
                        className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Number of Filters
                      </label>
                      <input
                        type="number"
                        value={calc.numFilters}
                        onChange={(e) => setCalc({ ...calc, numFilters: parseInt(e.target.value) || 0 })}
                        className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                  </div>

                  <button
                    onClick={calculateConvOutput}
                    className="w-full mt-6 bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 rounded-lg font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all flex items-center justify-center gap-2"
                  >
                    <PlayCircle className="w-5 h-5" />
                    Calculate Output
                  </button>
                </div>

                {/* Results */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-700 mb-4">Results</h3>
                  
                  {results.outputHeight > 0 && (
                    <div className="space-y-4">
                      {/* Output Dimensions */}
                      <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-6 border-2 border-green-200">
                        <h4 className="font-semibold text-gray-800 mb-3">📏 Output Dimensions</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-gray-600">Height:</span>
                            <span className="font-bold text-lg text-green-700">{results.outputHeight} pixels</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-600">Width:</span>
                            <span className="font-bold text-lg text-green-700">{results.outputWidth} pixels</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-600">Channels:</span>
                            <span className="font-bold text-lg text-green-700">{results.outputChannels}</span>
                          </div>
                          <div className="mt-4 pt-4 border-t border-green-300">
                            <div className="text-center">
                              <div className="text-sm text-gray-600 mb-1">Complete Output Shape</div>
                              <div className="font-mono text-xl font-bold text-green-800">
                                {results.outputHeight} × {results.outputWidth} × {results.outputChannels}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Parameters */}
                      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-6 border-2 border-blue-200">
                        <h4 className="font-semibold text-gray-800 mb-3">⚙️ Parameters</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-gray-600">Total Parameters:</span>
                            <span className="font-bold text-lg text-blue-700">
                              {results.parameters.toLocaleString()}
                            </span>
                          </div>
                          <div className="text-xs text-gray-500 mt-2">
                            Formula: (K×K×C_in×C_out) + C_out
                          </div>
                          <div className="text-xs font-mono bg-white p-2 rounded border border-blue-200 mt-2">
                            ({calc.kernelSize}×{calc.kernelSize}×{calc.inputChannels}×{calc.numFilters}) + {calc.numFilters}
                          </div>
                        </div>
                      </div>

                      {/* Computational Cost */}
                      <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6 border-2 border-purple-200">
                        <h4 className="font-semibold text-gray-800 mb-3">🔥 Computational Cost</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-gray-600">FLOPs:</span>
                            <span className="font-bold text-lg text-purple-700">
                              {(results.flops / 1e6).toFixed(2)}M
                            </span>
                          </div>
                          <div className="text-xs text-gray-500 mt-2">
                            Floating Point Operations (Multiply-Adds)
                          </div>
                        </div>
                      </div>

                      {/* Memory Calculation */}
                      <div className="bg-gradient-to-br from-orange-50 to-amber-50 rounded-lg p-6 border-2 border-orange-200">
                        <h4 className="font-semibold text-gray-800 mb-3">💾 Memory Usage</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-gray-600">Output Feature Map:</span>
                            <span className="font-bold text-lg text-orange-700">
                              {((results.outputHeight * results.outputWidth * results.outputChannels * 4) / 1024).toFixed(2)} KB
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-600">Weights Memory:</span>
                            <span className="font-bold text-lg text-orange-700">
                              {((results.parameters * 4) / 1024).toFixed(2)} KB
                            </span>
                          </div>
                          <div className="text-xs text-gray-500 mt-2">
                            Assuming 32-bit (4 bytes) floating point values
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {results.outputHeight === 0 && (
                    <div className="bg-gray-50 rounded-lg p-12 text-center">
                      <Calculator className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                      <p className="text-gray-500">
                        Enter parameters and click "Calculate Output" to see results
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Examples Tab */}
        {activeTab === 'examples' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <TrendingUp className="w-6 h-6 text-blue-600" />
                Real-World Examples from Famous Architectures
              </h2>

              <div className="grid md:grid-cols-2 gap-6">
                {commonExamples.map((example, idx) => (
                  <div
                    key={idx}
                    className="bg-gradient-to-br from-gray-50 to-blue-50 rounded-lg p-6 border-2 border-gray-200 hover:border-blue-400 transition-all hover:shadow-lg"
                  >
                    <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                      <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                        {idx + 1}
                      </div>
                      {example.name}
                    </h3>
                    
                    <div className="space-y-3">
                      <div className="bg-white rounded-lg p-3 border border-blue-200">
                        <div className="text-xs font-semibold text-blue-600 mb-1">INPUT</div>
                        <div className="font-mono text-sm font-bold text-gray-800">{example.input}</div>
                      </div>

                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-white rounded p-2 border border-gray-300">
                          <div className="text-xs text-gray-600">Kernel</div>
                          <div className="font-mono text-sm font-semibold">{example.kernel}</div>
                        </div>
                        <div className="bg-white rounded p-2 border border-gray-300">
                          <div className="text-xs text-gray-600">Stride</div>
                          <div className="font-mono text-sm font-semibold">{example.stride}</div>
                        </div>
                        <div className="bg-white rounded p-2 border border-gray-300">
                          <div className="text-xs text-gray-600">Padding</div>
                          <div className="font-mono text-sm font-semibold">{example.padding}</div>
                        </div>
                        <div className="bg-white rounded p-2 border border-gray-300">
                          <div className="text-xs text-gray-600">Filters</div>
                          <div className="font-mono text-sm font-semibold">{example.filters}</div>
                        </div>
                      </div>

                      <div className="bg-green-50 rounded-lg p-3 border-2 border-green-300">
                        <div className="text-xs font-semibold text-green-700 mb-1">OUTPUT</div>
                        <div className="font-mono text-sm font-bold text-green-800">{example.output}</div>
                      </div>

                      <div className="bg-purple-50 rounded-lg p-3 border border-purple-300">
                        <div className="text-xs font-semibold text-purple-700 mb-1">PARAMETERS</div>
                        <div className="font-mono text-xs text-purple-800">{example.params}</div>
                      </div>
                    </div>

                    <button
                      onClick={() => {
                        if (example.name.includes('Pool')) return;
                        const [h, w, c] = example.input.split('×').map(v => parseInt(v));
                        const k = parseInt(example.kernel.split('×')[0]);
                        setCalc({
                          inputHeight: h,
                          inputWidth: w,
                          inputChannels: c,
                          kernelSize: k,
                          stride: example.stride,
                          padding: example.padding,
                          numFilters: example.filters,
                        });
                        setActiveTab('calculator');
                      }}
                      className="w-full mt-4 bg-blue-600 text-white py-2 rounded-lg text-sm font-semibold hover:bg-blue-700 transition-all"
                      disabled={example.name.includes('Pool')}
                    >
                      {example.name.includes('Pool') ? 'Pooling Layer' : 'Try in Calculator →'}
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Reference */}
            <div className="bg-white rounded-lg shadow-md p-8">
              <h2 className="text-xl font-bold text-gray-800 mb-4">📚 Quick Reference</h2>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                  <h3 className="font-semibold text-blue-800 mb-2">Common Kernel Sizes</h3>
                  <ul className="text-sm space-y-1 text-gray-700">
                    <li>• 1×1: Channel mixing</li>
                    <li>• 3×3: Standard feature extraction</li>
                    <li>• 5×5: Larger receptive field</li>
                    <li>• 7×7, 11×11: Initial layers</li>
                  </ul>
                </div>
                <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                  <h3 className="font-semibold text-green-800 mb-2">Padding Strategies</h3>
                  <ul className="text-sm space-y-1 text-gray-700">
                    <li>• Valid: No padding (P=0)</li>
                    <li>• Same: Output = Input size</li>
                    <li>• P = (K-1)/2 for Same padding</li>
                    <li>• Full: Maximum padding</li>
                  </ul>
                </div>
                <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                  <h3 className="font-semibold text-purple-800 mb-2">Stride Effects</h3>
                  <ul className="text-sm space-y-1 text-gray-700">
                    <li>• S=1: No downsampling</li>
                    <li>• S=2: 50% size reduction</li>
                    <li>• S=3: 66% size reduction</li>
                    <li>• Larger S = Faster inference</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Data Transformation Tab */}
        {activeTab === 'transformation' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <Image className="w-6 h-6 text-blue-600" />
                Data Transformation Through Network Layers
              </h2>
              <p className="text-gray-600 mb-8">
                Watch how a sample image transforms as it passes through different layers of a neural network, 
                with actual dimension changes and sample values at each step.
              </p>

              {/* Convolutional Layer Transformation */}
              <div className="mb-12">
                <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                  <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">1</div>
                  Convolutional Layer Transformation
                </h3>

                <div className="grid md:grid-cols-3 gap-6 mb-6">
                  {/* Input */}
                  <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-6 border-2 border-green-300">
                    <h4 className="font-bold text-green-800 mb-4 flex items-center gap-2">
                      📥 INPUT
                    </h4>
                    <div className="space-y-3">
                      <div className="bg-white rounded-lg p-3 border border-green-200">
                        <div className="text-sm font-semibold text-green-700 mb-2">Shape: 5×5×1</div>
                        <div className="text-xs text-gray-600 mb-2">Grayscale image patch</div>
                        <div className="grid grid-cols-5 gap-1">
                          {[
                            [0, 0, 255, 255, 0],
                            [0, 255, 0, 0, 255],
                            [255, 0, 255, 0, 255],
                            [0, 255, 0, 255, 0],
                            [0, 0, 255, 255, 0]
                          ].map((row, i) => (
                            row.map((val, j) => (
                              <div 
                                key={`${i}-${j}`}
                                className="w-8 h-8 flex items-center justify-center text-xs font-mono border border-gray-300 rounded"
                                style={{ backgroundColor: `rgb(${val}, ${val}, ${val})`, color: val > 127 ? 'black' : 'white' }}
                              >
                                {val}
                              </div>
                            ))
                          ))}
                        </div>
                      </div>
                      <div className="text-xs font-mono bg-white p-2 rounded border border-green-200">
                        Total values: 25 pixels
                      </div>
                    </div>
                  </div>

                  {/* Convolution Operation */}
                  <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-6 border-2 border-blue-300 flex flex-col justify-center">
                    <h4 className="font-bold text-blue-800 mb-4 flex items-center gap-2">
                      ⚙️ OPERATION
                    </h4>
                    <div className="space-y-4">
                      <div className="bg-white rounded-lg p-3 border border-blue-200">
                        <div className="text-sm font-semibold text-blue-700 mb-2">3×3 Kernel (Edge Detector)</div>
                        <div className="grid grid-cols-3 gap-1 mb-2">
                          {[-1, 0, 1, -2, 0, 2, -1, 0, 1].map((val, i) => (
                            <div 
                              key={i}
                              className={`w-8 h-8 flex items-center justify-center text-xs font-bold border-2 rounded ${
                                val < 0 ? 'bg-red-100 border-red-300' : val > 0 ? 'bg-green-100 border-green-300' : 'bg-gray-100 border-gray-300'
                              }`}
                            >
                              {val}
                            </div>
                          ))}
                        </div>
                        <div className="text-xs text-gray-600">Sobel vertical edge detector</div>
                      </div>
                      
                      <div className="flex items-center justify-center">
                        <ArrowRight className="w-6 h-6 text-blue-600" />
                      </div>

                      <div className="bg-white rounded-lg p-3 border border-blue-200">
                        <div className="text-sm font-semibold text-blue-700 mb-2">Parameters</div>
                        <ul className="text-xs space-y-1 text-gray-700">
                          <li>• Stride: 1</li>
                          <li>• Padding: 0 (Valid)</li>
                          <li>• Filters: 2</li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  {/* Output */}
                  <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6 border-2 border-purple-300">
                    <h4 className="font-bold text-purple-800 mb-4 flex items-center gap-2">
                      📤 OUTPUT
                    </h4>
                    <div className="space-y-3">
                      <div className="bg-white rounded-lg p-3 border border-purple-200">
                        <div className="text-sm font-semibold text-purple-700 mb-2">Shape: 3×3×2</div>
                        <div className="text-xs text-gray-600 mb-2">Feature Map (Filter 1)</div>
                        <div className="grid grid-cols-3 gap-1 mb-3">
                          {[510, 255, -510, 765, 0, -765, 510, 255, -510].map((val, i) => (
                            <div 
                              key={i}
                              className="w-10 h-10 flex items-center justify-center text-xs font-mono border-2 border-purple-300 rounded bg-white"
                            >
                              {val}
                            </div>
                          ))}
                        </div>
                        <div className="text-xs text-gray-600 mb-2">Feature Map (Filter 2)</div>
                        <div className="grid grid-cols-3 gap-1">
                          {[255, 510, 255, 0, 0, 0, -255, -510, -255].map((val, i) => (
                            <div 
                              key={i}
                              className="w-10 h-10 flex items-center justify-center text-xs font-mono border-2 border-purple-300 rounded bg-white"
                            >
                              {val}
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="text-xs font-mono bg-white p-2 rounded border border-purple-200">
                        Total values: 18 (3×3×2)
                      </div>
                    </div>
                  </div>
                </div>

                {/* Mathematical Explanation */}
                <div className="bg-amber-50 rounded-lg p-6 border-2 border-amber-300">
                  <h4 className="font-bold text-amber-800 mb-3">🔢 Step-by-Step Calculation</h4>
                  <div className="space-y-2 text-sm text-gray-700">
                    <p><strong>Formula:</strong> Output[i,j] = Σ(Input[i:i+k, j:j+k] ⊙ Kernel)</p>
                    <p><strong>Example for position [0,0]:</strong></p>
                    <div className="bg-white rounded p-3 font-mono text-xs">
                      = (0×-1) + (0×0) + (255×1) + (0×-2) + (255×0) + (0×2) + (255×-1) + (0×0) + (255×1)
                      <br/>= 0 + 0 + 255 + 0 + 0 + 0 - 255 + 0 + 255 = 255
                    </div>
                    <p className="text-xs text-gray-600 mt-2">
                      Each output value is the element-wise multiplication of a 3×3 region with the kernel, then summed.
                    </p>
                  </div>
                </div>
              </div>

              {/* Pooling Layer Transformation */}
              <div className="mb-12">
                <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                  <div className="w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold">2</div>
                  Max Pooling Layer Transformation
                </h3>

                <div className="grid md:grid-cols-3 gap-6 mb-6">
                  {/* Input */}
                  <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-6 border-2 border-blue-300">
                    <h4 className="font-bold text-blue-800 mb-4">📥 INPUT (From Conv)</h4>
                    <div className="bg-white rounded-lg p-3 border border-blue-200">
                      <div className="text-sm font-semibold text-blue-700 mb-2">Shape: 4×4×1</div>
                      <div className="grid grid-cols-4 gap-1">
                        {[12, 20, 8, 15, 18, 24, 6, 10, 5, 14, 22, 9, 11, 7, 19, 13].map((val, i) => (
                          <div 
                            key={i}
                            className="w-10 h-10 flex items-center justify-center text-xs font-bold border-2 border-blue-300 rounded bg-white"
                          >
                            {val}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Operation */}
                  <div className="bg-gradient-to-br from-orange-50 to-amber-50 rounded-lg p-6 border-2 border-orange-300 flex flex-col justify-center">
                    <h4 className="font-bold text-orange-800 mb-4">⚙️ MAX POOLING</h4>
                    <div className="space-y-4">
                      <div className="bg-white rounded-lg p-4 border border-orange-200">
                        <div className="text-sm font-semibold text-orange-700 mb-3">2×2 Window</div>
                        <div className="grid grid-cols-2 gap-2 mb-3">
                          {[12, 20, 18, 24].map((val, i) => (
                            <div 
                              key={i}
                              className={`w-12 h-12 flex items-center justify-center text-sm font-bold border-2 rounded ${
                                val === 24 ? 'bg-green-100 border-green-400' : 'bg-gray-50 border-gray-300'
                              }`}
                            >
                              {val}
                            </div>
                          ))}
                        </div>
                        <div className="text-center text-lg font-bold text-green-600">↓ Max = 24</div>
                      </div>
                      
                      <div className="bg-white rounded-lg p-3 border border-orange-200">
                        <ul className="text-xs space-y-1 text-gray-700">
                          <li>• Stride: 2</li>
                          <li>• Takes maximum value</li>
                          <li>• Reduces spatial dimensions</li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  {/* Output */}
                  <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-6 border-2 border-green-300">
                    <h4 className="font-bold text-green-800 mb-4">📤 OUTPUT</h4>
                    <div className="bg-white rounded-lg p-3 border border-green-200">
                      <div className="text-sm font-semibold text-green-700 mb-2">Shape: 2×2×1</div>
                      <div className="text-xs text-gray-600 mb-3">50% size reduction</div>
                      <div className="grid grid-cols-2 gap-2">
                        {[24, 15, 14, 22].map((val, i) => (
                          <div 
                            key={i}
                            className="w-16 h-16 flex items-center justify-center text-lg font-bold border-2 border-green-400 rounded bg-white"
                          >
                            {val}
                          </div>
                        ))}
                      </div>
                      <div className="text-xs font-mono bg-green-50 p-2 rounded border border-green-200 mt-3">
                        Values reduced: 16 → 4
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-indigo-50 rounded-lg p-6 border-2 border-indigo-300">
                  <h4 className="font-bold text-indigo-800 mb-3">💡 Key Insights</h4>
                  <ul className="space-y-2 text-sm text-gray-700">
                    <li>• <strong>Dimension Reduction:</strong> 4×4 → 2×2 (75% fewer values)</li>
                    <li>• <strong>Feature Preservation:</strong> Keeps strongest activations (max values)</li>
                    <li>• <strong>Translation Invariance:</strong> Small shifts in input don't change max</li>
                    <li>• <strong>No Parameters:</strong> No learning involved, just selection</li>
                  </ul>
                </div>
              </div>

              {/* Fully Connected Layer */}
              <div className="mb-12">
                <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                  <div className="w-8 h-8 bg-orange-600 text-white rounded-full flex items-center justify-center font-bold">3</div>
                  Fully Connected Layer Transformation
                </h3>

                <div className="grid md:grid-cols-3 gap-6 mb-6">
                  {/* Input */}
                  <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6 border-2 border-purple-300">
                    <h4 className="font-bold text-purple-800 mb-4">📥 FLATTENED INPUT</h4>
                    <div className="bg-white rounded-lg p-4 border border-purple-200">
                      <div className="text-sm font-semibold text-purple-700 mb-3">Shape: [4] (1D vector)</div>
                      <div className="space-y-2">
                        {[24, 15, 14, 22].map((val, i) => (
                          <div key={i} className="flex items-center gap-2">
                            <div className="w-8 h-8 bg-purple-100 rounded flex items-center justify-center text-xs font-bold">
                              x{i}
                            </div>
                            <div className="flex-1 bg-purple-50 rounded p-2 text-center font-mono text-sm font-bold">
                              {val}
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="text-xs text-gray-600 mt-3">
                        From 2×2 pooling output, flattened to vector
                      </div>
                    </div>
                  </div>

                  {/* Operation */}
                  <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-6 border-2 border-blue-300">
                    <h4 className="font-bold text-blue-800 mb-4">⚙️ MATRIX MULTIPLY</h4>
                    <div className="space-y-4">
                      <div className="bg-white rounded-lg p-3 border border-blue-200">
                        <div className="text-sm font-semibold text-blue-700 mb-2">Weight Matrix [4×3]</div>
                        <div className="text-xs font-mono bg-blue-50 p-2 rounded">
                          W = [<br/>
                          &nbsp;&nbsp;[0.5, -0.3, 0.8],<br/>
                          &nbsp;&nbsp;[0.2, 0.6, -0.4],<br/>
                          &nbsp;&nbsp;[-0.7, 0.9, 0.3],<br/>
                          &nbsp;&nbsp;[0.4, -0.2, 0.5]<br/>
                          ]
                        </div>
                      </div>
                      
                      <div className="bg-white rounded-lg p-3 border border-blue-200">
                        <div className="text-sm font-semibold text-blue-700 mb-2">Bias [3]</div>
                        <div className="text-xs font-mono bg-blue-50 p-2 rounded">
                          b = [1.0, -0.5, 2.0]
                        </div>
                      </div>

                      <div className="text-xs text-gray-600 bg-white p-2 rounded border border-blue-200">
                        y = Wx + b
                      </div>
                    </div>
                  </div>

                  {/* Output */}
                  <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-6 border-2 border-green-300">
                    <h4 className="font-bold text-green-800 mb-4">📤 OUTPUT</h4>
                    <div className="bg-white rounded-lg p-4 border border-green-200">
                      <div className="text-sm font-semibold text-green-700 mb-3">Shape: [3] (3 classes)</div>
                      <div className="space-y-3">
                        {[
                          { label: 'Class 0', value: 15.8, color: 'bg-red-100 border-red-300' },
                          { label: 'Class 1', value: 22.4, color: 'bg-green-100 border-green-300' },
                          { label: 'Class 2', value: 18.1, color: 'bg-blue-100 border-blue-300' }
                        ].map((item, i) => (
                          <div key={i} className={`${item.color} rounded-lg p-3 border-2`}>
                            <div className="flex justify-between items-center">
                              <span className="text-xs font-semibold">{item.label}</span>
                              <span className="text-lg font-bold">{item.value}</span>
                            </div>
                            {i === 1 && (
                              <div className="text-xs text-green-700 font-semibold mt-1">✓ Highest (Predicted)</div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-amber-50 rounded-lg p-6 border-2 border-amber-300">
                  <h4 className="font-bold text-amber-800 mb-3">🔢 Calculation Example (Class 1)</h4>
                  <div className="bg-white rounded p-4 font-mono text-sm">
                    y[1] = (24 × 0.5) + (15 × 0.2) + (14 × -0.7) + (22 × 0.4) + 1.0
                    <br/>
                    y[1] = 12.0 + 3.0 - 9.8 + 8.8 + 1.0 = <strong className="text-green-600">15.0</strong>
                  </div>
                  <p className="text-xs text-gray-600 mt-3">
                    Each output neuron computes a weighted sum of all inputs plus bias. The highest value indicates the predicted class.
                  </p>
                </div>
              </div>

              {/* Batch Normalization Layer */}
              <div className="mb-12">
                <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                  <div className="w-8 h-8 bg-teal-600 text-white rounded-full flex items-center justify-center font-bold">4</div>
                  Batch Normalization Transformation
                </h3>

                <div className="grid md:grid-cols-3 gap-6 mb-6">
                  {/* Input */}
                  <div className="bg-gradient-to-br from-red-50 to-rose-50 rounded-lg p-6 border-2 border-red-300">
                    <h4 className="font-bold text-red-800 mb-4">📥 UNNORMALIZED INPUT</h4>
                    <div className="bg-white rounded-lg p-4 border border-red-200">
                      <div className="text-sm font-semibold text-red-700 mb-3">Batch of 4 samples</div>
                      <div className="space-y-2">
                        {[
                          { label: 'Sample 1', value: 120, color: 'bg-red-200' },
                          { label: 'Sample 2', value: 85, color: 'bg-red-100' },
                          { label: 'Sample 3', value: 150, color: 'bg-red-300' },
                          { label: 'Sample 4', value: 95, color: 'bg-red-100' }
                        ].map((item, i) => (
                          <div key={i} className={`${item.color} rounded p-3 border-2 border-red-400 animate-pulse-scale`} style={{ animationDelay: `${i * 0.2}s` }}>
                            <div className="flex justify-between items-center">
                              <span className="text-xs font-semibold">{item.label}</span>
                              <span className="text-lg font-bold">{item.value}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="mt-3 p-2 bg-red-50 rounded border border-red-200">
                        <div className="text-xs text-red-700">
                          <div>Mean (μ): 112.5</div>
                          <div>Std Dev (σ): 27.5</div>
                          <div className="font-semibold mt-1">⚠️ High variance!</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Operation */}
                  <div className="bg-gradient-to-br from-teal-50 to-cyan-50 rounded-lg p-6 border-2 border-teal-300 flex flex-col justify-center">
                    <h4 className="font-bold text-teal-800 mb-4">⚙️ NORMALIZATION</h4>
                    <div className="space-y-4">
                      <div className="bg-white rounded-lg p-4 border border-teal-200 animate-normalize">
                        <div className="text-sm font-semibold text-teal-700 mb-3">Step 1: Standardize</div>
                        <div className="text-xs font-mono bg-teal-50 p-3 rounded border border-teal-200">
                          x̂ = (x - μ) / √(σ² + ε)
                        </div>
                        <div className="text-xs text-gray-600 mt-2">
                          Subtract mean, divide by std dev
                        </div>
                      </div>

                      <div className="flex items-center justify-center">
                        <ArrowRight className="w-6 h-6 text-teal-600 animate-flow-arrow" />
                      </div>

                      <div className="bg-white rounded-lg p-4 border border-teal-200">
                        <div className="text-sm font-semibold text-teal-700 mb-3">Step 2: Scale & Shift</div>
                        <div className="text-xs font-mono bg-teal-50 p-3 rounded border border-teal-200">
                          y = γ·x̂ + β
                        </div>
                        <div className="text-xs text-gray-600 mt-2">
                          γ (gamma) = 1.0, β (beta) = 0.0
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Output */}
                  <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-6 border-2 border-green-300">
                    <h4 className="font-bold text-green-800 mb-4">📤 NORMALIZED OUTPUT</h4>
                    <div className="bg-white rounded-lg p-4 border border-green-200">
                      <div className="text-sm font-semibold text-green-700 mb-3">Standardized values</div>
                      <div className="space-y-2">
                        {[
                          { label: 'Sample 1', value: 0.27, color: 'bg-green-100' },
                          { label: 'Sample 2', value: -1.00, color: 'bg-green-100' },
                          { label: 'Sample 3', value: 1.36, color: 'bg-green-100' },
                          { label: 'Sample 4', value: -0.64, color: 'bg-green-100' }
                        ].map((item, i) => (
                          <div key={i} className={`${item.color} rounded p-3 border-2 border-green-400 animate-slide-right`} style={{ animationDelay: `${i * 0.1}s` }}>
                            <div className="flex justify-between items-center">
                              <span className="text-xs font-semibold">{item.label}</span>
                              <span className="text-lg font-bold">{item.value.toFixed(2)}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="mt-3 p-2 bg-green-50 rounded border border-green-200">
                        <div className="text-xs text-green-700">
                          <div>Mean (μ): ~0.0</div>
                          <div>Std Dev (σ): ~1.0</div>
                          <div className="font-semibold mt-1 text-green-600">✓ Normalized!</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-cyan-50 rounded-lg p-6 border-2 border-cyan-300">
                  <h4 className="font-bold text-cyan-800 mb-3">🎯 Benefits of Batch Normalization</h4>
                  <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-700">
                    <ul className="space-y-2">
                      <li>• <strong>Faster Training:</strong> Reduces internal covariate shift</li>
                      <li>• <strong>Higher Learning Rates:</strong> Can train with larger steps</li>
                      <li>• <strong>Regularization:</strong> Slight noise acts as regularizer</li>
                    </ul>
                    <ul className="space-y-2">
                      <li>• <strong>Gradient Flow:</strong> Prevents vanishing/exploding gradients</li>
                      <li>• <strong>Less Sensitive:</strong> Reduces dependency on initialization</li>
                      <li>• <strong>Better Generalization:</strong> Improves model performance</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Dropout Layer */}
              <div className="mb-12">
                <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                  <div className="w-8 h-8 bg-red-600 text-white rounded-full flex items-center justify-center font-bold">5</div>
                  Dropout Layer Transformation (Training Mode)
                </h3>

                <div className="grid md:grid-cols-3 gap-6 mb-6">
                  {/* Input */}
                  <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-6 border-2 border-blue-300">
                    <h4 className="font-bold text-blue-800 mb-4">📥 ALL NEURONS ACTIVE</h4>
                    <div className="bg-white rounded-lg p-4 border border-blue-200">
                      <div className="text-sm font-semibold text-blue-700 mb-3">8 Neurons</div>
                      <div className="grid grid-cols-4 gap-2">
                        {[0.8, 1.2, 0.5, 1.5, 0.9, 1.1, 0.7, 1.3].map((val, i) => (
                          <div 
                            key={i}
                            className="bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg p-3 text-white text-center animate-pulse-scale"
                            style={{ animationDelay: `${i * 0.1}s` }}
                          >
                            <div className="text-xs mb-1">N{i+1}</div>
                            <div className="font-bold text-sm">{val}</div>
                          </div>
                        ))}
                      </div>
                      <div className="text-xs text-gray-600 mt-3">
                        All neurons passing their values
                      </div>
                    </div>
                  </div>

                  {/* Operation */}
                  <div className="bg-gradient-to-br from-red-50 to-orange-50 rounded-lg p-6 border-2 border-red-300 flex flex-col justify-center">
                    <h4 className="font-bold text-red-800 mb-4">⚙️ DROPOUT (p=0.5)</h4>
                    <div className="space-y-4">
                      <div className="bg-white rounded-lg p-4 border border-red-200">
                        <div className="text-sm font-semibold text-red-700 mb-3">Random Mask</div>
                        <div className="grid grid-cols-4 gap-1 mb-2">
                          {[1, 0, 1, 0, 1, 1, 0, 1].map((mask, i) => (
                            <div 
                              key={i}
                              className={`w-10 h-10 flex items-center justify-center font-bold border-2 rounded ${
                                mask === 1 ? 'bg-green-100 border-green-400 text-green-700' : 'bg-red-100 border-red-400 text-red-700 animate-dropout-flicker'
                              }`}
                            >
                              {mask === 1 ? '✓' : '✗'}
                            </div>
                          ))}
                        </div>
                        <div className="text-xs text-gray-600">
                          50% randomly set to zero
                        </div>
                      </div>

                      <div className="bg-white rounded-lg p-3 border border-red-200">
                        <div className="text-sm font-semibold text-red-700 mb-2">Formula</div>
                        <div className="text-xs font-mono bg-red-50 p-2 rounded">
                          output = input × mask / (1-p)
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Output */}
                  <div className="bg-gradient-to-br from-gray-50 to-slate-50 rounded-lg p-6 border-2 border-gray-300">
                    <h4 className="font-bold text-gray-800 mb-4">📤 SOME NEURONS DROPPED</h4>
                    <div className="bg-white rounded-lg p-4 border border-gray-200">
                      <div className="text-sm font-semibold text-gray-700 mb-3">8 Neurons (4 active)</div>
                      <div className="grid grid-cols-4 gap-2">
                        {[
                          { val: 1.6, active: true },
                          { val: 0.0, active: false },
                          { val: 1.0, active: true },
                          { val: 0.0, active: false },
                          { val: 1.8, active: true },
                          { val: 2.2, active: true },
                          { val: 0.0, active: false },
                          { val: 2.6, active: true }
                        ].map((item, i) => (
                          <div 
                            key={i}
                            className={`rounded-lg p-3 text-center border-2 ${
                              item.active 
                                ? 'bg-gradient-to-br from-green-400 to-green-600 border-green-700 text-white' 
                                : 'bg-gray-200 border-gray-400 text-gray-500 opacity-50'
                            }`}
                          >
                            <div className="text-xs mb-1">N{i+1}</div>
                            <div className="font-bold text-sm">{item.val}</div>
                          </div>
                        ))}
                      </div>
                      <div className="text-xs text-gray-600 mt-3">
                        Scaled by 1/(1-p) = 2.0 to maintain expected value
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-orange-50 rounded-lg p-6 border-2 border-orange-300">
                  <h4 className="font-bold text-orange-800 mb-3">🔥 Dropout Effects</h4>
                  <div className="grid md:grid-cols-2 gap-6 text-sm text-gray-700">
                    <div>
                      <div className="font-semibold text-orange-700 mb-2">Training Mode:</div>
                      <ul className="space-y-1">
                        <li>• Randomly drops neurons (probability p)</li>
                        <li>• Forces network to not rely on specific neurons</li>
                        <li>• Creates ensemble effect (different sub-networks)</li>
                        <li>• Scales remaining by 1/(1-p)</li>
                      </ul>
                    </div>
                    <div>
                      <div className="font-semibold text-orange-700 mb-2">Testing/Inference Mode:</div>
                      <ul className="space-y-1">
                        <li>• All neurons active (no dropout)</li>
                        <li>• Outputs scaled by (1-p)</li>
                        <li>• Ensures expected value matches training</li>
                        <li>• Deterministic predictions</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* ReLU Activation Layer */}
              <div className="mb-12">
                <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                  <div className="w-8 h-8 bg-yellow-600 text-white rounded-full flex items-center justify-center font-bold">6</div>
                  ReLU Activation Function Transformation
                </h3>

                <div className="grid md:grid-cols-3 gap-6 mb-6">
                  {/* Input */}
                  <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6 border-2 border-purple-300">
                    <h4 className="font-bold text-purple-800 mb-4">📥 RAW ACTIVATIONS</h4>
                    <div className="bg-white rounded-lg p-4 border border-purple-200">
                      <div className="text-sm font-semibold text-purple-700 mb-3">Mixed positive & negative</div>
                      <div className="space-y-2">
                        {[2.5, -1.3, 4.2, -0.8, 1.7, -2.1, 3.6, 0.5].map((val, i) => (
                          <div key={i} className="flex items-center gap-2 animate-slide-right" style={{ animationDelay: `${i * 0.05}s` }}>
                            <div className={`flex-1 rounded p-2 border-2 ${
                              val >= 0 
                                ? 'bg-green-50 border-green-300' 
                                : 'bg-red-50 border-red-300'
                            }`}>
                              <div className="flex justify-between items-center">
                                <span className="text-xs">Value {i+1}:</span>
                                <span className={`font-bold ${val >= 0 ? 'text-green-700' : 'text-red-700'}`}>
                                  {val > 0 ? '+' : ''}{val.toFixed(1)}
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Operation */}
                  <div className="bg-gradient-to-br from-yellow-50 to-amber-50 rounded-lg p-6 border-2 border-yellow-300 flex flex-col justify-center">
                    <h4 className="font-bold text-yellow-800 mb-4">⚙️ ReLU FUNCTION</h4>
                    <div className="space-y-4">
                      <div className="bg-white rounded-lg p-4 border border-yellow-200">
                        <div className="text-sm font-semibold text-yellow-700 mb-3">Formula</div>
                        <div className="text-lg font-mono bg-yellow-50 p-3 rounded border border-yellow-200 text-center">
                          f(x) = max(0, x)
                        </div>
                        <div className="text-xs text-gray-600 mt-2 text-center">
                          Keep positive, zero out negative
                        </div>
                      </div>

                      <div className="bg-white rounded-lg p-4 border border-yellow-200">
                        <div className="text-sm font-semibold text-yellow-700 mb-2">Decision</div>
                        <div className="space-y-2 text-xs">
                          <div className="flex items-center gap-2">
                            <div className="w-4 h-4 bg-green-500 rounded"></div>
                            <span>If x {'>'} 0: output = x</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-4 h-4 bg-red-500 rounded"></div>
                            <span>If x ≤ 0: output = 0</span>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center justify-center">
                        <ArrowRight className="w-6 h-6 text-yellow-600 animate-flow-arrow" />
                      </div>
                    </div>
                  </div>

                  {/* Output */}
                  <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-6 border-2 border-green-300">
                    <h4 className="font-bold text-green-800 mb-4">📤 NON-NEGATIVE OUTPUT</h4>
                    <div className="bg-white rounded-lg p-4 border border-green-200">
                      <div className="text-sm font-semibold text-green-700 mb-3">Only positive values</div>
                      <div className="space-y-2">
                        {[2.5, 0.0, 4.2, 0.0, 1.7, 0.0, 3.6, 0.5].map((val, i) => (
                          <div key={i} className="flex items-center gap-2 animate-fade-in" style={{ animationDelay: `${i * 0.05}s` }}>
                            <div className={`flex-1 rounded p-2 border-2 ${
                              val > 0 
                                ? 'bg-green-100 border-green-400' 
                                : 'bg-gray-100 border-gray-300 opacity-60'
                            }`}>
                              <div className="flex justify-between items-center">
                                <span className="text-xs">Value {i+1}:</span>
                                <span className={`font-bold ${val > 0 ? 'text-green-700' : 'text-gray-500'}`}>
                                  {val.toFixed(1)}
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="text-xs text-gray-600 mt-3">
                        50% sparsity (4 of 8 active)
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-indigo-50 rounded-lg p-6 border-2 border-indigo-300">
                  <h4 className="font-bold text-indigo-800 mb-3">⚡ Why ReLU is Popular</h4>
                  <div className="grid md:grid-cols-3 gap-4 text-sm text-gray-700">
                    <div>
                      <div className="font-semibold text-indigo-700 mb-2">Computational:</div>
                      <ul className="space-y-1">
                        <li>• Very fast to compute</li>
                        <li>• Simple max(0, x) operation</li>
                        <li>• No exponentials like sigmoid</li>
                      </ul>
                    </div>
                    <div>
                      <div className="font-semibold text-indigo-700 mb-2">Training:</div>
                      <ul className="space-y-1">
                        <li>• Mitigates vanishing gradient</li>
                        <li>• Constant gradient for x {'>'} 0</li>
                        <li>• Sparse activations (many zeros)</li>
                      </ul>
                    </div>
                    <div>
                      <div className="font-semibold text-indigo-700 mb-2">Biological:</div>
                      <ul className="space-y-1">
                        <li>• Similar to neuron firing</li>
                        <li>• Neurons either fire or don't</li>
                        <li>• One-sided nonlinearity</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* Complete Network Flow */}
              <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg p-8 text-white">
                <h3 className="text-2xl font-bold mb-6">🔄 Complete Data Flow with All Layers</h3>
                <div className="grid md:grid-cols-4 lg:grid-cols-8 gap-3 mb-6">
                  {[
                    { step: '1️⃣', name: 'Input', size: '5×5×1', vals: '25' },
                    { step: '2️⃣', name: 'Conv', size: '3×3×2', vals: '18' },
                    { step: '3️⃣', name: 'ReLU', size: '3×3×2', vals: '18' },
                    { step: '4️⃣', name: 'BatchNorm', size: '3×3×2', vals: '18' },
                    { step: '5️⃣', name: 'Pool', size: '2×2×1', vals: '4' },
                    { step: '6️⃣', name: 'Dropout', size: '[4]', vals: '2-4' },
                    { step: '7️⃣', name: 'FC', size: '[3]', vals: '3' },
                    { step: '8️⃣', name: 'Softmax', size: '[3]', vals: '3' }
                  ].map((layer, i) => (
                    <div key={i} className="bg-white bg-opacity-20 rounded-lg p-3 backdrop-blur animate-fade-in" style={{ animationDelay: `${i * 0.1}s` }}>
                      <div className="font-bold mb-1 text-sm">{layer.step} {layer.name}</div>
                      <div className="text-xs">{layer.size}</div>
                      <div className="text-xs opacity-75">{layer.vals} vals</div>
                    </div>
                  ))}
                </div>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div className="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur">
                    <strong>Transformation Pipeline:</strong>
                    <div className="mt-2 opacity-90">
                      Raw pixels → Features → Activation → Normalization → Compression → Regularization → Classification
                    </div>
                  </div>
                  <div className="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur">
                    <strong>Key Operations:</strong>
                    <div className="mt-2 opacity-90">
                      Convolution (feature extraction) → ReLU (non-linearity) → BatchNorm (stability) → Pooling (downsampling) → Dropout (regularization) → FC (decision making)
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
