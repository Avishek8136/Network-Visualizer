import { useState } from 'react';
import { Info, Brain, Home, Network, Activity } from 'lucide-react';
import ModelSelector from './components/ModelSelector';
import NetworkVisualizer from './components/NetworkVisualizer';
import ModelInfoPanel from './components/ModelInfoPanel';
import Chatbot from './components/Chatbot';
import Dashboard from './components/Dashboard';
import InferenceExplorer from './components/InferenceExplorer';
import { neuralNetworkModels } from './data/models';

function App() {
  const [selectedModel, setSelectedModel] = useState(neuralNetworkModels[0]);
  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState<'dashboard' | 'visualizer' | 'inference'>('dashboard');

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-3 rounded-xl">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">Neural Network Visualizer</h1>
                <p className="text-sm text-gray-600">Explore and understand deep learning architectures</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              {/* Navigation Buttons */}
              <button
                onClick={() => setCurrentPage('dashboard')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors shadow-md hover:shadow-lg ${
                  currentPage === 'dashboard'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Home className="w-5 h-5" />
                Dashboard
              </button>
              <button
                onClick={() => setCurrentPage('visualizer')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors shadow-md hover:shadow-lg ${
                  currentPage === 'visualizer'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Network className="w-5 h-5" />
                Visualizer
              </button>
              <button
                onClick={() => setCurrentPage('inference')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors shadow-md hover:shadow-lg ${
                  currentPage === 'inference'
                    ? 'bg-purple-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Activity className="w-5 h-5" />
                Inference Explorer
              </button>
              
              {currentPage === 'visualizer' && (
                <button
                  onClick={() => setIsInfoOpen(true)}
                  className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors shadow-md hover:shadow-lg"
                >
                  <Info className="w-5 h-5" />
                  Model Info
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      {currentPage === 'dashboard' ? (
        <Dashboard />
      ) : currentPage === 'inference' ? (
        <InferenceExplorer />
      ) : (
        <main className="max-w-7xl mx-auto px-6 py-8">
          {/* Model Selector */}
          <div className="mb-6">
            <ModelSelector
              models={neuralNetworkModels}
              selectedModel={selectedModel}
              onModelChange={(model) => {
                setSelectedModel(model);
                setIsInfoOpen(false); // Close info panel when changing models
              }}
            />
          </div>

          {/* Visualization */}
          <div className="h-[calc(100vh-280px)] min-h-[600px]">
            <NetworkVisualizer model={selectedModel} />
          </div>

          {/* Info Panel */}
          <ModelInfoPanel
            model={selectedModel}
            isOpen={isInfoOpen}
            onClose={() => setIsInfoOpen(false)}
          />
        </main>
      )}

      {/* Floating Chatbot - Available on all pages */}
      <Chatbot currentModel={currentPage === 'visualizer' ? selectedModel.name : undefined} />
    </div>
  );
}

export default App;
