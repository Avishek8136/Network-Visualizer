import React from 'react';
import { X, Calendar, Target, Layers, TrendingUp, ExternalLink, CheckCircle } from 'lucide-react';
import type { ModelInfo } from '../types';

interface ModelInfoPanelProps {
  model: ModelInfo;
  isOpen: boolean;
  onClose: () => void;
}

const ModelInfoPanel: React.FC<ModelInfoPanelProps> = ({ model, isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed right-0 top-0 h-full w-96 bg-white shadow-2xl z-50 overflow-y-auto border-l border-gray-200">
      <div className="sticky top-0 bg-white border-b border-gray-200 p-4 flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">{model.name}</h2>
        <button
          onClick={onClose}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          aria-label="Close panel"
        >
          <X className="w-6 h-6 text-gray-500" />
        </button>
      </div>

      <div className="p-6 space-y-6">
        {/* Year */}
        <div className="flex items-start gap-3">
          <Calendar className="w-5 h-5 text-blue-600 mt-0.5" />
          <div>
            <h3 className="font-semibold text-gray-900">Published</h3>
            <p className="text-gray-600">{model.year}</p>
          </div>
        </div>

        {/* Description */}
        <div>
          <h3 className="font-semibold text-gray-900 mb-2">Description</h3>
          <p className="text-gray-600 text-sm leading-relaxed">{model.description}</p>
        </div>

        {/* Key Features */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle className="w-5 h-5 text-green-600" />
            <h3 className="font-semibold text-gray-900">Key Features</h3>
          </div>
          <ul className="space-y-2">
            {model.keyFeatures.map((feature, idx) => (
              <li key={idx} className="flex items-start gap-2 text-sm text-gray-600">
                <span className="text-blue-500 mt-1">•</span>
                <span>{feature}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Architecture Stats */}
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-4 space-y-3">
          <div className="flex items-start gap-3">
            <Layers className="w-5 h-5 text-blue-600 mt-0.5" />
            <div>
              <h3 className="font-semibold text-gray-900">Total Parameters</h3>
              <p className="text-gray-600">{model.totalParameters}</p>
            </div>
          </div>

          <div className="flex items-start gap-3">
            <Target className="w-5 h-5 text-blue-600 mt-0.5" />
            <div>
              <h3 className="font-semibold text-gray-900">Input Size</h3>
              <p className="text-gray-600">{model.inputSize}</p>
            </div>
          </div>

          <div className="flex items-start gap-3">
            <TrendingUp className="w-5 h-5 text-blue-600 mt-0.5" />
            <div>
              <h3 className="font-semibold text-gray-900">Accuracy</h3>
              <p className="text-gray-600">{model.accuracy}</p>
            </div>
          </div>
        </div>

        {/* Use Cases */}
        <div>
          <h3 className="font-semibold text-gray-900 mb-3">Common Use Cases</h3>
          <div className="flex flex-wrap gap-2">
            {model.useCases.map((useCase, idx) => (
              <span
                key={idx}
                className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium"
              >
                {useCase}
              </span>
            ))}
          </div>
        </div>

        {/* Paper Link */}
        {model.paperLink && (
          <div>
            <a
              href={model.paperLink}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-blue-600 hover:text-blue-700 font-medium transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              Read Original Paper
            </a>
          </div>
        )}

        {/* Layer Details */}
        <div>
          <h3 className="font-semibold text-gray-900 mb-3">Layer Information</h3>
          <div className="space-y-3">
            {model.layers.map((layer, idx) => (
              <div
                key={idx}
                className="border border-gray-200 rounded-lg p-3 hover:border-blue-300 transition-colors"
              >
                <div className="flex items-start justify-between mb-1">
                  <h4 className="font-medium text-gray-900">{layer.name}</h4>
                  <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                    {layer.type}
                  </span>
                </div>
                <p className="text-sm text-gray-600 mb-1">
                  Output: <span className="font-mono text-blue-600">{layer.outputShape}</span>
                </p>
                {layer.parameters && (
                  <p className="text-xs text-gray-500">
                    Parameters: {layer.parameters.toLocaleString()}
                  </p>
                )}
                {layer.description && (
                  <p className="text-xs text-gray-500 mt-1">{layer.description}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelInfoPanel;
