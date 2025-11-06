import React from 'react';
import { ChevronDown } from 'lucide-react';
import type { ModelInfo } from '../types';

interface ModelSelectorProps {
  models: ModelInfo[];
  selectedModel: ModelInfo;
  onModelChange: (model: ModelInfo) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModel,
  onModelChange,
}) => {
  return (
    <div className="relative inline-block">
      <label htmlFor="model-select" className="block text-sm font-medium text-gray-700 mb-2">
        Select Neural Network Model
      </label>
      <div className="relative">
        <select
          id="model-select"
          value={selectedModel.id}
          onChange={(e) => {
            const model = models.find((m) => m.id === e.target.value);
            if (model) onModelChange(model);
          }}
          className="block w-full px-4 py-3 pr-10 text-base border border-gray-300 rounded-lg shadow-sm 
                     focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 
                     bg-white cursor-pointer appearance-none hover:border-gray-400 transition-colors"
        >
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name} ({model.year})
            </option>
          ))}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500">
          <ChevronDown className="h-5 w-5" />
        </div>
      </div>
      <p className="mt-2 text-sm text-gray-500">
        {selectedModel.totalParameters} parameters • {selectedModel.inputSize} input
      </p>
    </div>
  );
};

export default ModelSelector;
