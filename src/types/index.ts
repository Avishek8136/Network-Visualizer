export interface LayerInfo {
  name: string;
  type: string;
  outputShape?: string;
  parameters?: number;
  description?: string;
  kernelSize?: string;
  stride?: string;
  padding?: string;
  activation?: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  year: number;
  description: string;
  keyFeatures: string[];
  totalParameters: string;
  inputSize: string;
  outputClasses: number;
  accuracy: string;
  layers: LayerInfo[];
  paperLink?: string;
  useCases: string[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}
