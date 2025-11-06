import React, { useCallback, useMemo, useEffect, useState } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  BackgroundVariant,
  MarkerType,
  Panel,
  Handle,
  Position,
} from '@xyflow/react';
import type { Node, Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import type { ModelInfo, LayerInfo } from '../types';
import { X, Info } from 'lucide-react';

interface NetworkVisualizerProps {
  model: ModelInfo;
}

interface ExtendedLayerInfo extends LayerInfo {
  index: number;
}

// Custom node component with detailed hover info
const LayerNode: React.FC<{ 
  data: ExtendedLayerInfo & { 
    showDetails?: boolean;
    onToggleDetails?: () => void;
  }; 
  selected?: boolean;
}> = ({ data, selected }) => {
  const showDetails = data.showDetails || false;
  const onToggleDetails = data.onToggleDetails || (() => {});

  const getNodeColor = (type: string) => {
    const colors: Record<string, string> = {
      'Input': 'bg-green-100 border-green-500 text-green-900',
      'Convolution': 'bg-blue-100 border-blue-500 text-blue-900',
      'MaxPooling': 'bg-purple-100 border-purple-500 text-purple-900',
      'GlobalAvgPooling': 'bg-purple-100 border-purple-500 text-purple-900',
      'Fully Connected': 'bg-orange-100 border-orange-500 text-orange-900',
      'Flatten': 'bg-yellow-100 border-yellow-500 text-yellow-900',
      'Dropout': 'bg-red-100 border-red-500 text-red-900',
      'Residual Block': 'bg-indigo-100 border-indigo-500 text-indigo-900',
      'Inception Module': 'bg-pink-100 border-pink-500 text-pink-900',
      'Convolution Block': 'bg-cyan-100 border-cyan-500 text-cyan-900',
    };
    return colors[type] || 'bg-gray-100 border-gray-500 text-gray-900';
  };

  const getLayerDescription = (type: string): string => {
    const descriptions: Record<string, string> = {
      'Input': 'Entry point for the network, receives raw image data',
      'Convolution': 'Extracts spatial features using learnable filters',
      'MaxPooling': 'Downsamples by taking maximum value, provides translation invariance',
      'GlobalAvgPooling': 'Reduces each feature map to a single value by averaging',
      'Fully Connected': 'Dense layer where every neuron connects to all inputs',
      'Flatten': 'Reshapes multi-dimensional data into 1D vector',
      'Dropout': 'Randomly deactivates neurons during training to prevent overfitting',
      'Residual Block': 'Skip connection allows gradient flow, enables very deep networks',
      'Inception Module': 'Parallel convolutions of different sizes for multi-scale features',
      'Convolution Block': 'Multiple stacked convolutional layers',
    };
    return descriptions[type] || 'Neural network layer';
  };

  return (
    <div className="relative group">
      {/* Input handle (top) */}
      <Handle
        type="target"
        position={Position.Top}
        style={{ background: '#3b82f6', width: 10, height: 10 }}
      />
      
      <div 
        className={`px-4 py-3 border-2 rounded-lg shadow-md min-w-[220px] transition-all cursor-pointer
          ${getNodeColor(data.type)} 
          ${selected ? 'ring-4 ring-blue-300 scale-105' : 'hover:scale-105 hover:shadow-xl'}`}
        onClick={onToggleDetails}
      >
        {/* Layer Name & Type */}
        <div className="flex items-center justify-between mb-1">
          <div className="font-bold text-sm">{data.name}</div>
          <Info className="w-4 h-4 opacity-50" />
        </div>
        <div className="text-xs font-semibold opacity-70 mb-2">{data.type}</div>
        
        {/* Output Shape */}
        {data.outputShape && (
          <div className="text-xs font-mono bg-white bg-opacity-60 px-2 py-1 rounded mb-1">
            Out: {data.outputShape}
          </div>
        )}
        
        {/* Key Parameters - Quick View */}
        <div className="text-xs space-y-0.5 mt-2">
          {data.kernelSize && (
            <div className="flex justify-between">
              <span className="opacity-70">Kernel:</span>
              <span className="font-semibold">{data.kernelSize}</span>
            </div>
          )}
          {data.stride && (
            <div className="flex justify-between">
              <span className="opacity-70">Stride:</span>
              <span className="font-semibold">{data.stride}</span>
            </div>
          )}
          {data.parameters && (
            <div className="flex justify-between">
              <span className="opacity-70">Params:</span>
              <span className="font-semibold">{data.parameters.toLocaleString()}</span>
            </div>
          )}
        </div>
      </div>

      {/* Detailed Information Popup */}
      {showDetails && (
        <div className="absolute top-0 left-full ml-4 z-50 w-80 bg-white rounded-lg shadow-2xl border-2 border-gray-300 p-4">
          <div className="flex justify-between items-start mb-3">
            <h3 className="font-bold text-lg text-gray-900">{data.name}</h3>
            <button 
              onClick={(e) => {
                e.stopPropagation();
                onToggleDetails();
              }}
              className="text-gray-500 hover:text-gray-700"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          
          <div className="space-y-3 text-sm">
            {/* Layer Type & Description */}
            <div>
              <div className="font-semibold text-blue-700">{data.type}</div>
              <p className="text-gray-600 text-xs mt-1 leading-relaxed">
                {getLayerDescription(data.type)}
              </p>
            </div>

            {/* Technical Specifications */}
            <div className="bg-gray-50 rounded p-3 space-y-2">
              <div className="font-semibold text-gray-700">Specifications:</div>
              
              {data.outputShape && (
                <div className="flex justify-between text-xs">
                  <span className="text-gray-600">Output Shape:</span>
                  <span className="font-mono font-semibold text-blue-600">{data.outputShape}</span>
                </div>
              )}
              
              {data.kernelSize && (
                <div className="flex justify-between text-xs">
                  <span className="text-gray-600">Kernel Size:</span>
                  <span className="font-mono font-semibold">{data.kernelSize}</span>
                </div>
              )}
              
              {data.stride && (
                <div className="flex justify-between text-xs">
                  <span className="text-gray-600">Stride:</span>
                  <span className="font-mono font-semibold">{data.stride}</span>
                </div>
              )}
              
              {data.padding && (
                <div className="flex justify-between text-xs">
                  <span className="text-gray-600">Padding:</span>
                  <span className="font-mono font-semibold">{data.padding}</span>
                </div>
              )}
              
              {data.activation && (
                <div className="flex justify-between text-xs">
                  <span className="text-gray-600">Activation:</span>
                  <span className="font-semibold text-green-600">{data.activation}</span>
                </div>
              )}
              
              {data.parameters && (
                <div className="flex justify-between text-xs">
                  <span className="text-gray-600">Parameters:</span>
                  <span className="font-semibold text-orange-600">{data.parameters.toLocaleString()}</span>
                </div>
              )}
            </div>

            {/* Additional Description */}
            {data.description && (
              <div>
                <div className="font-semibold text-gray-700 mb-1">Details:</div>
                <p className="text-xs text-gray-600 leading-relaxed">{data.description}</p>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Output handle (bottom) */}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: '#3b82f6', width: 10, height: 10 }}
      />
    </div>
  );
};

const nodeTypes = {
  layerNode: LayerNode,
};

const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({ model }) => {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const handleToggleDetails = useCallback((nodeId: string) => {
    setSelectedNodeId(prev => prev === nodeId ? null : nodeId);
  }, []);

  // Generate nodes and edges with special handling for ResNet
  const { initialNodes, initialEdges } = useMemo(() => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    
    // Check if this is ResNet (has Residual Blocks)
    const isResNet = model.id.includes('resnet') || model.layers.some(l => l.type === 'Residual Block');
    
    if (isResNet) {
      // Special layout for ResNet with skip connections
      let yPos = 0;
      
      model.layers.forEach((layer, index) => {
        if (layer.type === 'Residual Block') {
          // Create main path node (slightly to the right)
          nodes.push({
            id: `${model.id}-layer-${index}`,
            type: 'layerNode',
            position: { x: 500, y: yPos },
            data: { ...layer, index },
          });
          
          // Create skip connection node (to the left)
          nodes.push({
            id: `${model.id}-skip-${index}`,
            type: 'layerNode',
            position: { x: 200, y: yPos },
            data: {
              name: `Skip ${index}`,
              type: 'Convolution',
              outputShape: layer.outputShape,
              description: '1×1 conv for dimension matching (identity shortcut)',
              kernelSize: '1×1',
              stride: '1 or 2',
              index: index,
            } as unknown as Record<string, unknown>,
          } as Node);
          
          // Add skip connection edge (curved, different color)
          if (index > 0) {
            edges.push({
              id: `${model.id}-skip-edge-${index}`,
              source: `${model.id}-layer-${index - 1}`,
              target: `${model.id}-skip-${index}`,
              type: 'smoothstep',
              animated: false,
              style: { 
                stroke: '#f59e0b',  // Orange for skip
                strokeWidth: 2,
                strokeDasharray: '5,5',
              },
              markerEnd: {
                type: MarkerType.ArrowClosed,
                width: 15,
                height: 15,
                color: '#f59e0b',
              },
              label: 'Skip',
              labelStyle: { fill: '#f59e0b', fontSize: 10 },
            });
          }
          
          yPos += 250;  // More space for residual blocks
        } else {
          // Regular layer
          nodes.push({
            id: `${model.id}-layer-${index}`,
            type: 'layerNode',
            position: { x: 500, y: yPos },
            data: { ...layer, index },
          });
          yPos += 150;
        }
        
        // Add main path edge
        if (index > 0) {
          edges.push({
            id: `${model.id}-edge-${index}`,
            source: `${model.id}-layer-${index - 1}`,
            target: `${model.id}-layer-${index}`,
            type: 'smoothstep',
            animated: true,
            style: { 
              stroke: '#3b82f6', 
              strokeWidth: 3,
            },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              width: 20,
              height: 20,
              color: '#3b82f6',
            },
          });
        }
      });
    } else {
      // Standard sequential layout for other models
      model.layers.forEach((layer, index) => {
        nodes.push({
          id: `${model.id}-layer-${index}`,
          type: 'layerNode',
          position: { x: 400, y: index * 150 },
          data: { ...layer, index },
        });
        
        if (index > 0) {
          edges.push({
            id: `${model.id}-edge-${index}`,
            source: `${model.id}-layer-${index - 1}`,
            target: `${model.id}-layer-${index}`,
            type: 'smoothstep',
            animated: true,
            style: { 
              stroke: '#3b82f6', 
              strokeWidth: 3,
            },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              width: 20,
              height: 20,
              color: '#3b82f6',
            },
            label: layer.stride ? `stride ${layer.stride}` : undefined,
            labelStyle: { fill: '#1e40af', fontSize: 10, fontWeight: 600 },
          });
        }
      });
    }

    return { initialNodes: nodes, initialEdges: edges };
  }, [model]);

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(initialEdges);

  // Reset selected node when model changes
  useEffect(() => {
    setSelectedNodeId(null);
  }, [model.id]);

  // Update nodes and edges when model or selectedNodeId changes
  useEffect(() => {
    const enrichedNodes = initialNodes.map((node) => ({
      ...node,
      data: {
        ...node.data,
        showDetails: node.id === selectedNodeId,
        onToggleDetails: () => handleToggleDetails(node.id),
      },
    }));
    setNodes(enrichedNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, selectedNodeId, handleToggleDetails, setNodes, setEdges]);

  const onInit = useCallback(() => {
    console.log('Flow initialized');
  }, []);

  return (
    <div className="w-full h-full bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-6 py-4">
        <h2 className="text-xl font-bold">{model.name} Architecture</h2>
        <p className="text-sm opacity-90 mt-1">
          {model.layers.length} layers • {model.totalParameters} parameters
        </p>
      </div>
      
      <div style={{ width: '100%', height: 'calc(100% - 80px)' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onInit={onInit}
          nodeTypes={nodeTypes}
          fitView
          minZoom={0.1}
          maxZoom={1.5}
          defaultViewport={{ x: 0, y: 0, zoom: 0.7 }}
        >
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#e5e7eb" />
          <Controls />
          <MiniMap
            nodeColor={(node) => {
              const type = (node.data as unknown as LayerInfo).type;
              const colorMap: Record<string, string> = {
                'Input': '#86efac',
                'Convolution': '#93c5fd',
                'MaxPooling': '#d8b4fe',
                'Fully Connected': '#fdba74',
                'Flatten': '#fde047',
                'Dropout': '#fca5a5',
                'Residual Block': '#c7d2fe',
                'Inception Module': '#fbcfe8',
              };
              return colorMap[type] || '#e5e7eb';
            }}
            maskColor="rgba(0, 0, 0, 0.1)"
          />
          
          {/* Legend Panel */}
          <Panel position="top-right" className="bg-white rounded-lg shadow-lg p-4 m-4 text-xs">
            <div className="font-bold mb-2 text-gray-800">Legend</div>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-8 h-0.5 bg-blue-500"></div>
                <span className="text-gray-600">Main Path</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-0.5 bg-orange-500 border-dashed border-t-2 border-orange-500"></div>
                <span className="text-gray-600">Skip Connection</span>
              </div>
              <div className="mt-3 pt-2 border-t border-gray-200">
                <div className="text-gray-600 font-semibold mb-1">Interactions:</div>
                <div className="text-gray-500">• Click node for details</div>
                <div className="text-gray-500">• Hover for quick info</div>
                <div className="text-gray-500">• Zoom & pan to explore</div>
              </div>
            </div>
          </Panel>
        </ReactFlow>
      </div>
    </div>
  );
};

export default NetworkVisualizer;
