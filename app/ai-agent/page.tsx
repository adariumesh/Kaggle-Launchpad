'use client';

import React from 'react';
import { AIAgentDashboard } from '@/components/ai-agent-dashboard';

export default function AIAgentPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-black relative overflow-hidden">
      {/* Organic flowing background shapes */}
      <div className="bg-shape bg-shape-1 -top-32 -left-32" />
      <div className="bg-shape bg-shape-2 top-1/4 -right-40" />
      <div className="bg-shape bg-shape-3 bottom-1/4 -left-20" />
      <div className="bg-shape bg-shape-4 top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
      <div className="bg-shape bg-shape-5 -bottom-32 -right-32" />
      
      {/* Content layer */}
      <div className="relative z-10">
        <div className="container mx-auto px-4 py-8">
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-white mb-2">
              🤖 AI Agent Dashboard
            </h1>
            <p className="text-lg text-gray-300">
              Monitor your intelligent Kaggle competition assistant
            </p>
          </div>
          
          <AIAgentDashboard />
        </div>
      </div>
    </div>
  );
}