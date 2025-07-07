'use client';

import React from 'react';
import { AIAgentDashboard } from '@/components/ai-agent-dashboard';

export default function AIAgentPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-950 to-slate-900">
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
  );
}