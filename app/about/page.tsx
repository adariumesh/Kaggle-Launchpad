'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  Brain, 
  Target, 
  Users, 
  Zap, 
  Award, 
  TrendingUp,
  Code,
  Database,
  Lightbulb,
  Rocket
} from 'lucide-react';

export default function AboutPage() {
  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Analysis',
      description: 'Advanced machine learning algorithms analyze competition data and generate optimized solutions.'
    },
    {
      icon: Code,
      title: 'Production-Ready Code',
      description: 'Clean, well-documented code that follows best practices and is ready for submission.'
    },
    {
      icon: Database,
      title: 'Comprehensive EDA',
      description: 'Detailed exploratory data analysis with visualizations and insights.'
    },
    {
      icon: Zap,
      title: 'Rapid Generation',
      description: 'Get your complete project setup in minutes, not hours or days.'
    }
  ];

  const stats = [
    { label: 'Projects Generated', value: '10,000+', icon: Rocket },
    { label: 'Competitions Supported', value: '500+', icon: Target },
    { label: 'Happy Users', value: '2,500+', icon: Users },
    { label: 'Average Score Improvement', value: '15%', icon: TrendingUp }
  ];

  const team = [
    {
      name: 'Dr. Sarah Chen',
      role: 'AI Research Lead',
      description: 'Former Kaggle Grandmaster with expertise in deep learning and feature engineering.'
    },
    {
      name: 'Marcus Rodriguez',
      role: 'Platform Engineer',
      description: 'Full-stack developer specializing in scalable ML infrastructure and cloud platforms.'
    },
    {
      name: 'Dr. Aisha Patel',
      role: 'Data Science Advisor',
      description: 'PhD in Statistics with 10+ years experience in competitive machine learning.'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-black relative">
      {/* Background shapes */}
      <div className="bg-shape bg-shape-1 -top-32 -left-32" />
      <div className="bg-shape bg-shape-2 top-1/4 -right-40" />
      <div className="bg-shape bg-shape-3 bottom-1/4 -left-20" />
      <div className="bg-shape bg-shape-4 top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
      
      <div className="relative z-10">
        <div className="container mx-auto px-4 py-12 max-w-6xl">
          {/* Header */}
          <div className="text-center mb-16">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-6">
              About Kaggle Launchpad
            </h1>
            <p className="text-xl text-gray-200 max-w-3xl mx-auto leading-relaxed">
              We're revolutionizing how data scientists approach Kaggle competitions by providing 
              AI-powered tools that generate production-ready notebooks and project scaffolds in minutes.
            </p>
          </div>

          {/* Mission Section */}
          <Card className="glass-card border-white/20 mb-12">
            <CardHeader className="text-center">
              <CardTitle className="text-2xl text-white flex items-center justify-center">
                <Target className="h-6 w-6 mr-2 text-blue-400" />
                Our Mission
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-200 text-lg leading-relaxed text-center max-w-4xl mx-auto">
                To democratize competitive machine learning by providing intelligent automation tools 
                that help data scientists focus on innovation rather than repetitive setup tasks. 
                We believe everyone should have access to world-class ML practices and winning techniques.
              </p>
            </CardContent>
          </Card>

          {/* Stats Section */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16">
            {stats.map((stat, index) => (
              <Card key={index} className="glass-card border-white/20 text-center">
                <CardContent className="pt-6">
                  <stat.icon className="h-8 w-8 text-purple-400 mx-auto mb-2" />
                  <div className="text-2xl font-bold text-white mb-1">{stat.value}</div>
                  <div className="text-sm text-gray-300">{stat.label}</div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Features Section */}
          <div className="mb-16">
            <h2 className="text-3xl font-bold text-white text-center mb-8">
              What Makes Us Different
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {features.map((feature, index) => (
                <Card key={index} className="glass-card border-white/20">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center">
                      <feature.icon className="h-5 w-5 mr-2 text-purple-400" />
                      {feature.title}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-200">{feature.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          {/* Story Section */}
          <Card className="glass-card border-white/20 mb-16">
            <CardHeader>
              <CardTitle className="text-2xl text-white flex items-center">
                <Lightbulb className="h-6 w-6 mr-2 text-yellow-400" />
                Our Story
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-gray-200 leading-relaxed">
                Kaggle Launchpad was born from the frustration of spending countless hours on repetitive 
                setup tasks instead of focusing on the creative aspects of machine learning. Our founders, 
                all experienced Kaggle competitors, noticed that 80% of competition work involved the same 
                foundational steps: data exploration, baseline modeling, and feature engineering.
              </p>
              <p className="text-gray-200 leading-relaxed">
                We set out to build an AI system that could handle these repetitive tasks intelligently, 
                incorporating best practices and winning techniques from thousands of successful submissions. 
                The result is a platform that generates production-ready code tailored to each competition's 
                unique characteristics.
              </p>
              <p className="text-gray-200 leading-relaxed">
                Today, Kaggle Launchpad serves thousands of data scientists worldwide, from beginners 
                learning the ropes to Grandmasters looking to accelerate their workflow. We're proud 
                to be part of the competitive ML community and committed to pushing the boundaries of 
                what's possible with AI-assisted development.
              </p>
            </CardContent>
          </Card>

          {/* Team Section */}
          <div className="mb-16">
            <h2 className="text-3xl font-bold text-white text-center mb-8">
              Meet Our Team
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {team.map((member, index) => (
                <Card key={index} className="glass-card border-white/20">
                  <CardHeader className="text-center">
                    <div className="w-16 h-16 bg-gradient-to-r from-purple-600 to-blue-600 rounded-full mx-auto mb-4 flex items-center justify-center">
                      <Users className="h-8 w-8 text-white" />
                    </div>
                    <CardTitle className="text-white">{member.name}</CardTitle>
                    <CardDescription className="text-purple-300">{member.role}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-200 text-sm text-center">{member.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          {/* Values Section */}
          <Card className="glass-card border-white/20">
            <CardHeader>
              <CardTitle className="text-2xl text-white text-center">Our Values</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <Award className="h-8 w-8 text-yellow-400 mx-auto mb-3" />
                  <h3 className="text-white font-semibold mb-2">Excellence</h3>
                  <p className="text-gray-300 text-sm">
                    We strive for the highest quality in everything we build, 
                    from code generation to user experience.
                  </p>
                </div>
                <div className="text-center">
                  <Users className="h-8 w-8 text-green-400 mx-auto mb-3" />
                  <h3 className="text-white font-semibold mb-2">Community</h3>
                  <p className="text-gray-300 text-sm">
                    We're part of the Kaggle community and committed to 
                    giving back and supporting fellow data scientists.
                  </p>
                </div>
                <div className="text-center">
                  <Rocket className="h-8 w-8 text-blue-400 mx-auto mb-3" />
                  <h3 className="text-white font-semibold mb-2">Innovation</h3>
                  <p className="text-gray-300 text-sm">
                    We continuously push the boundaries of what's possible 
                    with AI-assisted development and automation.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}