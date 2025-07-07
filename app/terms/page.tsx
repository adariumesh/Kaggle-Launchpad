'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  FileText, 
  Shield, 
  AlertTriangle, 
  CheckCircle,
  Clock
} from 'lucide-react';

export default function TermsPage() {
  const lastUpdated = "January 1, 2024";

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-black relative">
      {/* Background shapes */}
      <div className="bg-shape bg-shape-1 -top-32 -left-32" />
      <div className="bg-shape bg-shape-2 top-1/4 -right-40" />
      <div className="bg-shape bg-shape-3 bottom-1/4 -left-20" />
      
      <div className="relative z-10">
        <div className="container mx-auto px-4 py-12 max-w-4xl">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Terms of Service
            </h1>
            <p className="text-xl text-gray-200 max-w-2xl mx-auto">
              Please read these terms carefully before using Kaggle Launchpad.
            </p>
            <div className="flex items-center justify-center mt-4 text-gray-300">
              <Clock className="h-4 w-4 mr-2" />
              <span className="text-sm">Last updated: {lastUpdated}</span>
            </div>
          </div>

          {/* Quick Summary */}
          <Card className="glass-card border-white/20 mb-8">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <CheckCircle className="h-5 w-5 mr-2 text-green-400" />
                Quick Summary
              </CardTitle>
              <CardDescription className="text-gray-200">
                The key points you should know about using our service
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <h4 className="text-white font-medium">✅ You Can:</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• Use generated code in competitions</li>
                    <li>• Modify and customize outputs</li>
                    <li>• Share notebooks publicly</li>
                    <li>• Use for educational purposes</li>
                  </ul>
                </div>
                <div className="space-y-2">
                  <h4 className="text-white font-medium">❌ You Cannot:</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• Resell our service</li>
                    <li>• Reverse engineer our AI</li>
                    <li>• Use for illegal activities</li>
                    <li>• Violate Kaggle's rules</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Main Terms */}
          <Card className="glass-card border-white/20">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <FileText className="h-5 w-5 mr-2 text-blue-400" />
                Terms and Conditions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[600px] pr-4">
                <div className="space-y-6 text-gray-200">
                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">1. Acceptance of Terms</h3>
                    <p className="leading-relaxed">
                      By accessing and using Kaggle Launchpad ("the Service"), you accept and agree to be bound by the terms and provision of this agreement. If you do not agree to abide by the above, please do not use this service.
                    </p>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">2. Description of Service</h3>
                    <p className="leading-relaxed mb-3">
                      Kaggle Launchpad is an AI-powered platform that generates machine learning notebooks and project scaffolds for Kaggle competitions. The service includes:
                    </p>
                    <ul className="list-disc list-inside space-y-1 ml-4">
                      <li>Automated exploratory data analysis (EDA)</li>
                      <li>Baseline model generation</li>
                      <li>Feature engineering suggestions</li>
                      <li>Code optimization and best practices</li>
                      <li>Competition-specific customizations</li>
                    </ul>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">3. User Responsibilities</h3>
                    <p className="leading-relaxed mb-3">You agree to:</p>
                    <ul className="list-disc list-inside space-y-1 ml-4">
                      <li>Provide accurate information when using the service</li>
                      <li>Use the service only for lawful purposes</li>
                      <li>Respect intellectual property rights</li>
                      <li>Comply with Kaggle's terms of service and competition rules</li>
                      <li>Not attempt to reverse engineer or copy our AI models</li>
                      <li>Not use the service to generate content that violates any laws</li>
                    </ul>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">4. Intellectual Property</h3>
                    <p className="leading-relaxed mb-3">
                      The generated code and notebooks are provided under an open-source license. You are free to:
                    </p>
                    <ul className="list-disc list-inside space-y-1 ml-4">
                      <li>Use, modify, and distribute the generated code</li>
                      <li>Submit generated notebooks to Kaggle competitions</li>
                      <li>Share your modifications with the community</li>
                    </ul>
                    <p className="leading-relaxed mt-3">
                      However, the underlying AI models, algorithms, and service infrastructure remain our proprietary technology.
                    </p>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">5. Privacy and Data</h3>
                    <p className="leading-relaxed">
                      We are committed to protecting your privacy. Competition URLs and email addresses are used solely for service delivery. We do not store or analyze competition data permanently. For detailed information, please refer to our Privacy Policy.
                    </p>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">6. Service Availability</h3>
                    <p className="leading-relaxed">
                      While we strive for 99.9% uptime, we cannot guarantee uninterrupted service. We reserve the right to modify, suspend, or discontinue the service with reasonable notice. We are not liable for any downtime or service interruptions.
                    </p>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">7. Limitation of Liability</h3>
                    <p className="leading-relaxed">
                      The service is provided "as is" without warranties of any kind. We are not responsible for the performance of generated code in competitions or any losses resulting from its use. Users are responsible for testing and validating all generated content.
                    </p>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">8. Prohibited Uses</h3>
                    <p className="leading-relaxed mb-3">You may not use the service to:</p>
                    <ul className="list-disc list-inside space-y-1 ml-4">
                      <li>Violate any applicable laws or regulations</li>
                      <li>Infringe on intellectual property rights</li>
                      <li>Generate malicious or harmful code</li>
                      <li>Attempt to overwhelm our servers</li>
                      <li>Resell or redistribute our service</li>
                      <li>Create competing AI services using our outputs</li>
                    </ul>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">9. Termination</h3>
                    <p className="leading-relaxed">
                      We reserve the right to terminate or suspend access to the service immediately, without prior notice, for conduct that we believe violates these terms or is harmful to other users or the service.
                    </p>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">10. Changes to Terms</h3>
                    <p className="leading-relaxed">
                      We reserve the right to modify these terms at any time. Changes will be effective immediately upon posting. Continued use of the service after changes constitutes acceptance of the new terms.
                    </p>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">11. Contact Information</h3>
                    <p className="leading-relaxed">
                      If you have any questions about these Terms of Service, please contact us at legal@kagglelaunchpad.com.
                    </p>
                  </section>
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Important Notice */}
          <Card className="glass-card border-yellow-500/30 mt-8">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <AlertTriangle className="h-5 w-5 mr-2 text-yellow-400" />
                Important Notice
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-200">
                These terms are legally binding. By using Kaggle Launchpad, you acknowledge that you have read, understood, and agree to be bound by these terms. If you do not agree with any part of these terms, you must not use our service.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}