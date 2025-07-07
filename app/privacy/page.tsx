'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { 
  Shield, 
  Lock, 
  Eye, 
  Database, 
  Clock,
  CheckCircle,
  AlertCircle,
  Mail
} from 'lucide-react';

export default function PrivacyPage() {
  const lastUpdated = "January 1, 2024";

  const dataTypes = [
    {
      type: "Competition URLs",
      purpose: "Service delivery",
      retention: "30 days",
      sharing: "Never shared"
    },
    {
      type: "Email Addresses", 
      purpose: "Notifications",
      retention: "Until unsubscribe",
      sharing: "Never shared"
    },
    {
      type: "Usage Analytics",
      purpose: "Service improvement",
      retention: "12 months",
      sharing: "Anonymized only"
    },
    {
      type: "Generated Code",
      purpose: "Temporary processing",
      retention: "24 hours",
      sharing: "Never stored"
    }
  ];

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
              Privacy Policy
            </h1>
            <p className="text-xl text-gray-200 max-w-2xl mx-auto">
              Your privacy is important to us. Learn how we collect, use, and protect your data.
            </p>
            <div className="flex items-center justify-center mt-4 text-gray-300">
              <Clock className="h-4 w-4 mr-2" />
              <span className="text-sm">Last updated: {lastUpdated}</span>
            </div>
          </div>

          {/* Privacy Highlights */}
          <Card className="glass-card border-white/20 mb-8">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Shield className="h-5 w-5 mr-2 text-green-400" />
                Privacy Highlights
              </CardTitle>
              <CardDescription className="text-gray-200">
                Key points about how we handle your data
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-400" />
                    <span className="text-gray-200 text-sm">No competition data stored permanently</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-400" />
                    <span className="text-gray-200 text-sm">Email used only for notifications</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-400" />
                    <span className="text-gray-200 text-sm">No personal data sold or shared</span>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-400" />
                    <span className="text-gray-200 text-sm">GDPR and CCPA compliant</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-400" />
                    <span className="text-gray-200 text-sm">Data encrypted in transit and at rest</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-400" />
                    <span className="text-gray-200 text-sm">Right to delete your data anytime</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Data Collection Table */}
          <Card className="glass-card border-white/20 mb-8">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Database className="h-5 w-5 mr-2 text-blue-400" />
                Data We Collect
              </CardTitle>
              <CardDescription className="text-gray-200">
                Transparent overview of all data types we collect and how we use them
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {dataTypes.map((item, index) => (
                  <div key={index} className="p-4 rounded-lg bg-white/5 border border-white/10">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                      <div>
                        <h4 className="text-white font-medium mb-1">{item.type}</h4>
                        <Badge variant="outline" className="border-blue-400/50 text-blue-300">
                          {item.purpose}
                        </Badge>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm mb-1">Retention</p>
                        <p className="text-gray-200 text-sm">{item.retention}</p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm mb-1">Sharing</p>
                        <p className="text-gray-200 text-sm">{item.sharing}</p>
                      </div>
                      <div className="flex items-center">
                        <Lock className="h-4 w-4 text-green-400 mr-1" />
                        <span className="text-green-300 text-sm">Encrypted</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Detailed Privacy Policy */}
          <Card className="glass-card border-white/20">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Eye className="h-5 w-5 mr-2 text-purple-400" />
                Detailed Privacy Policy
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[600px] pr-4">
                <div className="space-y-6 text-gray-200">
                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">1. Information We Collect</h3>
                    <p className="leading-relaxed mb-3">
                      We collect minimal information necessary to provide our service:
                    </p>
                    <ul className="list-disc list-inside space-y-1 ml-4">
                      <li><strong>Competition URLs:</strong> To analyze and generate appropriate code</li>
                      <li><strong>Email addresses:</strong> To send you notifications when your notebook is ready</li>
                      <li><strong>Usage analytics:</strong> Anonymized data to improve our service</li>
                      <li><strong>Technical data:</strong> IP addresses, browser type, and device information for security</li>
                    </ul>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">2. How We Use Your Information</h3>
                    <p className="leading-relaxed mb-3">Your information is used exclusively for:</p>
                    <ul className="list-disc list-inside space-y-1 ml-4">
                      <li>Generating personalized Kaggle notebooks and code</li>
                      <li>Sending email notifications about your projects</li>
                      <li>Improving our AI models and service quality</li>
                      <li>Ensuring security and preventing abuse</li>
                      <li>Complying with legal obligations</li>
                    </ul>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">3. Data Storage and Security</h3>
                    <p className="leading-relaxed mb-3">
                      We implement industry-standard security measures:
                    </p>
                    <ul className="list-disc list-inside space-y-1 ml-4">
                      <li>All data is encrypted in transit using TLS 1.3</li>
                      <li>Data at rest is encrypted using AES-256</li>
                      <li>Access to data is restricted to authorized personnel only</li>
                      <li>Regular security audits and penetration testing</li>
                      <li>Competition data is processed temporarily and not stored permanently</li>
                    </ul>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">4. Data Sharing and Disclosure</h3>
                    <p className="leading-relaxed mb-3">
                      We do not sell, trade, or rent your personal information. We may share data only in these limited circumstances:
                    </p>
                    <ul className="list-disc list-inside space-y-1 ml-4">
                      <li>With your explicit consent</li>
                      <li>To comply with legal obligations or court orders</li>
                      <li>To protect our rights, property, or safety</li>
                      <li>In connection with a business transfer (with notice)</li>
                    </ul>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">5. Your Rights and Choices</h3>
                    <p className="leading-relaxed mb-3">You have the right to:</p>
                    <ul className="list-disc list-inside space-y-1 ml-4">
                      <li><strong>Access:</strong> Request a copy of your personal data</li>
                      <li><strong>Rectification:</strong> Correct inaccurate or incomplete data</li>
                      <li><strong>Erasure:</strong> Request deletion of your personal data</li>
                      <li><strong>Portability:</strong> Receive your data in a machine-readable format</li>
                      <li><strong>Objection:</strong> Object to processing of your personal data</li>
                      <li><strong>Restriction:</strong> Request limitation of processing</li>
                    </ul>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">6. Cookies and Tracking</h3>
                    <p className="leading-relaxed">
                      We use minimal cookies for essential functionality only. We do not use tracking cookies or third-party analytics that compromise your privacy. You can disable cookies in your browser settings, though this may affect service functionality.
                    </p>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">7. Data Retention</h3>
                    <p className="leading-relaxed">
                      We retain your data only as long as necessary for the purposes outlined in this policy. Competition URLs are deleted after 30 days, generated code is not stored permanently, and email addresses are retained until you unsubscribe.
                    </p>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">8. International Data Transfers</h3>
                    <p className="leading-relaxed">
                      Our servers are located in secure data centers with appropriate safeguards. If we transfer data internationally, we ensure adequate protection through standard contractual clauses or adequacy decisions.
                    </p>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">9. Children's Privacy</h3>
                    <p className="leading-relaxed">
                      Our service is not intended for children under 13. We do not knowingly collect personal information from children under 13. If we become aware of such collection, we will delete the information immediately.
                    </p>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">10. Changes to This Policy</h3>
                    <p className="leading-relaxed">
                      We may update this privacy policy to reflect changes in our practices or legal requirements. We will notify you of significant changes via email or prominent notice on our website.
                    </p>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold text-white mb-3">11. Contact Us</h3>
                    <p className="leading-relaxed">
                      If you have questions about this privacy policy or want to exercise your rights, contact us at privacy@kagglelaunchpad.com or use our contact form.
                    </p>
                  </section>
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Contact for Privacy */}
          <Card className="glass-card border-blue-500/30 mt-8">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Mail className="h-5 w-5 mr-2 text-blue-400" />
                Privacy Questions?
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-200 mb-4">
                We're committed to transparency about our privacy practices. If you have any questions or concerns about how we handle your data, please don't hesitate to reach out.
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                <div className="flex items-center space-x-2">
                  <Mail className="h-4 w-4 text-blue-400" />
                  <span className="text-gray-200">privacy@kagglelaunchpad.com</span>
                </div>
                <div className="flex items-center space-x-2">
                  <AlertCircle className="h-4 w-4 text-yellow-400" />
                  <span className="text-gray-200">Response within 48 hours</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}