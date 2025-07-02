import './globals.css';
import type { Metadata } from 'next/types';
import { Inter } from 'next/font/google';
import { Toaster } from '@/components/ui/sonner';

const inter = Inter({ 
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter'
});

export const metadata: Metadata = {
  title: 'Kaggle Launchpad - AI-Powered Competition Project Generator',
  description: 'Generate ready-to-run project scaffolds for Kaggle competitions with AI-powered templates, EDA notebooks, and baseline models.',
  keywords: 'kaggle, machine learning, data science, project generator, AI, competition',
  authors: [{ name: 'Kaggle Launchpad' }],
  viewport: 'width=device-width, initial-scale=1',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className={`${inter.className} antialiased`}>
        {children}
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: 'hsl(var(--card))',
              color: 'hsl(var(--card-foreground))',
              border: '1px solid hsl(var(--border))',
            },
          }}
        />
      </body>
    </html>
  );
}