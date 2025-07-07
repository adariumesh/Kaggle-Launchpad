# Kaggle Launchpad - Frontend Client

A modern React/Next.js frontend for the Kaggle Launchpad AI-powered competition project generator.

## Architecture

This repository contains the frontend client application that communicates with a separate backend service for AI agent functionality and project generation.

### Frontend Responsibilities
- User interface for project creation and management
- Real-time dashboard for monitoring AI agent status
- Project workflow visualization
- File and notebook downloads
- Local caching of project data

### Backend Responsibilities (Separate Repository)
- AI agent intelligence and learning
- Competition analysis and code generation
- Project workflow orchestration
- File generation and storage
- Knowledge base management

## Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Backend service running (see backend repository)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd kaggle-launchpad-frontend
```

2. Install dependencies:
```bash
npm install
```

3. Configure environment variables:
```bash
cp .env.example .env.local
```

Edit `.env.local` with your backend service URL and API key:
```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_API_KEY=your_api_key_here
```

4. Start the development server:
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
├── app/                    # Next.js app directory
│   ├── ai-agent/          # AI agent dashboard page
│   ├── globals.css        # Global styles
│   └── layout.tsx         # Root layout
├── components/            # React components
│   ├── ui/               # shadcn/ui components
│   └── ai-agent-dashboard.tsx
├── lib/                  # Utility libraries
│   ├── api-client.ts     # Backend API client
│   ├── client-storage.ts # Local storage utilities
│   ├── supabase-client.ts # Supabase client (optional)
│   └── utils.ts          # General utilities
└── supabase/             # Database migrations (optional)
```

## API Integration

The frontend communicates with the backend through a REST API. Key endpoints:

- `POST /api/projects` - Create new project
- `GET /api/projects` - List all projects
- `GET /api/projects/{id}` - Get project details
- `POST /api/projects/{id}/cancel` - Cancel project
- `GET /api/projects/{id}/download` - Download project files
- `GET /api/agent/status` - Get AI agent status

See `lib/api-client.ts` for complete API documentation.

## Features

### AI Agent Dashboard
- Real-time monitoring of active workflows
- Agent status and learning progress
- Knowledge domain coverage
- Recent activity feed
- Configuration management

### Project Management
- Create new Kaggle competition projects
- Monitor generation progress
- Download generated files and notebooks
- Cancel running projects
- View project history

### Offline Support
- Local caching of project data
- Graceful fallback when backend is unavailable
- Persistent storage across browser sessions

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

### Code Style

This project uses:
- TypeScript for type safety
- Tailwind CSS for styling
- shadcn/ui for component library
- ESLint for code linting

## Deployment

### Build for Production

```bash
npm run build
```

### Environment Variables

Required environment variables for production:

```env
NEXT_PUBLIC_BACKEND_URL=https://your-backend-api.com
NEXT_PUBLIC_API_KEY=your_production_api_key
```

### Static Export

The project is configured for static export:

```bash
npm run build
```

This generates a `out/` directory that can be deployed to any static hosting service.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.