# CodeAnt Dashboard - Interactive Web Interface

## 🎯 Overview

Modern, interactive web dashboard for CodeAnt Agent built with SvelteKit, providing advanced visualizations of code analysis, real-time updates, and role-based interfaces.

## 🚀 Features

### High Priority Features (MVP)

#### 📊 Technical Debt Dashboard
- **Debt Evolution Timeline**: Historical visualization of technical debt
- **Component Distribution**: Debt breakdown by modules
- **ROI Calculator**: Calculate return on investment for debt remediation
- **Payment Roadmap**: Optimized plan for addressing technical debt
- **Impact Analysis**: Understand the business impact of technical debt

#### 🔧 Automatic Fix Panel
- **AI-Powered Fixes**: One-click application of code improvements
- **Fix Preview**: See changes before applying
- **Batch Management**: Apply multiple fixes at once
- **Fix History**: Track all applied fixes with rollback capability
- **Safety Validation**: Pre-application security checks

#### 🏢 Multi-Project View
- **Portfolio Health**: Organization-wide code quality metrics
- **Cross-Project Analytics**: Compare metrics across projects
- **Trend Analysis**: Organization-level quality trends
- **Project Matrix**: Side-by-side project comparison
- **Aggregated KPIs**: Combined metrics and insights

#### 🚀 CI/CD Integration Dashboard
- **DORA Metrics**: DevOps performance indicators
  - Deployment Frequency
  - Lead Time for Changes
  - Change Failure Rate
  - Time to Restore Service
- **Pipeline Monitor**: Real-time pipeline status
- **Deployment History**: Track all deployments
- **Failure Analysis**: Understand and prevent failures

### Medium Priority Features

#### 🧠 Semantic Analysis View
- **Natural Language Search**: Find code by intent
- **Code Similarity Map**: Visualize related code clusters
- **Embeddings Explorer**: Interactive semantic relationships
- **Cross-Language Search**: Find similar patterns across languages

#### 📝 Custom Rules Dashboard
- **Natural Language Input**: Create rules in Spanish or English
- **Visual Rule Builder**: Intuitive rule creation interface
- **Performance Metrics**: Track rule effectiveness
- **Template Gallery**: Reusable rule templates
- **Testing Sandbox**: Test rules before deployment

## 🛠️ Technology Stack

- **Framework**: SvelteKit 2.0+ with TypeScript
- **State Management**: Svelte Stores (native)
- **Styling**: Tailwind CSS + Svelte native CSS
- **Charts**: LayerCake (Svelte-native) + D3.js
- **Real-time**: Server-Sent Events (SSE) + WebSocket
- **Build**: Vite (included in SvelteKit)
- **Testing**: Vitest + Playwright

## 📁 Project Structure

```
codeant-dashboard/
├── src/
│   ├── lib/
│   │   ├── components/      # Reusable UI components
│   │   ├── stores/          # Svelte stores for state
│   │   ├── types/           # TypeScript definitions
│   │   └── utils/           # Helper functions
│   ├── routes/              # SvelteKit routes
│   │   ├── +page.svelte     # Main dashboard
│   │   ├── +page.ts         # Page data loading
│   │   └── api/             # API endpoints
│   └── app.html             # HTML template
├── tests/                   # E2E tests with Playwright
├── static/                  # Static assets
└── package.json
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- npm or pnpm

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 🧪 Testing

### Unit Tests
```bash
# Run unit tests
npm run test:unit

# Run with coverage
npm run test:coverage
```

### E2E Tests
```bash
# Run E2E tests
npm run test:e2e

# Run in UI mode
npm run test:e2e:ui
```

### Test Coverage Requirements
- Unit Tests: 95% minimum coverage
- Integration Tests: All API integrations tested
- E2E Tests: All user flows covered
- Performance Tests: <2s load time
- Accessibility: WCAG 2.1 AA compliance

## 📊 Performance Targets

- **Initial Load**: <2 seconds
- **Bundle Size**: <150KB gzipped
- **Chart Rendering**: <500ms for 1000+ data points
- **Real-time Updates**: <50ms latency
- **Mobile Performance**: <3 seconds on 3G

## 🎨 Design Guidelines

### Responsive Design
- Mobile-first approach
- Breakpoints: 480px, 768px, 1024px, 1280px
- Touch-friendly interfaces
- Optimized for all device sizes

### Accessibility
- WCAG 2.1 AA compliant
- Keyboard navigation support
- Screen reader optimized
- High contrast mode
- Reduced motion support

### Internationalization
- Spanish and English support
- RTL language ready
- Localized number/date formats
- Cultural adaptations

## 🔒 Security

- Input validation on all forms
- XSS protection
- CSRF tokens
- Secure API communication
- Role-based access control

## 📚 API Integration

The dashboard integrates with the CodeAnt backend API:

```typescript
// Example API calls
/api/technical-debt          // Get technical debt metrics
/api/fixes/{projectId}       // Get available fixes
/api/projects                // Get all projects
/api/dora-metrics           // Get DORA metrics
/api/embeddings/search      // Semantic search
/api/rules/generate         // Generate custom rules
```

## 🚢 Deployment

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["node", "build"]
```

### Environment Variables
```env
PUBLIC_API_URL=https://api.codeant.com
PUBLIC_WS_URL=wss://ws.codeant.com
NODE_ENV=production
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Style
- Follow ESLint configuration
- Use Prettier for formatting
- Write meaningful commit messages
- Add tests for new features

## 📄 License

This project is part of the CodeAnt Agent system.

## 🆘 Support

- Documentation: [docs.codeant.com](https://docs.codeant.com)
- Issues: GitHub Issues
- Discord: [discord.gg/codeant](https://discord.gg/codeant)

## 🎯 Roadmap

- [ ] Additional visualizations
- [ ] Mobile app
- [ ] Plugin system
- [ ] Advanced analytics
- [ ] AI-powered insights

---

Built with ❤️ by the CodeAnt Team