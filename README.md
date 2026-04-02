# Medical Image Database Search & Analysis Platform

AI-powered search across publicly available medical imaging datasets with intelligent insights.

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+

### Development Setup

1. **Clone and setup**
```bash
git clone <repo-url>
cd medical-image-search
cp .env.example .env
# Edit .env with your configuration
```

2. **Backend setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
uvicorn src.main:app --reload
```

3. **Frontend setup**
```bash
cd frontend
npm install
npm start
```

4. **Access the app**
- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs

## Project Structure

```
├── backend/          # FastAPI application
├── frontend/         # React TypeScript application
├── infra/            # Docker, Kubernetes configs
├── tests/            # Unit, integration, e2e tests
├── docs/             # Documentation
└── design.md         # Architecture and design decisions
```

## Features

- 🔍 **AI-powered search** by modality and semantic similarity
- 📊 **Dataset insights** with automatic statistics generation
- 🏥 **Multiple data sources** (NIH ChestX-ray, TCIA, OpenNeuro, etc.)
- 🚀 **Scalable architecture** with async processing and caching
- 📱 **Responsive UI** built with React and Material-UI

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file.