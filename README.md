# DeepTutor 🎓

An intelligent tutoring system powered by large language models, designed to provide personalized learning experiences through interactive document-based Q&A.

> Fork of [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor)

## ✨ Features

- 📄 **Document Understanding** — Upload PDFs and interact with your study materials
- 🤖 **AI-Powered Tutoring** — Leverages state-of-the-art LLMs for intelligent responses
- 🔍 **RAG Pipeline** — Retrieval-Augmented Generation for accurate, grounded answers
- 💬 **Interactive Chat** — Conversational interface for natural learning flow
- 🌐 **Multi-language Support** — Supports both English and Chinese interfaces
- 🐳 **Docker Ready** — Easy deployment with Docker and Docker Compose

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/DeepTutor.git
   cd DeepTutor
   ```

2. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   python app.py
   ```

### Docker Deployment

```bash
docker compose up --build
```

## ⚙️ Configuration

Copy `.env.example` to `.env` and configure the following:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `MODEL_NAME` | LLM model to use (e.g. `gpt-4o`) | Yes |
| `EMBEDDING_MODEL` | Embedding model name | Yes |
| `VECTOR_STORE_PATH` | Path to persist vector store | No |
| `MAX_UPLOAD_SIZE_MB` | Maximum file upload size in MB | No |

For Chinese users, refer to `.env.example_CN` for region-specific settings.

> **Personal note:** I've been running this with `MODEL_NAME=gpt-4o-mini` to keep API costs low for casual studying — works well for most Q&A on lecture notes and textbook chapters. I also bumped `MAX_UPLOAD_SIZE_MB` to `50` since some of my textbook PDFs are on the larger side. If you're using this for research papers, `gpt-4o` gives noticeably better answers for technical content.

## 🏗️ Architecture

```
DeepTutor/
├── app.py              # Main application entry point
├── pipeline/           # RAG pipeline components
│   ├── ingestion.py    # Document ingestion & chunking
│   ├── retrieval.py    # Vector search & retrieval
│   └── generation.py   # LLM response generation
├── ui/                 # Frontend interface
├── utils/              # Utility functions
└── tests/              # Test suite
```

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see the [LICEN