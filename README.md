# 🎙️ Audio Analysis API for Render

AI-powered call recording analysis service using OpenAI Whisper and GPT-4.

**Status**: Deploying to Render

## 🚀 Quick Deploy to Render

### Option 1: One-Click Deploy (Recommended)

1. Fork this repository
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click **New +** → **Web Service**
4. Connect your GitHub repository
5. Render will auto-detect `render.yaml` and configure everything
6. Add your `OPENAI_API_KEY` in Environment Variables
7. Click **Deploy**

### Option 2: Manual Setup

1. Create a new Web Service on Render
2. Configure:
   - **Build Command**: `apt-get update && apt-get install -y ffmpeg && pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port 10000`
3. Add Environment Variables:
   - `OPENAI_API_KEY` = your_key_here
   - `WHISPER_MODEL` = whisper-1 (optional)
   - `ANALYSIS_MODEL` = gpt-4o-mini (optional)

## 📋 API Endpoints

### 1. Web UI (Testing)
```
GET /
```
Opens a simple web interface for uploading and testing audio files.

### 2. Analyze Uploaded File
```bash
POST /analyze
Content-Type: multipart/form-data

# Example with curl
curl -X POST https://your-render-url.onrender.com/analyze \
  -F "file=@sample.wav"
```

### 3. Analyze from URL
```bash
POST /analyze-url
Content-Type: application/json

# Example with curl
curl -X POST https://your-render-url.onrender.com/analyze-url \
  -H "Content-Type: application/json" \
  -d '{"recording_url": "https://example.com/recording.mp3"}'
```

### 4. Health Check
```bash
GET /health
```

## 📊 Response Format

```json
{
  "status": "success",
  "data": {
    "disposition": "interested",
    "interest_level": "high",
    "sentiment_score": 0.85,
    "sentiment": "positive",
    "summary": "Lead expressed strong interest in M.Tech CSE program...",
    "transcript": [
      {"role": "agent", "text": "Hello, this is..."},
      {"role": "user", "text": "Yes, I'm interested..."}
    ],
    "ai_recommendations": {
      "stage_update": {
        "recommended_stage": "Interested",
        "recommended_sub_stage": "Interested lead"
      },
      "follow_up": {
        "follow_up_mode": "call",
        "next_action_date": "2026-05-20",
        "next_action_time": "15:00"
      },
      "auto_fill": {
        "form_fields": {
          "qualification": "B.Tech",
          "city": "Mumbai",
          "course": "M.Tech CSE"
        }
      }
    }
  }
}
```

## ⚙️ Configuration

Create a `.env` file (for local testing):

```bash
OPENAI_API_KEY=sk-...
WHISPER_MODEL=whisper-1
ANALYSIS_MODEL=gpt-4o-mini
```

## 🧪 Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure ffmpeg is installed
# macOS: brew install ffmpeg
# Ubuntu: sudo apt-get install ffmpeg
# Windows: download from ffmpeg.org

# Run locally
uvicorn app:app --reload --port 8000

# Test
curl -X POST http://localhost:8000/analyze \
  -F "file=@test_audio.wav"
```

## ⚠️ Important Notes

### Render Free Tier Limitations
- **Cold starts**: 30-60 seconds after inactivity
- **Sleeps**: After 15 minutes of inactivity
- **Not for production**: Use paid tier for production workloads

### File Size Limits
- Recommended: < 25MB per file
- Whisper API limit: 25MB

### Processing Time
- Typical: 10-60 seconds depending on audio length
- Includes: download → enhance → transcribe → analyze

## 🔧 Troubleshooting

### "ffmpeg not found"
Make sure the build command includes ffmpeg installation:
```bash
apt-get update && apt-get install -y ffmpeg && pip install -r requirements.txt
```

### "OPENAI_API_KEY not configured"
Add the environment variable in Render dashboard under Environment tab.

### Cold Start Delays
First request after inactivity takes 30-60 seconds. Consider:
- Using a paid Render plan (no sleep)
- Setting up a cron job to ping the service every 10 minutes

### Timeout Errors
For very long recordings (>10 minutes), consider:
- Splitting the audio file
- Upgrading to a paid Render plan with longer timeouts

## 📚 Tech Stack

- **FastAPI**: Modern Python web framework
- **OpenAI Whisper**: Speech-to-text transcription
- **GPT-4**: Structured data extraction
- **FFmpeg**: Audio enhancement and normalization
- **Pydantic**: Data validation and serialization

## 🔐 Security

- Never commit `.env` files
- Use Render's environment variables for secrets
- API keys are not logged or exposed in responses

## 📞 Support

For issues or questions:
1. Check the `/health` endpoint
2. Review Render logs in the dashboard
3. Verify ffmpeg and OpenAI API key configuration

## 📄 License

MIT License - feel free to use and modify as needed.
