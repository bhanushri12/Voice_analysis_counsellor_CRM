"""FastAPI application for audio analysis deployment on Render."""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import shutil
import os
import tempfile
import logging
from pydantic import BaseModel
from typing import Optional

# Import your analysis functions
from vox_client import analyze_recording, transcribe_recording, analyze_transcript

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio Analysis API",
    description="AI-powered call recording analysis service",
    version="1.0.0"
)


class AnalyzeURLRequest(BaseModel):
    """Request model for URL-based analysis."""
    recording_url: str
    lead_id: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def upload_page():
    """Simple HTML form for testing file uploads."""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Audio Analysis Tester</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                }
                h2 { color: #333; }
                .form-group {
                    margin: 20px 0;
                }
                input[type="file"], input[type="text"] {
                    padding: 10px;
                    width: 100%;
                    margin: 10px 0;
                }
                button {
                    background: #007bff;
                    color: white;
                    padding: 12px 30px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 16px;
                }
                button:hover {
                    background: #0056b3;
                }
                .result {
                    margin-top: 20px;
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 4px;
                    white-space: pre-wrap;
                }
            </style>
        </head>
        <body>
            <h2>🎙️ Upload Call Recording for Analysis</h2>
            
            <div class="form-group">
                <h3>Option 1: Upload Audio File</h3>
                <form action="/analyze" method="post" enctype="multipart/form-data">
                    <input name="file" type="file" accept="audio/*" required/>
                    <button type="submit">Analyze File</button>
                </form>
            </div>
            
            <div class="form-group">
                <h3>Option 2: Analyze from URL</h3>
                <form action="/analyze-url" method="post" id="urlForm">
                    <input type="text" id="recording_url" placeholder="https://example.com/recording.mp3" required/>
                    <button type="button" onclick="analyzeURL()">Analyze URL</button>
                </form>
            </div>
            
            <div id="result" class="result" style="display:none;">
                <h3>Result:</h3>
                <pre id="resultContent"></pre>
            </div>
            
            <script>
                async function analyzeURL() {
                    const url = document.getElementById('recording_url').value;
                    const resultDiv = document.getElementById('result');
                    const resultContent = document.getElementById('resultContent');
                    
                    resultDiv.style.display = 'block';
                    resultContent.textContent = 'Processing...';
                    
                    try {
                        const response = await fetch('/analyze-url', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({recording_url: url})
                        });
                        const data = await response.json();
                        resultContent.textContent = JSON.stringify(data, null, 2);
                    } catch (error) {
                        resultContent.textContent = 'Error: ' + error.message;
                    }
                }
            </script>
        </body>
    </html>
    """


@app.post("/analyze")
async def analyze_audio_file(file: UploadFile = File(...)):
    """
    Analyze an uploaded audio file.
    
    Args:
        file: Audio file (WAV, MP3, etc.)
    
    Returns:
        JSON with analysis results including transcript, sentiment, disposition, etc.
    """
    logger.info(f"Received file: {file.filename} ({file.content_type})")
    
    # Check if OpenAI API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    
    # Check if ffmpeg is available
    if not shutil.which("ffmpeg"):
        raise HTTPException(status_code=503, detail="ffmpeg not installed on server")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_path = temp_file.name
        try:
            # Save uploaded file
            shutil.copyfileobj(file.file, temp_file)
            logger.info(f"Saved to temporary file: {temp_path}")
            
            # Transcribe the audio
            logger.info("Starting transcription...")
            transcript = transcribe_recording(f"file://{temp_path}")
            
            if transcript is None:
                raise HTTPException(status_code=502, detail="Transcription failed")
            
            logger.info(f"Transcription complete: {len(transcript)} characters")
            
            # Analyze the transcript
            logger.info("Starting analysis...")
            result = analyze_transcript(transcript)
            
            if result is None:
                raise HTTPException(status_code=502, detail="Analysis failed")
            
            logger.info("Analysis complete")
            
            # Format response in a clean, readable structure
            response = {
                "status": "success",
                "call_analysis": {
                    "disposition": result.get("disposition"),
                    "interest_level": result.get("interest_level"),
                    "sentiment": result.get("sentiment"),
                    "sentiment_score": result.get("sentiment_score"),
                    "summary": result.get("summary")
                },
                "conversation": {
                    "transcript": result.get("transcript", [])
                },
                "recommendations": {
                    "stage": {
                        "current": result.get("ai_recommendations", {}).get("stage_update", {}).get("current_stage", ""),
                        "recommended": result.get("ai_recommendations", {}).get("stage_update", {}).get("recommended_stage", ""),
                        "sub_stage": result.get("ai_recommendations", {}).get("stage_update", {}).get("recommended_sub_stage", "")
                    },
                    "follow_up": result.get("ai_recommendations", {}).get("follow_up", {}),
                },
                "extracted_information": {
                    "contact": {
                        k: v for k, v in result.get("ai_recommendations", {}).get("auto_fill", {}).get("form_fields", {}).items()
                        if k in ["phone", "mobile_number", "alternate_mobile_number", "email", "alternate_email"] and v
                    },
                    "personal": {
                        k: v for k, v in result.get("ai_recommendations", {}).get("auto_fill", {}).get("form_fields", {}).items()
                        if k in ["city", "first_line_add", "best_time_to_call"] and v
                    },
                    "education": {
                        k: v for k, v in result.get("ai_recommendations", {}).get("auto_fill", {}).get("form_fields", {}).items()
                        if k in ["qualification", "degree", "field_of_study", "institute_school", "course"] and v
                    },
                    "professional": {
                        k: v for k, v in result.get("ai_recommendations", {}).get("auto_fill", {}).get("form_fields", {}).items()
                        if k in ["experience", "company_name", "salary_ctc", "salary_increment"] and v
                    },
                    "program_interest": {
                        k: v for k, v in result.get("ai_recommendations", {}).get("auto_fill", {}).get("form_fields", {}).items()
                        if k in ["budget_range", "pain_points"] and v
                    },
                    "remarks": result.get("ai_recommendations", {}).get("auto_fill", {}).get("remarks", {})
                }
            }
            
            # Remove empty sections
            response["extracted_information"] = {
                k: v for k, v in response["extracted_information"].items() if v
            }
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {e}")


@app.post("/analyze-url")
async def analyze_audio_url(request: AnalyzeURLRequest):
    """
    Analyze an audio file from a URL.
    
    Args:
        request: JSON body with recording_url and optional lead_id
    
    Returns:
        JSON with analysis results including transcript, sentiment, disposition, etc.
    """
    logger.info(f"Analyzing recording from URL: {request.recording_url}")
    
    # Check if OpenAI API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    
    # Check if ffmpeg is available
    if not shutil.which("ffmpeg"):
        raise HTTPException(status_code=503, detail="ffmpeg not installed on server")
    
    try:
        # Use the analyze_recording function which handles download + transcribe + analyze
        result = analyze_recording(request.recording_url)
        
        if result is None:
            raise HTTPException(status_code=502, detail="Analysis failed")
        
        logger.info("Analysis complete")
        
        # Format response in a clean, readable structure
        response = {
            "status": "success",
            "lead_id": request.lead_id,
            "call_analysis": {
                "disposition": result.get("disposition"),
                "interest_level": result.get("interest_level"),
                "sentiment": result.get("sentiment"),
                "sentiment_score": result.get("sentiment_score"),
                "summary": result.get("summary")
            },
            "conversation": {
                "transcript": result.get("transcript", [])
            },
            "recommendations": {
                "stage": {
                    "current": result.get("ai_recommendations", {}).get("stage_update", {}).get("current_stage", ""),
                    "recommended": result.get("ai_recommendations", {}).get("stage_update", {}).get("recommended_stage", ""),
                    "sub_stage": result.get("ai_recommendations", {}).get("stage_update", {}).get("recommended_sub_stage", "")
                },
                "follow_up": result.get("ai_recommendations", {}).get("follow_up", {}),
            },
            "extracted_information": {
                "contact": {
                    k: v for k, v in result.get("ai_recommendations", {}).get("auto_fill", {}).get("form_fields", {}).items()
                    if k in ["phone", "mobile_number", "alternate_mobile_number", "email", "alternate_email"] and v
                },
                "personal": {
                    k: v for k, v in result.get("ai_recommendations", {}).get("auto_fill", {}).get("form_fields", {}).items()
                    if k in ["city", "first_line_add", "best_time_to_call"] and v
                },
                "education": {
                    k: v for k, v in result.get("ai_recommendations", {}).get("auto_fill", {}).get("form_fields", {}).items()
                    if k in ["qualification", "degree", "field_of_study", "institute_school", "course"] and v
                },
                "professional": {
                    k: v for k, v in result.get("ai_recommendations", {}).get("auto_fill", {}).get("form_fields", {}).items()
                    if k in ["experience", "company_name", "salary_ctc", "salary_increment"] and v
                },
                "program_interest": {
                    k: v for k, v in result.get("ai_recommendations", {}).get("auto_fill", {}).get("form_fields", {}).items()
                    if k in ["budget_range", "pain_points"] and v
                },
                "remarks": result.get("ai_recommendations", {}).get("auto_fill", {}).get("remarks", {})
            }
        }
        
        # Remove empty sections
        response["extracted_information"] = {
            k: v for k, v in response["extracted_information"].items() if v
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing URL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    checks = {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "ffmpeg_available": bool(shutil.which("ffmpeg")),
    }
    
    if not checks["openai_configured"] or not checks["ffmpeg_available"]:
        checks["status"] = "degraded"
    
    return checks


@app.get("/info")
def api_info():
    """API information and usage instructions."""
    return {
        "name": "Audio Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/": "Web UI for testing",
            "/analyze": "POST - Upload audio file for analysis",
            "/analyze-url": "POST - Analyze audio from URL",
            "/health": "GET - Health check",
            "/info": "GET - API information",
        },
        "usage": {
            "file_upload": "POST /analyze with multipart/form-data, field name: 'file'",
            "url_analysis": "POST /analyze-url with JSON body: {\"recording_url\": \"https://...\"}",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
