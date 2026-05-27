"""AI call analysis — transcribes a recording and extracts structured CRM data.

analyze_recording(url) is the single public function:
1. Downloads the audio file
2. Enhances it via FFmpeg (normalize, compress, mono 16 kHz WAV)
3. Transcribes via OpenAI Whisper
4. Extracts structured CRM intelligence via GPT-4o Structured Outputs

Returns the full analysis dict on success, None on any failure.
"""
import os
import logging
import subprocess
import tempfile
import requests
from datetime import datetime, timezone, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
from openai import OpenAI

_IST = timezone(timedelta(hours=5, minutes=30))
logger = logging.getLogger("VoxClient")

# ── Enums (constrain GPT output — no hallucination) ─────────────────

class DispositionEnum(str, Enum):
    interested    = "interested"
    not_interested = "not interested"
    callback      = "callback"
    invalid       = "invalid"
    not_answered  = "not answered"

class InterestLevelEnum(str, Enum):
    high   = "high"
    medium = "medium"
    low    = "low"

class StageNameEnum(str, Enum):
    Re_allocate       = "Re-allocate"
    RnR               = "RnR"
    Follow_up         = "Follow-up"
    Interested        = "Interested"
    Application_Status = "Application Status"
    Enrolled          = "Enrolled"
    Drop              = "Drop"
    Invalid_Leads     = "Invalid Leads"
    Call_back         = "Call back"
    Reference         = "Reference"
    Language_barrier  = "Language barrier"
    New               = "New"

# ── Pydantic models for GPT Structured Outputs ───────────────────────

class TranscriptTurn(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: str = Field(description="Speaker role: 'agent' or 'user'")
    text: str = Field(description="What the speaker said")

class FormFields(BaseModel):
    model_config = ConfigDict(extra='forbid')
    phone:                  str = Field(default="", description="Extract ONLY if explicitly mentioned")
    mobile_number:          str = Field(default="", description="Extract ONLY if explicitly mentioned")
    alternate_mobile_number: str = Field(default="", description="Extract ONLY if explicitly mentioned")
    email:                  str = Field(default="", description="Extract ONLY if explicitly mentioned")
    alternate_email:        str = Field(default="", description="Extract ONLY if explicitly mentioned")
    city:                   str = Field(default="", description="Extract ONLY if explicitly mentioned")
    first_line_add:         str = Field(default="", description="First line of address, extract ONLY if explicitly mentioned")
    qualification:          str = Field(default="", description="Extract ONLY if explicitly mentioned")
    degree:                 str = Field(default="", description="Extract ONLY if explicitly mentioned")
    field_of_study:         str = Field(default="", description="Extract ONLY if explicitly mentioned")
    institute_school:       str = Field(default="", description="Institute or school name, extract ONLY if explicitly mentioned")
    experience:             str = Field(default="", description="Extract ONLY if explicitly mentioned (in years)")
    company_name:           str = Field(default="", description="Extract ONLY if explicitly mentioned")
    salary_ctc:             str = Field(default="", description="Current salary or CTC, extract ONLY if explicitly mentioned")
    salary_increment:       str = Field(default="", description="Expected salary increment, extract ONLY if explicitly mentioned")
    budget_range:           str = Field(default="", description="Extract budget/fees discussed. If not mentioned return empty string.")
    course:                 str = Field(default="", description="Extract ONLY if explicitly mentioned")
    best_time_to_call:      str = Field(default="", description="Preferred time to be called, extract ONLY if explicitly mentioned")
    pain_points:            str = Field(default="", description="Pain points or concerns expressed by the user")

class Remarks(BaseModel):
    model_config = ConfigDict(extra='forbid')
    funding_option:    str = Field(default="", description="Funding option discussed")
    decision_timeline: str = Field(default="", description="Timeline for decision")
    preferred_intake:  str = Field(default="", description="Preferred intake/batch")
    designation:       str = Field(default="", description="Designation mentioned")

class StageUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    current_stage:         str = Field(default="", description="Current stage of the lead")
    recommended_stage:     StageNameEnum = Field(description="AI-recommended CRM stage")
    recommended_sub_stage: str = Field(default="", description="AI-recommended sub-stage")

class FollowUp(BaseModel):
    model_config = ConfigDict(extra='forbid')
    follow_up_mode:    str = Field(default="", description="Mode of follow-up (call/email/whatsapp)")
    next_action_date:  str = Field(default="", description="Next action date (YYYY-MM-DD, IST timezone)")
    next_action_time:  str = Field(default="", description="Next action time (HH:MM, IST timezone, 24-hour)")
    display_time:      str = Field(default="", description="Human-readable display time in IST (e.g. '3:30 PM IST')")
    follow_up_comment: str = Field(default="", description="Comment for follow-up action")

class AutoFill(BaseModel):
    model_config = ConfigDict(extra='forbid')
    form_fields: FormFields = Field(default_factory=FormFields)
    remarks:     Remarks    = Field(default_factory=Remarks)

class AIRecommendations(BaseModel):
    model_config = ConfigDict(extra='forbid')
    stage_update: StageUpdate
    follow_up:    FollowUp  = Field(default_factory=FollowUp)
    auto_fill:    AutoFill  = Field(default_factory=AutoFill)

class AnalysisData(BaseModel):
    model_config = ConfigDict(extra='forbid')
    disposition:        DispositionEnum
    interest_level:     InterestLevelEnum
    sentiment_score:    float = Field(description="Overall sentiment of the user in the call, from 0.0 (very negative) to 1.0 (very positive).")
    summary:            str = Field(description="Max 2-3 sentences. Capture user intent, interest, and outcome.")
    transcript:         list[TranscriptTurn] = Field(description="Full conversation as [{role, text}] turns. role must be 'agent' or 'user'.")
    ai_recommendations: AIRecommendations

# ── System prompt (overridable via ANALYSIS_SYSTEM_PROMPT in .env) ───

_DEFAULT_PROMPT = """You are an AI assistant extracting structured CRM data from a sales call.

STRICT INSTRUCTIONS:
- Extract ONLY from USER responses
- DO NOT infer or guess
- DO NOT hallucinate missing values
- If not explicitly mentioned → return ""
- Use ONLY allowed values where specified
- All dates and times are in IST (India Standard Time, UTC+5:30)
- Today's date in IST is provided at the top of the transcript — use it as the reference when computing any relative follow-up dates (e.g. "next Monday", "call in 3 days")

FIELDS TO CLASSIFY:
disposition: "interested" | "not interested" | "callback" | "invalid" | "not answered"
interest_level: "high" | "medium" | "low"
sentiment_score: float 0.0–1.0 (0.0 = very negative, 0.5 = neutral, 1.0 = very positive) — score the user's overall tone and sentiment during the call

STAGE CLASSIFICATION RULES (STRICT):
If call did not connect:
→ stage = "RnR", sub_stage = one of: "Ringing no response", "Not reachable", "Number busy", "Switched off", "Called but disconnected", "Not able to connect"

If user asks to be contacted later:
→ stage = "Follow-up", sub_stage = one of: "Call back after few days", "Will discuss and get back", "Need time to decide", "Parents dependency", "Next session", "Arranging funds"

If user shows interest:
→ stage = "Interested", sub_stage = one of: "Interested lead", "Asked to send application link", "Call back later, interested in course"

If user is applying:
→ stage = "Application Status", sub_stage = one of: "Will apply", "Application form filled", "Fee pending"

If user has paid:
→ stage = "Enrolled", sub_stage = "Fee paid"

If user rejects:
→ stage = "Drop", sub_stage = one of: "Not interested", "Fees high", "Distance issue", "Not eligible", "Parents denied"

If invalid:
→ stage = "Invalid Leads", sub_stage = "Invalid number"

If language issue:
→ stage = "Language barrier"

FORM FIELDS — Extract ONLY if explicitly mentioned:
phone, mobile_number, alternate_mobile_number, email, alternate_email,
city, first_line_add, qualification, degree, field_of_study, institute_school,
experience, company_name, salary_ctc, salary_increment, budget_range, course,
best_time_to_call, pain_points

REMARKS — Put all extra info here:
funding_option, decision_timeline, preferred_intake, designation

summary: Max 2-3 sentences capturing intent, interest, and outcome.

TRANSCRIPT SPEAKER IDENTIFICATION:
The raw input is a flat transcription of a TWO-party call merged into one stream with no speaker labels.
Your job: split it into turns and assign each turn the correct role — "agent" or "user".

HARD RULES:
- Determine who speaks first from context:
  • Outgoing call: the AGENT speaks first — greets and introduces themselves/university.
  • Incoming call: the USER speaks first with a bare greeting ("Hello?", "Haan?", "Ji?") and the AGENT's FIRST substantive turn introduces themselves/university.
- Roles strictly alternate unless one party delivers a long monologue — in that case split only at a clear topic/sentence boundary.
- Every question must be followed by an answer from the OTHER party; do not assign two consecutive questions to the same role.
- Coherence check: if a turn starts with a direct answer to the previous turn's question, it must be the opposite role.

AGENT signals — assign "agent" when the turn:
- Opens with a greeting and self-introduction: "Hello, am I speaking with...?", "Good morning/afternoon, this is [name] from [university]"
- Mentions the institution or program by name: "I'm calling from Manipal / Amity / LPU / DY Patil..."
- Asks structured qualification questions: "What is your current qualification?", "How many years of experience do you have?"
- Explains course details, fees, batch dates, EMI options, or the admission process
- Confirms or summarises: "So as I understand, you're interested in MBA...", "Let me share the application link"
- Uses formal sales phrases: "The program offers...", "We have a scholarship of...", "The next batch starts..."
- Proposes follow-up action: "I'll call you back on...", "Please check your WhatsApp for the link"

USER signals — assign "user" when the turn:
- Responds with short confirmations or acknowledgements: "Yes", "Okay", "Haan", "Theek hai", "I see", "Hmm"
- Gives personal details: name, city, current job, qualification, years of experience, company name
- Asks about cost, eligibility, or logistics: "What are the fees?", "Is it online?", "Can I pay in installments?"
- Expresses hesitation or conditions: "Let me check with my family", "The fees are a bit high", "I'm busy right now"
- States their availability or preference: "Call me after 6 PM", "I'm free on weekends"
- Uses informal or colloquial language; may mix Hindi and English

AMBIGUITY RESOLUTION (in order of priority):
1. Apply coherence: a direct answer to the previous question → opposite role.
2. Apply agent/user signals above.
3. If still ambiguous, assign to whichever role has spoken less recently (maintains alternation).

transcript: Array of {role, text} turns. role must be exactly "agent" or "user".
"""

_env_prompt = os.getenv("ANALYSIS_SYSTEM_PROMPT", "")
SYSTEM_PROMPT = _env_prompt.replace('\\n', '\n') if _env_prompt else _DEFAULT_PROMPT

# ── Public functions ─────────────────────────────────────────────────

def transcribe_recording(recording_url: str) -> str | None:
    """Download, enhance via FFmpeg, and transcribe a recording with Whisper.
    
    Returns the raw transcript string, or None on any failure.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping transcription")
        return None
    
    openai_client = OpenAI(api_key=api_key)
    
    try:
        with tempfile.TemporaryDirectory() as tmp:
            raw_path      = os.path.join(tmp, "raw.tmp")
            enhanced_path = os.path.join(tmp, "enhanced.wav")
            
            # 1. Download
            logger.info(f"Downloading recording: {recording_url}")
            
            # Handle file:// URLs for local files
            if recording_url.startswith("file://"):
                local_path = recording_url[7:]  # Remove file:// prefix
                import shutil
                shutil.copy(local_path, raw_path)
            else:
                resp = requests.get(recording_url, stream=True, timeout=60)
                resp.raise_for_status()
                with open(raw_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            # 2. Enhance via FFmpeg
            filters = (
                "loudnorm=I=-16:TP=-1.5:LRA=11,"
                "acompressor=threshold=-21dB:ratio=4:makeup=2,"
                "equalizer=f=300:width_type=o:width=2:g=3,"
                "aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono"
            )
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", raw_path, "-af", filters, enhanced_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            if proc.returncode != 0:
                logger.error(f"FFmpeg failed: {proc.stderr[:300]}")
                return None
            
            # 3. Transcribe via gpt-4o-transcribe (or whisper-1 fallback)
            stt_model = os.getenv("WHISPER_MODEL", "gpt-4o-transcribe")
            logger.info("Transcribing via OpenAI %s...", stt_model)
            # prompt biases toward domain vocabulary and reduces hallucination on
            # low-quality audio. No `language` param so the model auto-detects
            # per-segment — correct for Hindi/English code-switched calls.
            stt_prompt = (
                "Sales call between a university admissions counselor and a prospective "
                "student. Indian English, Hindi, or code-switched Hindi-English. Topics: "
                "MBA, BBA, degree programs, fees, EMI, admission process, qualifications."
            )
            with open(enhanced_path, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model=stt_model,
                    file=audio_file,
                    prompt=stt_prompt,
                ).text.strip()
            
            logger.info("Transcription complete (%d chars)", len(transcript))
            return transcript
    
    except FileNotFoundError:
        logger.error("ffmpeg not found — install ffmpeg on the server")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download recording: {e}")
    except Exception as exc:
        logger.error(f"Transcription failed: {exc}")
    
    return None


def analyze_transcript(transcript: str) -> dict | None:
    """Run GPT Structured Outputs on a raw transcript string.
    
    Returns the full AnalysisData dict, or None on any failure.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping analysis")
        return None
    
    openai_client = OpenAI(api_key=api_key)
    
    try:
        gpt_model = os.getenv("ANALYSIS_MODEL", "gpt-4o-mini")
        today_ist = datetime.now(_IST).strftime("%Y-%m-%d")
        
        logger.info("Extracting CRM intelligence via GPT...")
        completion = openai_client.beta.chat.completions.parse(
            model=gpt_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"[Today's date (IST): {today_ist}]\n\n{transcript}"},
            ],
            response_format=AnalysisData,
            temperature=0.0,
        )
        
        result = completion.choices[0].message.parsed.model_dump()
        score = result.get("sentiment_score", 0.5)
        result["sentiment"] = (
            "positive" if score >= 0.6 else
            "negative" if score <= 0.4 else
            "neutral"
        )
        
        logger.info(
            "Analysis complete — disposition=%s, interest_level=%s, sentiment=%s (%.2f)",
            result.get("disposition"), result.get("interest_level"),
            result.get("sentiment"), score,
        )
        return result
    
    except Exception as exc:
        logger.error(f"GPT analysis failed: {exc}")
        return None


def analyze_recording(recording_url: str) -> dict | None:
    """Download, enhance, transcribe, and analyze a call recording.
    
    Returns the full AnalysisData dict, or None on any failure.
    """
    transcript = transcribe_recording(recording_url)
    if transcript is None:
        return None
    
    return analyze_transcript(transcript)
