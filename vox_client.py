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
    text: str = Field(description="What the speaker said, preserved in the original spoken language. Hindi/regional words stay in native script; English words stay in English. Do NOT translate.")

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

LANGUAGE PRESERVATION (CRITICAL):
- The transcript array MUST be written in the ORIGINAL SPOKEN LANGUAGE — do NOT translate to English.
- Hindi words → Devanagari script. Telugu → Telugu script. Tamil → Tamil script. etc.
- English words that were spoken in English MUST remain in English (Latin script).
- Code-switched sentences (Hinglish, Tenglish, etc.) must preserve both scripts exactly as spoken.
- Only the summary, form_fields, and remarks should be written in English for CRM readability.

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

⚠️ CRITICAL — NEVER output the entire conversation as a single turn. That is ALWAYS wrong.
A real call has 6–40+ turns. If your transcript array has only 1 or 2 turns, you have failed to split.
Every time the speaker changes, you MUST start a new {role, text} object.

TURN SPLITTING RULES:
- Short acknowledgements like "हाँ", "हाँ जी", "okay", "theek hai", "ठीक है", "अच्छा", "हलो", "नहीं"
  spoken by the OTHER party are ALWAYS their own separate turn — do NOT merge them into the previous turn.
- After the agent asks a question, the very next words are always a USER turn (even if short).
- After the user answers, any explanation, pitch, or follow-up is always an AGENT turn.
- If one party speaks for a long time on one topic, keep it as one turn. But the moment the
  other party interjects (even a single word), start a new turn for them.

HARD RULES:
- Determine who speaks first from context:
  • Outgoing call: the AGENT speaks first — greets and introduces themselves/university.
  • Incoming call: the USER speaks first with a bare greeting ("Hello?", "Haan?", "Ji?") and the AGENT's FIRST substantive turn introduces themselves/university.
- Every question must be followed by an answer from the OTHER party; do not assign two consecutive questions to the same role.
- Coherence check: if a turn starts with a direct answer to the previous turn's question, it must be the opposite role.

HINDI / HINGLISH TURN-BOUNDARY SIGNALS:
USER turn starts when you see: हाँ | हाँ जी | अच्छा | ठीक है | नहीं | हलो | okay | hmm | ji | एक सेकंड | देखते हैं | मैं समझ रहा हूँ
AGENT turn starts when you see: तो | देखो | मैंने आपको | हमारे यहाँ | इस programme में | fee है | आपको apply करना होगा | मैं share कर देती/देता हूँ

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
A valid transcript for a 2-minute call has at least 6 turns. Fewer than 4 turns means you did not split properly.
"""

_env_prompt = os.getenv("ANALYSIS_SYSTEM_PROMPT", "")
SYSTEM_PROMPT = _env_prompt.replace('\\n', '\n') if _env_prompt else _DEFAULT_PROMPT

# ── Public functions ─────────────────────────────────────────────────

def transcribe_recording(recording_url: str, language: str | None = None) -> str | None:
    """Download, enhance via FFmpeg, and transcribe a recording with Whisper.

    Args:
        recording_url: URL or file:// path to the audio file.
        language: BCP-47 language code (e.g. "te", "hi", "ta"). When provided
            it is passed directly to the transcription API so the model outputs
            in the correct script instead of defaulting to English.  Leave None
            to let the model auto-detect (acceptable for clearly Hindi calls;
            unreliable for South-Indian languages).

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
            
            # 3. Transcribe via OpenAI audio.transcriptions
            stt_model = os.getenv("WHISPER_MODEL", "gpt-4o-transcribe")
            logger.info("Transcribing via OpenAI %s...", stt_model)
            # prompt steers the model toward domain vocabulary and reduces hallucination
            # on low-quality audio. No `language` param so the model auto-detects
            # per-segment — this correctly handles Hindi/English code-switching.
            # The prompt must begin with native-language text so gpt-4o-transcribe
            # infers the correct output script instead of defaulting to English.
            stt_prompt = (
                # Hindi primer — steers the model toward native-script output
                "यह एक भारतीय विश्वविद्यालय और छात्र के बीच की प्रवेश परामर्श कॉल है। "
                "कृपया मूल भाषा में ही लिखें, अनुवाद न करें। "
                # English instructions
                "TRANSCRIBE IN THE ORIGINAL SPOKEN LANGUAGE — DO NOT TRANSLATE TO ENGLISH. "
                "The call may be in Hindi, Telugu, Tamil, Kannada, Marathi, Bengali, "
                "Malayalam, Gujarati, or Punjabi, often mixed with English. "
                "Write Indian-language words in their native script (Devanagari, Telugu, "
                "Tamil, Kannada, etc.). Write English words in English (Latin) script "
                "exactly as spoken — never transliterate them into any Indian script. "
                "Topics: MBA, BBA, B.Sc, M.Tech, fees, EMI, admission, qualifications, "
                "university, online, distance learning."
            )
            with open(enhanced_path, "rb") as audio_file:
                transcription_kwargs = dict(
                    model=stt_model,
                    file=audio_file,
                    prompt=stt_prompt,
                )
                if language:
                    transcription_kwargs["language"] = language
                transcript = openai_client.audio.transcriptions.create(
                    **transcription_kwargs
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
        user_message = f"[Today's date (IST): {today_ist}]\n\n{transcript}"

        def _parse(temperature: float) -> "AnalysisData":
            return openai_client.beta.chat.completions.parse(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                response_format=AnalysisData,
                temperature=temperature,
            ).choices[0].message.parsed

        logger.info("Extracting CRM intelligence via GPT...")
        parsed = _parse(temperature=0.0)

        # If GPT collapsed the whole conversation into ≤2 turns, retry with a
        # stronger split instruction so the dialog is properly separated.
        if len(parsed.transcript) <= 2:
            logger.warning(
                "Transcript has only %d turn(s) — retrying with forced split prompt",
                len(parsed.transcript),
            )
            split_suffix = (
                "\n\n⚠️ IMPORTANT: Your previous response returned the entire conversation "
                "as a single turn. That is incorrect. You MUST split the transcript into "
                "individual agent/user turns. Every speaker change = a new turn. "
                "A 2-minute call should have at least 6–10 turns. Re-analyse and split properly."
            )
            parsed = openai_client.beta.chat.completions.parse(
                model=gpt_model,
                messages=[
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_message},
                    {"role": "assistant", "content": parsed.model_dump_json()},
                    {"role": "user",      "content": split_suffix},
                ],
                response_format=AnalysisData,
                temperature=0.2,
            ).choices[0].message.parsed
            logger.info("Retry produced %d turns", len(parsed.transcript))

        result = parsed.model_dump()
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


def analyze_recording(recording_url: str, language: str | None = None) -> dict | None:
    """Download, enhance, transcribe, and analyze a call recording.

    Args:
        language: BCP-47 code forwarded to transcribe_recording (e.g. "te", "hi").
    Returns the full AnalysisData dict, or None on any failure.
    """
    transcript = transcribe_recording(recording_url, language=language)
    if transcript is None:
        return None

    return analyze_transcript(transcript)
