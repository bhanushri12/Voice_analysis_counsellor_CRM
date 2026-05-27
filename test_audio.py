"""Complete test with actual audio file"""
import os
import sys

# Check prerequisites
print("Checking prerequisites...")
print("=" * 60)

# 1. Check OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY not set!")
    print("Please set it in .env file or run:")
    print("export OPENAI_API_KEY=your-key-here")
    sys.exit(1)
print("✓ OPENAI_API_KEY is set")

# 2. Check ffmpeg
import shutil
if not shutil.which("ffmpeg"):
    print("❌ ffmpeg not found!")
    print("Install it with: sudo apt-get install ffmpeg")
    sys.exit(1)
print("✓ ffmpeg is installed")

# 3. Check audio file
audio_file = "audio (2).mp3"
if not os.path.exists(audio_file):
    print(f"❌ Audio file '{audio_file}' not found!")
    sys.exit(1)
print(f"✓ Audio file found: {audio_file}")

print("\n" + "=" * 60)
print("Starting full audio analysis test...")
print("=" * 60 + "\n")

# Import after checks
from vox_client import transcribe_recording, analyze_transcript

# Step 1: Transcribe
print("Step 1: Transcribing audio...")
file_url = f"file://{os.path.abspath(audio_file)}"
print(f"File URL: {file_url}")

transcript = transcribe_recording(file_url)

if not transcript:
    print("❌ Transcription failed!")
    sys.exit(1)

print(f"✅ Transcription successful! ({len(transcript)} characters)")
print(f"\nTranscript preview:\n{transcript[:500]}...\n")

# Step 2: Analyze
print("Step 2: Analyzing transcript with GPT...")
result = analyze_transcript(transcript)

if not result:
    print("❌ Analysis failed!")
    sys.exit(1)

print("✅ Analysis successful!")
print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"Disposition: {result.get('disposition')}")
print(f"Interest Level: {result.get('interest_level')}")
print(f"Sentiment: {result.get('sentiment')} (score: {result.get('sentiment_score')})")
print(f"\nSummary:\n{result.get('summary')}")

# Extract key fields
ai_rec = result.get('ai_recommendations', {})
stage = ai_rec.get('stage_update', {})
auto_fill = ai_rec.get('auto_fill', {})
form_fields = auto_fill.get('form_fields', {})

print(f"\nRecommended Stage: {stage.get('recommended_stage')}")
print(f"Sub-stage: {stage.get('recommended_sub_stage')}")

print("\nExtracted Information:")
for field, value in form_fields.items():
    if value:
        print(f"  - {field}: {value}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
