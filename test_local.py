"""Quick local test of the audio analysis"""
import os
from dotenv import load_dotenv
from vox_client import analyze_transcript

# Load environment variables from .env file
load_dotenv()

# Test with a simple transcript
test_transcript = """
Agent: Hello, this is John from XYZ University. Am I speaking with Rahul?
User: Yes, this is Rahul speaking.
Agent: Great! I'm calling regarding your interest in our M.Tech Computer Science program. Do you have a few minutes to discuss?
User: Yes, sure. I'm very interested in the program.
Agent: Excellent! Can you tell me about your educational background?
User: I completed my B.Tech in Computer Science from Mumbai University in 2020. I've been working as a software engineer at TCS for the past 3 years.
Agent: That's great experience. What's your current CTC?
User: I'm currently earning 8 lakhs per annum.
Agent: And what's your budget for the program?
User: I'm looking at around 5 to 8 lakhs. I might need an education loan.
Agent: We can definitely help with that. When would you like to start?
User: I'm interested in the Fall 2026 intake.
Agent: Perfect. Can I get your email address to send you more details?
User: Sure, it's rahul.sharma@gmail.com
Agent: Great! And what's the best time to reach you for follow-up?
User: Evenings between 6 to 8 PM work best for me.
Agent: Excellent. I'll send you the application link and we can schedule a detailed discussion next week. Does that work?
User: Yes, that sounds good. Thank you!
Agent: Thank you, Rahul. Have a great day!
"""

print("Testing audio analysis...")
print("=" * 50)

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY not set!")
    print("Set it with: export OPENAI_API_KEY=your-key-here")
    exit(1)

print("✓ OPENAI_API_KEY is set")
print("\nAnalyzing transcript...")

result = analyze_transcript(test_transcript)

if result:
    print("\n✅ SUCCESS! Analysis completed:")
    print("=" * 50)
    print(f"Disposition: {result.get('disposition')}")
    print(f"Interest Level: {result.get('interest_level')}")
    print(f"Sentiment: {result.get('sentiment')} ({result.get('sentiment_score')})")
    print(f"\nSummary: {result.get('summary')}")
    print(f"\nRecommended Stage: {result.get('ai_recommendations', {}).get('stage_update', {}).get('recommended_stage')}")
    print(f"Course Extracted: {result.get('ai_recommendations', {}).get('auto_fill', {}).get('form_fields', {}).get('course')}")
    print(f"Email: {result.get('ai_recommendations', {}).get('auto_fill', {}).get('form_fields', {}).get('email')}")
else:
    print("\n❌ FAILED: Analysis returned None")
    print("Check the logs above for errors")
