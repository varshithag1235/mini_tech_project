# app.py
import os
import json
import csv
import re
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, render_template_string, jsonify
from groq import Groq  # Groq Python SDK

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in your environment or .env file")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Choose a model available on Groq (LLama-3.3 / others exist; change if needed).
MODEL = "llama-3.3-70b-versatile"

CSV_FILE = Path("call_analysis.csv")

app = Flask(__name__)

HTML_FORM = """
<!doctype html>
<title>Call Transcript Analyzer</title>
<h2>Paste a customer call transcript</h2>
<form method=post action="/analyze">
  <textarea name=transcript rows=12 cols=100 placeholder="Paste transcript here..."></textarea><br>
  <button type=submit>Analyze</button>
</form>
{% if result %}
<hr>
<h3>Result</h3>
<b>Transcript:</b>
<pre>{{ result.transcript }}</pre>
<b>Summary:</b> {{ result.summary }} <br>
<b>Sentiment:</b> {{ result.sentiment }}
{% endif %}
"""

def extract_json_from_text(text: str):
    """
    Try to parse JSON from the model's output robustly.
    Returns dict or None.
    """
    # direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # extract first {...} block
    m = re.search(r'(\{[\s\S]*\})', text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None

def fallback_parse(text: str):
    """
    If JSON extraction fails, attempt simple regex parsing for 'Summary' and 'Sentiment'.
    """
    summary = None
    sentiment = None
    # look for 'Summary:' block (take up to two sentences)
    m_summary = re.search(r'(Summary[:\-\s]+)(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m_summary:
        summary = m_summary.group(2).strip()
    # sentiment
    m_sent = re.search(r'(Sentiment[:\-\s]+)(\w+)', text, re.IGNORECASE)
    if m_sent:
        sentiment = m_sent.group(2).strip().capitalize()
    # fallback: if no keys, pick first 2 sentences as summary and classify sentiment by keywords
    if not summary:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if sentences:
            summary = " ".join(sentences[:2]).strip()
    if not sentiment:
        # simple keyword mapping
        low = text.lower()
        if any(k in low for k in ["angry", "frustrat", "upset", "not happy", "rude", "bad"]):
            sentiment = "Negative"
        elif any(k in low for k in ["thank", "good", "great", "happy", "satisfied", "awesome"]):
            sentiment = "Positive"
        else:
            sentiment = "Neutral"
    return {"summary": summary, "sentiment": sentiment}

def analyze_transcript_with_groq(transcript: str):
    """
    Calls the Groq Chat API asking for a JSON with 'summary' and 'sentiment'.
    """
    system_prompt = (
        "You are a concise assistant that reads a customer call transcript and returns a JSON object "
        "with EXACTLY two fields: 'summary' and 'sentiment'.\n"
        "- 'summary': 2-3 sentence concise summary of the customer's main issue.\n"
        "- 'sentiment': one of 'Positive', 'Neutral', or 'Negative'.\n"
        "Return only valid JSON with those keys. Do not add extra commentary.\n"
    )

    user_prompt = f"Transcript:\n{transcript}\n\nReturn the JSON described above."

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=MODEL,
            max_output_tokens=500,
            temperature=0.0,
        )
        # Groq python SDK returns pydantic model; content usually at choices[0].message.content
        content = None
        try:
            content = response.choices[0].message.content
        except Exception:
            # fallback: try accessing text property
            content = getattr(response, "text", None) or str(response)

        # Try to get JSON
        parsed = extract_json_from_text(content)
        if parsed is None:
            parsed = fallback_parse(content)

        # normalize sentiment
        sent = parsed.get("sentiment") if isinstance(parsed, dict) else None
        if sent:
            sent = sent.strip().capitalize()
            if sent.lower() in ["negative", "neg", "frustrated", "angry"]:
                sentiment = "Negative"
            elif sent.lower() in ["positive", "pos", "satisfied", "happy"]:
                sentiment = "Positive"
            else:
                sentiment = "Neutral"
        else:
            sentiment = "Neutral"

        summary = parsed.get("summary") if isinstance(parsed, dict) else parsed
        if not summary:
            summary = "No summary produced."

        return {"summary": summary.strip(), "sentiment": sentiment}
    except Exception as e:
        # Provide a clear error to the user; in production you'd add better error handling/logging
        return {"error": str(e)}

def append_to_csv(file_path: Path, transcript: str, summary: str, sentiment: str):
    header = ["Transcript", "Summary", "Sentiment"]
    write_header = not file_path.exists()
    with file_path.open(mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([transcript, summary, sentiment])

@app.route("/", methods=["GET"])
def form():
    return render_template_string(HTML_FORM)

@app.route("/analyze", methods=["POST"])
def analyze():
    # Accept form or JSON
    if request.is_json:
        transcript = request.json.get("transcript", "")
    else:
        transcript = request.form.get("transcript", "") or request.values.get("transcript", "")

    if not transcript or len(transcript.strip()) == 0:
        return "Please provide a non-empty transcript.", 400

    result = analyze_transcript_with_groq(transcript)
    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    summary = result["summary"]
    sentiment = result["sentiment"]

    # print to console (deliverable)
    print("\n----- Transcript -----")
    print(transcript)
    print("----- Summary -----")
    print(summary)
    print("----- Sentiment -----")
    print(sentiment)
    print("----------------------\n")

    # save to CSV
    append_to_csv(CSV_FILE, transcript, summary, sentiment)

    # If request JSON, return JSON; else show result in HTML
    if request.is_json:
        return jsonify({"transcript": transcript, "summary": summary, "sentiment": sentiment})
    else:
        return render_template_string(HTML_FORM, result={"transcript": transcript, "summary": summary, "sentiment": sentiment})

if __name__ == "__main__":
    # Run Flask app (for demo/local)
    app.run(host="127.0.0.1", port=5000, debug=True)
