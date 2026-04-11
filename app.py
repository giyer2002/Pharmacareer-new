import os
import json
import logging
import requests
import re
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/')
def home():
    return send_file('index.html')


# ── Job Search Endpoint ─────────────────────────────────────────────

@app.route('/api/search', methods=['POST'])
def search_jobs():
    data = request.json or {}
    title = data.get('title', '').strip()
    location = data.get('location', '').strip() or "United States"
    num_jobs = min(int(data.get('num_jobs', 30)), 30)

    if not title:
        return jsonify({"error": "Missing title"}), 400

    serp_api_key = os.environ.get("SERPAPI_KEY")
    if not serp_api_key:
        logger.error("SERPAPI_KEY is missing!")
        return jsonify({"error": "Server configuration error"}), 500

    try:
        params = {
            "engine": "google_jobs",
            "q": f"{title} {location}",
            "hl": "en",
            "gl": "us",
            "chips": "date_posted:week",
            "api_key": serp_api_key,
        }

        jobs = []
        while len(jobs) < num_jobs:
            response = requests.get("https://serpapi.com/search", params=params, timeout=30)
            response.raise_for_status()
            results = response.json()

            page_jobs = results.get("jobs_results", [])
            if not page_jobs:
                break

            jobs.extend(page_jobs)

            next_token = results.get("serpapi_pagination", {}).get("next_page_token")
            if not next_token:
                break

            params["next_page_token"] = next_token

        jobs = jobs[:num_jobs]

        formatted_jobs = []
        for job in jobs:
            direct_link = job.get("apply_link") or \
                         (job.get("apply_options")[0].get("link") if job.get("apply_options") else None) or \
                         job.get("share_link") or job.get("link")

            formatted_jobs.append({
                "title": job.get("title"),
                "company": job.get("company_name"),
                "location": job.get("location"),
                "link": direct_link,
                "description": job.get("description") or job.get("snippet") or "No description available.",
                "salary": job.get("salary"),
                "posted_date": job.get("date_posted"),
                "remote_hybrid": "Remote" if any(x in str(job).lower() for x in ["remote", "work from home"]) else
                                 "Hybrid" if "hybrid" in str(job).lower() else "On-site"
            })

        return jsonify(formatted_jobs)

    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify({"error": "Search failed. Please try again."}), 500


# ── AI Match Scoring Endpoint (NOW WITH ROBUST JSON CLEANING) ───────────────────────────────

@app.route('/api/match', methods=['POST'])
def match_score():
    data = request.json or {}
    candidate_profile = data.get('candidate_profile', '').strip()
    job_title         = data.get('job_title', '').strip()
    job_company       = data.get('job_company', '').strip()
    job_description   = data.get('job_description', '').strip()

    google_api_key = os.environ.get("GOOGLE_API_KEY")

    logger.info(f"LOADED GOOGLE_API_KEY: {google_api_key[:15]}...{google_api_key[-8:] if google_api_key else 'MISSING'}")

    if not candidate_profile or not job_description:
        return jsonify({"error": "candidate_profile and job_description are required"}), 400

    if not google_api_key or len(google_api_key) < 30:
        logger.error("GOOGLE_API_KEY is missing or too short!")
        return jsonify({"error": "Server configuration error (GOOGLE_API_KEY missing)"}), 500

    system_prompt = """You are an expert pharmaceutical/biotech recruiter.
Score the candidate against the job on a 0-100 scale.
Return **ONLY** valid JSON with this exact structure and nothing else (no markdown, no extra text):

{
  "overall": number,
  "skills": number,
  "seniority": number,
  "therapeutic": number,
  "role_type": number,
  "key_matches": ["short bullet 1", "short bullet 2"],
  "insight": "One short sentence explaining the match",
  "insight_type": "positive" or "caution"
}"""

    user_message = f"""CANDIDATE PROFILE:
{candidate_profile}

JOB TITLE: {job_title}
COMPANY: {job_company}

JOB DESCRIPTION:
{job_description}"""

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={google_api_key}"

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
                "max_output_tokens": 800
            }
        }

        resp = requests.post(url, json=payload, timeout=25)
        resp.raise_for_status()

        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

        # ── ROBUST JSON CLEANING (fixes your current error) ──
        # 1. Remove markdown code blocks
        raw = re.sub(r'```(?:json)?\s*', '', raw)
        raw = re.sub(r'```\s*$', '', raw)

        # 2. Extract only the JSON object
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        # 3. Fix common issues (single quotes → double quotes, trailing commas)
        raw = raw.replace("'", '"')
        raw = re.sub(r',\s*}', '}', raw)   # remove trailing comma before }

        # 4. Final parse
        scores = json.loads(raw)

        # Clamp scores
        for key in ["overall", "skills", "seniority", "therapeutic", "role_type"]:
            if key in scores:
                scores[key] = max(0, min(100, int(scores.get(key, 0))))

        if "key_matches" not in scores or not isinstance(scores["key_matches"], list):
            scores["key_matches"] = []

        return jsonify(scores)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {str(e)}\nRaw response was: {raw[:500]}", exc_info=True)
        return jsonify({
            "error": "Scoring failed",
            "detail": "AI model returned invalid format — please try again in a few seconds"
        }), 500

    except Exception as e:
        logger.error(f"Match scoring FAILED: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Scoring failed",
            "detail": "Please wait 30 seconds and try again (rate limit or temporary issue)"
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
