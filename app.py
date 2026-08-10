import os
import json
import logging
import requests
import re
import time
import threading
from datetime import datetime

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
                "remote_hybrid": "Remote" if any(x in str(job).lower() for x in ["remote", "work from home"]) else "Hybrid" if "hybrid" in str(job).lower() else "On-site"
            })
        return jsonify(formatted_jobs)
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify({"error": "Search failed. Please try again."}), 500

# ── AI Match Scoring Endpoint ───────────────────────────────────────
@app.route('/api/match', methods=['POST'])
def match_score():
    data = request.json or {}
    candidate_profile = data.get('candidate_profile', '').strip()
    job_title = data.get('job_title', '').strip()
    job_company = data.get('job_company', '').strip()
    job_description = data.get('job_description', '').strip()

    google_api_key = os.environ.get("GOOGLE_API_KEY")

    if not candidate_profile or not job_description:
        return jsonify({"error": "candidate_profile and job_description are required"}), 400

    if not google_api_key or len(google_api_key) < 30:
        logger.error("GOOGLE_API_KEY is missing or too short!")
        return jsonify({"error": "Server configuration error (GOOGLE_API_KEY missing)"}), 500

    system_prompt = """You are an expert pharmaceutical/biotech recruiter. Score the candidate against the job on a 0-100 scale. Return **ONLY** valid JSON with this exact structure and nothing else (no markdown, no extra text):
{ "overall": number, "skills": number, "seniority": number, "therapeutic": number, "role_type": number, "key_matches": ["short bullet 1", "short bullet 2"], "insight": "One short sentence explaining the match", "insight_type": "positive" or "caution" }"""

    user_message = f"""CANDIDATE PROFILE: {candidate_profile}
JOB TITLE: {job_title}
COMPANY: {job_company}
JOB DESCRIPTION: {job_description}"""

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

        raw = re.sub(r'```(?:json)?\s*', '', raw)
        raw = re.sub(r'```\s*$', '', raw)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)
        raw = raw.replace("'", '"')
        raw = re.sub(r',\s*}', '}', raw)

        scores = json.loads(raw)

        for key in ["overall", "skills", "seniority", "therapeutic", "role_type"]:
            if key in scores:
                scores[key] = max(0, min(100, int(scores.get(key, 0))))
        if "key_matches" not in scores or not isinstance(scores["key_matches"], list):
            scores["key_matches"] = []

        return jsonify(scores)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Scoring failed", "detail": "AI model returned invalid format — please try again in a few seconds"}), 500
    except Exception as e:
        logger.error(f"Match scoring FAILED: {str(e)}", exc_info=True)
        return jsonify({"error": "Scoring failed", "detail": "Please wait 30 seconds and try again (rate limit or temporary issue)"}), 500

# ── SEO & CACHING LAYER (Makes Google index your site) ──────────────
SEO_CACHE = {"jobs": [], "last_update": 0}
SEO_LOCK = threading.Lock()

def get_seo_jobs():
    try:
        if time.time() - SEO_CACHE["last_update"] > 43200 or not SEO_CACHE["jobs"]:
            with SEO_LOCK:
                if time.time() - SEO_CACHE["last_update"] > 43200 or not SEO_CACHE["jobs"]:
                    _seed_seo_jobs()
    except Exception as e:
        logger.error(f"get_seo_jobs error: {e}")
    return SEO_CACHE["jobs"]

def _seed_seo_jobs():
    global SEO_CACHE
    serp_api_key = os.environ.get("SERPAPI_KEY")
    if not serp_api_key:
        logger.error("SEO seed skipped: SERPAPI_KEY missing")
        return

    queries = [
        "pharmacovigilance jobs", "regulatory affairs jobs",
        "clinical research jobs", "quality assurance pharma jobs",
        "medical affairs jobs", "biostatistics jobs"
    ]
    all_jobs, seen = [], set()

    for q in queries:
        try:
            params = {"engine": "google_jobs", "q": q, "hl": "en", "gl": "us", "api_key": serp_api_key}
            r = requests.get("https://serpapi.com/search", params=params, timeout=15)
            if r.status_code == 200:
                for job in r.json().get("jobs_results", [])[:8]:
                    link = job.get("apply_link") or job.get("share_link") or job.get("link")
                    if link and link not in seen:
                        seen.add(link)
                        slug = re.sub(r'[^a-z0-9]+', '-', (job.get("title", "") + "-" + job.get("company_name", "")).lower()).strip('-')[:80] or "job"
                        all_jobs.append({
                            "slug": slug, "title": job.get("title") or "Pharma Role",
                            "company": job.get("company_name") or "Life Sciences Company",
                            "location": job.get("location") or "USA",
                            "description": job.get("description") or "",
                            "link": link, "posted": job.get("date_posted") or "Recently"
                        })
        except Exception as e:
            logger.error(f"SEO seed error for '{q}': {e}")

    SEO_CACHE["jobs"] = all_jobs
    SEO_CACHE["last_update"] = time.time()
    logger.info(f"Seeded {len(all_jobs)} jobs for SEO")

def _start_seed():
    try:
        get_seo_jobs()
    except Exception as e:
        logger.error(f"seed thread error: {e}")

threading.Thread(target=_start_seed, daemon=True).start()

@app.route('/robots.txt')
def robots_txt():
    return "User-agent: *\nAllow: /\nSitemap: https://www.pharmacareerhub.com/sitemap.xml\n", 200, {'Content-Type': 'text/plain'}

@app.route('/sitemap.xml')
def sitemap_xml():
    jobs = get_seo_jobs()
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    xml += "  <url><loc>https://www.pharmacareerhub.com/</loc></url>\n"
    for j in jobs:
        xml += f"  <url><loc>https://www.pharmacareerhub.com/job/{j['slug']}</loc></url>\n"
    xml += "</urlset>"
    return xml, 200, {'Content-Type': 'application/xml'}

@app.route('/jobs')
def all_jobs_page():
    jobs = get_seo_jobs()
    html = "<html><head><title>Live Pharma & Biotech Jobs</title></head><body style='font-family:system-ui;max-width:800px;margin:40px auto;padding:20px'>"
    html += "<a href='/'>← Home</a><h1>Live Pharma & Biotech Jobs</h1><ul>"
    for j in jobs:
        html += f"<li><a href='/job/{j['slug']}'>{j['title']} at {j['company']} ({j['location']})</a></li>"
    html += "</ul></body></html>"
    return html

@app.route('/job/<slug>')
def job_page(slug):
    jobs = get_seo_jobs()
    job = next((j for j in jobs if j['slug'] == slug), None)
    if not job:
        return "Job not found", 404

    schema = {
        "@context": "https://schema.org", "@type": "JobPosting",
        "title": job['title'], "description": job['description'],
        "datePosted": datetime.utcnow().strftime("%Y-%m-%d"),
        "hiringOrganization": {"@type": "Organization", "name": job['company']},
        "jobLocation": {"@type": "Place", "address": {"@type": "PostalAddress", "addressLocality": job['location']}},
        "directApplyUrl": job['link']
    }

    html = f"""<!DOCTYPE html><html><head>
    <title>{job['title']} at {job['company']} | PharmaCareer Hub</title>
    <meta name="description" content="{job['description'][:150]}">
    <script type="application/ld+json">{json.dumps(schema)}</script>
    <style>body{{font-family:system-ui;max-width:800px;margin:40px auto;padding:20px;color:#0a1f1a}}a{{color:#00d4aa}}.btn{{background:#00d4aa;color:#000;padding:12px 24px;text-decoration:none;border-radius:8px;display:inline-block;margin-top:20px}}</style>
    </head><body>
    <a href="/">← Back to PharmaCareer Hub</a>
    <h1>{job['title']}</h1>
    <p><strong>{job['company']}</strong> • {job['location']} • {job['posted']}</p>
    <p>{job['description']}</p>
    <a href="{job['link']}" target="_blank" rel="noopener" class="btn">Apply on Company Site →</a>
    </body></html>"""
    return html

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)