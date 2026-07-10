# 
"""
SQA Testing Agent - Flask Backend
=================================
Automated quality-assurance testing service for websites and GitHub
repositories. Given a URL it fetches the page, checks all links, detects
the technology stack, scores UX/trust signals, and produces an AI-written
recommendation using a fine-tuned TinyLlama model (with a rule-based
fallback if the model is unavailable or its output is unusable).
"""

import os
import re
import time
import random
import warnings
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
import torch
import urllib3
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from flask_cors import CORS
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from auth import auth_bp
from database import get_db, init_db
from models import (
    save_ai_suggestion,
    save_test_result,
    get_all_users,
    get_all_tests,
    get_admin_stats,
    delete_user,
    delete_test,
    get_user_stats,
    get_user_tests,
    get_test_full_detail,
)

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False
    print("Warning: cloudscraper not installed. Run: pip install cloudscraper")

# We intentionally fetch third-party sites that may present self-signed or
# misconfigured certificates during testing, so SSL verification is relaxed
# for the outbound test requests only. Silence the resulting urllib3 warning
# instead of letting it spam the console on every request.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_ADAPTER_PATH = os.path.join(BASE_DIR, "llm-finetune", "sqa_pro_final")
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_NEW_TOKENS = 300
REQUEST_TIMEOUT = 20
LINK_TIMEOUT = 5

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
]

BROWSER_HEADERS = {
    "User-Agent": random.choice(USER_AGENTS),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# ---------------------------------------------------------------------------
# Technology fingerprints
# ---------------------------------------------------------------------------
# FIXED: the original version matched short/generic words as plain
# substrings anywhere in the page (e.g. "react" matched inside GitHub's
# "reactions" emoji feature, causing a false "React" detection). This
# version uses regex with word boundaries for ambiguous words, and keeps
# only specific, low-collision signatures for the rest, so a technology is
# only reported when a real, unique marker for it is present.
TECH_MAP = {
    "React": [
        r"data-reactroot",
        r"data-reactid",
        r"react-dom",
        r"react\.production\.min",
        r"__react",
        r"_reactRootContainer",
    ],
    "Next.js": ["__next", "_next/static", "next/dist"],
    "Vue.js": [r"\bvue\.js\b", r"\bvue\.min\.js\b", "data-v-", "v-bind:", "v-model", "__vue__"],
    "Angular": ["ng-version", "ng-app", r"angular\.min\.js", "angular.io"],
    "WordPress": ["wp-content", "wp-includes", "wp-json"],
    "Shopify": ["cdn.shopify.com", "myshopify.com", "shopify-section", "shopify.theme"],
    "jQuery": [r"jquery(\.min)?\.js", r"jquery-\d"],
    "Bootstrap": [r"bootstrap(\.min)?\.css", "bootstrap.bundle"],
    "Tailwind CSS": [r"tailwind(css)?(\.min)?\.css"],
    "Laravel": ["laravel_session", r"\blaravel\b"],
    "Django": ["csrfmiddlewaretoken", "__django"],
    "PHP": [r"\.php\?", "phpsessid"],
    "Node.js": [r"x-powered-by:\s*express", "express-session"],
}

SOCIAL_PLATFORMS = ["facebook", "twitter", "instagram", "linkedin", "youtube"]

SKIP_EXT = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".ico",
    ".mp4", ".mp3", ".pdf", ".zip",
}

# ---------------------------------------------------------------------------
# LLM Setup - fine-tuned TinyLlama adapter used for AI recommendations
# ---------------------------------------------------------------------------
print("Loading LLM model...")

_base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float16)
llm_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
llm_tokenizer.pad_token = llm_tokenizer.eos_token

try:
    llm_model = PeftModel.from_pretrained(_base_model, LLM_ADAPTER_PATH)
    print("Fine-tuned adapter loaded.")
except Exception as exc:
    print(f"Adapter load failed: {exc}. Using base model.")
    llm_model = _base_model

print("LLM ready.\n")

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
app.register_blueprint(auth_bp)
init_db()

# ---------------------------------------------------------------------------
# URL Helpers
# ---------------------------------------------------------------------------
def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def to_absolute_url(base_url: str, href: str) -> str | None:
    href = href.strip()
    if href.startswith(("http://", "https://")):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return urljoin(base_url, href)
    if href.startswith(("#", "mailto:", "javascript:", "tel:", "data:")):
        return None
    return urljoin(base_url, href)

# ---------------------------------------------------------------------------
# Page Fetcher - requests -> cloudscraper -> Playwright fallback chain
# ---------------------------------------------------------------------------
def fetch_page_with_retry(url: str) -> tuple[str | None, dict | None, float]:
    """
    Fetch page HTML using progressively stronger methods:
      1. Plain requests with a randomised browser User-Agent.
      2. Cloudscraper, to get past basic Cloudflare challenges.
      3. Playwright (headless Chromium), for JS-rendered pages.
    Returns (html, response_headers, load_time_seconds). html is None if
    every method failed.
    """
    headers = BROWSER_HEADERS.copy()
    headers["User-Agent"] = random.choice(USER_AGENTS)

    t0 = time.time()

    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, verify=False)
        if resp.status_code == 200 and len(resp.text) > 500:
            print(f"Fetched with requests: {len(resp.text)} chars")
            return resp.text, dict(resp.headers), round(time.time() - t0, 2)
    except Exception as exc:
        print(f"Requests failed: {exc}")

    if CLOUDSCRAPER_AVAILABLE:
        try:
            scraper = cloudscraper.create_scraper()
            resp = scraper.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and len(resp.text) > 500:
                print(f"Fetched with cloudscraper: {len(resp.text)} chars")
                return resp.text, dict(resp.headers), round(time.time() - t0, 2)
        except Exception as exc:
            print(f"Cloudscraper failed: {exc}")

    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.5"})
            page.goto(url, wait_until="networkidle", timeout=REQUEST_TIMEOUT * 1000)
            html = page.content()
            browser.close()
            print(f"Fetched with Playwright: {len(html)} chars")
            return html, None, round(time.time() - t0, 2)
    except ImportError:
        print("Playwright not installed.")
    except Exception as exc:
        print(f"Playwright failed: {exc}")

    return None, None, round(time.time() - t0, 2)

# ---------------------------------------------------------------------------
# Technology Detection - returns every stack detected, not just the first
# ---------------------------------------------------------------------------
def detect_technologies(html: str, headers: dict | None = None) -> list[str]:
    """
    FIXED VERSION.
    Uses regex (with word boundaries for ambiguous words) instead of plain
    substring checks, so a word like "react" can no longer match inside
    unrelated text such as GitHub's "reactions" emoji feature. Only real,
    specific signatures for each technology are matched.
    """
    combined = html.lower()
    if headers:
        combined += " " + " ".join(f"{k.lower()}: {v.lower()}" for k, v in headers.items())

    detected = []
    for tech, patterns in TECH_MAP.items():
        for pattern in patterns:
            if re.search(pattern, combined):
                detected.append(tech)
                break

    return detected or ["Unknown"]

# ---------------------------------------------------------------------------
# Link Checker - scans every link found on the page, no cap
# ---------------------------------------------------------------------------
def _probe_link(url: str) -> bool:
    """HEAD first (cheap), GET fallback if the server rejects HEAD."""
    try:
        resp = requests.head(url, headers=BROWSER_HEADERS, timeout=LINK_TIMEOUT, allow_redirects=True)
        if resp.status_code == 405:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=LINK_TIMEOUT, stream=True)
            resp.close()
        return resp.status_code < 400
    except Exception:
        return False


def check_links(base_url: str, soup: BeautifulSoup) -> dict:
    urls = set()
    for tag in soup.find_all("a", href=True):
        abs_url = to_absolute_url(base_url, tag["href"])
        if not abs_url or not is_valid_url(abs_url):
            continue
        if any(urlparse(abs_url).path.lower().endswith(ext) for ext in SKIP_EXT):
            continue
        urls.add(abs_url)

    print(f"Total links found: {len(urls)}")

    working, broken = [], []
    for link in urls:
        (working if _probe_link(link) else broken).append(link)

    return {
        "total": len(urls),
        "working": len(working),
        "broken": len(broken),
        "broken_list": broken[:20],
    }

# ---------------------------------------------------------------------------
# Emotional / UX Analysis
# ---------------------------------------------------------------------------
def analyze_emotions(url: str, soup: BeautifulSoup, load_time: float) -> dict:
    trust = 25 if url.startswith("https") else 0
    trust_details = ["SSL certificate present"] if url.startswith("https") else ["Missing SSL certificate"]

    if soup.find("a", href=re.compile(r"contact", re.I)) or soup.find("a", string=re.compile(r"contact", re.I)):
        trust += 15
        trust_details.append("Contact information found")
    else:
        trust_details.append("No contact information found")

    if soup.find("a", href=re.compile(r"about", re.I)):
        trust += 10
        trust_details.append("About page found")
    else:
        trust_details.append("No About page found")

    for platform in SOCIAL_PLATFORMS:
        if soup.find("a", href=re.compile(platform, re.I)):
            trust += 5

    images = len(soup.find_all("img"))
    excitement = min(images * 2, 30)
    excitement_details = [f"{images} images on page"]

    videos = soup.find_all("video") + soup.find_all("iframe", src=re.compile(r"youtube|vimeo", re.I))
    if videos:
        excitement += 20
        excitement_details.append(f"{len(videos)} video(s) present")

    if soup.find(attrs={"class": re.compile(r"hero|banner", re.I)}):
        excitement += 15
        excitement_details.append("Hero/banner section found")

    pro_score, pro_details = 0, []

    meta = soup.find("meta", attrs={"name": "description"})
    if meta and len(meta.get("content", "")) > 50:
        pro_score += 15
        pro_details.append("Good meta description (length > 50 chars)")
    else:
        pro_details.append("Meta description missing or too short")

    title = soup.find("title")
    if title and len(title.text) > 10:
        pro_score += 10
        pro_details.append("Descriptive page title found")
    else:
        pro_details.append("Page title missing or too short")

    if load_time < 2:
        pro_score += 15
        pro_details.append(f"Fast loading ({load_time:.1f}s)")
    elif load_time < 4:
        pro_score += 7
        pro_details.append(f"Moderate loading ({load_time:.1f}s)")
    else:
        pro_details.append(f"Slow loading ({load_time:.1f}s) - needs optimization")

    return {
        "trust_score": min(trust, 100),
        "trust_details": trust_details,
        "excitement_score": min(excitement, 100),
        "excitement_details": excitement_details,
        "professionalism_score": min(pro_score, 100),
        "professional_indicators": pro_details,
        "response_time": load_time,
    }

# ---------------------------------------------------------------------------
# AI Suggestion Generator - real LLM inference with a rule-based fallback
# ---------------------------------------------------------------------------
def _score_label(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 50:
        return "Average"
    if score >= 30:
        return "Poor"
    return "Critical"


def _build_prompt(url: str, score: int, total: int, working: int, broken: int,
                   load_time: float, techs: list, trust: int,
                   excitement: int, professionalism: int) -> str:
    link_status = "all links healthy" if broken == 0 else f"{broken} broken out of {total}"
    tech_str = ", ".join(techs) if techs and techs != ["Unknown"] else "not detected"

    return f"""### Instruction:
You are a senior QA engineer. Based on the website test report below, write exactly 4 specific and actionable recommendations. Each recommendation must be based on the ACTUAL data provided.

### Website Test Report:
URL: {url}
Overall Score: {score}/100 ({_score_label(score)})
Load Time: {load_time}s
Links: {total} total | {working} working | {link_status}
Technologies: {tech_str}
Trust Score: {trust}/100
Excitement Score: {excitement}/100
Professionalism Score: {professionalism}/100

### Rules:
- If broken == 0, do NOT mention fixing broken links
- If score >= 75, acknowledge good performance
- If load_time > 3, recommend performance improvements
- If trust < 50, recommend SSL, contact page, About section
- Use priority labels: [CRITICAL] [HIGH] [MEDIUM] [LOW]
- Reference actual numbers from the report
- Do not use emojis

### Recommendations:
"""


def _rule_based_suggestion(url: str, score: int, total: int, working: int, broken: int,
                            load_time: float, techs: list, trust: int,
                            fetch_failed: bool = False) -> str:
    """Deterministic fallback used when the LLM is unavailable or its
    output is too short/unusable to show the user."""
    if fetch_failed:
        return """CRITICAL: Website could not be fetched or analyzed.

- The site may be blocking automated requests
- Try testing a different URL or check if the website is accessible
- 0 links found - unable to perform complete analysis

Recommendation: Test with a simpler website like https://example.com first"""

    lines = []
    if score >= 90:
        lines.append(f"[LOW] Excellent website quality! Score {score}/100")
    elif score >= 70:
        lines.append(f"[MEDIUM] Good website with minor issues. Score {score}/100")
    elif score >= 50:
        lines.append(f"[HIGH] Website needs significant improvements. Score {score}/100")
    else:
        lines.append(f"[CRITICAL] Website has serious quality issues! Score {score}/100")

    if total == 0:
        lines.append("No <a href> links were found on this page - it may rely on "
                      "JavaScript-rendered navigation, so link health could not be scored")
    else:
        lines.append(f"Link Analysis: {total} total links, {working} working, {broken} broken")
        if broken > 0:
            lines.append(f"[CRITICAL] Fix {broken} broken link(s) immediately - these hurt SEO and user trust")
        else:
            lines.append("[LOW] All links are healthy - good job!")

    if load_time > 3:
        lines.append(f"[CRITICAL] Page load time is {load_time}s - compress images, enable caching, use a CDN")
    elif load_time > 2:
        lines.append(f"[HIGH] Load time {load_time}s - consider optimizing images and minifying CSS/JS")
    else:
        lines.append(f"[LOW] Load time {load_time}s is good - continue monitoring")

    if not url.startswith("https"):
        lines.append("[HIGH] Add an SSL certificate for security - currently using HTTP")

    if trust < 40:
        lines.append("[HIGH] Trust score is low - add SSL, a contact page, and an About section")
    elif trust < 70:
        lines.append("[MEDIUM] Improve trust signals - add social links, privacy policy, testimonials")

    if techs and techs != ["Unknown"]:
        lines.append(f"[LOW] Technologies detected: {', '.join(techs[:5])}")
    else:
        lines.append("[MEDIUM] Technology stack could not be detected - verify meta tags")

    return "\n".join(lines)


def _looks_hallucinated(raw: str, total: int) -> bool:
    """
    Small models like TinyLlama can ignore the prompt data entirely and
    default to a stock 'could not fetch the site' answer regardless of
    what actually happened. If the page WAS fetched (total > 0 links
    were found) but the model's text claims otherwise, the output
    contradicts the real data and must not be shown to the user.
    """
    if total <= 0:
        return False  # nothing to contradict - a "could not fetch" style answer is plausible

    red_flags = [
        "could not be fetched",
        "could not fetch",
        "unable to fetch",
        "could not be analyzed",
        "0 links found",
        "blocking automated requests",
        "no links found",
        "website may be blocking",
    ]
    lowered = raw.lower()
    return any(flag in lowered for flag in red_flags)


def generate_suggestion(url: str, score: int, total: int, working: int, broken: int,
                         load_time: float, techs: list, trust: int,
                         excitement: int = 0, professionalism: int = 0,
                         fetch_failed: bool = False) -> str:
    """
    Generate the AI recommendation shown to the user.
    Tries the fine-tuned LLM first; falls back to the deterministic
    rule engine if the model errors out, returns something too short to
    be useful, or contradicts the data we actually measured (see
    `_looks_hallucinated`).
    """
    if fetch_failed:
        return _rule_based_suggestion(url, score, total, working, broken, load_time, techs, trust,
                                       fetch_failed=True)

    try:
        prompt = _build_prompt(url, score, total, working, broken, load_time,
                                techs, trust, excitement, professionalism)
        inputs = llm_tokenizer(prompt, return_tensors="pt")
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=llm_tokenizer.eos_token_id,
        )
        raw = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        for marker in ["### Recommendations:", "Recommendations:"]:
            if marker in raw:
                raw = raw.split(marker)[-1].strip()
                break

        if raw and len(raw) >= 40 and not _looks_hallucinated(raw, total):
            return raw

        if raw and _looks_hallucinated(raw, total):
            print(f"LLM output contradicted the measured data (total={total}), using rule-based fallback.")
        else:
            print("LLM output too short/empty, using rule-based fallback.")

    except Exception as exc:
        print(f"LLM generation failed: {exc}. Using rule-based fallback.")

    return _rule_based_suggestion(url, score, total, working, broken, load_time, techs, trust)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return "SQA Agent Backend - Running"


@app.get("/api/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/api/test/website")
def test_website():
    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    user_id = data.get("user_id", 1)

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    html, headers, load_time = fetch_page_with_retry(url)

    if not html:
        result = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "score": 0,
            "load_time": load_time,
            "technologies": ["Unknown"],
            "links": {"total": 0, "working": 0, "broken": 0, "broken_list": []},
            "recommendations": ["Could not fetch page - website may be blocking automated requests"],
            "emotions": {},
            "ai_suggestion": generate_suggestion(url, 0, 0, 0, 0, load_time, [], 0, fetch_failed=True),
        }
        test_id = save_test_result(user_id=user_id, result_data=result)
        result["test_id"] = test_id
        return jsonify(result), 200

    soup = BeautifulSoup(html, "html.parser")
    links = check_links(url, soup)
    techs = detect_technologies(html, headers)
    total, working, broken = links["total"], links["working"], links["broken"]
    score = min(int((working / total) * 100) if total > 0 else 70, 100)
    emotions = analyze_emotions(url, soup, load_time)

    suggestion = generate_suggestion(
        url, score, total, working, broken, load_time, techs,
        emotions["trust_score"], emotions["excitement_score"], emotions["professionalism_score"],
    )

    result = {
        "url": url,
        "timestamp": datetime.now().isoformat(),
        "score": score,
        "load_time": load_time,
        "technologies": techs,
        "links": links,
        "recommendations": [
            f"Fix {broken} broken links" if broken else "All links are healthy",
            f"Technologies detected: {', '.join(techs[:5])}" if techs != ["Unknown"] else "Technology stack could not be detected",
            "Add SSL certificate" if not url.startswith("https") else "SSL is active",
        ],
        "emotions": emotions,
        "ai_suggestion": suggestion,
    }

    test_id = save_test_result(user_id=user_id, result_data=result)
    result["test_id"] = test_id
    save_ai_suggestion(test_result_id=test_id, url=url, score=score, broken_count=broken, suggestion=suggestion)

    return jsonify(result), 200


@app.post("/api/test/github")
def test_github():
    data = request.get_json(force=True)
    repo_url = data.get("url", "").strip()
    match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)

    if not match:
        return jsonify({"error": "Invalid GitHub URL"}), 400

    owner, repo = match.groups()
    base = f"https://api.github.com/repos/{owner}/{repo}"
    gh_headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        repo_resp = requests.get(base, headers=gh_headers, timeout=REQUEST_TIMEOUT)
        if repo_resp.status_code != 200:
            return jsonify({"error": "Repository not found"}), 404

        repo_data = repo_resp.json()
        languages = requests.get(f"{base}/languages", headers=gh_headers, timeout=REQUEST_TIMEOUT).json()
        has_readme = requests.get(f"{base}/readme", headers=gh_headers, timeout=REQUEST_TIMEOUT).status_code == 200

        stars = repo_data.get("stargazers_count", 0)
        forks = repo_data.get("forks_count", 0)
        issues = repo_data.get("open_issues_count", 0)
        has_description = bool(repo_data.get("description"))
        has_license = bool(repo_data.get("license"))

        score = 50
        score += 15 if stars > 100 else (10 if stars > 50 else 0)
        score += 10 if forks > 50 else 0
        score += 10 if issues < 5 else 0
        score += 5 if has_description else 0
        score += 5 if has_license else 0
        score += 5 if has_readme else 0
        score = min(score, 100)

        recommendations = []
        if not has_description:
            recommendations.append("Add repository description")
        if not has_license:
            recommendations.append("Add license file")
        if not has_readme:
            recommendations.append("Add README.md file")
        if issues > 20:
            recommendations.append(f"Address {issues} open issues")
        if not recommendations:
            recommendations.append("Repository is well-maintained")

        suggestion = generate_suggestion(repo_url, score, 0, 0, issues, 0, list(languages.keys()), 50)

        return jsonify({
            "url": repo_url,
            "name": repo_data.get("name"),
            "owner": owner,
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "stars": stars,
            "forks": forks,
            "open_issues": issues,
            "has_readme": has_readme,
            "has_license": has_license,
            "language": repo_data.get("language", "Unknown"),
            "all_languages": languages,
            "recommendations": recommendations,
            "ai_suggestion": suggestion,
        }), 200

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/ai-suggest")
def ai_suggest():
    data = request.get_json(force=True)
    url = data.get("url", "")
    score = int(data.get("score", 0))
    total = int(data.get("total", 0))
    working = int(data.get("working", 0))
    broken = int(data.get("broken", 0))
    test_id = int(data.get("test_id", 0))
    load_time = float(data.get("load_time", 0))
    techs = data.get("technologies", [])
    trust = int(data.get("trust_score", 0))

    suggestion = generate_suggestion(url, score, total, working, broken, load_time, techs, trust)

    if not test_id:
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM test_results WHERE url = ? ORDER BY id DESC LIMIT 1", (url,))
            row = cursor.fetchone()
            conn.close()
            test_id = row["id"] if row else 0
        except Exception:
            test_id = 0

    save_ai_suggestion(test_id, url, score, broken, suggestion)
    return jsonify({"suggestion": suggestion})

# ---------------------------------------------------------------------------
# Admin Dashboard Routes
# ---------------------------------------------------------------------------
@app.get("/api/admin/users")
def admin_get_users():
    return jsonify(get_all_users()), 200


@app.get("/api/admin/tests")
def admin_get_tests():
    return jsonify(get_all_tests()), 200


@app.get("/api/admin/stats")
def admin_get_stats():
    return jsonify(get_admin_stats()), 200


@app.delete("/api/admin/users/<int:user_id>")
def admin_delete_user(user_id):
    delete_user(user_id)
    return jsonify({"message": "User deleted"}), 200


@app.delete("/api/admin/tests/<int:test_id>")
def admin_delete_test(test_id):
    delete_test(test_id)
    return jsonify({"message": "Test deleted"}), 200

# ---------------------------------------------------------------------------
# User Dashboard Routes
# ---------------------------------------------------------------------------
@app.get("/api/user/<int:user_id>/stats")
def user_get_stats(user_id):
    return jsonify(get_user_stats(user_id)), 200


@app.get("/api/user/<int:user_id>/tests")
def user_get_tests(user_id):
    return jsonify(get_user_tests(user_id)), 200


@app.get("/api/test/<int:test_id>")
def get_test_detail_route(test_id):
    detail = get_test_full_detail(test_id)
    if not detail:
        return jsonify({"error": "Test not found"}), 404
    return jsonify(detail), 200

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print("SQA TESTING AGENT BACKEND")
    print("=" * 50)
    print("Unlimited link scanning   - ACTIVE")
    print("Multi-technology detection - ACTIVE")
    print("AI suggestions (LLM)      - ACTIVE")
    print("=" * 50)
    print(f"Server: http://0.0.0.0:{port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)