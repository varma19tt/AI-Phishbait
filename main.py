# main.py - Advanced Backend
import os
import re
import asyncio
import hashlib
import json
import uuid
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict
from gpt4all import GPT4All
from playwright.async_api import async_playwright
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from cachetools import TTLCache
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Configuration
MODEL_PATH = "Yi-1.5-9B-Chat-16K-Q4_0.gguf"
MAX_TOKENS = 1024
TEMPERATURE = 0.75
SCRAPE_TIMEOUT = 30000
MAX_CONCURRENT_TASKS = 5
DB_PATH = "phishbait.db"

# Suppress CUDA warnings
os.environ["GFILT_IGNORE_CUDA"] = "1"

# Initialize FastAPI
app = FastAPI(
    title="AI-PhishBait Pro",
    description="Advanced AI-powered phishing simulation platform",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url=None
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Models
class TargetRequest(BaseModel):
    platform: str = "linkedin"  # linkedin, twitter, custom
    identifier: str  # URL for linkedin/twitter, JSON for custom
    custom_prompt: Optional[str] = None
    template_type: Optional[str] = None
    social_engineering_tactic: Optional[str] = "authority"

class CampaignRequest(BaseModel):
    name: str
    targets: List[TargetRequest]
    schedule: Optional[Dict] = None  # {type: "immediate"/"scheduled", datetime: optional}

class User(BaseModel):
    id: str
    name: str
    email: str
    credits: int
    tier: str  # free, pro, enterprise

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 id TEXT PRIMARY KEY,
                 name TEXT,
                 email TEXT,
                 credits INTEGER,
                 tier TEXT,
                 api_key TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS history (
                 id TEXT PRIMARY KEY,
                 user_id TEXT,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                 platform TEXT,
                 identifier TEXT,
                 prompt TEXT,
                 generated_email TEXT,
                 credits_used INTEGER,
                 performance_data TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS campaigns (
                 id TEXT PRIMARY KEY,
                 user_id TEXT,
                 name TEXT,
                 targets TEXT,  
                 schedule TEXT,
                 status TEXT,
                 results TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS analytics (
                 id TEXT PRIMARY KEY,
                 campaign_id TEXT,
                 open_rate REAL,
                 click_rate REAL,
                 report_rate REAL,
                 effectiveness_score REAL)''')
    
    # Create demo user if not exists
    c.execute("SELECT * FROM users WHERE id = 'demo'")
    if not c.fetchone():
        c.execute('''INSERT INTO users (id, name, email, credits, tier, api_key)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  ('demo', 'Demo User', 'demo@phishbait.com', 100, 'pro', 'demo_key'))
    
    conn.commit()
    conn.close()

init_db()

# Mock data for social engineering tactics
SOCIAL_ENGINEERING_TACTICS = {
    "authority": {
        "name": "Authority Exploitation",
        "description": "Leverage respect for authority figures",
        "icon": "crown"
    },
    "urgency": {
        "name": "Time Sensitivity",
        "description": "Create time-sensitive scenarios",
        "icon": "clock"
    },
    "familiarity": {
        "name": "Familiarity Principle",
        "description": "Mimic trusted relationships",
        "icon": "handshake"
    },
    "scarcity": {
        "name": "Scarcity Principle",
        "description": "Use limited availability",
        "icon": "ticket"
    },
    "consensus": {
        "name": "Social Consensus",
        "description": "Suggest others are doing it",
        "icon": "users"
    }
}

# Email Templates
TEMPLATES = {
    "job_offer": {
        "name": "Job Offer",
        "description": "Convincing job offer from a reputable company",
        "effectiveness": 0.85
    },
    "security_alert": {
        "name": "Security Alert",
        "description": "Urgent security notification requiring action",
        "effectiveness": 0.92
    },
    "document_share": {
        "name": "Document Share",
        "description": "Document collaboration request",
        "effectiveness": 0.78
    },
    "social_engineering": {
        "name": "Advanced Social Engineering",
        "description": "Sophisticated psychological manipulation",
        "effectiveness": 0.95
    },
    "password_reset": {
        "name": "Password Reset",
        "description": "Urgent password reset request",
        "effectiveness": 0.88
    },
    "invoice": {
        "name": "Invoice Payment",
        "description": "Overdue invoice notification",
        "effectiveness": 0.82
    }
}

# Global instances
model = None
playwright = None
browser = None
task_queue = asyncio.Queue()
current_tasks = 0

# Profile cache
profile_cache = TTLCache(maxsize=200, ttl=3600)

# Startup Event
@app.on_event("startup")
async def startup_event():
    global model, playwright, browser
    
    print("⚡ Starting AI-PhishBait Pro Server...")
    
    try:
        print("🧠 Initializing AI model...")
        model = GPT4All(
            model_name=MODEL_PATH,
            model_path='.',
            allow_download=False,
            n_threads=os.cpu_count() or 8
        )
        print("✅ AI model initialized")
        
        print("🌐 Launching browser...")
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--single-process'
            ],
            timeout=60000
        )
        print("✅ Browser launched")
        
        # Start task processing
        asyncio.create_task(process_tasks())
        
        print("🚀 AI-PhishBait Pro is ready!")
    except Exception as e:
        print(f"🔥 Startup failed: {str(e)}")
        raise

# Background task processor
async def process_tasks():
    global current_tasks
    
    while True:
        if current_tasks < MAX_CONCURRENT_TASKS:
            task = await task_queue.get()
            current_tasks += 1
            asyncio.create_task(execute_task(task))
        await asyncio.sleep(0.1)

async def execute_task(task):
    try:
        await task["function"](*task["args"])
    finally:
        global current_tasks
        current_tasks -= 1
        task_queue.task_done()

# Shutdown Event
@app.on_event("shutdown")
async def shutdown_event():
    if browser:
        await browser.close()
    if playwright:
        await playwright.stop()
    print("🔌 Resources cleaned up")

# Database helper functions
def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_user(api_key: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE api_key = ?", (api_key,))
    user = c.fetchone()
    conn.close()
    
    if user:
        return {
            "id": user[0],
            "name": user[1],
            "email": user[2],
            "credits": user[3],
            "tier": user[4],
            "api_key": user[5]
        }
    return None

def update_user_credits(user_id: str, credits: int):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE users SET credits = ? WHERE id = ?", (credits, user_id))
    conn.commit()
    conn.close()

def save_to_history(user_id: str, data: dict):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''INSERT INTO history 
                 (id, user_id, platform, identifier, prompt, generated_email, credits_used, performance_data)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (
                  str(uuid.uuid4()),
                  user_id,
                  data["platform"],
                  data["identifier"],
                  data["prompt"],
                  data["generated_email"],
                  data["credits_used"],
                  json.dumps(data["performance_data"])
              ))
    conn.commit()
    conn.close()

def get_user_history(user_id: str, limit: int = 10):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
    history = c.fetchall()
    conn.close()
    
    return [{
        "id": row[0],
        "timestamp": row[2],
        "platform": row[3],
        "identifier": row[4],
        "prompt": row[5],
        "generated_email": row[6],
        "credits_used": row[7],
        "performance_data": json.loads(row[8])
    } for row in history]

# Platform scraping functions
async def scrape_linkedin(url: str) -> dict:
    cache_key = hashlib.md5(url.encode()).hexdigest()
    if cache_key in profile_cache:
        return profile_cache[cache_key]

    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        viewport={"width": 1920, "height": 1080}
    )
    page = await context.new_page()
    
    try:
        await page.goto(url, timeout=SCRAPE_TIMEOUT, wait_until="networkidle")
        
        try:
            await page.wait_for_selector(".top-card-layout__entity-info, .sign-in-form", timeout=20000)
        except:
            raise HTTPException(status_code=400, detail="LinkedIn structure not recognized")

        if await page.query_selector(".sign-in-form"):
            raise HTTPException(status_code=403, detail="LinkedIn requires authentication")

        profile_data = {
            "name": await page.evaluate('''() => {
                return document.querySelector(".top-card-layout__title")?.innerText?.trim() || "";
            }'''),
            "headline": await page.evaluate('''() => {
                return document.querySelector(".top-card-layout__headline")?.innerText?.trim() || "";
            }'''),
            "location": await page.evaluate('''() => {
                return document.querySelector(".top-card__subline-item")?.innerText?.trim() || "";
            }'''),
            "about": await page.evaluate('''() => {
                return document.querySelector(".core-section-container__content .bio")?.innerText?.trim() || "";
            }'''),
            "recent_activity": []
        }

        try:
            await page.click('button:has-text("Show more")', timeout=3000)
            await asyncio.sleep(1)
        except:
            pass

        posts = await page.query_selector_all(".feed-shared-update-v2")
        for post in posts[:5]:
            try:
                text = await post.evaluate('node => node.innerText')
                if text:
                    clean_text = re.sub(r'\s+', ' ', text).strip()
                    profile_data["recent_activity"].append(clean_text[:500])
            except:
                continue

        profile_cache[cache_key] = profile_data
        return profile_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LinkedIn scraping failed: {str(e)}")
    finally:
        await context.close()

async def scrape_twitter(url: str) -> dict:
    cache_key = hashlib.md5(url.encode()).hexdigest()
    if cache_key in profile_cache:
        return profile_cache[cache_key]

    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        viewport={"width": 1920, "height": 1080}
    )
    page = await context.new_page()
    
    try:
        await page.goto(url, timeout=SCRAPE_TIMEOUT, wait_until="networkidle")
        
        try:
            await page.wait_for_selector("[data-testid='UserName']", timeout=20000)
        except:
            raise HTTPException(status_code=400, detail="Twitter structure not recognized")

        profile_data = {
            "name": await page.evaluate('''() => {
                return document.querySelector("[data-testid='UserName']")?.innerText?.trim() || "";
            }'''),
            "handle": await page.evaluate('''() => {
                return document.querySelector("[data-testid='UserHandle']")?.innerText?.trim() || "";
            }'''),
            "bio": await page.evaluate('''() => {
                return document.querySelector("[data-testid='UserDescription']")?.innerText?.trim() || "";
            }'''),
            "location": await page.evaluate('''() => {
                return document.querySelector("[data-testid='UserLocation']")?.innerText?.trim() || "";
            }'''),
            "website": await page.evaluate('''() => {
                return document.querySelector("[data-testid='UserUrl']")?.innerText?.trim() || "";
            }'''),
            "join_date": await page.evaluate('''() => {
                return document.querySelector("[data-testid='UserJoinDate']")?.innerText?.trim() || "";
            }'''),
            "tweets": []
        }

        # Scroll to load more tweets
        for _ in range(2):
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            await asyncio.sleep(1)

        tweets = await page.query_selector_all("[data-testid='tweet']")
        for tweet in tweets[:10]:
            try:
                text = await tweet.evaluate('node => node.innerText')
                if text:
                    clean_text = re.sub(r'\s+', ' ', text).strip()
                    profile_data["tweets"].append(clean_text[:280])
            except:
                continue

        profile_cache[cache_key] = profile_data
        return profile_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Twitter scraping failed: {str(e)}")
    finally:
        await context.close()

# AI Generation Functions
def generate_prompt(profile: dict, request: TargetRequest) -> str:
    template = TEMPLATES.get(request.template_type or "social_engineering", TEMPLATES["social_engineering"])
    tactic = SOCIAL_ENGINEERING_TACTICS.get(request.social_engineering_tactic, SOCIAL_ENGINEERING_TACTICS["authority"])
    
    # Platform-specific context
    context = ""
    if request.platform == "linkedin":
        activity_context = "\nRecent Activity:\n- " + "\n- ".join(profile["recent_activity"][:3]) if profile.get("recent_activity") else ""
        context = f"""
Target Profile (LinkedIn):
- Name: {profile.get('name', 'Unknown')}
- Headline: {profile.get('headline', '')}
- Location: {profile.get('location', '')}
- About: {profile.get('about', '')[:300]}{activity_context}"""
    elif request.platform == "twitter":
        tweets_context = "\nRecent Tweets:\n- " + "\n- ".join(profile["tweets"][:3]) if profile.get("tweets") else ""
        context = f"""
Target Profile (Twitter):
- Name: {profile.get('name', 'Unknown')}
- Handle: {profile.get('handle', '')}
- Bio: {profile.get('bio', '')}
- Location: {profile.get('location', '')}
- Website: {profile.get('website', '')}
- Join Date: {profile.get('join_date', '')}{tweets_context}"""
    else:  # Custom
        context = f"Custom Target Data:\n{json.dumps(profile, indent=2)}"
    
    return f"""
**RED TEAM PHISHING GENERATION - ADVANCED MODE**
Template: {template['name']} ({template['description']})
Psychological Tactic: {tactic['name']} ({tactic['description']})

{context}

Requirements:
1. EXPLOIT: {tactic['name']} tactic effectively
2. Reference specific profile details
3. Create URGENT but plausible scenario requiring immediate action
4. Include [LINK] placeholder in natural context
5. Professional yet friendly tone matching target's industry
6. Subject line < 50 characters
7. Include 1-2 human imperfections (slight typo, colloquial phrase)
8. Add plausible reason why email looks informal

Additional Instructions:
{request.custom_prompt or "Focus on exploiting human psychology rather than technical vulnerabilities"}
"""

def generate_email(prompt: str) -> str:
    try:
        with model.chat_session():
            response = model.generate(
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temp=TEMPERATURE,
                streaming=False,
                n_batch=512,
                repeat_penalty=1.1
            )
        return response.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

# Analytics Functions
def generate_analytics_report(history: list):
    if not history:
        return None
    
    df = pd.DataFrame(history)
    
    # Generate some metrics
    report = {
        "total_emails": len(df),
        "credits_used": df["credits_used"].sum(),
        "platform_distribution": df["platform"].value_counts().to_dict(),
        "templates_used": {},
        "effectiveness_estimate": round(np.mean([TEMPLATES.get(t, {"effectiveness": 0.75})["effectiveness"] 
                                               for t in df["performance_data"].apply(lambda x: x.get("template_type", "social_engineering"))]), 2)
    }
    
    # Generate a chart
    plt.figure(figsize=(10, 6))
    df["platform"].value_counts().plot(kind='bar', color='skyblue')
    plt.title("Platform Distribution")
    plt.xlabel("Platform")
    plt.ylabel("Count")
    plt.tight_layout()
    
    # Save the chart
    chart_path = "static/analytics_chart.png"
    plt.savefig(chart_path)
    plt.close()
    
    report["chart_path"] = chart_path
    return report

# API Endpoints
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/api/generate")
@limiter.limit("10/minute")
async def generate_email_endpoint(
    request: Request,
    target_request: TargetRequest,
    api_key: str = Header(default="demo_key"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    start_time = time.time()
    user = get_user(api_key)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if user["credits"] <= 0:
        raise HTTPException(status_code=402, detail="Insufficient credits")
    
    # Update credits immediately
    new_credits = user["credits"] - 1
    update_user_credits(user["id"], new_credits)
    
    try:
        # Scrape profile based on platform
        profile = {}
        if target_request.platform == "linkedin":
            if not re.match(r'^https?://(www\.)?linkedin\.com/in/.+', target_request.identifier):
                raise HTTPException(status_code=400, detail="Invalid LinkedIn URL")
            profile = await scrape_linkedin(target_request.identifier)
        elif target_request.platform == "twitter":
            if not re.match(r'^https?://(www\.)?twitter\.com/.+', target_request.identifier):
                raise HTTPException(status_code=400, detail="Invalid Twitter URL")
            profile = await scrape_twitter(target_request.identifier)
        else:  # Custom
            try:
                profile = json.loads(target_request.identifier)
            except:
                raise HTTPException(status_code=400, detail="Invalid custom data format")
        
        scrape_time = time.time() - start_time
        
        # Generate prompt
        prompt = generate_prompt(profile, target_request)
        
        # Generate email
        email = generate_email(prompt)
        gen_time = time.time() - start_time - scrape_time
        
        # Add disclaimer
        disclaimer = (
            "\n\n---\nDISCLAIMER: This content is generated for authorized security "
            "testing only. Unauthorized phishing attacks are illegal. By using this service, "
            "you agree to comply with all applicable laws and ethical guidelines."
        )
        
        # Prepare response
        response_data = {
            "email": email + disclaimer,
            "profile_data": {
                "name": profile.get("name") or "Unknown",
                "headline": profile.get("headline") or profile.get("bio") or "No title"
            },
            "performance": {
                "scrape_time": f"{scrape_time:.2f}s",
                "generation_time": f"{gen_time:.2f}s",
                "total_time": f"{time.time() - start_time:.2f}s"
            },
            "credits_remaining": new_credits,
            "status": "success"
        }
        
        # Save to history in background
        history_data = {
            "platform": target_request.platform,
            "identifier": target_request.identifier,
            "prompt": prompt,
            "generated_email": response_data["email"],
            "credits_used": 1,
            "performance_data": {
                "scrape_time": scrape_time,
                "generation_time": gen_time,
                "template_type": target_request.template_type or "social_engineering",
                "tactic": target_request.social_engineering_tactic
            }
        }
        background_tasks.add_task(save_to_history, user["id"], history_data)
        
        return response_data
        
    except HTTPException:
        # Refund credits on error
        update_user_credits(user["id"], user["credits"])
        raise
    except Exception as e:
        update_user_credits(user["id"], user["credits"])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/campaign")
@limiter.limit("5/minute")
async def create_campaign(
    request: Request,
    campaign_request: CampaignRequest,
    api_key: str = Header(default="demo_key"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    user = get_user(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    required_credits = len(campaign_request.targets)
    if user["credits"] < required_credits:
        raise HTTPException(status_code=402, detail=f"Insufficient credits. Required: {required_credits}")
    
    # Deduct credits
    new_credits = user["credits"] - required_credits
    update_user_credits(user["id"], new_credits)
    
    campaign_id = str(uuid.uuid4())
    
    # Process campaign in background
    async def process_campaign():
        results = []
        for target in campaign_request.targets:
            try:
                # Similar to single generation but without response
                profile = {}
                if target.platform == "linkedin":
                    profile = await scrape_linkedin(target.identifier)
                elif target.platform == "twitter":
                    profile = await scrape_twitter(target.identifier)
                else:
                    profile = json.loads(target.identifier)
                
                prompt = generate_prompt(profile, target)
                email = generate_email(prompt)
                
                # Save to history
                history_data = {
                    "platform": target.platform,
                    "identifier": target.identifier,
                    "prompt": prompt,
                    "generated_email": email,
                    "credits_used": 1,
                    "performance_data": {
                        "template_type": target.template_type or "social_engineering",
                        "tactic": target.social_engineering_tactic,
                        "campaign_id": campaign_id
                    }
                }
                save_to_history(user["id"], history_data)
                
                results.append({
                    "target": target.identifier,
                    "status": "success",
                    "email": email
                })
            except Exception as e:
                results.append({
                    "target": target.identifier,
                    "status": "error",
                    "error": str(e)
                })
        
        # Save campaign results to DB
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''INSERT INTO campaigns 
                     (id, user_id, name, targets, schedule, status, results)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (
                      campaign_id,
                      user["id"],
                      campaign_request.name,
                      json.dumps([t.dict() for t in campaign_request.targets]),
                      json.dumps(campaign_request.schedule),
                      "completed",
                      json.dumps(results)
                  ))
        conn.commit()
        conn.close()
    
    # Add to task queue
    await task_queue.put({
        "function": process_campaign,
        "args": []
    })
    
    return {
        "campaign_id": campaign_id,
        "status": "processing",
        "message": "Campaign is being processed in the background",
        "credits_remaining": new_credits
    }

@app.get("/api/history")
@limiter.limit("20/minute")
async def get_history(
    request: Request,
    api_key: str = Header(default="demo_key"),
    limit: int = 10
):
    user = get_user(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    history = get_user_history(user["id"], limit)
    return history

@app.get("/api/analytics")
@limiter.limit("10/minute")
async def get_analytics(
    request: Request,
    api_key: str = Header(default="demo_key")
):
    user = get_user(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    history = get_user_history(user["id"], 100)
    report = generate_analytics_report(history)
    
    return report or {"message": "No analytics data available"}

@app.get("/api/templates")
async def get_templates():
    return TEMPLATES

@app.get("/api/tactics")
async def get_tactics():
    return SOCIAL_ENGINEERING_TACTICS

@app.get("/api/user")
@limiter.limit("30/minute")
async def get_user_info(
    request: Request,
    api_key: str = Header(default="demo_key")
):
    user = get_user(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "credits": user["credits"],
        "tier": user["tier"]
    }

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=60)
