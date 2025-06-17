# AI-PhishBait: AI-Powered Phishing Simulation Framework

## Project Overview

**Tagline:** "Turn Your Phishing Emails Into AI-Generated Masterpieces"

AI-PhishBait is an advanced red teaming tool that generates hyper-personalized phishing emails by leveraging AI and OSINT techniques. The system scrapes target profiles from LinkedIn/Twitter and uses natural language generation to create convincing phishing lures for authorized security testing.

## Key Features

- **Hyper-Personalized Email Generation**: Creates tailored phishing emails using target's public profile data
- **OSINT Integration**: Automatically scrapes LinkedIn/Twitter profiles for personalization data
- **AI-Powered Content**: Uses GPT4All (Yi-1.5-9B model) for natural-sounding email generation
- **Red Team Focus**: Designed specifically for authorized penetration testing
- **Web Interface**: User-friendly dashboard for generating and managing campaigns
- **API Support**: REST API for integration with other security tools

## Technical Stack

### Core Components
- **Backend**: Python + FastAPI
- **Frontend**: HTML5, TailwindCSS, Vanilla JS
- **AI Model**: Yi-1.5-9B-Chat-16K-Q4_0 (GPT4All)
- **Web Scraping**: Playwright + BeautifulSoup
- **Database**: SQLite (for user data and history)

### Development Tools
- Playwright for browser automation
- Pydantic for data validation
- NetworkX for attack graph visualization (in advanced version)
- Matplotlib for analytics charts

## Project Structure
phishbait/
├── backend/ # FastAPI application
│ ├── main.py # Core API implementation
│ ├── requirements.txt # Python dependencies
│ └── ... # Other backend files
├── frontend/ # Web interface
│ ├── index.html # Main interface
│ ├── styles/ # CSS files
│ └── ... # Other frontend assets
├── docs/ # Documentation
├── samples/ # Example outputs
└── README.md # This file

## My Contributions

As the sole developer of this project, I:

- Designed and implemented the full-stack architecture
- Developed the AI prompt engineering system for generating convincing phishing emails
- Created the OSINT scraping functionality using Playwright
- Built the REST API with FastAPI
- Designed the responsive frontend interface
- Implemented security measures and ethical use safeguards
- Optimized performance for the local AI model

## Skills Demonstrated

- **Programming**: Python, JavaScript, HTML/CSS
- **Frameworks**: FastAPI, TailwindCSS
- **AI/ML**: Prompt engineering, NLP, GPT4All integration
- **Security**: Red team tool development, ethical hacking principles
- **Web Scraping**: Playwright, BeautifulSoup
- **DevOps**: Local deployment, dependency management

## How to Run Locally

*Note: This project was developed for educational purposes and not deployed publicly.*

1. Clone the repository:
   ```bash
   git clone https://github.com/varma19tt/phishbait.git
   cd phishbait
2.  Set up backend:
    cd backend
    pip install -r requirements.txt
    playwright install chromium
 3. Download the AI model (Yi-1.5-9B-Chat-16K-Q4_0.gguf) and place in backend directory
 4. Run the backend:
      ```bash
      uvicorn main:app --reload
 5. Open frontend in browser:
      Open frontend/index.html in browser
    
##Ethical Considerations
  This tool was developed strictly for:

  - Authorized penetration testing
  - Security research
  - Red team training exercises

Unauthorized use for actual phishing attacks is strictly prohibited and illegal. The project includes built-in ethical use disclaimers in all generated content.

##Future Enhancements
Planned improvements include:
  - Twitter integration for additional OSINT data
  - Campaign management system
  - Advanced analytics dashboard
  - Enterprise API support
  - Detectability scoring system
