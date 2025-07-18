# Raseed

## Description
Brief description of your project.

## Setup

1. Clone the repository
2. Copy `.env.template` to `.env` and fill in your configuration
3. Create virtual environment: `python -m venv .venv`
4. Activate virtual environment: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`

## Usage

### Client Setup
Firebase console configs in .env

### Server Setup

**SERVER FILE**: server/run.py
```
.venv\Scripts\activate
pip install -r requirements.txt

#important to do this, ask for serviceAccountKey.json. Must be kept inside server folder
$env:FIREBASE_CREDENTIALS = Get-Content "FULL PATH TO serviceAccountKey.json" -Raw 
export FIREBASE_CREDENTIALS="$(cat FULL PATH TO serviceAccountKey.json)"


uvicorn server.run:app --host 0.0.0.0 --port 8000


## Must have firebase-sdk.json file in root
```
## Scripts
For market_analytics.py you need to add the gcp project id to your terminal
$env:GCP_PROJECT_ID="your-gcp-project-id"
export GCP_PROJECT_ID="your-gcp-project-id"

<details>
<summary><strong>📁 Project Structure</strong></summary>

```
Raseed/
├── agents/
│   └── __init__.py
├── client/
│   ├── node_modules/
│   ├── public/
│   │   └── vite.svg
│   ├── src/
│   │   ├── assets/
│   │   │   └── react.svg
│   │   ├── components/
│   │   │   ├── ActionButton.jsx
│   │   │   ├── PassCard.jsx
│   │   │   ├── QuickActions.jsx
│   │   │   ├── RecentPasses.jsx
│   │   │   └── StatsCard.jsx
│   │   ├── constants/
│   │   │   └── pages.jsx
│   │   ├── pages/
│   │   │   ├── AskRaseed.jsx
│   │   │   ├── CaptureReceipt.jsx
│   │   │   └── Dashboard.jsx
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── README.md
│   ├── eslint.config.js
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   └── vite.config.js
├── config/
│   └── init.py
├── core/
│   └── init.py
├── data/
├── docs/
├── models/
│   └── init.py
├── server/
│   └── init.py
│   └── run.py
├── tests/
├── ui/
│   └── init.py
├── utils/
│   └── init.py
├── LICENSE
├── README.md
```
</details>

## Contributing

TODO: Add contributing guidelines

## License

TODO: Add license information
