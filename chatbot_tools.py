import ipaddress
import os
import smtplib
import socket
import sqlite3
from datetime import datetime
from email.message import EmailMessage
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
 
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import (
    ArxivQueryRun,
    DuckDuckGoSearchResults,
    PubmedQueryRun,
    WikipediaQueryRun,
    YouTubeSearchTool,
)
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper

REQUEST_TIMEOUT = 10
MAX_CHARS = 6000
TOOLS_DB = "tools_data.db"

## tool for web search
web_search = DuckDuckGoSearchResults(
    name="web_search",
    description=(
        "Search the web for current events, news, prices, or anything that may "
        "have changed recently. Input: a short search query. Returns snippets and links."
    ),
    max_results=5,
    api_wrapper=DuckDuckGoSearchAPIWrapper(region="in-en"),
)

## tool for wikipedia 
wikipedia = WikipediaQueryRun(
    name="wikipedia",
    description=(
        "Look up stable encyclopedic facts about people, places, concepts and "
        "history. Input: a topic name."
    ),
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000),
)


## tool for dealing with research papers
arxiv = ArxivQueryRun(
    name="arxiv_search",
    description="Find research papers (physics, CS, maths, AI) on arXiv. Input: a topic or paper title.",
)

## tool for search related to medical and life science
pubmed = PubmedQueryRun(
    name="pubmed_search",
    description="Search medical and life-science literature on PubMed. Input: a medical topic.",
)

## tool for youtube search
youtube = YouTubeSearchTool(
    name="youtube_search",
    description="Find YouTube videos. Input: 'search terms' or 'search terms,number_of_results'.",
)

def _is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        for info in socket.getaddrinfo(parsed.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except socket.gaierror:
        return False
    return True


## tool for fetch url and extract text
@tool
def fetch_url(url: str) -> str:
    """
    Read the text content of a web page. Use when the user pastes a link
    and wants it summarised or asks a question about it.
    """
    if not _is_safe_url(url):
        return "Error: URL is invalid or not allowed."
    try:
        resp = requests.get(
            url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0 (ChatBot)"}
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Error fetching page: {e}"
 
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
    text = " ".join(soup.get_text(separator=" ").split())
    return f"Title: {title}\n\n{text[:MAX_CHARS]}"

@tool
def get_current_datetime(timezone: str = "Asia/Kolkata") -> str:
    """Get the current date and time. Optional IANA timezone, e.g. 'Asia/Kolkata',
    'America/New_York', 'Europe/London'."""
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception:
        return f"Unknown timezone '{timezone}'. Use an IANA name like 'Asia/Kolkata'."
    return now.strftime(f"%A, %d %B %Y, %H:%M:%S ({timezone})")
 

## tool to fetch weather 
@tool
def get_weather(city: str) -> str:
    """Get current weather and a 3-day forecast for a city. Input: city name."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=REQUEST_TIMEOUT,
        ).json()
        if not geo.get("results"):
            return f"Could not find a city named '{city}'."
        place = geo["results"][0]
 
        data = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "forecast_days": 3,
                "timezone": "auto",
            },
            timeout=REQUEST_TIMEOUT,
        ).json()
    except (requests.RequestException, ValueError) as e:
        return f"Weather service error: {e}"
 
    cur, daily = data["current"], data["daily"]
    lines = [
        f"Weather in {place['name']}, {place.get('country', '')}:",
        f"Now: {cur['temperature_2m']}°C, humidity {cur['relative_humidity_2m']}%, "
        f"wind {cur['wind_speed_10m']} km/h, precipitation {cur['precipitation']} mm",
        "Forecast:",
    ]
    for i, day in enumerate(daily["time"]):
        lines.append(
            f"  {day}: {daily['temperature_2m_min'][i]}°C to {daily['temperature_2m_max'][i]}°C, "
            f"rain {daily['precipitation_sum'][i]} mm"
        )
    return "\n".join(lines)
 

## tool to convert currency 
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert money between currencies using live rates.
    Use 3-letter codes, e.g. amount=100, from_currency='USD', to_currency='INR'."""
    try:
        resp = requests.get(
            "https://api.frankfurter.dev/v1/latest",
            params={
                "amount": amount,
                "from": from_currency.upper(),
                "to": to_currency.upper(),
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        return f"Currency service error: {e}"
    converted = data["rates"][to_currency.upper()]
    return f"{amount} {from_currency.upper()} = {converted} {to_currency.upper()} (rate date: {data['date']})"
 
 
stock_news = YahooFinanceNewsTool(
    name="stock_news",
    description="Get recent news for a stock. Input: a ticker symbol such as 'AAPL' or 'TCS.NS'.",
)
 
 
## too to use calculator
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression exactly, e.g. '23 * (4 + 5)', 'sqrt(144)',
    '2**10'. Always use this for arithmetic instead of calculating yourself."""
    import numexpr
 
    try:
        result = numexpr.evaluate(expression.strip()).item()
        return str(result)
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"
 

## tool to convert unit 
@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units, e.g. value=5, from_unit='km', to_unit='miles'.
    Works for length, mass, temperature (degC, degF, kelvin), volume, speed, etc."""
    import pint
 
    ureg = pint.UnitRegistry()
    try:
        qty = ureg.Quantity(value, from_unit).to(to_unit)
        return f"{value} {from_unit} = {qty.magnitude:.6g} {to_unit}"
    except Exception as e:
        return f"Conversion failed: {e}"

tools = [
    # search & knowledge
    web_search,
    wikipedia,
    arxiv,
    pubmed,
    youtube,
    fetch_url,
    # real-time data
    get_current_datetime,
    get_weather,
    convert_currency,
    stock_news,
    # computation
    calculator,
    unit_converter,
]