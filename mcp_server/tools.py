"""
External tool integrations for MCP server
"""

import os
import asyncio
import json
from typing import Dict, Any, List, Optional
import httpx
from loguru import logger


class BraveSearchTool:
    """Brave Search API integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('BRAVE_API_KEY')
        self.base_url = "https://api.search.brave.com/res/v1"
        self.client = httpx.AsyncClient()
    
    async def search_web(self, query: str, count: int = 5, country: str = "US") -> Dict[str, Any]:
        """Search the web using Brave Search API"""
        if not self.api_key:
            return {"error": "Brave API key not configured"}
        
        try:
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key
            }
            
            params = {
                "q": query,
                "country": country,
                "search_lang": "en",
                "count": count,
                "offset": 0,
                "safesearch": "moderate",
                "freshness": "pw",  # past week
                "text_decorations": False,
                "spellcheck": True
            }
            
            response = await self.client.get(
                f"{self.base_url}/web/search",
                headers=headers,
                params=params,
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Format results
                results = []
                if "web" in data and "results" in data["web"]:
                    for result in data["web"]["results"]:
                        results.append({
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "description": result.get("description", ""),
                            "published": result.get("age", "")
                        })
                
                return {
                    "success": True,
                    "query": query,
                    "results": results,
                    "total_count": len(results)
                }
            else:
                return {"error": f"Search failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Brave search error: {e}")
            return {"error": f"Search failed: {str(e)}"}


class WeatherTool:
    """Weather API integration (using OpenWeatherMap or similar)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('OPENWEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.client = httpx.AsyncClient()
    
    async def get_weather(self, location: str, units: str = "metric") -> Dict[str, Any]:
        """Get current weather for a location"""
        if not self.api_key:
            return {"error": "OpenWeather API key not configured"}
        
        try:
            params = {
                "q": location,
                "appid": self.api_key,
                "units": units
            }
            
            response = await self.client.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    "success": True,
                    "location": data["name"],
                    "country": data["sys"]["country"],
                    "temperature": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "humidity": data["main"]["humidity"],
                    "description": data["weather"][0]["description"],
                    "wind_speed": data["wind"]["speed"],
                    "units": units
                }
            else:
                return {"error": f"Weather request failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return {"error": f"Weather request failed: {str(e)}"}


class CalculatorTool:
    """Mathematical calculation tool"""
    
    @staticmethod
    async def calculate(expression: str) -> Dict[str, Any]:
        """Safely evaluate mathematical expressions"""
        try:
            # Simple math expressions only - security conscious
            import ast
            import operator
            
            # Allowed operations
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }
            
            def eval_expr(node):
                if isinstance(node, ast.Constant):  # numbers
                    return node.value
                elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
                    return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):  # <operator> <operand>
                    return ops[type(node.op)](eval_expr(node.operand))
                else:
                    raise TypeError(f"Unsupported operation: {type(node)}")
            
            # Parse and evaluate
            tree = ast.parse(expression, mode='eval')
            result = eval_expr(tree.body)
            
            return {
                "success": True,
                "expression": expression,
                "result": result
            }
            
        except Exception as e:
            return {
                "error": f"Calculation failed: {str(e)}",
                "expression": expression
            }


class NewsSearcTool:
    """News search using NewsAPI or similar"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2"
        self.client = httpx.AsyncClient()
    
    async def search_news(self, query: str, count: int = 5, language: str = "en") -> Dict[str, Any]:
        """Search for news articles"""
        if not self.api_key:
            return {"error": "News API key not configured"}
        
        try:
            params = {
                "q": query,
                "language": language,
                "sortBy": "publishedAt",
                "pageSize": count,
                "apiKey": self.api_key
            }
            
            response = await self.client.get(
                f"{self.base_url}/everything",
                params=params,
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                results = []
                for article in data.get("articles", []):
                    results.append({
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", {}).get("name", ""),
                        "published": article.get("publishedAt", "")
                    })
                
                return {
                    "success": True,
                    "query": query,
                    "results": results,
                    "total_results": data.get("totalResults", 0)
                }
            else:
                return {"error": f"News search failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"News search error: {e}")
            return {"error": f"News search failed: {str(e)}"}


class ToolRegistry:
    """Registry for all external tools"""
    
    def __init__(self):
        self.brave_search = BraveSearchTool()
        self.weather = WeatherTool()
        self.calculator = CalculatorTool()
        self.news = NewsSearcTool()
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name"""
        try:
            if tool_name == "web_search":
                query = kwargs.get("query", "")
                count = kwargs.get("count", 5)
                return await self.brave_search.search_web(query, count)
            
            elif tool_name == "weather":
                location = kwargs.get("location", "")
                units = kwargs.get("units", "metric")
                return await self.weather.get_weather(location, units)
            
            elif tool_name == "calculate":
                expression = kwargs.get("expression", "")
                return await self.calculator.calculate(expression)
            
            elif tool_name == "news_search":
                query = kwargs.get("query", "")
                count = kwargs.get("count", 5)
                return await self.news.search_news(query, count)
            
            else:
                return {"error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [
            {
                "name": "web_search",
                "description": "Search the web using Brave Search",
                "parameters": ["query", "count"]
            },
            {
                "name": "weather",
                "description": "Get current weather for a location",
                "parameters": ["location", "units"]
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": ["expression"]
            },
            {
                "name": "news_search",
                "description": "Search for news articles",
                "parameters": ["query", "count"]
            }
        ]


# Global tool registry instance
tool_registry = ToolRegistry()