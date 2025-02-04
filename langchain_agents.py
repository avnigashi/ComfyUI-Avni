"""
custom_nodes/langchain_nodes.py
Simplified but powerful ComfyUI nodes using LangChain
"""

import os
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.agents import Tool, AgentExecutor, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult, AgentAction, AgentFinish
from langchain_core.tools import Tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.llms.ollama import Ollama

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('LangChainNodes')

class ComfyCallbackHandler(BaseCallbackHandler):
    """Callback handler for tracking LangChain events"""

    def __init__(self):
        self.events = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self.events.append({"type": "llm_start", "prompts": prompts})

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.events.append({"type": "llm_end", "response": response})

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        self.events.append({"type": "tool_start", "input": input_str})

    def on_agent_action(self, action: AgentAction, **kwargs) -> Any:
        self.events.append({"type": "agent_action", "action": action})

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        self.events.append({"type": "agent_finish", "finish": finish})

class OllamaLLMNode:
    """Node for setting up Ollama LLM using LangChain"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "llama2"}),
                "base_url": ("STRING", {"default": "http://localhost:11434"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("LLM",)
    FUNCTION = "setup_ollama_llm"
    CATEGORY = "LangChain"

    def setup_ollama_llm(self, model_name: str, base_url: str, temperature: float,
                         repeat_penalty: float, top_k: int, top_p: float):
        callback_handler = ComfyCallbackHandler()

        try:
            llm = Ollama(
                base_url=base_url,
                model=model_name,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
                top_k=top_k,
                top_p=top_p,
                callbacks=[callback_handler],
            )
            return (llm,)

        except Exception as e:
            logger.error(f"Error setting up Ollama LLM: {str(e)}")
            raise
class LLMNode:
    """Node for setting up LangChain LLMs"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (["openai", "anthropic"], {"default": "openai"}),
                "model_name": (["gpt-4o", "gpt-4o-mini", "claude-3-opus", "claude-3-sonnet"], {"default": "gpt-3.5-turbo"}),
                "api_key": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("LLM",)
    FUNCTION = "setup_llm"
    CATEGORY = "LangChain"

    def setup_llm(self, provider: str, model_name: str, api_key: str, temperature: float):
        if not api_key:
            raise ValueError("API key is required")

        callback_handler = ComfyCallbackHandler()

        try:
            if provider == "openai":
                llm = ChatOpenAI(
                    model_name=model_name,
                    openai_api_key=api_key,
                    temperature=temperature,
                    callbacks=[callback_handler]
                )
            else:  # anthropic
                llm = ChatAnthropic(
                    model=model_name,
                    anthropic_api_key=api_key,
                    temperature=temperature,
                    callbacks=[callback_handler]
                )

            return (llm,)

        except Exception as e:
            logger.error(f"Error setting up LLM: {str(e)}")
            raise

class MemoryNode:
    """Node for setting up LangChain memory systems"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory_type": (["buffer", "summary", "vector"], {"default": "buffer"}),
                "memory_key": ("STRING", {"default": "chat_history"}),
            },
            "optional": {
                "openai_api_key": ("STRING", {"default": ""}),  # Required for vector memory
            }
        }

    RETURN_TYPES = ("MEMORY",)
    FUNCTION = "setup_memory"
    CATEGORY = "LangChain"

    def setup_memory(self, memory_type: str, memory_key: str, openai_api_key: str = ""):
        try:
            if memory_type == "buffer":
                memory = ConversationBufferMemory(
                    memory_key=memory_key,
                    return_messages=True
                )
            elif memory_type == "summary":
                memory = ConversationSummaryMemory(
                    memory_key=memory_key,
                    return_messages=True
                )
            else:  # vector
                if not openai_api_key:
                    raise ValueError("OpenAI API key required for vector memory")

                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vectorstore = FAISS.from_texts([""], embeddings)

                memory = ConversationBufferMemory(
                    memory_key=memory_key,
                    return_messages=True,
                    output_key="output"
                )

            return (memory,)

        except Exception as e:
            logger.error(f"Error setting up memory: {str(e)}")
            raise


class SearchToolNode:
    """Node for setting up search tools"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "search_provider": (["google", "tavily", "serper"], {"default": "google"}),
                "k": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "google_api_key": ("STRING", {"default": ""}),
                "google_cse_id": ("STRING", {"default": ""}),
                "tavily_api_key": ("STRING", {"default": ""}),
                "serper_api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("TOOL",)
    FUNCTION = "setup_search"
    CATEGORY = "LangChain"

    def setup_search(self, search_provider: str, k: int = 5,
                    google_api_key: str = "", google_cse_id: str = "",
                    tavily_api_key: str = "", serper_api_key: str = ""):
        try:
            if search_provider == "google":
                if not google_api_key or not google_cse_id:
                    raise ValueError("Both Google API key and CSE ID are required for Google Search")

                search = GoogleSearchAPIWrapper(
                    google_api_key=google_api_key,
                    google_cse_id=google_cse_id,
                    k=k
                )
                description = "Search Google for recent information. Returns top results."

            elif search_provider == "tavily":
                if not tavily_api_key:
                    raise ValueError("Tavily API key is required")

                search = TavilySearchAPIWrapper(
                    tavily_api_key=tavily_api_key,
                    k=k
                )
                description = "Search the web using Tavily's AI-powered search. Good for recent and analytical results."

            else:  # serper
                if not serper_api_key:
                    raise ValueError("Serper API key is required")

                os.environ["SERPER_API_KEY"] = serper_api_key
                search = SerpAPIWrapper(k=k)
                description = "Search the web using Serper. Provides comprehensive search results."

            tool = Tool(
                name=f"{search_provider}_search",
                description=description,
                func=search.run
            )

            return (tool,)

        except Exception as e:
            logger.error(f"Error setting up search tool: {str(e)}")
            raise

class WebSearchNode:
    """Node for web search implementations using googlesearch-python and tavily"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "search_provider": (["google", "tavily", "serper"], {"default": "google"}),
                "max_results": ("INT", {"default": 5, "min": 1, "max": 100}),
                "search_depth": (["basic", "advanced"], {"default": "basic"}),
                "search_topic": (["general", "news"], {"default": "general"}),
            },
            "optional": {
                # Google-specific options
                "language": ("STRING", {"default": "en"}),
                "region": ("STRING", {"default": "us"}),
                "safe_search": ("BOOLEAN", {"default": True}),
                "advanced_google": ("BOOLEAN", {"default": True}),
                "sleep_interval": ("FLOAT", {"default": 0.0}),
                "proxy": ("STRING", {"default": ""}),

                # Tavily-specific options
                "tavily_api_key": ("STRING", {"default": ""}),
                "include_answer": ("BOOLEAN", {"default": False}),
                "include_raw_content": ("BOOLEAN", {"default": False}),
                "include_images": ("BOOLEAN", {"default": False}),
                "days_limit": ("INT", {"default": 3}),
                "include_domains": ("STRING", {"default": ""}),  # Comma-separated list
                "exclude_domains": ("STRING", {"default": ""}),  # Comma-separated list

                # Serper-specific options
                "serper_api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("TOOL",)
    FUNCTION = "create_search_tool"
    CATEGORY = "LangChain"

    def create_search_tool(self, search_provider: str, max_results: int = 5,
                         search_depth: str = "basic", search_topic: str = "general",
                         language: str = "en", region: str = "us",
                         safe_search: bool = True, advanced_google: bool = True,
                         sleep_interval: float = 0.0, proxy: str = "",
                         tavily_api_key: str = "", include_answer: bool = False,
                         include_raw_content: bool = False, include_images: bool = False,
                         days_limit: int = 3, include_domains: str = "",
                         exclude_domains: str = "", serper_api_key: str = "") -> tuple:
        """Creates a search tool that can be used by the agent"""

        def search_func(query: str) -> str:
            try:
                if search_provider == "google":
                    return self._google_search(
                        query=query,
                        max_results=max_results,
                        language=language,
                        region=region,
                        safe_search=safe_search,
                        advanced=advanced_google,
                        sleep_interval=sleep_interval,
                        proxy=proxy
                    )
                elif search_provider == "tavily":
                    return self._tavily_search(
                        query=query,
                        api_key=tavily_api_key,
                        max_results=max_results,
                        search_depth=search_depth,
                        topic=search_topic,
                        include_answer=include_answer,
                        include_raw_content=include_raw_content,
                        include_images=include_images,
                        days=days_limit,
                        include_domains=include_domains.split(",") if include_domains else None,
                        exclude_domains=exclude_domains.split(",") if exclude_domains else None
                    )
                else:  # serper
                    return self._serper_search(query, max_results, serper_api_key)
            except Exception as e:
                return f"Search error: {str(e)}"

        tool = Tool(
            name=f"{search_provider}_search",
            description=f"Search the web using {search_provider}. Input should be a search query.",
            func=search_func
        )

        return (tool,)

    def _google_search(self, query: str, max_results: int, language: str,
                      region: str, safe_search: bool, advanced: bool,
                      sleep_interval: float, proxy: str) -> str:
        """Google search implementation using googlesearch-python"""
        try:
            from googlesearch import search

            # Configure search parameters
            search_params = {
                "num_results": max_results,
                "lang": language,
                "region": region,
                "advanced": advanced,
                "sleep_interval": sleep_interval if sleep_interval > 0 else None
            }

            # Add optional parameters
            if not safe_search:
                search_params["safe"] = None
            if proxy:
                search_params["proxy"] = proxy
                search_params["ssl_verify"] = False

            # Perform search
            results = search(query, **search_params)

            # Format results based on advanced mode
            formatted_results = []
            if advanced:
                for result in results:
                    formatted_results.append(
                        f"Title: {result.title}\n"
                        f"URL: {result.url}\n"
                        f"Description: {result.description}\n"
                    )
            else:
                formatted_results = [str(result) for result in results]

            return "\n---\n".join(formatted_results)

        except ImportError:
            return "Please install googlesearch-python package: pip install googlesearch-python"
        except Exception as e:
            return f"Google search error: {str(e)}"

    def _tavily_search(self, query: str, api_key: str, max_results: int,
                      search_depth: str = "basic", topic: str = "general",
                      include_answer: bool = False, include_raw_content: bool = False,
                      include_images: bool = False, days: int = 3,
                      include_domains: List[str] = None,
                      exclude_domains: List[str] = None) -> str:
        """Tavily search implementation"""
        if not api_key:
            return "Tavily requires an API key"

        try:
            from tavily import TavilyClient

            # Initialize client
            client = TavilyClient(api_key=api_key)

            # Prepare search parameters
            search_params = {
                "search_depth": search_depth,
                "topic": topic,
                "max_results": max_results,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "include_images": include_images,
            }

            # Add optional parameters
            if topic == "news":
                search_params["days"] = days
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains

            # Perform search
            response = client.search(query, **search_params)

            # Format results
            formatted_results = []

            # Add answer if included
            if include_answer and response.get("answer"):
                formatted_results.append(f"Answer: {response['answer']}\n")

            # Add images if included
            if include_images and response.get("images"):
                formatted_results.append("Images:")
                formatted_results.extend([f"- {img}" for img in response["images"]])
                formatted_results.append("")

            # Add search results
            for result in response.get("results", []):
                result_text = [
                    f"Title: {result.get('title', 'No title')}",
                    f"URL: {result.get('url', 'No URL')}",
                    f"Content: {result.get('content', 'No content')}",
                    f"Score: {result.get('score', 0)}"
                ]

                if include_raw_content and result.get("raw_content"):
                    result_text.append(f"Raw Content: {result['raw_content']}")

                if topic == "news" and result.get("published_date"):
                    result_text.append(f"Published: {result['published_date']}")

                formatted_results.append("\n".join(result_text))

            return "\n---\n".join(formatted_results)

        except ImportError:
            return "Please install tavily-python package: pip install tavily-python"
        except Exception as e:
            return f"Tavily search error: {str(e)}"

    def _serper_search(self, query: str, max_results: int, api_key: str) -> str:
        """Serper search implementation"""
        if not api_key:
            return "Serper requires an API key"

        try:
            import requests

            url = "https://google.serper.dev/search"
            headers = {
                'X-API-KEY': api_key,
                'Content-Type': 'application/json'
            }
            payload = {
                'q': query,
                'num': max_results
            }

            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            results = response.json()

            formatted_results = []
            if 'organic' in results:
                for item in results['organic']:
                    formatted_results.append(
                        f"Title: {item.get('title', 'No title')}\n"
                        f"Link: {item.get('link', 'No link')}\n"
                        f"Snippet: {item.get('snippet', 'No snippet')}\n"
                    )

            return "\n---\n".join(formatted_results)

        except Exception as e:
            return f"Serper search error: {str(e)}"

class AgentNode:
    """Node for setting up and running LangChain agents"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm": ("LLM",),
                "tools": ("TOOL",),
                "memory": ("MEMORY",),
                "agent_type": (["zero-shot-react-description", "chat-conversational-react-description"],
                             {"default": "chat-conversational-react-description"}),
                "input": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "system_message": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "AGENT_STATE")
    FUNCTION = "run_agent"
    CATEGORY = "LangChain"

    def run_agent(self, llm: Any, tools: List[Tool], memory: Any, agent_type: str,
                 input: str, system_message: str = ""):
        try:
            # Initialize agent with callbacks
            callback_handler = ComfyCallbackHandler()

            # Custom prompt template for the agent
            prefix = """You are a helpful AI assistant with access to various tools.
            Answer the user's questions thoughtfully using the tools when needed.

            When you need to use a tool, use the following format:
            Thought: I need to use a tool to help with this
            Action: tool_name
            Action Input: input for the tool

            When you have the final answer, use:
            Thought: I have the answer
            Final Answer: your answer here

            Begin!"""

            if system_message:
                prefix = system_message + "\n\n" + prefix

            # Initialize agent with proper configuration
            agent = initialize_agent(
                tools=tools if isinstance(tools, list) else [tools],
                llm=llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True,
                callbacks=[callback_handler],
                agent_kwargs={
                    "prefix": prefix,
                    "format_instructions": """Use the following format:

Thought: I need to figure out what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the tool
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I know what to do
Final Answer: the final answer to the original input question""",
                    "system_message": prefix
                }
            )

            # Run agent
            response = agent.run(input)

            # Collect state
            state = {
                "events": callback_handler.events,
                "memory": memory.chat_memory.messages if hasattr(memory, 'chat_memory') else [],
            }

            return (response, state)

        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            error_response = f"Error: {str(e)}\n\nPlease try rephrasing your question or try again."
            return (error_response, {"error": str(e)})

class ResultFormatterNode:
    """Node for formatting agent results"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "result": ("STRING",),
                "format_type": (["text", "json", "markdown"], {"default": "text"}),  # Changed 'format' to 'format_type'
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "format_result"
    CATEGORY = "LangChain"

    def format_result(self, result: str, format_type: str):  # Changed parameter name to match INPUT_TYPES
        try:
            if format_type == "json":
                try:
                    # First try to parse result as JSON
                    data = json.loads(result)
                    return (json.dumps(data, indent=2),)
                except:
                    # If not valid JSON, wrap in a simple structure
                    formatted = {
                        "result": result
                    }
                    return (json.dumps(formatted, indent=2),)

            elif format_type == "markdown":
                # Simple markdown formatting
                md = f"# Result\n\n{result}\n"
                return (md,)

            else:  # text
                # Return as plain text
                return (str(result),)

        except Exception as e:
            logger.error(f"Error formatting result: {str(e)}")
            return (f"Error formatting result: {str(result)}",)  # Return original with error note


class GoalOrientedAgentNode:
    """Node for running LangChain agents until a specific goal is reached"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm": ("LLM",),
                "tools": ("TOOL",),
                "memory": ("MEMORY",),
                "goal": ("STRING", {"default": "", "multiline": True}),
                "input": ("STRING", {"default": "", "multiline": True}),
                "max_iterations": ("INT", {"default": 5, "min": 1, "max": 20}),
                "success_criteria": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "system_message": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "AGENT_STATE", "BOOLEAN")
    FUNCTION = "run_goal_oriented_agent"
    CATEGORY = "LangChain"

    def run_goal_oriented_agent(self, llm: Any, tools: List[Tool], memory: Any,
                              goal: str, input: str, max_iterations: int,
                              success_criteria: str, system_message: str = ""):
        try:
            callback_handler = ComfyCallbackHandler()

            # Create a goal-oriented system message
            goal_prefix = f"""You are a persistent AI assistant focused on achieving a specific goal.
            Your current goal is: {goal}

            You should:
            1. Keep searching and exploring until you either:
               - Find information that satisfies the goal
               - Reach the maximum number of iterations
               - Determine the goal cannot be achieved
            2. Use the available tools strategically to gather information
            3. Maintain context between iterations using your memory
            4. Evaluate each finding against the success criteria

            Success Criteria:
            {success_criteria}

            After each action, evaluate if you've met the success criteria.
            If yes, provide your final answer.
            If no, continue searching with refined queries based on what you've learned.

            Previous context is available in your memory - use it to avoid repetition.
            """

            if system_message:
                goal_prefix = system_message + "\n\n" + goal_prefix

            # Initialize agent
            agent = initialize_agent(
                tools=tools if isinstance(tools, list) else [tools],
                llm=llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True,
                callbacks=[callback_handler],
                agent_kwargs={
                    "prefix": goal_prefix,
                    "system_message": goal_prefix
                }
            )

            # Track iterations and findings
            iterations = 0
            final_response = ""
            goal_reached = False

            # Run agent iterations
            while iterations < max_iterations and not goal_reached:
                iterations += 1

                # Construct iteration prompt
                if iterations == 1:
                    current_prompt = f"{input}\n\nEvaluate if this satisfies the goal: {goal}"
                else:
                    current_prompt = (
                        f"Continue searching for: {goal}\n"
                        f"Previous findings: {final_response}\n"
                        "Evaluate if we've met the success criteria or need to continue searching."
                    )

                # Run iteration
                response = agent.run(current_prompt)

                # Append to final response
                final_response += f"\n\nIteration {iterations}:\n{response}"

                # Check if goal is reached using the LLM
                goal_check_prompt = f"""
                Based on the success criteria:
                {success_criteria}

                And the current findings:
                {response}

                Has the goal been achieved? Reply with just 'YES' or 'NO'.
                """

                goal_check = llm.predict(goal_check_prompt).strip().upper()
                goal_reached = goal_check == "YES"

                if goal_reached:
                    final_response += "\n\nGoal has been achieved! ✓"
                elif iterations == max_iterations:
                    final_response += "\n\nMaximum iterations reached without achieving goal."

            # Collect state
            state = {
                "events": callback_handler.events,
                "memory": memory.chat_memory.messages if hasattr(memory, 'chat_memory') else [],
                "iterations": iterations,
            }

            return (final_response, state, goal_reached)

        except Exception as e:
            logger.error(f"Error running goal-oriented agent: {str(e)}")
            error_response = f"Error: {str(e)}\n\nPlease try rephrasing your goal or try again."
            return (error_response, {"error": str(e)}, False)

# Node Mappings
NODE_CLASS_MAPPINGS = {
    "LangChainLLM": LLMNode,
    "LangChainMemory": MemoryNode,
    "LangChainSearchTool": WebSearchNode,
    "LangChainAgent": AgentNode,
    "ResultFormatter": ResultFormatterNode,
    "GoalOrientedAgentNode": GoalOrientedAgentNode,
    "OllamaLLMNode": OllamaLLMNode,


}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LangChainLLM": "avni - 🦜 LangChain LLM",
    "LangChainMemory": "avni - 🧠 LangChain Memory",
    "LangChainSearchTool": "avni - 🔍 LangChain Search",
    "LangChainAgent": "avni - 🤖 LangChain Agent",
    "ResultFormatter": "avni - 📝 Result Formatter",
    "GoalOrientedAgentNode": "avni - 📝 GoalOrientedAgentNode ",
    "OLLAMALLMNODE": "AVNI - 🦙 OLLAMA LLM",

}