import sys
import os
from typing import Any, Dict, List
import json
import folder_paths

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

class LangChainBaseNode:
    """Base class for all LangChain nodes"""
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = False
    CATEGORY = "LangChain"

    def __init__(self):
        super().__init__()

class LLMNode(LangChainBaseNode):
    """Node for creating LLM instances"""
    RETURN_TYPES = ("LLM",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "create_llm"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "gpt-3.5-turbo"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "openai_api_key": ("STRING", {"default": ""})
            },
        }

    def create_llm(self, model_name, temperature, openai_api_key):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        llm = OpenAI(model_name=model_name, temperature=temperature)
        return (llm,)

class PromptTemplateNode(LangChainBaseNode):
    """Node for creating prompt templates"""
    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "create_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {"default": ""}),
                "input_variables": ("STRING", {"default": "topic,context"})
            },
        }

    def create_prompt(self, template, input_variables):
        variables = [v.strip() for v in input_variables.split(",")]
        prompt = PromptTemplate(
            template=template,
            input_variables=variables
        )
        return (prompt,)

class LLMChainNode(LangChainBaseNode):
    """Node for creating and running LLM chains"""
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "run_chain"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm": ("LLM",),
                "prompt": ("PROMPT",),
                "inputs": ("STRING", {"default": '{"topic": "AI", "context": "brief"}'})
            },
        }

    def run_chain(self, llm, prompt, inputs):
        chain = LLMChain(llm=llm, prompt=prompt)
        input_dict = json.loads(inputs)
        with get_openai_callback() as cb:
            result = chain.run(**input_dict)
        return (result,)

class ConversationMemoryNode(LangChainBaseNode):
    """Node for managing conversation memory"""
    RETURN_TYPES = ("MEMORY",)
    RETURN_NAMES = ("memory",)
    FUNCTION = "create_memory"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "k": ("INT", {"default": 5, "min": 1, "max": 50}),
            },
        }

    def create_memory(self, k):
        memory = ConversationBufferMemory(k=k)
        return (memory,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "LLMNode": LLMNode,
    "PromptTemplateNode": PromptTemplateNode,
    "LLMChainNode": LLMChainNode,
    "ConversationMemoryNode": ConversationMemoryNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMNode": "LangChain LLM",
    "PromptTemplateNode": "Prompt Template",
    "LLMChainNode": "LLM Chain",
    "ConversationMemoryNode": "Conversation Memory"
}
