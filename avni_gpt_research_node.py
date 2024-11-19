import os
import asyncio
from typing import Literal, Optional
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from gpt_researcher import GPTResearcher

class GPTResearcherNode:
    """Custom node for running GPT Researcher in ComfyUI"""

    def __init__(self):
        self.researcher = None
        self.output_data = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {"default": "", "multiline": True}),
                "report_type": (["research_report", "resource_report", "outline_report"], {"default": "research_report"}),
                "report_format": (["default", "APA", "MLA", "Chicago"], {"default": "default"}),
                "report_source": (["web", "local", "hybrid"], {"default": "web"}),
                "tone": ("STRING", {"default": "formal and objective"}),
                "max_subtopics": ("INT", {"default": 5, "min": 1, "max": 10}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "openai_api_key": ("STRING", {"default": ""}),
                "tavily_api_key": ("STRING", {"default": ""}),
                "verify_ssl": ("BOOLEAN", {"default": True}),
                "doc_path": ("STRING", {"default": "./my-docs"}),
                "retriever": ("STRING", {"default": "", "multiline": False, "label": "Retrievers (comma-separated)"}),
                "retriever_endpoint": ("STRING", {"default": ""}),
                "retriever_args": ("STRING", {"default": "", "multiline": True, "label": "Retriever Arguments (KEY=VALUE per line)"}),
                "openai_api_base": ("STRING", {"default": ""}),
                "ollama_base_url": ("STRING", {"default": ""}),
                "fast_llm": ("STRING", {"default": "openai:gpt-4o"}),
                "smart_llm": ("STRING", {"default": "openai:gpt-4o"}),
                "embedding": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("report", "research_context", "costs", "sources", "images", "image_tensor")
    FUNCTION = "run_research"
    CATEGORY = "research"

    def download_and_process_image(self, url: str) -> Optional[torch.Tensor]:
        """Download and process an image from URL into a tensor."""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')

                target_size = (512, 512)
                img = img.resize(target_size, Image.Resampling.LANCZOS)

                img_np = np.array(img)
                img_tensor = torch.from_numpy(img_np).float() / 255.0
                # Rearrange dimensions to match ComfyUI format (B, H, W, C)
                img_tensor = img_tensor.unsqueeze(0)
                return img_tensor
            return None
        except Exception as e:
            print(f"Error downloading image {url}: {str(e)}")
            return None

    def setup_environment(self, openai_api_key: str = "",
                         tavily_api_key: str = "",
                         doc_path: str = "./my-docs",
                         retriever: str = "",
                         retriever_endpoint: str = "",
                         retriever_args: str = "",
                         openai_api_base: str = "",
                         ollama_base_url: str = "",
                         fast_llm: str = "",
                         smart_llm: str = "",
                         embedding: str = "",
                         verify_ssl: bool = True):
        """Setup environment variables"""
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if openai_api_base:
            os.environ["OPENAI_API_BASE"] = openai_api_base
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
        if doc_path:
            os.environ["DOC_PATH"] = doc_path
        if retriever:
            os.environ["RETRIEVER"] = retriever
        if retriever_endpoint:
            os.environ["RETRIEVER_ENDPOINT"] = retriever_endpoint
        if retriever_args:
            for line in retriever_args.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        os.environ[f"RETRIEVER_ARG_{key}"] = value
        if ollama_base_url:
            os.environ["OLLAMA_BASE_URL"] = ollama_base_url
        if fast_llm:
            os.environ["FAST_LLM"] = fast_llm
        if smart_llm:
            os.environ["SMART_LLM"] = smart_llm
        if embedding:
            os.environ["EMBEDDING"] = embedding
        os.environ["VERIFY_SSL"] = "true" if verify_ssl else "false"

    def run_research(self, query: str,
                    report_type: Literal["research_report", "resource_report", "outline_report"],
                    report_format: str = "default",
                    report_source: str = "web",
                    tone: str = "formal and objective",
                    max_subtopics: int = 5,
                    verbose: bool = False,
                    openai_api_key: str = "",
                    tavily_api_key: str = "",
                    verify_ssl: bool = False,
                    doc_path: str = "./my-docs",
                    retriever: str = "",
                    retriever_endpoint: str = "",
                    retriever_args: str = "",
                    openai_api_base: str = "",
                    ollama_base_url: str = "",
                    fast_llm: str = "",
                    smart_llm: str = "",
                    embedding: str = ""):
        """Run the research process"""

        try:
            self.setup_environment(
                openai_api_key=openai_api_key,
                tavily_api_key=tavily_api_key,
                doc_path=doc_path,
                retriever=retriever,
                retriever_endpoint=retriever_endpoint,
                retriever_args=retriever_args,
                openai_api_base=openai_api_base,
                ollama_base_url=ollama_base_url,
                fast_llm=fast_llm,
                smart_llm=smart_llm,
                embedding=embedding,
                verify_ssl=verify_ssl
            )

            self.researcher = GPTResearcher(
                query=query,
                report_type=report_type,
                report_format=report_format,
                report_source=report_source,
                tone=tone,
                max_subtopics=max_subtopics,
                verbose=verbose
            )

            async def run():
                try:
                    await self.researcher.conduct_research()
                    report = await self.researcher.write_report()

                    context = self.researcher.get_research_context()
                    costs = self.researcher.get_costs()
                    sources = self.researcher.get_research_sources()
                    image_urls = self.researcher.get_research_images()

                    # Process images
                    image_tensors = []
                    for url in image_urls:
                        img_tensor = self.download_and_process_image(url)
                        if img_tensor is not None:
                            image_tensors.append(img_tensor)

                    # Create batch tensor if images found
                    if image_tensors:
                        try:
                            batch_tensor = torch.cat(image_tensors, dim=0)
                        except Exception as e:
                            print(f"Error concatenating image tensors: {str(e)}")
                            # Return first valid image if concatenation fails
                            batch_tensor = image_tensors[0]
                    else:
                        # Return empty tensor with correct dimensions
                        batch_tensor = torch.zeros((1, 512, 512, 3))

                    return (
                        report,
                        str(context),
                        str(costs),
                        str(sources),
                        str(image_urls),
                        batch_tensor
                    )
                except Exception as e:
                    print(f"Error during research: {str(e)}")
                    raise

            return asyncio.run(run())

        except Exception as e:
            print(f"Error in run_research: {str(e)}")
            return ("", "", "", "", "", torch.zeros((1, 512, 512, 3)))

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Return hash of input parameters to detect changes"""
        return hash(str(kwargs))

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate input parameters"""
        if not kwargs["query"].strip():
            return "Query cannot be empty"
        if kwargs.get("max_subtopics", 5) < 1:
            return "max_subtopics must be at least 1"
        if kwargs.get("report_source") in ["local", "hybrid"] and not kwargs.get("doc_path"):
            return "Document path must be provided for local or hybrid research"
        if kwargs.get("retriever") == "custom" and not kwargs.get("retriever_endpoint"):
            return "Retriever endpoint must be provided for custom retriever"
        return True

NODE_CLASS_MAPPINGS = {
    "GPTResearcher": GPTResearcherNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTResearcher": "GPT Researcher"
}
