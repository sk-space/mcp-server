import os, torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from transformers import AutoModel, AutoTokenizer, pipeline
from openai import OpenAI
from logger import get_logger

logger = get_logger(__name__)

load_dotenv()


class HuggingFaceClientWrapper:
    """Wrapper to make HuggingFace API compatible with LangChain interface"""

    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("HF_API_URL"),
            api_key=os.getenv("HF_TOKEN"),
        )
        self._model_name = os.getenv("HF_MODEL_NAME")

    def invoke(self, prompt):
        """Implement the invoke method for LangChain compatibility"""
        try:
            completion = self.client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=150,
                temperature=0.1
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.info(f"Error calling HuggingFace API: {e}")
            raise e

    def __call__(self, prompt):
        """Support for direct calling"""
        return self.invoke(prompt)


class HFInferenceClient:

    def __init__(self):
        self.use_huggingface_api = True
        self.use_openai = False
        self._model_name = os.getenv("HF_MODEL_NAME")
        self._api_key = os.getenv("HF_TOKEN")

    def load_model(self):
        """Load the model either via HuggingFace API or locally"""
        if self.use_huggingface_api:
            return self._load_huggingface_api_model()
        elif self.use_openai:
            return self._load_huggingface_openai_model()
        else:
            return self._load_local_model()


    def _load_huggingface_api_model(self):
        logger.info(f"Loading model via HuggingFace API: {self._model_name}")
        # Check if HF_TOKEN is available
        if not self._api_key:
            logger.info("Warning: HuggingFace API KEY not found in environment variables. Using fallback.")

        try:
            llm = HuggingFaceEndpoint(
                model=self._model_name,
                task="text-generation",
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
                provider="auto",
                huggingfacehub_api_token=self._api_key
            )
            # return llm
            chat_model = ChatHuggingFace(llm=llm)
            return chat_model
        except Exception as e:
            logger.info(f"Error loading model via HuggingFace API: {e}")



    def _load_huggingface_openai_model(self):
        """Load model using HuggingFace API (recommended)"""
        logger.info(f"Loading model via OpenAI: {self._model_name}")

        # Check if HF_TOKEN is available
        if not self._api_key:
            logger.info("Warning: OpenAI API KEY not found in environment variables. Using fallback.")

        try:
            return HuggingFaceClientWrapper()
        except Exception as e:
            logger.info(f"Error loading model via API: {e}")


    def _load_local_model(self):
        """Load the HuggingFace model locally with fallbacks"""
        try:
            logger.info(f"Loading local model: {self._model_name}")

            tokenizer = AutoTokenizer.from_pretrained(self._model_name)

            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModel.from_pretrained(
                self._model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=150,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            logger.info("Local model loaded successfully")

            return HuggingFacePipeline(pipeline=pipe)

        except Exception as e:
            logger.info(f"Error loading local model: {self._model_name}: {e}")


# Create global instance
hf_interface = HFInferenceClient()
