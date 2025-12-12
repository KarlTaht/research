"""HuggingFace-compatible text generation interfaces."""

from typing import Protocol, Optional
import torch


class Generator(Protocol):
    """Protocol for text generation backends.

    This protocol defines the interface that all generation backends must implement.
    It allows for seamless switching between local HuggingFace models and API-based
    services like OpenAI or Anthropic.
    """

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate a response from a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic, 1 = creative)
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        ...


class HFGenerator:
    """HuggingFace transformers backend for local model inference."""

    def __init__(
        self,
        model,  # PreTrainedModel
        tokenizer,  # PreTrainedTokenizer
        device: str = "cuda",
    ):
        """Initialize with a HuggingFace model and tokenizer.

        Args:
            model: A PreTrainedModel (e.g., from AutoModelForCausalLM)
            tokenizer: A PreTrainedTokenizer (e.g., from AutoTokenizer)
            device: Device to run inference on ('cuda', 'cpu', 'mps')
        """
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer

        # Ensure tokenizer has required special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs,
    ) -> str:
        """Generate using HuggingFace model.generate().

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (False = greedy)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            Generated text response
        """
        # Apply chat template if available (most modern models have this)
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation for models without chat template
            prompt = self._format_messages_fallback(messages)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=getattr(self.tokenizer, "model_max_length", 4096)
            - max_new_tokens,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode only the generated tokens (exclude prompt)
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _format_messages_fallback(self, messages: list[dict]) -> str:
        """Format messages for models without chat templates."""
        parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            parts.append(f"{role}: {content}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        torch_dtype=None,
        **kwargs,
    ) -> "HFGenerator":
        """Create generator from a HuggingFace model name.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on
            torch_dtype: Optional dtype (e.g., torch.float16)
            **kwargs: Additional arguments for from_pretrained

        Returns:
            Configured HFGenerator instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch_dtype is None:
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, **kwargs
        )

        return cls(model, tokenizer, device)


class APIGenerator:
    """API-based backend for OpenAI-compatible services."""

    def __init__(self, client, model: str = "gpt-4"):
        """Initialize with an API client.

        Args:
            client: An OpenAI-compatible client (OpenAI, Anthropic, etc.)
            model: Model identifier to use
        """
        self.client = client
        self.model = model

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate using API call.

        Args:
            messages: List of message dicts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional API parameters

        Returns:
            Generated text response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content


class DummyGenerator:
    """Dummy generator for testing without a real model."""

    def __init__(self, response_prefix: str = "This is a test response"):
        """Initialize with a fixed response prefix.

        Args:
            response_prefix: Prefix for generated responses
        """
        self.response_prefix = response_prefix
        self.call_count = 0

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate a dummy response.

        Args:
            messages: List of message dicts (used to include user message in response)
            max_new_tokens: Ignored
            temperature: Ignored
            **kwargs: Ignored

        Returns:
            Dummy response string
        """
        self.call_count += 1
        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "no user message",
        )
        return f"{self.response_prefix} to: {user_msg[:50]}... (call #{self.call_count})"
