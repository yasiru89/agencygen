"""
Shared configuration and defaults for AgencyGen.
"""

# Default model to use for agents
# This can be changed to any Gemini model
DEFAULT_MODEL = "gemini-2.0-flash"

# Available Gemini models for diversity in multi-agent systems
# Check available models: https://ai.google.dev/gemini-api/docs/models/gemini
AVAILABLE_MODELS = [
    "gemini-2.0-flash",           # Fast, good for most tasks (default)
    "gemini-2.0-flash-lite",      # Faster, lighter version
    "gemini-2.5-flash-preview-05-20",  # Preview of next version
]

# Default diverse model set for voting/council patterns
DEFAULT_COUNCIL_MODELS = [
    "gemini-2.0-flash",           # Main model
    "gemini-2.0-flash-lite",      # Lighter variant - different trade-offs
    "gemini-2.0-flash",           # Third voter uses main model again
    # Note: For true diversity like LLM Council, you'd want different
    # model families (GPT, Claude, Gemini) but here the focus is Gemini
]

