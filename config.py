from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Azure OpenAI
    azure_openai_api_key: str = Field(alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: str = Field(alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field(alias="AZURE_OPENAI_API_VERSION")

    # MCP
    mcp_server_endpoint: str = Field(alias="MCP_SERVER_ENDPOINT")
    mcp_server_timeout: int = Field(default=500, alias="MCP_SERVER_TIMEOUT")
    # Token used at startup to fetch tool schemas once. Per-request auth is
    # still forwarded to the gateway at tool-execution time.
    mcp_bootstrap_auth: str = Field(default="", alias="MCP_BOOTSTRAP_AUTH")
    # Meta-tool names for aggregator MCP gateways. If the list meta-tool is
    # missing, we fall back to using gateway tools directly.
    mcp_list_tools_tool: str = Field(default="list_internal_tools", alias="MCP_LIST_TOOLS_TOOL")
    mcp_schema_tool: str = Field(default="get_tool_schema", alias="MCP_SCHEMA_TOOL")
    mcp_call_tool_tool: str = Field(default="call_internal_tool", alias="MCP_CALL_TOOL_TOOL")
    mcp_schema_concurrency: int = Field(default=10, alias="MCP_SCHEMA_CONCURRENCY")
    mcp_cache_file: str = Field(default="./mcp_cache.json", alias="MCP_CACHE_FILE")
    mcp_cache_ttl: int = Field(default=3600, alias="MCP_CACHE_TTL")  # seconds

    # Session checkpointer: "sqlite" | "redis" | "memory"
    checkpointer: str = Field(default="sqlite", alias="CHECKPOINTER")
    sqlite_path: str = Field(default="./sessions.sqlite", alias="SQLITE_PATH")
    redis_url: str = Field(default="", alias="REDIS_URL")

    # Auth
    itc_token_secret: str = Field(default="", alias="ITC_TOKEN_SECRET")

    # Agent
    agent_max_iterations: int = Field(default=30, alias="AGENT_MAX_ITERATIONS")
    agent_verbose: bool = Field(default=False, alias="AGENT_VERBOSE")
    # Hard cap on tools handed to the react agent. OpenAI enforces <= 128.
    # A small value (~40-80) both satisfies the limit and reduces LLM confusion.
    agent_max_tools: int = Field(default=64, alias="AGENT_MAX_TOOLS")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8080, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="dev", alias="ENVIRONMENT")
    reload: bool = Field(default=True, alias="RELOAD")


@lru_cache
def get_settings() -> Settings:
    return Settings()
