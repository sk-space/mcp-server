import os
from dataclasses import dataclass

from fastmcp import FastMCP

from agent_generator import NL2SQLAgent
from logger import setup_file_logging, get_logger

setup_file_logging("server.log")

logger = get_logger(__name__)


# Initialize MCP
mcp = FastMCP("NL2SQL Agent")


@dataclass
class MCPConfig:
    # MCP Server Configuration
    MCP_SERVER_TRANSPORT = os.getenv("MCP_SERVER_TRANSPORT", "stdio")
    MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
    MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", 8000))

    def validate(self):
        """Validate configuration"""
        required_vars = {
            "MCP_TRANSPORT": self.MCP_SERVER_TRANSPORT,
            "MCP_HOST": self.MCP_SERVER_HOST,
            "MCP_PORT": self.MCP_SERVER_PORT
        }

        missing = [var for var, value in required_vars.items() if not value]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")


@mcp.tool(title="NL2SQL Converter")
async def tool_convert_to_sql(query: str, schema_context: dict) -> dict:
    """
        Converts a natural language query into a valid SQL statement.

        This tool accepts a user-provided natural language question describing
        a data retrieval requirement and generates a syntactically correct,
        schema-aware SQL query based on the current database structure.

        The function:
        - Loads the active database schema context
        - Uses an NL-to-SQL agent to translate the natural language query
        - Returns the generated SQL without executing it

        Parameters:
            query (str): A natural language query describing the desired data,
                         such as "List all active projects in the HR department".

        Returns:
            dict: A structured response containing:
                - success (bool): Indicates whether SQL generation was successful
                - query (str): The original natural language query
                - sql (str): The generated SQL query (if successful)
                - error (str): Error message (if failed)
    """
    try:
        logger.info(f"Server: Converting query to SQL: {query}")
        logger.info(f"Server: Provided schema context: {schema_context}")
        nl2sql_agent = NL2SQLAgent()
        sql = nl2sql_agent.generate_sql(query, schema_context)
        return {
            "success": True,
            "query": query,
            "sql": sql
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }




if __name__ == "__main__":
    try:
        config = MCPConfig()
        config.validate()
        logger.info("MCP Configuration validated")
    except ValueError as e:
        logger.error(f"MCP Configuration error: {e}")
        exit(1)

    # Run MCP server
    logger.info(
        "Starting MCP server on %s://%s:%s",
        config.MCP_SERVER_TRANSPORT,
        config.MCP_SERVER_HOST,
        config.MCP_SERVER_PORT,
    )

    if config.MCP_SERVER_TRANSPORT == "http":
        mcp.run(transport="http", host=config.MCP_SERVER_HOST, port=config.MCP_SERVER_PORT)
    elif config.MCP_SERVER_TRANSPORT == "stdio":
        mcp.run(transport="stdio")
    else:
        raise ValueError("Invalid MCP_SERVER_TRANSPORT")
