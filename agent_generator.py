import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

from agent_validator import sql_validator
from llm import hf_interface
from logger import setup_file_logging, get_logger

setup_file_logging("server.log")
logger = get_logger(__name__)

load_dotenv()


class SQLOutputParser(BaseOutputParser):
    """Custom output parser to extract SQL from LLM response"""

    def parse(self, text: str):
        if not text:
            return None

        # Remove code block markers if present
        text = re.sub(r"```sql|```", "", text, flags=re.IGNORECASE)

        # Normalize whitespace: remove line breaks and extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Match SQL starting keywords
        match = re.search(
            r"\b(SELECT|INSERT|UPDATE|DELETE|WITH)\b.*",
            text,
            flags=re.IGNORECASE
        )

        if not match:
            return None

        sql_query = match.group(0).strip()

        # Remove trailing semicolon (optional consistency)
        if sql_query.endswith(";"):
            sql_query = sql_query[:-1]

        return sql_query




class NL2SQLAgent:
    def __init__(self):
        self.MAX_ITERATIONS = 5
        self.output_parser = SQLOutputParser()
        self.current_model = hf_interface.load_model()
        self.base_url = os.getenv("HF_API_URL")
        self.headers = {
            "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
            "Content-Type": "application/json"
        }
        self.client = hf_interface.load_model()
        if not self.client:
            raise RuntimeError("Failed to load LLM model for NL2SQLAgent")

    def generate_sql(self, natural_language_query, schema_context):
        """Generate SQL from natural language query"""
        logger.info(f"GQ NLQ: {self.current_model}")
        logger.info(f"Schema: {schema_context}")
        prompt_template = PromptTemplate(
            input_variables=["schema", "question"],
            template="""
                            You are a SQL expert. Convert the following natural language question into a SQL query using the database schema below.

                            Database Schema:
                            {schema}

                            Natural Language Question: {question}

                            Instructions:
                                1. Generate only the SQL query without any explanations.
                                2. Use proper MySQL SQL syntax.
                                3. Only query the tables that are necessary.
                                4. STRICT SCHEMA RULE: You MUST use ONLY tables and columns that appear in the Database Schema above.
                                   - Do NOT invent columns (no hallucinations).
                                   - Do NOT invent tables.
                                   - If a requested field does not exist, omit it or approximate with existing fields.
                                5. Prefer explicit JOIN paths based on foreign keys in the schema.
                                   - Example: carts does NOT have product_id/quantity/unit_price; those are in cart_items.
                                6. Every column in every SELECT clause must have an explicit alias, including:
                                   - Columns in the main query
                                   - Columns in all CTEs (WITH clauses)
                                   - Columns in all subqueries
                                   - Computed columns (like SUM(), COUNT(), CASE, arithmetic expressions, etc.)
                                7. Use consistent, readable aliases for all columns.
                                8. Return the query as a single, complete SQL statement.
                                9. Do NOT use Postgres-only syntax such as ILIKE. For pattern matching use LIKE.
                                   - If case-insensitive matching is required, use: LOWER(column) LIKE LOWER('%pattern%')
                                10. Avoid duplicate projections in JOIN queries:
                                   - Do NOT select the same logical key from both sides of a JOIN (e.g., select product_id from only ONE table).
                                   - Ensure every output column alias is UNIQUE.
                                   - Do NOT use SELECT * when joining tables.

                            SQL Query:
                            """
        )

        # Use the modern LangChain approach
        try:
            # Method 1: Use invoke (preferred in newer versions)
            if hasattr(self.current_model, 'invoke'):
                logger.info("Method 1: Using invoke method for LLM call")
                response = self.current_model.invoke(
                    prompt_template.format(
                        schema=schema_context,
                        question=natural_language_query
                    )
                )
                logger.info(f"Raw response: {response}")

                # Handle different response types
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
            else:
                # Method 2: Use __call__ for older compatibility
                logger.info("Method 2: Using direct method for LLM call")
                response_text = self.current_model(
                    prompt_template.format(
                        schema=schema_context,
                        question=natural_language_query
                    )
                )

            sql_query = self.output_parser.parse(response_text)
            logger.info(f"Response query: {sql_query}")

            return self.validate_fix_and_optimize_sql(sql_query, schema_context).replace("NULLS LAST", "")

        except Exception as e:
            logger.info(f"Error generating SQL with LLM: {e}")
            # Fallback to rule-based generation
            return self.generate_sql_rule_based(natural_language_query)

    def validate_fix_and_optimize_sql(self, sql, schema):
        history = []
        for attempt in range(self.MAX_ITERATIONS):
            result = sql_validator.validate_mysql_sql(sql, schema)
            logger.info(f"Result {attempt}: {result}")

            history.append({
                "attempt": attempt,
                "sql": sql,
                "validation": result
            })

            if result["valid"]:
                break

            # Feed structured errors to LLM
            sql = self.llm_fix_sql(
                sql=sql,
                errors=result["errors"],
                schema=schema
            )
        else:
            logger.info(f"History1: {history}")
            raise RuntimeError(
                "Failed to generate valid SQL after "
                f"{self.MAX_ITERATIONS} attempts"
            )

        final_check = sql_validator.validate_mysql_sql(sql, schema)

        logger.info(f"History2: {history}")

        if not final_check["valid"]:
            return final_check

        return sql.replace("NULLS LAST", "")


    def llm_fix_sql(self, sql: str, errors: list, schema: dict) -> str:
        """
        Calls LLM to fix broken SQL.
        Returns SQL only (no markdown, no text).
        """

        prompt = """
                    You are a MySQL SQL repair engine.

                    TASK:
                    Fix the provided SQL query using ONLY the validation errors below.

                    INSTRUCTIONS:
                    1. Generate only the SQL query without any explanations.
                    2. Use proper MySQL SQL syntax.
                    3. Only query the tables that are necessary.
                    4. STRICT SCHEMA RULE: You MUST use ONLY tables and columns that appear in the SCHEMA section.
                       - If you see UNKNOWN_TABLE / UNKNOWN_COLUMN errors, remove/replace the invalid references using the schema.
                       - Do NOT invent new columns or tables while fixing.
                    5. If a table is missing a needed attribute, locate the correct table via foreign keys.
                       - Example: cart line item fields (product_id, quantity, unit_price) live in cart_items (joined by cart_id), not in carts.
                    6. Every column in every SELECT clause must have an explicit alias, including:
                       - Columns in the main query
                       - Columns in all CTEs (WITH clauses)
                       - Columns in all subqueries
                       - Computed columns (like SUM(), COUNT(), CASE, arithmetic expressions, etc.)
                    7. Use consistent, readable aliases for all columns.
                    8. Return the query as a single, complete SQL statement.
                    9. Ensure every output column alias is UNIQUE. Remove duplicate projections introduced by JOINs (e.g., don't select the join key from both tables).

                    CONSTRAINTS:
                    - Do NOT change query intent
                    - Do NOT add new tables or columns that are not in schema
                    - Do NOT remove required logic
                    - Output ONLY valid MySQL SQL
                    - Do NOT use Postgres-only syntax such as ILIKE. For pattern matching use LIKE.
                      If case-insensitive matching is required, use: LOWER(col) LIKE LOWER(pattern)
                    - Avoid SELECT * in JOIN queries (explicitly list required columns)
                    - No explanations, no markdown

                    SCHEMA:
                    {schema}

                    ORIGINAL SQL:
                    {sql}

                    VALIDATION ERRORS (JSON):
                    {errors}

                    Return the corrected SQL only.
                """.strip()

        prompt_template = PromptTemplate(
            input_variables=["schema", "sql", "errors"],
            template=prompt
        )

        response = self.current_model.invoke(
                    prompt_template.format(
                        schema=schema,
                        sql=sql,
                        errors=errors,
                    )
                )

        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)

        sql_query = self.output_parser.parse(response_text)
        logger.info(f"LLM fixed query: {sql_query}")

        return sql_query

    def generate_sql_rule_based(self, natural_language_query):
        """Simple rule-based SQL generation as fallback"""
        query_lower = natural_language_query.lower()

        rules = [
            (["engineer", "engineering"], "SELECT * FROM employees WHERE department = 'Engineering'"),
            (["hr", "human resources"], "SELECT * FROM employees WHERE department = 'HR'"),
            (["market", "marketing"], "SELECT * FROM employees WHERE department = 'Marketing'"),
            (["sales"], "SELECT * FROM employees WHERE department = 'Sales'"),
            (["average", "avg", "salary"], "SELECT department, AVG(salary) as average_salary FROM employees GROUP BY department"),
            (["high", "highest", "max", "salary"], "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1"),
            (["low", "lowest", "min", "salary"], "SELECT name, salary FROM employees ORDER BY salary ASC LIMIT 1"),
            (["recent", "new", "hired", "hire date"], "SELECT name, hire_date FROM employees ORDER BY hire_date DESC"),
            (["project", "projects"], "SELECT p.name as project_name, d.name as department_name FROM projects p JOIN departments d ON p.department_id = d.id"),
            (["count", "how many", "employees"], "SELECT COUNT(*) as total_employees FROM employees"),
            (["department", "departments", "list"], "SELECT name, budget FROM departments"),
        ]

        for keywords, sql in rules:
            if any(keyword in query_lower for keyword in keywords):
                return sql

        return "SELECT * FROM employees LIMIT 10"


nl2sql_agent = NL2SQLAgent()