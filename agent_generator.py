import os
import re
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

from agent_validator import SQLValidator
import sqlglot
import sqlglot.expressions as exp

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
        # Local import so that importing this module doesn't initialize/load models.
        from llm import hf_interface

        self.MAX_ITERATIONS = 10
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
            raise RuntimeError(
                f"Failed to generate valid SQL. Error: {e}"
            )

    def validate_fix_and_optimize_sql(self, sql, schema):
        """
        Validate, fix, and optimize SQL query using comprehensive validation.

        The ComprehensiveSQLValidator returns a detailed response with:
        - syntax: {valid, ast/error}
        - structure: {valid, errors, warnings, metadata}
        - semantic: {valid}
        - types: {valid, errors, warnings, inferred_types}
        - dialect: {compatible, score, issues, suggestions}
        - performance: {performance_issues, score}
        - summary: {overall_valid, issue_count, warning_count, ...}
        """
        sql_validator = SQLValidator()
        sql_validator.load_schema(schema)
        history = []

        for attempt in range(self.MAX_ITERATIONS):
            result = sql_validator.validate_query(sql)
            logger.info(f"Result {attempt}: {result}")

            history.append({
                "attempt": attempt,
                "sql": sql,
                "validation": result
            })

            if result["valid"]:
                break

            sql = self.llm_fix_sql(
                sql=sql,
                errors=result["errors"],
                schema=schema
            )
        else:
            logger.warning(f"Failed to generate valid SQL after {self.MAX_ITERATIONS} attempts")
            logger.info(f"Validation history: {history}")
            raise RuntimeError(
                f"Failed to generate valid SQL after {self.MAX_ITERATIONS} attempts. "
            )

        # Optimization step (deterministic, no LLM):
        # - normalize/format
        # - enforce unique SELECT output aliases to satisfy downstream consumers and the validator
        optimized_sql = self.optimize_sql(sql)

        # Final validation check on optimized output
        final_result = sql_validator.validate_query(optimized_sql)
        final_valid = final_result["valid"]

        logger.info(f"Final validation (after optimize): valid={final_valid}")
        logger.info(f"Validation history: {history}")

        if not final_valid:
            return {
                "valid": False,
                "sql": optimized_sql,
                "result": final_result
            }

        return optimized_sql.replace("NULLS LAST", "")

    def optimize_sql(self, sql: str) -> str:
        """Best-effort SQL optimization/normalization.

        This is intentionally deterministic (no LLM):
        - Parse with sqlglot (mysql dialect)
        - Drop redundant duplicate JOIN-key projections (Option 2)
          Example: SELECT a.id, b.id FROM a JOIN b ON a.id = b.id  -> keep one
        - Ensure every SELECT projection has an explicit alias
        - Ensure SELECT output aliases are unique
        - Return a formatted SQL string
        """
        if not sql:
            return sql

        try:
            tree = sqlglot.parse_one(sql, dialect="mysql")
        except Exception:
            return sql

        for select in tree.find_all(exp.Select):
            # Option 2: remove redundant join-key projections first
            self._drop_duplicate_join_key_projections(select)
            # Then enforce explicit, unique output aliases
            self._enforce_unique_select_aliases(select)

        try:
            return tree.sql(dialect="mysql", pretty=False)
        except Exception:
            return sql

    def _drop_duplicate_join_key_projections(self, select: exp.Select) -> None:
        """Drop redundant projections when they are the *same join key*.

        We detect simple equi-join predicates of the form:
            <alias1>.<col> = <alias2>.<col>

        If both sides' columns are selected in the same SELECT list *without explicit aliases*,
        we keep the first occurrence and drop later duplicates.

        This is conservative:
        - Only handles equality joins between two columns.
        - Only drops when the projection is a plain Column (not an Alias, not an expression).
        """
        joins = select.args.get("joins") or []
        if not joins:
            return

        # Build a set of equivalent column pairs from join ON conditions.
        # We store as frozenset({(alias, col), (alias, col)}) so order doesn't matter.
        equiv_pairs: set[frozenset[tuple[str, str]]] = set()

        def record_equivalence(left: exp.Expression, right: exp.Expression) -> None:
            if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                if left.table and right.table and left.name and right.name:
                    equiv_pairs.add(frozenset({(left.table, left.name), (right.table, right.name)}))

        # Extract join predicate equalities
        for j in joins:
            on_expr = j.args.get("on")
            if not on_expr:
                continue

            # Find all equality expressions within ON (handles AND chains)
            for eq in on_expr.find_all(exp.EQ):
                record_equivalence(eq.this, eq.expression)

        if not equiv_pairs:
            return

        projections = list(select.expressions or [])
        if not projections:
            return

        # Track first seen join-key side we kept, keyed by normalized equivalence pair
        kept_for_pair: dict[frozenset[tuple[str, str]], tuple[str, str]] = {}
        new_projs: list[exp.Expression] = []

        for proj in projections:
            # Only drop plain columns (no explicit alias). If user aliased it, keep.
            if isinstance(proj, exp.Alias):
                new_projs.append(proj)
                continue

            if not isinstance(proj, exp.Column) or not proj.table or not proj.name:
                new_projs.append(proj)
                continue

            this_side = (proj.table, proj.name)
            matched_pair: Optional[frozenset[tuple[str, str]]] = None
            for p in equiv_pairs:
                if this_side in p:
                    matched_pair = p
                    break

            if matched_pair is None:
                new_projs.append(proj)
                continue

            if matched_pair not in kept_for_pair:
                kept_for_pair[matched_pair] = this_side
                new_projs.append(proj)
                continue

        select.set("expressions", new_projs)

    def _enforce_unique_select_aliases(self, select: exp.Select) -> None:
        """Mutates select.expressions so each projection has a unique explicit alias.

        After dropping join-key duplicates we still want a stable output schema:
        - Every projection becomes `expr AS alias` (unless it already is an Alias)
        - Aliases are made unique by prefixing with table alias and/or numeric suffix
        """
        projections = list(select.expressions or [])
        if not projections:
            return

        used: set[str] = set()
        new_projs: list[exp.Expression] = []

        for idx, proj in enumerate(projections, start=1):
            expr = proj
            alias: Optional[str] = None

            if isinstance(proj, exp.Alias):
                expr = proj.this
                alias = proj.alias
            else:
                if isinstance(proj, exp.Column):
                    alias = proj.name
                else:
                    alias = f"expr_{idx}"

            base = alias or f"expr_{idx}"
            candidate = base

            if candidate in used:
                if isinstance(expr, exp.Column) and expr.table:
                    candidate = f"{expr.table}_{base}"
                if candidate in used:
                    n = 2
                    while f"{candidate}_{n}" in used:
                        n += 1
                    candidate = f"{candidate}_{n}"

            used.add(candidate)

            if isinstance(proj, exp.Alias) and proj.alias == candidate:
                new_projs.append(proj)
            else:
                new_projs.append(exp.alias_(expr, candidate, quoted=False))

        select.set("expressions", new_projs)

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