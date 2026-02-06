import sqlglot
import sqlglot.expressions as exp
from typing import Dict, List, Set, Tuple, Optional, Any


class SQLValidator:
    """A scope-aware MySQL-ish SQL validator using sqlglot.

    Goals (pragmatic, not a full SQL engine):
    - Build correct alias scopes for SELECTs (including joins)
    - Support CTEs / derived tables (subqueries in FROM/JOIN) as sources
    - Validate columns & table aliases in SELECT/WHERE/JOIN/GROUP BY/HAVING/ORDER BY
    - Allow SELECT aliases only where MySQL allows them (ORDER BY, HAVING, GROUP BY)
    - Detect ambiguous unqualified column references
    - Warn on reserved keywords as identifiers
    """

    AGG_FUNCS = {"SUM", "COUNT", "AVG", "MIN", "MAX", "GROUP_CONCAT"}

    def __init__(self):
        self.schema: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

        # Populated per validate_query run
        self.cte_definitions: Dict[str, Dict[str, Any]] = {}  # cte_name -> columns dict
        self.cte_order: List[str] = []

        # For debug / reporting (outermost select only)
        self.table_aliases: Dict[str, Any] = {}

    def load_schema(self, schema: Dict) -> None:
        # Accept either:
        # - {table: {"columns": {col: {...}}}}
        # - {"tables": {table: {"columns": [col, ...]}}}  (MCP format)
        # - {table: {"columns": [col, ...]}}
        schema = schema or {}
        if isinstance(schema.get("tables"), dict):
            schema = schema.get("tables") or {}

        normalized: Dict[str, Any] = {}
        for table_name, tinfo in (schema or {}).items():
            if not isinstance(tinfo, dict):
                normalized[table_name] = {"columns": {}}
                continue

            cols = tinfo.get("columns")
            if isinstance(cols, list):
                normalized_cols = {c: {} for c in cols}
            elif isinstance(cols, dict):
                normalized_cols = cols
            else:
                normalized_cols = {}

            normalized[table_name] = {**tinfo, "columns": normalized_cols}

        self.schema = normalized

    def validate_query(self, query: str) -> Dict[str, Any]:
        self.errors = []
        self.warnings = []
        self.cte_definitions = {}
        self.cte_order = []
        self.table_aliases = {}

        try:
            parsed = sqlglot.parse_one(query, dialect="mysql")

            self._validate_reserved_keywords(parsed)

            # Build CTE definitions for outer query
            self._process_ctes(parsed)

            # Validate the outer statement (and nested queries)
            self._validate_expression_as_statement(parsed, outer_ctes=self.cte_definitions)

        except sqlglot.errors.ParseError as e:
            self.errors.append(f"SQL Parse Error: {str(e)}")
        except Exception as e:
            self.errors.append(f"Validation Error: {str(e)}")

        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "table_aliases": self.table_aliases,
            "cte_definitions": self.cte_definitions,
        }

    # -------------------------
    # CTE handling
    # -------------------------

    def _process_ctes(self, expression: exp.Expression) -> None:
        with_clause = expression.find(exp.With)
        if not with_clause:
            return

        for cte in with_clause.expressions or []:
            if not isinstance(cte, exp.CTE):
                continue

            cte_name = cte.alias_or_name
            if not cte_name:
                continue

            cols = self._extract_select_output_columns(cte.this)
            self.cte_definitions[cte_name] = cols
            self.cte_order.append(cte_name)

    def _extract_select_output_columns(self, expression: exp.Expression) -> Dict[str, Any]:
        """Infer output columns for Select/Subquery/CTE body."""
        select = expression
        if isinstance(select, exp.Subquery):
            select = select.this

        select = select.find(exp.Select) if isinstance(select, exp.Expression) else None
        if not select:
            return {}

        cols: Dict[str, Any] = {}
        for i, proj in enumerate(select.expressions or [], start=1):
            out_name = None
            if isinstance(proj, exp.Alias):
                out_name = proj.alias
            elif isinstance(proj, exp.Column):
                out_name = proj.name
            elif isinstance(proj, exp.Identifier):
                out_name = proj.name

            if out_name:
                cols[out_name] = {"position": i, "expression": proj}
            elif isinstance(proj, exp.Star):
                cols["*"] = {"position": i, "expression": proj}
            else:
                cols[f"expr_{i}"] = {"position": i, "expression": proj}

        return cols

    # -------------------------
    # Scope building
    # -------------------------

    def _build_select_scope(
        self,
        select: exp.Select,
        *,
        outer_scope: Optional[Dict[str, Any]] = None,
        ctes: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build a scope for a single SELECT.

        scope = {
          'sources': { alias -> source_info },
          'select_aliases': { alias -> proj_expr },
          'outer_scope': outer_scope
        }

        source_info:
          {'type': 'table'|'cte'|'derived', 'name': real_name, 'columns': set|dict|None}
        """
        scope: Dict[str, Any] = {
            "sources": {},
            "select_aliases": {},
            "outer_scope": outer_scope,
            "ctes": ctes or {},
        }

        # FROM clause is 'from_' in sqlglot. It can be an exp.From where:
        # - from_.this is the first source
        # - from_.expressions may contain additional sources (rare in MySQL, but possible)
        from_clause = select.args.get("from_")
        if from_clause is not None:
            from_this = getattr(from_clause, "this", None)
            if isinstance(from_this, exp.Expression):
                self._add_source_from_table_expr(from_this, scope)
            for e in getattr(from_clause, "expressions", None) or []:
                if isinstance(e, exp.Expression):
                    self._add_source_from_table_expr(e, scope)

        # JOIN targets
        for j in select.args.get("joins") or []:
            self._add_source_from_table_expr(getattr(j, "this", None), scope)

        return scope

    def _add_source_from_table_expr(self, node: Optional[exp.Expression], scope: Dict[str, Any]) -> None:
        if node is None:
            return

        sources = scope.get("sources") or {}

        # Derived table
        if isinstance(node, exp.Subquery):
            alias = node.alias_or_name
            if not alias:
                self.errors.append("Derived table/subquery in FROM/JOIN must have an alias")
                return

            if alias in sources:
                self.errors.append(f"Duplicate table alias '{alias}'")
                return

            cols = self._extract_select_output_columns(node.this)
            sources[alias] = {
                "type": "derived",
                "name": f"derived_{alias}",
                "columns": cols,
            }
            scope["sources"] = sources
            return

        # Table reference
        if isinstance(node, exp.Table):
            table_name = node.name
            actual_table = table_name.split(".")[-1]
            alias = node.alias_or_name or actual_table

            if alias in sources:
                self.errors.append(f"Duplicate table alias '{alias}'")
                return

            # determine columns
            if (scope.get("ctes") or {}).get(actual_table) is not None:
                cte_cols = (scope.get("ctes") or {}).get(actual_table) or {}
                sources[alias] = {"type": "cte", "name": actual_table, "columns": cte_cols}
            else:
                cols = None
                if actual_table in self.schema and isinstance(self.schema[actual_table], dict):
                    cols = (self.schema[actual_table].get("columns") or {})
                sources[alias] = {"type": "table", "name": actual_table, "columns": cols}

                if actual_table not in self.schema and actual_table not in (scope.get("ctes") or {}):
                    # Schema may be partial; alias validation should still work.
                    self.warnings.append(f"Table '{actual_table}' not found in schema")

            scope["sources"] = sources
            return

        # Other nodes: ignore.

    # -------------------------
    # Statement validation
    # -------------------------

    def _validate_expression_as_statement(self, expression: exp.Expression, *, outer_ctes: Dict[str, Any]) -> None:
        # Validate all selects found; each select builds its own scope
        for select in expression.find_all(exp.Select):
            self._validate_select(select, outer_ctes=outer_ctes)

        # Also validate any subqueries not reached via find_all(exp.Select) edge cases
        for subq in expression.find_all(exp.Subquery):
            if isinstance(subq.this, exp.Expression):
                self._validate_expression_as_statement(subq.this, outer_ctes=outer_ctes)

    def _validate_select(self, select: exp.Select, *, outer_ctes: Dict[str, Any]) -> None:
        # Build scope for this select - note: correlated subqueries should see outer scope; we do minimal support
        scope = self._build_select_scope(select, outer_scope=None, ctes=outer_ctes)

        # Expose outermost scope for debug
        if not self.table_aliases:
            self.table_aliases = {k: v for k, v in scope["sources"].items()}

        # SELECT projections + collect select aliases
        self._validate_select_list(select, scope)

        # JOIN ON / USING
        for j in select.args.get("joins") or []:
            on_expr = j.args.get("on")
            if on_expr is not None:
                self._validate_expr(on_expr, scope, context="JOIN", allow_select_alias=False)

            using = j.args.get("using")
            if using is not None:
                for c in using.expressions or []:
                    if isinstance(c, exp.Column):
                        # USING uses unqualified columns resolved across both sides; still validate existence/ambiguity
                        self._validate_column(c, scope, context="JOIN")

        # WHERE
        where = select.args.get("where")
        if where is not None:
            self._validate_expr(where.this, scope, context="WHERE", allow_select_alias=False)
            if self._contains_aggregate_function(where.this):
                self.errors.append("Aggregate functions are not allowed in WHERE clause")

        # GROUP BY (MySQL allows referencing select aliases)
        group = select.args.get("group")
        if group is not None:
            for gexpr in group.expressions or []:
                self._validate_expr(gexpr, scope, context="GROUP BY", allow_select_alias=True)

        # HAVING (MySQL allows referencing select aliases)
        having = select.args.get("having")
        if having is not None:
            self._validate_expr(having.this, scope, context="HAVING", allow_select_alias=True)

        # ORDER BY (MySQL allows referencing select aliases)
        order = select.args.get("order")
        if order is not None:
            for oexpr in order.expressions or []:
                target = oexpr.this if isinstance(oexpr, exp.Ordered) else oexpr
                self._validate_expr(target, scope, context="ORDER BY", allow_select_alias=True)

        # Ambiguous column references (unqualified)
        self._check_ambiguous_unqualified_columns(select, scope)

        # Duplicate projection (same column from multiple tables with no alias)
        self._check_duplicate_join_key_projection(select)

    # -------------------------
    # Projection / alias rules
    # -------------------------

    def _validate_select_list(self, select: exp.Select, scope: Dict[str, Any]) -> None:
        select_aliases: Dict[str, exp.Expression] = {}

        for proj in select.expressions or []:
            # Track output alias
            if isinstance(proj, exp.Alias):
                alias = proj.alias
                if alias in select_aliases:
                    self.errors.append(f"Duplicate column alias in SELECT: {alias}")
                select_aliases[alias] = proj.this

            # Validate expression tree; in SELECT we do NOT allow referencing select aliases
            self._validate_expr(proj, scope, context="SELECT", allow_select_alias=False)

        scope["select_aliases"] = select_aliases

    # -------------------------
    # Expression validation
    # -------------------------

    def _validate_expr(self, node: exp.Expression, scope: Dict[str, Any], *, context: str, allow_select_alias: bool) -> None:
        if node is None:
            return

        # SELECT alias references appear as Identifier in ORDER/GROUP/HAVING
        if allow_select_alias and isinstance(node, exp.Identifier):
            name = node.name
            if name in (scope.get("select_aliases") or {}):
                return

        # Column
        if isinstance(node, exp.Column):
            self._validate_column(node, scope, context=context, allow_select_alias=allow_select_alias)
            return

        # Alias
        if isinstance(node, exp.Alias):
            self._validate_expr(node.this, scope, context=context, allow_select_alias=allow_select_alias)
            return

        # Function
        if isinstance(node, exp.Func):
            for arg in node.expressions or []:
                self._validate_expr(arg, scope, context=context, allow_select_alias=allow_select_alias)
            return

        # Generic: walk through children
        for child in node.args.values():
            if isinstance(child, list):
                for c in child:
                    if isinstance(c, exp.Expression):
                        self._validate_expr(c, scope, context=context, allow_select_alias=allow_select_alias)
            elif isinstance(child, exp.Expression):
                self._validate_expr(child, scope, context=context, allow_select_alias=allow_select_alias)

    def _validate_column(self, col: exp.Column, scope: Dict[str, Any], *, context: str, allow_select_alias: bool = False) -> None:
        table_ref = col.table
        col_name = col.name

        # Unqualified column might refer to SELECT alias in ORDER/GROUP/HAVING
        if allow_select_alias and not table_ref:
            if col_name in (scope.get("select_aliases") or {}):
                return

        sources: Dict[str, Any] = scope.get("sources") or {}

        if table_ref:
            if table_ref not in sources:
                self.errors.append(f"Unknown table alias '{table_ref}' referenced in {context} clause")
                return

            src = sources[table_ref]
            self._validate_column_in_source(src, col_name, context=context, table_alias=table_ref)
            return

        # Unqualified: resolve against sources
        matches: List[str] = []
        for alias, src in sources.items():
            if self._source_has_column(src, col_name):
                matches.append(alias)

        if len(matches) == 0:
            self.errors.append(f"Column '{col_name}' not found in any table/CTE in {context} clause")
        elif len(matches) > 1:
            self.errors.append(
                f"Ambiguous column '{col_name}' in {context} clause - exists in multiple tables: {', '.join(matches)}"
            )

    def _source_has_column(self, src: Dict[str, Any], col_name: str) -> bool:
        cols = src.get("columns")
        if cols is None:
            # unknown schema -> cannot prove
            return False
        if "*" in cols:
            return True
        if isinstance(cols, dict):
            return col_name in cols
        if isinstance(cols, set):
            return col_name in cols
        return False

    def _validate_column_in_source(self, src: Dict[str, Any], col_name: str, *, context: str, table_alias: str) -> None:
        cols = src.get("columns")
        if cols is None:
            # schema not loaded for this table
            return
        if col_name == "*":
            return
        if "*" in cols:
            return

        ok = False
        if isinstance(cols, dict):
            ok = col_name in cols
        elif isinstance(cols, set):
            ok = col_name in cols

        if not ok:
            self.errors.append(
                f"Column '{col_name}' not found in table '{src.get('name')}' (alias '{table_alias}') in {context} clause"
            )

    # -------------------------
    # Extra checks
    # -------------------------

    def _check_ambiguous_unqualified_columns(self, select: exp.Select, scope: Dict[str, Any]) -> None:
        # Already handled by _validate_column for general column nodes.
        # Here, add a specific check for SELECT list implicit ambiguity from joins.
        for proj in select.expressions or []:
            c = None
            if isinstance(proj, exp.Column) and not proj.table:
                c = proj
            elif isinstance(proj, exp.Alias) and isinstance(proj.this, exp.Column) and not proj.this.table:
                c = proj.this
            if c is None:
                continue

            sources = scope.get("sources") or {}
            matches = [a for a, src in sources.items() if self._source_has_column(src, c.name)]
            if len(matches) > 1:
                self.errors.append(
                    f"Ambiguous column '{c.name}' in SELECT clause - specify table alias"
                )

    def _check_duplicate_join_key_projection(self, select: exp.Select) -> None:
        """Reject selecting the same output column name multiple times from different sources unless aliased.

        Rationale: In join queries, projecting the same logical field from multiple tables without unique aliases
        is ambiguous/inconvenient for consumers.

        This check is stricter than SQL itself, but matches the agent's contract.
        """
        # output_name -> set(table_aliases)
        seen: Dict[str, Set[str]] = {}

        for proj in select.expressions or []:
            # If aliased, treat as intentional/unique.
            if isinstance(proj, exp.Alias):
                continue

            if isinstance(proj, exp.Column):
                out_name = proj.name
                if not out_name:
                    continue
                if not proj.table:
                    # unqualified columns handled elsewhere (ambiguity)
                    continue
                seen.setdefault(out_name, set()).add(proj.table)

        for col_name, tables in seen.items():
            if len(tables) > 1:
                self.errors.append(
                    f"Duplicate projection '{col_name}' selected from multiple tables ({', '.join(sorted(tables))}); alias each projection uniquely"
                )

    def _contains_aggregate_function(self, expression: exp.Expression) -> bool:
        for func in expression.find_all(exp.Func):
            func_name = (func.sql().split("(")[0] or "").upper()
            if func_name in self.AGG_FUNCS:
                return True
        return False

    # -------------------------
    # Reserved keywords
    # -------------------------

    def _validate_reserved_keywords(self, parsed: exp.Expression) -> None:
        mysql_reserved = {
            'ACCESSIBLE', 'ADD', 'ALL', 'ALTER', 'ANALYZE', 'AND', 'AS', 'ASC',
            'ASENSITIVE', 'BEFORE', 'BETWEEN', 'BIGINT', 'BINARY', 'BLOB',
            'BOTH', 'BY', 'CALL', 'CASCADE', 'CASE', 'CHANGE', 'CHAR',
            'CHARACTER', 'CHECK', 'COLLATE', 'COLUMN', 'CONDITION',
            'CONSTRAINT', 'CONTINUE', 'CONVERT', 'CREATE', 'CROSS',
            'CURRENT_DATE', 'CURRENT_TIME', 'CURRENT_TIMESTAMP',
            'CURRENT_USER', 'CURSOR', 'DATABASE', 'DATABASES', 'DAY_HOUR',
            'DAY_MICROSECOND', 'DAY_MINUTE', 'DAY_SECOND', 'DEC', 'DECIMAL',
            'DECLARE', 'DEFAULT', 'DELAYED', 'DELETE', 'DESC', 'DESCRIBE',
            'DETERMINISTIC', 'DISTINCT', 'DISTINCTROW', 'DIV', 'DOUBLE',
            'DROP', 'DUAL', 'EACH', 'ELSE', 'ELSEIF', 'ENCLOSED', 'ESCAPED',
            'EXISTS', 'EXIT', 'EXPLAIN', 'FALSE', 'FETCH', 'FLOAT', 'FLOAT4',
            'FLOAT8', 'FOR', 'FORCE', 'FOREIGN', 'FROM', 'FULLTEXT', 'GENERATED',
            'GET', 'GRANT', 'GROUP', 'HAVING', 'HIGH_PRIORITY', 'HOUR_MICROSECOND',
            'HOUR_MINUTE', 'HOUR_SECOND', 'IF', 'IGNORE', 'IN', 'INDEX',
            'INFILE', 'INNER', 'INOUT', 'INSENSITIVE', 'INSERT', 'INT', 'INT1',
            'INT2', 'INT3', 'INT4', 'INT8', 'INTEGER', 'INTERVAL', 'INTO',
            'IO_AFTER_GTIDS', 'IO_BEFORE_GTIDS', 'IS', 'ITERATE', 'JOIN',
            'KEY', 'KEYS', 'KILL', 'LEADING', 'LEAVE', 'LEFT', 'LIKE', 'LIMIT',
            'LINEAR', 'LINES', 'LOAD', 'LOCALTIME', 'LOCALTIMESTAMP', 'LOCK',
            'LONG', 'LONGBLOB', 'LONGTEXT', 'LOOP', 'LOW_PRIORITY', 'MASTER_BIND',
            'MASTER_SSL_VERIFY_SERVER_CERT', 'MATCH', 'MAXVALUE', 'MEDIUMBLOB',
            'MEDIUMINT', 'MEDIUMTEXT', 'MIDDLEINT', 'MINUTE_MICROSECOND',
            'MINUTE_SECOND', 'MOD', 'MODIFIES', 'NATURAL', 'NOT', 'NO_WRITE_TO_BINLOG',
            'NULL', 'NUMERIC', 'ON', 'OPTIMIZE', 'OPTION', 'OPTIONALLY', 'OR',
            'ORDER', 'OUT', 'OUTER', 'OUTFILE', 'PARTITION', 'PRECISION',
            'PRIMARY', 'PROCEDURE', 'PURGE', 'RANGE', 'READ', 'READS',
            'READ_WRITE', 'REAL', 'REFERENCES', 'REGEXP', 'RELEASE', 'RENAME',
            'REPEAT', 'REPLACE', 'REQUIRE', 'RESIGNAL', 'RESTRICT', 'RETURN',
            'REVOKE', 'RIGHT', 'RLIKE', 'SCHEMA', 'SCHEMAS', 'SECOND_MICROSECOND',
            'SELECT', 'SENSITIVE', 'SEPARATE', 'SET', 'SHOW', 'SIGNAL', 'SMALLINT',
            'SPATIAL', 'SPECIFIC', 'SQL', 'SQLEXCEPTION', 'SQLSTATE', 'SQLWARNING',
            'SQL_BIG_RESULT', 'SQL_CALC_FOUND_ROWS', 'SQL_SMALL_RESULT', 'SSL',
            'STARTING', 'STORED', 'STRAIGHT_JOIN', 'TABLE', 'TERMINATED', 'THEN',
            'TINYBLOB', 'TINYINT', 'TINYTEXT', 'TO', 'TRAILING', 'TRIGGER', 'TRUE',
            'UNDO', 'UNION', 'UNIQUE', 'UNLOCK', 'UNSIGNED', 'UPDATE', 'USAGE',
            'USE', 'USING', 'UTC_DATE', 'UTC_TIME', 'UTC_TIMESTAMP', 'VALUES',
            'VARBINARY', 'VARCHAR', 'VARCHARACTER', 'VARYING', 'WHEN', 'WHERE',
            'WHILE', 'WINDOW', 'WITH', 'WRITE', 'XOR', 'YEAR_MONTH', 'ZEROFILL'
        }

        for identifier in parsed.find_all(exp.Identifier):
            if identifier.name and identifier.name.upper() in mysql_reserved:
                self.warnings.append(f"Reserved keyword used as identifier: {identifier.name}")


# Test function with comprehensive alias checking
def test_alias_validation():
    """Test the enhanced alias validation"""

    # Define schema
    schema = {
        'users': {
            'columns': {
                'id': {'type': 'INT', 'nullable': False},
                'name': {'type': 'VARCHAR(255)', 'nullable': True},
                'email': {'type': 'VARCHAR(255)', 'nullable': True},
                'created_at': {'type': 'TIMESTAMP', 'nullable': True},
                'status': {'type': 'VARCHAR(50)', 'nullable': True}
            }
        },
        'orders': {
            'columns': {
                'order_id': {'type': 'INT', 'nullable': False},
                'user_id': {'type': 'INT', 'nullable': True},
                'amount': {'type': 'DECIMAL(10,2)', 'nullable': True},
                'status': {'type': 'VARCHAR(50)', 'nullable': True},
                'order_date': {'type': 'DATE', 'nullable': True}
            }
        },
        'order_items': {
            'columns': {
                'id': {'type': 'INT', 'nullable': False},
                'item_id': {'type': 'INT', 'nullable': False},
                'order_id': {'type': 'INT', 'nullable': True},
                'product_id': {'type': 'INT', 'nullable': True},
                'quantity': {'type': 'INT', 'nullable': True},
                'price': {'type': 'DECIMAL(10,2)', 'nullable': True}
            }
        }
    }

    validator = SQLValidator()
    validator.load_schema(schema)

    # Test cases
    test_cases = [
        {
            'name': 'Valid query with table aliases',
            'query': """
                     SELECT u.id as user_id,
                            u.name,
                            o.order_id,
                            o.amount
                     FROM users u
                              JOIN orders o ON u.id = o.user_id
                     WHERE u.status = 'active'
                     ORDER BY u.name
                     """,
            'should_pass': True
        },
        {
            'name': 'Valid query with column aliases in ORDER BY',
            'query': """
                     SELECT u.id              as user_id,
                            u.name            as user_name,
                            COUNT(o.order_id) as order_count
                     FROM users u
                              LEFT JOIN orders o ON u.id = o.user_id
                     GROUP BY u.id, u.name
                     ORDER BY order_count DESC, user_name
                     """,
            'should_pass': True
        },
        {
            'name': 'Invalid - Unknown table alias',
            'query': """
                     SELECT u.id, x.name -- x is not defined
                     FROM users u
                     """,
            'should_pass': False
        },
        {
            'name': 'Invalid - Column not found with alias',
            'query': """
                     SELECT u.id, u.invalid_column -- invalid_column doesn't exist
                     FROM users u
                     """,
            'should_pass': False
        },
        {
            'name': 'Valid - CTE with alias usage',
            'query': """
                     WITH user_stats AS (SELECT user_id,
                                                COUNT(*)    as order_count,
                                                SUM(amount) as total_spent
                                         FROM orders
                                         GROUP BY user_id)
                     SELECT u.name,
                            us.order_count,
                            us.total_spent
                     FROM users u
                              JOIN user_stats us ON u.id = us.user_id
                     WHERE us.order_count > 5
                     ORDER BY us.total_spent DESC
                     """,
            'should_pass': True
        },
        {
            'name': 'Invalid - Ambiguous column without alias',
            'query': """
                     SELECT status -- ambiguous: exists in both users and orders
                     FROM users u
                              JOIN orders o ON u.id = o.user_id
                     """,
            'should_pass': False
        },
        {
            'name': 'Invalid - Explicit table alias resolves ambiguity, duplicate column selection from join tables',
            'query': """
                     SELECT u.status, o.status
                     FROM users u
                              JOIN orders o ON u.id = o.user_id
                     """,
            'should_pass': False
        },
        {
            'name': 'Invalid - Explicit table alias resolves ambiguity, duplicate column selection from join tables',
            'query': """
                     SELECT u.order_id, o.order_id
                     FROM order_items u
                              JOIN orders o ON u.order_id = o.user_id
                     """,
            'should_pass': False
        },
        {
            'name': 'Valid - Explicit column alias resolves ambiguity, unique column selection from join tables',
            'query': """
                     SELECT u.order_id u_id, o.order_id o_id
                     FROM order_items u
                              JOIN orders o ON u.order_id = o.user_id
                     """,
            'should_pass': True
        },
        {
            'name': 'Valid - Derived table with alias',
            'query': """
                     SELECT u.name,
                            dt.total_orders
                     FROM users u
                              JOIN (SELECT user_id, COUNT(*) as total_orders
                                    FROM orders
                                    GROUP BY user_id) dt ON u.id = dt.user_id
                     ORDER BY dt.total_orders DESC
                     """,
            'should_pass': True
        },
        {
            'name': 'Valid - Self-join with aliases',
            'query': """
                     SELECT e1.name as employee,
                            e2.name as manager
                     FROM employees e1
                              LEFT JOIN employees e2 ON e1.manager_id = e2.id
                     """,
            'should_pass': True
        },
        {
            'name': 'Invalid - Using SELECT alias in WHERE (not allowed)',
            'query': """
                     SELECT u.name            as user_name,
                            COUNT(o.order_id) as order_count
                     FROM users u
                              LEFT JOIN orders o ON u.id = o.user_id
                     WHERE order_count > 5 -- can't use SELECT alias in WHERE
                     GROUP BY u.id, u.name
                     """,
            'should_pass': False
        },
        {
            'name': 'Valid - Using SELECT alias in HAVING',
            'query': """
                     SELECT u.name            as user_name,
                            COUNT(o.order_id) as order_count
                     FROM users u
                              LEFT JOIN orders o ON u.id = o.user_id
                     GROUP BY u.id, u.name
                     HAVING order_count > 5 -- allowed in HAVING
                     """,
            'should_pass': True
        },
        {
            'name': 'Valid - Multiple JOINs with aliases',
            'query': """
                     SELECT u.name,
                            o.order_id,
                            oi.quantity,
                            oi.price
                     FROM users u
                              JOIN orders o ON u.id = o.user_id
                              JOIN order_items oi ON o.order_id = oi.order_id
                     WHERE o.status = 'completed'
                     ORDER BY o.order_date DESC
                     """,
            'should_pass': True
        },
        {
            'name': 'Valid - Column alias in GROUP BY',
            'query': """
                     SELECT
                         DATE (o.order_date) as order_day, COUNT (*) as daily_orders
                     FROM orders o
                     GROUP BY order_day -- using SELECT alias
                     ORDER BY order_day
                     """,
            'should_pass': True
        }
    ]

    print("Testing Enhanced Alias Validation")
    print("=" * 80)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 80)

        # Clean query
        query = ' '.join(test_case['query'].split())

        # Validate
        result = validator.validate_query(query)

        print(f"Query: {query[:100]}..." if len(query) > 100 else f"Query: {query}")
        print(f"\nExpected: {'PASS' if test_case['should_pass'] else 'FAIL'}")
        print(f"Actual: {'PASS' if result['valid'] else 'FAIL'}")

        if result['errors']:
            print("\nErrors:")
            for error in result['errors']:
                print(f"  - {error}")

        if result['warnings']:
            print("\nWarnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")

        if result['valid'] != test_case['should_pass']:
            print(f"\n❌ TEST FAILED!")
        else:
            print(f"\n✅ Test passed")

        print(f"\nTable aliases detected: {list(validator.table_aliases.keys())}")


if __name__ == "__main__":
    test_alias_validation()
