from typing import Any, Dict, List

from sqlglot import parse_one, exp
from sqlglot.errors import ParseError


class Scope:
    def __init__(self, tables: Dict[str, str], select_aliases: set):
        self.tables = tables
        self.select_aliases = select_aliases


class SQLValidator:
    def __init__(self, dialect: str = "mysql"):
        self.dialect = dialect

    # ============================================================
    # 1. SYNTAX
    # ============================================================
    def validate_mysql_syntax(self, sql: str) -> dict:
        try:
            ast = parse_one(sql, dialect=self.dialect)
            return {"valid": True, "ast": ast}
        except ParseError as e:
            return {
                "valid": False,
                "error": {
                    "type": "SYNTAX_ERROR",
                    "message": str(e)
                }
            }

    # ============================================================
    # 2. CTE HANDLING
    # ============================================================
    def _projection_output_name(self, proj: exp.Expression, idx: int) -> str | None:
        """Best-effort name of a SELECT projection output column."""
        # sqlglot usually exposes alias via alias_or_name
        name = getattr(proj, "alias_or_name", None)
        if name:
            return str(name)

        # Plain column reference without alias
        if isinstance(getattr(proj, "this", None), exp.Column):
            return proj.this.name

        # SELECT * -> no stable name(s) here
        if isinstance(getattr(proj, "this", None), exp.Star):
            return None

        # Expression without alias: generate a stable placeholder so downstream lookups don't crash.
        return f"expr_{idx}"

    def _extract_ctes(self, ast: exp.Expression) -> Dict[str, dict]:
        """Extract top-level CTEs and their output columns.

        Contract: returns {cte_name: {"columns": {col_name: {}}}}
        Values are empty dicts because we only need column *existence*.
        """
        ctes: Dict[str, dict] = {}

        # Prefer top-level WITH stored in args; fallback to a search for compatibility across sqlglot versions.
        with_expr = ast.args.get("with") or ast.find(exp.With)
        if not with_expr:
            return ctes

        for cte in with_expr.expressions:
            name = cte.alias_or_name
            if not name:
                continue

            cols: Dict[str, dict] = {}

            # Prefer explicit CTE column list: WITH cte(col1, col2) AS (...)
            explicit_cols = cte.args.get("columns")
            if explicit_cols:
                for c in explicit_cols:
                    if isinstance(c, exp.Identifier):
                        cols[c.name] = {}
                    else:
                        # Fallback: use SQL string
                        cols[str(c)] = {}
                ctes[name] = {"columns": cols}
                continue

            # Infer from the first SELECT in the CTE body (works for SELECT / UNION / nested queries)
            body = cte.this
            if isinstance(body, exp.Subquery):
                body = body.this

            select = body.find(exp.Select) if isinstance(body, exp.Expression) else None
            if select:
                for i, proj in enumerate(select.expressions or [], start=1):
                    out_name = self._projection_output_name(proj, i)
                    if out_name:
                        cols[out_name] = {}

            ctes[name] = {"columns": cols}

        return ctes

    def _normalize_table_def(self, table_def: Any) -> dict:
        """Ensure every table def has at least a {"columns": {...}} shape."""
        if not isinstance(table_def, dict):
            return {"columns": {}}
        cols = table_def.get("columns")
        if not isinstance(cols, dict):
            cols = {}
        normalized = dict(table_def)
        normalized["columns"] = cols
        return normalized

    def _build_effective_schema(self, schema: dict, ctes: Dict[str, dict]) -> dict:
        effective = {"tables": {}}

        for t, v in (schema.get("tables") or {}).items():
            effective["tables"][t] = self._normalize_table_def(v)

        # CTEs override base tables if same name
        for cte, defn in ctes.items():
            effective["tables"][cte] = self._normalize_table_def(defn)

        return effective

    def _outer_select(self, ast: exp.Expression) -> exp.Expression | None:
        """Return the outer query SELECT for both normal and WITH statements."""
        # For WITH statements, sqlglot places the main query under ast.this.
        main = getattr(ast, "this", None)
        if isinstance(main, exp.Select):
            return main

        # If main is a set operation (UNION/EXCEPT/INTERSECT), pick its left-most SELECT.
        if isinstance(main, exp.Expression):
            s = main.find(exp.Select)
            if s:
                return s

        # Fallback (non-WITH): first SELECT in the statement.
        return ast.find(exp.Select)

    # ============================================================
    # 3. SCOPE-SAFE EXTRACTION (CTE-AWARE)
    # ============================================================
    def _extract_scoped_tables(self, ast: exp.Expression) -> Dict[str, str]:
        tables: Dict[str, str] = {}
        main_select = self._outer_select(ast)
        if not main_select:
            return tables

        def visit(node: exp.Expression) -> None:
            if isinstance(node, exp.Subquery):
                return
            if isinstance(node, exp.Select) and node is not main_select:
                return

            if isinstance(node, exp.Table):
                tables[node.alias_or_name] = node.name
                return

            for child in node.args.values():
                if isinstance(child, list):
                    for c in child:
                        if isinstance(c, exp.Expression):
                            visit(c)
                elif isinstance(child, exp.Expression):
                    visit(child)

        visit(main_select)
        return tables

    def _extract_scoped_columns(self, ast: exp.Expression) -> List[dict]:
        cols: List[dict] = []
        main_select = self._outer_select(ast)
        if not main_select:
            return cols

        def visit(node: exp.Expression) -> None:
            if isinstance(node, exp.Subquery):
                return
            if isinstance(node, exp.Select) and node is not main_select:
                return
            if isinstance(node, exp.Column):
                cols.append({"table": node.table, "column": node.name})

            for child in node.args.values():
                if isinstance(child, list):
                    for c in child:
                        if isinstance(c, exp.Expression):
                            visit(c)
                elif isinstance(child, exp.Expression):
                    visit(child)

        visit(main_select)
        return cols

    def _extract_select_aliases(self, ast: exp.Expression) -> set:
        select = self._outer_select(ast)
        if not select:
            return set()

        # Ensure aliases are plain strings for membership checks.
        aliases = set()
        for p in select.expressions or []:
            a = getattr(p, "alias_or_name", None)
            if a:
                aliases.add(str(a))
        return aliases

    # ============================================================
    # 4. SYMBOL TABLE
    # ============================================================
    def _build_root_scope(self, ast: exp.Expression) -> Scope:
        return Scope(
            tables=self._extract_scoped_tables(ast),
            select_aliases=self._extract_select_aliases(ast)
        )

    # ============================================================
    # 5. CORE SEMANTIC VALIDATION
    # ============================================================
    def validate_semantics(self, ast: exp.Expression, schema: dict) -> List[dict]:
        errors: List[dict] = []

        ctes = self._extract_ctes(ast)
        effective_schema = self._build_effective_schema(schema, ctes)
        scope = self._build_root_scope(ast)

        tables_map = effective_schema.get("tables") or {}

        # ---- Dialect-specific operator checks
        errors.extend(self._validate_mysql_unsupported_operators(ast))

        # ---- Projection safety / duplicates
        errors.extend(self._validate_duplicate_select_outputs(ast))

        # ---- Table existence
        for _, table_name in scope.tables.items():
            if table_name not in tables_map:
                errors.append({"type": "UNKNOWN_TABLE", "table": table_name})

        def table_columns(table_name: str) -> Dict[str, Any]:
            return (tables_map.get(table_name) or {}).get("columns", {}) or {}

        # ---- Column existence
        for col in self._extract_scoped_columns(ast):
            table_ref = col["table"]
            col_name = col["column"]

            # SELECT alias allowed (ORDER BY / HAVING)
            if not table_ref and col_name in scope.select_aliases:
                continue

            if table_ref:
                if table_ref not in scope.tables:
                    errors.append({
                        "type": "UNKNOWN_TABLE_ALIAS",
                        "alias": table_ref,
                        "column": col_name
                    })
                    continue

                real_table = scope.tables[table_ref]
                if real_table not in tables_map:
                    errors.append({"type": "UNKNOWN_TABLE", "table": real_table})
                    continue

                if col_name not in table_columns(real_table):
                    errors.append({
                        "type": "UNKNOWN_COLUMN",
                        "table": real_table,
                        "column": col_name
                    })
            else:
                # Skip unknown tables while matching (already reported in table existence).
                candidates = [t for t in scope.tables.values() if t in tables_map]
                matches = [t for t in candidates if col_name in table_columns(t)]

                if len(matches) == 0:
                    errors.append({"type": "UNKNOWN_COLUMN", "column": col_name})
                elif len(matches) > 1:
                    errors.append({
                        "type": "AMBIGUOUS_COLUMN",
                        "column": col_name,
                        "candidates": matches
                    })

        errors.extend(self._validate_only_full_group_by(ast, scope, effective_schema))
        errors.extend(self._validate_order_by(ast, scope, effective_schema))
        errors.extend(self._validate_window_functions(ast))

        return errors

    def _validate_duplicate_select_outputs(self, ast: exp.Expression) -> List[dict]:
        """Detect duplicate output column names in the outer SELECT.

        Why:
        - In JOIN scenarios, the LLM often selects the same logical key from both sides
          (e.g., p.product_id and cc.product_id). Even if values are equal, returning both
          is redundant and can break consumers expecting unique column names.

        Scope:
        - Outer-most SELECT only (keeps it predictable and avoids false positives inside subqueries).
        """
        errors: List[dict] = []
        select = self._outer_select(ast)
        if not select:
            return errors

        projections = list(select.expressions or [])

        # If SELECT * appears and there are joins, duplicates are likely and we can't safely
        # expand without knowing the concrete schema at runtime here.
        has_join = bool(select.args.get("joins"))
        for p in projections:
            is_star = isinstance(p, exp.Star) or isinstance(getattr(p, "this", None), exp.Star)
            if is_star and has_join:
                errors.append({
                    "type": "UNSAFE_SELECT_STAR_WITH_JOIN",
                    "hint": "Avoid SELECT * in JOIN queries; explicitly list required columns to prevent duplicates.",
                })
                # Keep going to also flag any explicit duplicates present.

        # 1) Duplicate output names / aliases
        seen: Dict[str, int] = {}
        duplicates: List[dict] = []

        for idx, proj in enumerate(projections, start=1):
            out_name = self._projection_output_name(proj, idx)
            if not out_name:
                continue
            key = out_name.lower()
            if key in seen:
                duplicates.append({
                    "name": out_name,
                    "first_index": seen[key],
                    "duplicate_index": idx,
                })
            else:
                seen[key] = idx

        if duplicates:
            errors.append({
                "type": "DUPLICATE_SELECT_OUTPUT",
                "duplicates": duplicates,
                "hint": "Each output column name must be unique. Remove redundant projections (e.g., select the join key from only one table).",
            })

        # 2) Duplicate join keys (even if aliases differ)
        # Example: SELECT a.id AS a_id, b.id AS b_id FROM a JOIN b ON a.id = b.id
        # This is redundant in most APIs: keep only one.
        join_key_dups = self._detect_selected_join_key_duplicates(select)
        if join_key_dups:
            errors.append({
                "type": "DUPLICATE_JOIN_KEY_SELECTED",
                "duplicates": join_key_dups,
                "hint": "Join keys are equal by the JOIN condition; select the join key from only one side unless you explicitly need both.",
            })

        return errors

    def _detect_selected_join_key_duplicates(self, select: exp.Expression) -> List[dict]:
        """Return duplicates where both sides of an equi-join key are projected.

        We only inspect the outer SELECT and equi-join conditions of the form:
            <col> = <col>
        combined with:
            SELECT <left_col> ... <right_col> ...

        Notes:
        - This is intentionally conservative: it won't try to reason about expressions like
          COALESCE(a.id, 0) = b.id.
        - It ignores stars (handled separately).
        - It only treats it as a "duplicate join key" when both sides have the SAME column name
          (e.g., a.product_id = b.product_id). If the names differ (customer_id = product_id),
          that's far more likely to be a wrong join than an intentional equivalence.
        """
        dups: List[dict] = []

        # Build a set of projected column references (table alias + column name)
        projected_cols: set[tuple[str, str]] = set()
        for proj in list(select.expressions or []):
            # Extract any Column nodes referenced by the projection.
            # We count direct column projections and columns embedded in expressions.
            for c in proj.find_all(exp.Column) if isinstance(proj, exp.Expression) else []:
                if c.table and c.name:
                    projected_cols.add((c.table, c.name))

        joins = select.args.get("joins") or []
        for j in joins:
            on_expr = j.args.get("on")
            if not on_expr:
                continue

            # Find a = b patterns inside ON
            for eq in on_expr.find_all(exp.EQ):
                left = eq.left
                right = eq.right
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                    continue
                if not left.table or not right.table:
                    continue

                # Only treat same-named columns as join-key duplicates.
                if left.name != right.name:
                    continue

                left_key = (left.table, left.name)
                right_key = (right.table, right.name)

                if left_key in projected_cols and right_key in projected_cols:
                    dups.append({
                        "left": {"table_alias": left.table, "column": left.name},
                        "right": {"table_alias": right.table, "column": right.name},
                    })

        return dups

    def _validate_mysql_unsupported_operators(self, ast: exp.Expression) -> List[dict]:
        """Flag operators that are not supported in MySQL but may be emitted by the LLM."""
        errors: List[dict] = []

        # sqlglot may or may not expose exp.ILike depending on version.
        ILikeExp = getattr(exp, "ILike", None)
        if ILikeExp:
            for _ in ast.find_all(ILikeExp):
                errors.append({
                    "type": "UNSUPPORTED_OPERATOR",
                    "operator": "ILIKE",
                    "dialect": self.dialect,
                    "hint": "MySQL does not support ILIKE. Use LIKE (or LOWER(col) LIKE LOWER(pattern) for case-insensitive matching).",
                })
        else:
            # Conservative fallback: catch literal operator text if present.
            # This is not perfect, but it prevents silent acceptance.
            sql_text = ast.sql(dialect=self.dialect)
            if " ILIKE " in sql_text.upper():
                errors.append({
                    "type": "UNSUPPORTED_OPERATOR",
                    "operator": "ILIKE",
                    "dialect": self.dialect,
                    "hint": "MySQL does not support ILIKE. Use LIKE (or LOWER(col) LIKE LOWER(pattern) for case-insensitive matching).",
                })

        return errors

    # ============================================================
    # 6. ORDER BY
    # ============================================================
    def _validate_order_by(self, ast: exp.Expression, scope: Scope, schema: dict) -> List[dict]:
        errors: List[dict] = []
        order = ast.find(exp.Order)
        select = self._outer_select(ast)

        if not order or not select:
            return errors

        tables_map = (schema.get("tables") or {})

        for expr in order.expressions:
            e = expr.this

            if isinstance(e, exp.Literal) and e.is_int:
                pos = int(e.this)
                if pos < 1 or pos > len(select.expressions):
                    errors.append({"type": "INVALID_ORDER_BY_POSITION"})
                continue

            # ORDER BY alias
            if isinstance(e, exp.Column) and not e.table and e.name in scope.select_aliases:
                continue

            # Validate any column references inside ORDER BY expressions (e.g., a+b)
            for c in e.find_all(exp.Column) if isinstance(e, exp.Expression) else []:
                if c.table:
                    if c.table not in scope.tables:
                        errors.append({"type": "UNKNOWN_TABLE_ALIAS", "alias": c.table})
                        continue
                    real = scope.tables[c.table]
                    cols = (tables_map.get(real) or {}).get("columns", {}) or {}
                    if c.name not in cols:
                        errors.append({"type": "UNKNOWN_COLUMN", "table": real, "column": c.name})
                else:
                    # Unqualified in ORDER BY: allow SELECT alias
                    if c.name in scope.select_aliases:
                        continue
                    candidates = [t for t in scope.tables.values() if t in tables_map]
                    matches = [t for t in candidates if c.name in ((tables_map.get(t) or {}).get("columns", {}) or {})]
                    if len(matches) == 0:
                        errors.append({"type": "UNKNOWN_COLUMN", "column": c.name})
                    elif len(matches) > 1:
                        errors.append({"type": "AMBIGUOUS_COLUMN", "column": c.name, "candidates": matches})

        return errors

    # ============================================================
    # 7. ONLY_FULL_GROUP_BY
    # ============================================================
    def _validate_only_full_group_by(self, ast, scope, schema):
        errors = []

        select = self._outer_select(ast)
        if not select:
            return errors

        group = select.args.get("group")
        if not group:
            return errors

        grouped = {
            (g.table, g.name)
            for g in group.expressions
            if isinstance(g, exp.Column)
        }

        for proj in select.expressions:
            if isinstance(proj.this, exp.AggFunc):
                continue
            if isinstance(proj.this, exp.Column):
                key = (proj.this.table, proj.this.name)
                if key not in grouped:
                    errors.append({
                        "type": "ONLY_FULL_GROUP_BY_VIOLATION",
                        "column": proj.this.sql()
                    })

        return errors

    # ============================================================
    # 8. WINDOW FUNCTION PLACEMENT
    # ============================================================
    def _validate_window_functions(self, ast):
        errors = []
        for win in ast.find_all(exp.Window):
            parent = win.parent
            while parent:
                if isinstance(parent, exp.Where):
                    errors.append({"type": "INVALID_WINDOW_FUNCTION_SCOPE", "clause": "WHERE"})
                    break
                if isinstance(parent, exp.Group):
                    errors.append({"type": "INVALID_WINDOW_FUNCTION_SCOPE", "clause": "GROUP"})
                    break
                parent = parent.parent
        return errors

    # ============================================================
    # 9. PUBLIC API
    # ============================================================
    def validate_mysql_sql(self, sql: str, schema: dict) -> dict:
        syntax = self.validate_mysql_syntax(sql)
        if not syntax["valid"]:
            return {"valid": False, "stage": "syntax", "errors": [syntax["error"]]}

        ast = syntax["ast"]
        semantic_errors = self.validate_semantics(ast, schema)

        if semantic_errors:
            return {"valid": False, "stage": "semantic", "errors": semantic_errors}

        return {"valid": True, "stage": "ok", "errors": []}



sql_validator = SQLValidator()