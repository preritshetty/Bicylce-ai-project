import re
import pandas as pd
from typing import List
import warnings
warnings.filterwarnings('ignore')


class BasicCleaner:
    """Simple data hygiene operations for CSV data"""

    def __init__(self):
        self.cleaning_log = []

    # -----------------------------
    # Basic ops
    # -----------------------------
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicate rows"""
        df = df.copy()
        original_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = original_count - len(df)
        self.cleaning_log.append({
            'operation': 'remove_duplicates',
            'details': f"Removed {duplicates_removed} duplicate rows",
            'success': True
        })
        return df

    def clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns by removing extra whitespace"""
        df = df.copy()
        text_columns = df.select_dtypes(include=['object', 'string']).columns
        cleaned_columns = []
        for column in text_columns:
            try:
                mask = df[column].notna()
                if mask.any():
                    df.loc[mask, column] = df.loc[mask, column].astype(str).str.strip()
                    df.loc[mask, column] = df.loc[mask, column].str.replace(r'\s+', ' ', regex=True)
                    cleaned_columns.append(column)
            except Exception:
                # keep going if a single column misbehaves
                pass
        if cleaned_columns:
            self.cleaning_log.append({
                'operation': 'clean_text',
                'details': f"Cleaned text in {len(cleaned_columns)} columns",
                'success': True
            })
        return df

    def standardize_case(self, df: pd.DataFrame, case_type: str = 'title') -> pd.DataFrame:
        """Standardize text case for text columns"""
        df = df.copy()
        text_columns = df.select_dtypes(include=['object', 'string']).columns
        processed_columns = []
        for column in text_columns:
            try:
                mask = df[column].notna()
                if not mask.any():
                    continue
                if case_type == 'title':
                    df.loc[mask, column] = df.loc[mask, column].astype(str).str.title()
                elif case_type == 'upper':
                    df.loc[mask, column] = df.loc[mask, column].astype(str).str.upper()
                elif case_type == 'lower':
                    df.loc[mask, column] = df.loc[mask, column].astype(str).str.lower()
                processed_columns.append(column)
            except Exception:
                pass
        if processed_columns:
            self.cleaning_log.append({
                'operation': 'standardize_case',
                'details': f"Standardized case in {len(processed_columns)} columns to {case_type}",
                'success': True
            })
        return df

    # -----------------------------
    # Legacy helpers (kept for compatibility)
    # -----------------------------
    def remove_uniform_prefixes(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Handle 'Token_' style prefixes (e.g., Name_, Pilot_)"""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
        df = df.copy()
        changed_cols = []
        for col in df.select_dtypes(include=["object", "string"]).columns:
            s = df[col].astype(str).str.strip()
            tokens = s.str.extract(r'^([A-Za-z]+)_', expand=False)
            has_prefix = tokens.notna()
            if has_prefix.mean() >= threshold:
                top = tokens.mode(dropna=True)
                if not top.empty:
                    token = top.iloc[0]
                    df[col] = s.str.replace(rf'^{re.escape(token)}_', '', regex=True)
                    changed_cols.append((col, f"{token}_", round(100 * has_prefix.mean(), 2)))
        if changed_cols:
            self.cleaning_log.append({
                "operation": "remove_uniform_prefixes",
                "details": [{"column": c, "removed_prefix": p, "%rows": pct} for c, p, pct in changed_cols],
                "success": True
            })
        return df

    def remove_uniform_alpha_prefixes(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95,
        min_prefix_len: int = 3,
        max_prefix_len: int = 10
    ) -> pd.DataFrame:
        """Handle glued alphabetic prefixes (e.g., Claimjlo -> jlo, Locwfq -> wfq)"""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df

        def longest_common_prefix(strs: List[str]) -> str:
            if not strs:
                return ""
            s1, s2 = min(strs), max(strs)
            i = 0
            while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
                i += 1
            return s1[:i]

        df = df.copy()
        changed = []
        for col in df.select_dtypes(include=["object", "string"]).columns:
            s = df[col].astype(str).str.strip()
            if s.empty:
                continue
            tokens = s.str.extract(r'^([A-Za-z]+)', expand=False)
            token_presence = tokens.notna().mean()
            if token_presence < threshold:
                continue
            token_list = tokens.dropna().str.lower().tolist()
            candidate = longest_common_prefix(token_list).strip()
            if not candidate or len(candidate) < min_prefix_len:
                continue
            candidate = candidate[:max_prefix_len]
            starts_with = s.str[:len(candidate)].str.lower().eq(candidate)
            share = starts_with.mean()
            if share < threshold:
                continue
            remainder_nonempty = s.str[len(candidate):].str.len().gt(0).mean()
            if remainder_nonempty < 0.9:
                continue
            df[col] = s.where(~starts_with, s.str[len(candidate):])
            changed.append({"column": col, "removed_prefix": candidate, "%rows": round(100 * share, 2)})
        if changed:
            self.cleaning_log.append({
                "operation": "remove_uniform_alpha_prefixes",
                "details": changed,
                "success": True
            })
        return df

    # -----------------------------
    # Unified robust prefix remover (use this in the pipeline)
    # -----------------------------
    def strip_uniform_prefixes(
        self,
        df: pd.DataFrame,
        threshold: float = 0.85,          # tolerant to outliers
        min_prefix_len: int = 3,
        max_prefix_len: int = 12
    ) -> pd.DataFrame:
        """
        Remove a uniform leading alphabetic prefix across a column, whether:
          - 'Token_' (underscore),
          - 'Token ' / 'Token-' / 'Token:' (separators),
          - or glued 'TokenRest' (e.g., Gatezn, Claimjlo, Locwfq, Typeruh).
        Works on ALL columns by casting to string view.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df

        def lcp(strs: List[str]) -> str:
            if not strs:
                return ""
            a, b = min(strs), max(strs)
            i = 0
            while i < len(a) and i < len(b) and a[i] == b[i]:
                i += 1
            return a[:i]

        df = df.copy()
        changed = []

        for col in df.columns:
            s = df[col].astype(str).str.strip()
            if s.empty:
                continue

            # 1) Fast path: explicit Token_
            token_series = s.str.extract(r'^([A-Za-z]+)_', expand=False)
            mode_token = token_series.mode(dropna=True)
            if not mode_token.empty:
                token = mode_token.iloc[0]
                has = s.str.match(rf'^{re.escape(token)}_', na=False)
                share = has.mean()
                if share >= threshold and len(token) >= min_prefix_len:
                    df[col] = s.str.replace(rf'^{re.escape(token)}_', '', regex=True)
                    changed.append({"column": col, "removed_prefix": token + "_", "%rows": round(100*share, 2)})
                    continue  # done with this column

            # 2) Generic: leading letters (glued or with separators)
            alpha = s.str.extract(r'^([A-Za-z]+)', expand=False)
            if alpha.isna().all():
                continue

            tokens = alpha.dropna().str.lower().tolist()
            candidate = lcp(tokens).strip()
            if not candidate or len(candidate) < min_prefix_len:
                continue
            candidate = candidate[:max_prefix_len]

            starts = s.str[:len(candidate)].str.lower().eq(candidate)
            share = starts.mean()
            if share < threshold:
                # Try slightly shorter (helps 'Types' vs 'Type')
                cand2 = candidate[:-1]
                if len(cand2) >= min_prefix_len:
                    starts2 = s.str[:len(cand2)].str.lower().eq(cand2)
                    if starts2.mean() >= threshold:
                        candidate, starts, share = cand2, starts2, starts2.mean()
                    else:
                        continue
                else:
                    continue

            # Safety: ensure a remainder exists for most rows
            remainder_ok = s.loc[starts].str[len(candidate):].str.len().gt(0).mean() >= 0.90
            if not remainder_ok:
                continue

            # Remove candidate (case-insensitive via mask + slicing)
            df[col] = s.where(~starts, s.str[len(candidate):])
            changed.append({"column": col, "removed_prefix": candidate, "%rows": round(100*share, 2)})

        if changed:
            self.cleaning_log.append({
                "operation": "strip_uniform_prefixes",
                "details": changed,
                "success": True
            })
        return df

    # -----------------------------
    # Fallback: known prefixes for your schema (optional but handy)
    # -----------------------------
    def force_known_prefixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If most rows in specific known columns start with a short expected prefix,
        strip it even if the general threshold logic didn't trigger.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df

        df = df.copy()
        changed = []

        # map: lowercase column name -> expected lowercase prefix
        rules = {
            "gate": "gate",
            "baggage_claim": "claim",
            "layover_locations": "loc",
            "aircraft_type": "type",
        }

        cols_lc = {c.lower(): c for c in df.columns}
        for want_col_lc, prefix in rules.items():
            if want_col_lc not in cols_lc:
                continue
            col = cols_lc[want_col_lc]
            s = df[col].astype(str).str.strip()
            starts = s.str[:len(prefix)].str.lower().eq(prefix)
            share = starts.mean()   
            if share >= 0.70:  # softer threshold
                df[col] = s.where(~starts, s.str[len(prefix):])
                changed.append({"column": col, "removed_prefix": prefix, "%rows": round(100*share, 2)})

        if changed:
            self.cleaning_log.append({
                "operation": "force_known_prefixes",
                "details": changed,
                "success": True
            })
        return df

    # -----------------------------
    # Pipeline
    # -----------------------------
    def perform_basic_cleaning(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        clean_text: bool = True,
        case_type: str = "title"
    ) -> pd.DataFrame:
        """Perform all basic cleaning operations safely"""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df

        df = df.copy()
        self.cleaning_log = []  # Reset log

        # 1) Duplicates
        if remove_duplicates:
            df = self.remove_duplicates(df)

        # 2) Trim whitespace / normalize spaces
        if clean_text:
            df = self.clean_text_columns(df)

        # 2.5) Strip uniform prefixes (underscore / separated / glued)
        df = self.strip_uniform_prefixes(df, threshold=0.85)

        # 2.6) Safety net for known columns (gate/claim/loc/type)
        df = self.force_known_prefixes(df)

        # 3) Case standardization
        if case_type and case_type.lower() != "none":
            df = self.standardize_case(df, case_type)

        return df

    # -----------------------------
    # Reporting
    # -----------------------------
    def get_cleaning_report(self) -> pd.DataFrame:
        """Get a report of cleaning operations"""
        if not self.cleaning_log:
            return pd.DataFrame()
        return pd.DataFrame(self.cleaning_log)

    def reset_log(self):
        """Reset the cleaning log"""
        self.cleaning_log = []
