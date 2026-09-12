import os
import re
import json
import base64
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from groq_service import groq_service
from pdf_bw_report import generate_black_and_white_pdf

logger = logging.getLogger(__name__)

PIPELINE_STEPS_CONTEXT = {
    "upload": {
        "title": "Upload Data",
        "description": "Upload CSV, Excel, or JSON datasets into the cleaning pipeline.",
        "meaning": "This step ingests your raw tabular files, auto-detects column delimiters, character encodings (UTF-8, Latin-1), and parses initial headers and row counts.",
        "guidance": "If your file has strange characters, ensure UTF-8 encoding. If your file is Excel (.xlsx/.xls), select the sheet containing tabular data. You can upload via the page or directly in my chat box!"
    },
    "summary": {
        "title": "Data Summary & Column Profiling",
        "description": "Explores column data types, missing value ratios, and cardinality.",
        "meaning": "This step analyzes the health of your dataset: data types (numerical vs categorical vs datetime), null/missing percentages per column, unique value counts, and memory usage.",
        "guidance": "Look for columns with >30% missing values (which might need imputation or dropping), or numerical columns misidentified as text due to symbols or currency signs."
    },
    "configuration": {
        "title": "Cleaning Configuration",
        "description": "Configure deduplication, missing value imputation, type normalization, and label fixing.",
        "meaning": "Here you define how the pipeline should clean your data: whether to drop exact/fuzzy duplicates, impute missing values with mean/median/mode, or apply column encryption.",
        "guidance": "For skewed numerical data, median imputation is safer than mean. For categorical columns, use mode imputation or a 'Missing' label. You can ask me 'remove duplicates' or 'encrypt column X' to do it automatically!"
    },
    "outliers": {
        "title": "Outlier Detection & Treatment",
        "description": "Detect statistical divergence using IQR (Interquartile Range) or Z-score.",
        "meaning": "Outliers are extreme data points that can distort statistical models. We calculate Q1, Q3, and IQR = Q3 - Q1. Values beyond [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are flagged.",
        "guidance": "You can choose 'Winsorization' (caps extreme values to the 5th and 95th percentiles without losing rows) or 'Removal' (drops outlier records). Ask me 'handle outliers' to execute it!"
    },
    "weights": {
        "title": "Weights & Survey Estimation",
        "description": "Apply sample survey weights to calculate balanced population estimates.",
        "meaning": "Survey weights adjust for sampling bias where certain demographics or districts were over- or under-represented. Multiplying values by weights produces representative estimates.",
        "guidance": "Select the column containing sample weights (e.g. 'sample_weight' or 'district_weight') to calculate weighted means, standard errors, and 95% confidence intervals."
    },
    "results": {
        "title": "Results & Final Reports",
        "description": "Export the cleaned dataset in CSV, HTML, or Black & White audited PDF.",
        "meaning": "The final stage where all pipeline operations are consolidated into a reproducible audit trail with downloadable data and executive summary reports.",
        "guidance": "You can download the cleaned CSV for machine learning, view the interactive HTML report, or download the Black & White audited PDF report with embedded charts and borders!"
    }
}

class NarayanPipelineService:
    """
    Narayan - Master AI Agent for Data Cleaning Pipeline.
    Maintains pipeline context, executes Python/pandas transformations on demand,
    and exports in CSV, HTML, and Black & White PDF reports.
    """

    def __init__(self):
        self.active_df = None
        self.active_filename = None
        self.cleaning_log = []
        self.encrypted_columns = []
        self._load_default_sample()

    def _load_default_sample(self):
        """Pre-load sample dataset if available"""
        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        sample_path = os.path.join(upload_dir, 'district_health.csv')
        if os.path.exists(sample_path):
            try:
                self.active_df = pd.read_csv(sample_path)
                self.active_filename = "district_health.csv"
                self.cleaning_log.append("Loaded default district_health.csv dataset.")
            except Exception as e:
                logger.warning(f"Failed to load default sample: {e}")

    def load_dataset(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Load new dataset into Narayan workspace"""
        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext in ['.xlsx', '.xls']:
                self.active_df = pd.read_excel(file_path)
            elif ext == '.json':
                self.active_df = pd.read_json(file_path)
            else:
                encodings = ['utf-8', 'latin-1', 'cp1252']
                for enc in encodings:
                    try:
                        self.active_df = pd.read_csv(file_path, encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue

            self.active_filename = filename
            self.cleaning_log = [f"Loaded dataset '{filename}' ({len(self.active_df):,} rows, {len(self.active_df.columns)} columns)"]
            self.encrypted_columns = []

            return {
                "success": True,
                "filename": filename,
                "rows": len(self.active_df),
                "columns": len(self.active_df.columns),
                "column_names": list(self.active_df.columns)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def remove_duplicates(self, subset_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Deduplicate records using Python/pandas"""
        if self.active_df is None:
            return {"success": False, "error": "No dataset loaded."}
        
        initial_count = len(self.active_df)
        valid_cols = [c for c in subset_cols if c in self.active_df.columns] if subset_cols else None
        self.active_df = self.active_df.drop_duplicates(subset=valid_cols, keep='first')
        removed_count = initial_count - len(self.active_df)
        
        msg = f"Deduplication completed: Removed {removed_count:,} duplicate rows. Remaining rows: {len(self.active_df):,}."
        self.cleaning_log.append(msg)
        return {
            "success": True,
            "action": "remove_duplicates",
            "removed": removed_count,
            "remaining": len(self.active_df),
            "message": msg
        }

    def encrypt_columns(self, target_cols: List[str], method: str = "pbkdf2") -> Dict[str, Any]:
        """Apply cryptographic hashing / encryption to designated sensitive columns"""
        if self.active_df is None:
            return {"success": False, "error": "No dataset loaded."}

        cols_to_encrypt = [c for c in target_cols if c in self.active_df.columns]
        if not cols_to_encrypt:
            # Auto-detect PII if not specified
            for c in self.active_df.columns:
                lower = c.lower()
                if any(k in lower for k in ['phone', 'mobile', 'email', 'name', 'aadhaar', 'ssn', 'patient', 'doctor_id']):
                    cols_to_encrypt.append(c)

        if not cols_to_encrypt:
            # Fallback to first text column
            obj_cols = self.active_df.select_dtypes(include=['object']).columns.tolist()
            if obj_cols:
                cols_to_encrypt = [obj_cols[0]]

        salt = b'narayan_pipeline_enterprise_salt_2026'

        for col in cols_to_encrypt:
            def _encrypt_val(val):
                if pd.isna(val):
                    return val
                key = hashlib.pbkdf2_hmac('sha256', str(val).encode('utf-8'), salt, 50000)
                return "ENC_" + base64.b64encode(key).decode('utf-8')[:16]

            self.active_df[col] = self.active_df[col].apply(_encrypt_val)
            if col not in self.encrypted_columns:
                self.encrypted_columns.append(col)

        msg = f"Encryption applied: Secured columns {cols_to_encrypt} using PBKDF2-SHA256."
        self.cleaning_log.append(msg)
        return {
            "success": True,
            "action": "encryption",
            "encrypted_columns": cols_to_encrypt,
            "message": msg
        }

    def impute_missing(self, method: str = "median", columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Impute missing null values with mean, median, mode, or constant"""
        if self.active_df is None:
            return {"success": False, "error": "No dataset loaded."}

        num_cols = self.active_df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = [c for c in columns if c in num_cols] if columns else num_cols
        imputed_count = 0

        for col in target_cols:
            null_count = int(self.active_df[col].isnull().sum())
            if null_count > 0:
                if method == "mean":
                    val = self.active_df[col].mean()
                elif method == "median":
                    val = self.active_df[col].median()
                elif method == "mode":
                    mode_s = self.active_df[col].mode()
                    val = mode_s[0] if not mode_s.empty else 0
                else:
                    val = 0
                self.active_df[col].fillna(val, inplace=True)
                imputed_count += null_count

        msg = f"Missing value imputation ({method.upper()}): Imputed {imputed_count:,} missing cells across {len(target_cols)} columns."
        self.cleaning_log.append(msg)
        return {
            "success": True,
            "action": "impute_missing",
            "method": method,
            "imputed_cells": imputed_count,
            "message": msg
        }

    def handle_outliers(self, method: str = "winsorize", percentile: float = 5.0) -> Dict[str, Any]:
        """Winsorize or clip outliers using Python/pandas"""
        if self.active_df is None:
            return {"success": False, "error": "No dataset loaded."}

        num_cols = self.active_df.select_dtypes(include=[np.number]).columns.tolist()
        initial_rows = len(self.active_df)

        if method == "winsorize":
            for col in num_cols:
                lower = self.active_df[col].quantile(percentile / 100.0)
                upper = self.active_df[col].quantile(1 - percentile / 100.0)
                self.active_df[col] = self.active_df[col].clip(lower, upper)
            msg = f"Outliers winsorized at {percentile}th-{100-percentile}th percentiles across {len(num_cols)} numerical columns."
        else:
            # Remove outside 1.5*IQR
            for col in num_cols:
                q1 = self.active_df[col].quantile(0.25)
                q3 = self.active_df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lb = q1 - 1.5 * iqr
                    ub = q3 + 1.5 * iqr
                    self.active_df = self.active_df[(self.active_df[col] >= lb) & (self.active_df[col] <= ub)]
            removed = initial_rows - len(self.active_df)
            msg = f"Outliers removed: Dropped {removed:,} rows exceeding 1.5*IQR bounds. Remaining rows: {len(self.active_df):,}."

        self.cleaning_log.append(msg)
        return {
            "success": True,
            "action": "handle_outliers",
            "method": method,
            "message": msg,
            "remaining_rows": len(self.active_df)
        }

    def export_cleaned_csv(self) -> Dict[str, Any]:
        """Export the current transformed dataset as CSV"""
        if self.active_df is None:
            return {"success": False, "error": "No dataset loaded."}

        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        filename = f"narayan_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(upload_dir, filename)
        self.active_df.to_csv(filepath, index=False)

        return {
            "success": True,
            "format": "csv",
            "filename": filename,
            "download_url": f"/api/ai-assistant/download/{filename}",
            "rows": len(self.active_df),
            "columns": len(self.active_df.columns)
        }

    def export_pdf_report(self) -> Dict[str, Any]:
        """Generate Black & White audited PDF report with borders and embedded graph"""
        if self.active_df is None or self.active_df.empty:
            self._load_default_sample()
            
        if self.active_df is None or self.active_df.empty:
            return {"success": False, "error": "No dataset loaded."}

        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        filename = f"Narayan_Pipeline_Audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(upload_dir, filename)

        generate_black_and_white_pdf(self.active_df, filepath, self.active_filename or "dataset.csv", agent_name="Narayan")

        return {
            "success": True,
            "format": "pdf",
            "filename": filename,
            "download_url": f"/api/ai-assistant/download/{filename}",
            "theme": "detailed_black_and_white_with_borders_and_graph"
        }

    def export_html_report(self) -> Dict[str, Any]:
        """Generate interactive HTML summary report with actual data records"""
        if self.active_df is None or self.active_df.empty:
            self._load_default_sample()

        if self.active_df is None or self.active_df.empty:
            return {"success": False, "error": "No dataset loaded."}

        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        filename = f"Narayan_Pipeline_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(upload_dir, filename)

        rows, cols = self.active_df.shape
        num_cols = self.active_df.select_dtypes(include=[np.number]).columns.tolist()

        logs_html = "".join([f"<li>{entry}</li>" for entry in self.cleaning_log])

        # Generate actual data preview rows
        sample_rows_df = self.active_df.head(25)
        display_columns = list(self.active_df.columns)[:8]
        th_html = "".join([f"<th style='background:#000; color:#fff; padding:8px; border:1px solid #333; text-align:left;'>{col}</th>" for col in display_columns])

        rows_tr_html = ""
        for i, (_, r) in enumerate(sample_rows_df.iterrows()):
            bg = "#ffffff" if i % 2 == 0 else "#f8f9fa"
            cells = "".join([f"<td style='padding:6px 8px; border:1px solid #ddd;'>{str(r[c])[:35] if not pd.isna(r[c]) else '<i style=\"color:#888;\">null</i>'}</td>" for c in display_columns])
            rows_tr_html += f"<tr style='background:{bg};'>{cells}</tr>"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Narayan Pipeline Audit Report - {self.active_filename}</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #ffffff; color: #111; padding: 40px; margin: 0 auto; max-width: 960px; }}
        .border-box {{ border: 2px solid #000; padding: 25px; margin-bottom: 25px; }}
        h1 {{ font-size: 20px; text-transform: uppercase; border-bottom: 2px solid #000; padding-bottom: 8px; margin-top: 0; }}
        h2 {{ font-size: 13px; text-transform: uppercase; background: #000; color: #fff; padding: 6px 12px; margin-top: 25px; letter-spacing: 0.5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }}
        th, td {{ border: 1px solid #000; padding: 6px 10px; text-align: left; }}
        th {{ background: #f2f2f2; font-weight: bold; }}
        ul {{ padding-left: 20px; font-size: 13px; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; background: #e5e7eb; }}
    </style>
</head>
<body>
    <div class="border-box">
        <h1>Bhishma AI - Narayan Data Cleaning Pipeline Report</h1>
        <p><strong>Agent:</strong> Narayan (Pipeline Master) | <strong>Dataset:</strong> {self.active_filename} | <strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        <p><strong>Total Records:</strong> <span class="badge">{rows:,} Rows</span> | <strong>Total Columns:</strong> <span class="badge">{cols} Columns</span> | <strong>Encrypted Columns:</strong> {', '.join(self.encrypted_columns) if self.encrypted_columns else 'None'}</p>
        
        <h2>1. Pipeline Execution Audit Log</h2>
        <ul>{logs_html}</ul>

        <h2>2. Actual Dataset Records Inspection Table (First {len(sample_rows_df)} Rows)</h2>
        <p style="font-size:12px; color:#555; margin-top:4px;">Displaying sample data records from <strong>{self.active_filename}</strong> across key attributes.</p>
        <div style="overflow-x:auto; margin-top:10px; margin-bottom:20px;">
            <table style="width:100%; border-collapse:collapse;">
                <thead>
                    <tr>{th_html}</tr>
                </thead>
                <tbody>
                    {rows_tr_html}
                </tbody>
            </table>
        </div>

        <h2>3. Cleaned Metric Distributions & Summary Statistics</h2>
        <table>
            <tr><th>Attribute</th><th>Mean</th><th>Std Dev</th><th>Median</th><th>Min</th><th>Max</th></tr>
            {''.join([f"<tr><td><strong>{c}</strong></td><td>{self.active_df[c].mean():.2f}</td><td>{self.active_df[c].std():.2f}</td><td>{self.active_df[c].median():.2f}</td><td>{self.active_df[c].min():.2f}</td><td>{self.active_df[c].max():.2f}</td></tr>" for c in num_cols[:8]])}
        </table>
    </div>
</body>
</html>"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return {
            "success": True,
            "format": "html",
            "filename": filename,
            "download_url": f"/api/ai-assistant/download/{filename}"
        }

    def chat_with_narayan(self, user_message: str, current_step: str = "upload", conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Main reasoning & execution loop for Narayan.
        1. Contextual guidance for the current pipeline step.
        2. Automatic execution of data operations (dedup, encrypt, impute, outliers).
        3. Automatic export in CSV, HTML, and Black & White PDF.
        """
        msg_lower = user_message.lower().strip()
        executed_action = None
        action_result = None
        downloads = []

        # Check for executable transformation commands
        if any(kw in msg_lower for kw in ['remove duplicate', 'remove duplicates', 'drop duplicate', 'dedup']):
            action_result = self.remove_duplicates()
            executed_action = "remove_duplicates"

        elif any(kw in msg_lower for kw in ['encrypt', 'encryption', 'hash column', 'protect column', 'secure column', 'pbkdf2', 'aes']):
            # Look for specific columns or auto-detect
            cols = []
            if self.active_df is not None:
                for c in self.active_df.columns:
                    if c.lower() in msg_lower:
                        cols.append(c)
            action_result = self.encrypt_columns(cols)
            executed_action = "encryption"

        elif any(kw in msg_lower for kw in ['impute', 'fill missing', 'replace missing', 'handle missing']):
            method = "median" if "median" in msg_lower else ("mean" if "mean" in msg_lower else "mode")
            action_result = self.impute_missing(method)
            executed_action = "impute_missing"

        elif any(kw in msg_lower for kw in ['handle outlier', 'handle outliers', 'winsorize', 'remove outlier', 'remove outliers']):
            method = "remove" if "remove" in msg_lower else "winsorize"
            action_result = self.handle_outliers(method)
            executed_action = "handle_outliers"

        # Check for exports
        if any(kw in msg_lower for kw in ['export csv', 'download csv', 'save csv', 'output csv']):
            res = self.export_cleaned_csv()
            if res.get("success"):
                downloads.append(res)

        if any(kw in msg_lower for kw in ['pdf', 'black and white pdf', 'audit report', 'generate pdf']):
            res = self.export_pdf_report()
            if res.get("success"):
                downloads.append(res)

        if any(kw in msg_lower for kw in ['html', 'html report', 'interactive report']):
            res = self.export_html_report()
            if res.get("success"):
                downloads.append(res)

        # Build prompt with step context and dataset stats
        step_info = PIPELINE_STEPS_CONTEXT.get(current_step, PIPELINE_STEPS_CONTEXT["upload"])
        df_summary = "No dataset currently active."
        if self.active_df is not None:
            rows, cols = self.active_df.shape
            num_cols = self.active_df.select_dtypes(include=[np.number]).columns.tolist()
            df_summary = (
                f"Active Dataset: {self.active_filename}\n"
                f"- Rows: {rows:,}\n"
                f"- Columns ({cols}): {', '.join(list(self.active_df.columns)[:10])}...\n"
                f"- Missing Values: {int(self.active_df.isnull().sum().sum()):,}\n"
                f"- Encrypted Columns: {', '.join(self.encrypted_columns) if self.encrypted_columns else 'None'}\n"
                f"- Recent Log: {'; '.join(self.cleaning_log[-3:]) if self.cleaning_log else 'Initial load'}"
            )

        system_prompt = f"""You are Narayan, the Master AI Strategist of the Data Cleaning Pipeline in Bhishma AI.
You possess total mastery over every step of data preparation, cleaning, imputation, transformation, and validation.

CURRENT PIPELINE STEP: '{step_info['title']}' (ID: {current_step})
Step Overview: {step_info['description']}
Meaning of this Step: {step_info['meaning']}
Best-Practice Guidance: {step_info['guidance']}

ACTIVE DATASET CONTEXT:
{df_summary}

YOUR RESPONSIBILITIES AS NARAYAN:
1. When asked 'what is the meaning of this' or 'what does this page mean', clearly explain the purpose of the '{step_info['title']}' stage, what the controls do, and how it impacts data quality.
2. When asked 'if I want to do X, what should I do', provide direct, step-by-step guidance.
3. If an automated transformation was just executed, acknowledge it clearly with before/after statistics.
4. If the user asked for CSV, HTML, or PDF reports, confirm that the outputs are ready for download.
5. Tone: Knowledgeable, proactive, concise, professional, and empowering. Use Markdown bolding and bullet points.
"""

        # Call Groq / Gemini LLM with Narayan's specific persona
        prompt_with_action = user_message
        if action_result:
            prompt_with_action += f"\n\n[SYSTEM NOTIFICATION: Automated Action Executed: {json.dumps(action_result)}]"

        groq_resp = groq_service.ask_dataset(
            user_question=prompt_with_action,
            df=self.active_df,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            agent_name="Narayan"
        )
        llm_text = groq_resp.get("response", "")

        # If an automated action was executed, prepend it clearly
        if action_result and not llm_text.startswith("### ⚙️"):
            llm_text = f"### ⚙️ Action Executed by Narayan\n\n{action_result.get('message')}\n\n" + llm_text

        # If LLM didn't return text or user explicitly asked for step guidance
        if not llm_text:
            if action_result:
                llm_text = f"### ⚙️ Narayan Action Completed\n\n{action_result.get('message')}\n\nYour dataset has been updated in memory. You can continue cleaning or export to CSV, HTML, or PDF."
            elif any(w in msg_lower for w in ["meaning", "what does this mean", "what is this page"]):
                llm_text = f"### 📘 Meaning of '{step_info['title']}'\n\n{step_info['meaning']}\n\n**Key Actions on this Page:**\n• {step_info['guidance']}"
            else:
                llm_text = f"### 💡 Narayan Pipeline Guidance\n\nTo clean and prepare your dataset effectively on the **{step_info['title']}** page:\n\n• **What to do:** {step_info['guidance']}\n• **Quick commands:** Try asking me to *'remove duplicates'*, *'encrypt sensitive columns'*, *'fill missing values'*, or *'generate B&W PDF report'*!"

        return {
            "success": True,
            "agent": "Narayan",
            "current_step": current_step,
            "step_title": step_info["title"],
            "response": llm_text,
            "action_executed": executed_action,
            "action_result": action_result,
            "downloads": downloads,
            "dataset_info": {
                "filename": self.active_filename,
                "rows": len(self.active_df) if self.active_df is not None else 0,
                "columns": len(self.active_df.columns) if self.active_df is not None else 0,
                "encrypted_columns": self.encrypted_columns,
                "cleaning_log": self.cleaning_log[-5:]
            }
        }

# Global singleton
narayan_service = NarayanPipelineService()
