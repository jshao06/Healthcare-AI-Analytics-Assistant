"""UI for the healthcare analytics assistant workflow."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analysis.types import AnalysisResult

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.analyzer import DataAnalyzer
from src.assistant.assistant import HealthcareAssistant
from src.assistant.context import ContextBuilder
from src.assistant.request_builder import QueryRequestBuilder
from src.db.schema import SchemaManager
from src.db.sqlite_database import SQLiteDatabase
from src.sql.executor import SQLExecutor
from src.sql.generator import SQLGenerator
from src.sql.validator import SQLValidator
from src.visualization.visualizer import DataVisualizer

DB_PATH = ROOT / "data" / "tinyehr_mimic_format.db"


@st.cache_resource
def get_pipeline() -> tuple[HealthcareAssistant, DataAnalyzer, DataVisualizer]:
    """Build the app pipeline once and reuse it across UI interactions."""

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    database = SQLiteDatabase(DB_PATH)
    schema_manager = SchemaManager(database)
    context_builder = ContextBuilder.from_sqlite(DB_PATH)
    request_builder = QueryRequestBuilder(default_limit=100)
    sql_generator = SQLGenerator(schema_manager)
    sql_validator = SQLValidator(schema_manager)
    sql_executor = SQLExecutor(database, sql_validator, max_rows=500)
    assistant = HealthcareAssistant(
        context_builder=context_builder,
        request_builder=request_builder,
        sql_generator=sql_generator,
        sql_validator=sql_validator,
        sql_executor=sql_executor,
    )

    return assistant, DataAnalyzer(), DataVisualizer()


def _render_insights(insights: Iterable[object] | None) -> None:
    """Render insight objects in a compact list format."""

    insights_tuple = tuple(insights) if insights else ()
    if not insights_tuple:
        st.info("No insights were generated for this result.")
        return

    for insight in insights_tuple:
        severity = getattr(insight, "severity", None)
        if severity and str(severity).lower() == "critical":
            icon = "🚨"
        elif severity and str(severity).lower() == "warning":
            icon = "⚠️"
        else:
            icon = "ℹ️"
        st.markdown(f"- {icon} **{getattr(insight, 'title', 'Insight')}**: {getattr(insight, 'description', '')}")


def _render_chart(
    dataframe: pd.DataFrame,
    analysis_result: AnalysisResult,
    visualizer: DataVisualizer,
) -> None:
    """Render the best available chart for the query results."""

    try:
        chart_results = tuple(visualizer.visualize(dataframe, analysis_result))
    except ValueError:
        st.caption("No chart could be generated for this dataset.")
        return

    if not chart_results:
        st.caption("The dataset does not contain a chartable pattern.")
        return

    st.pyplot(chart_results[0].figure)


def main() -> None:
    """Run the app."""

    st.set_page_config(
        page_title="Healthcare AI Analytics Assistant",
        page_icon="🩺",
        layout="wide",
    )

    st.title("Healthcare AI Analytics Assistant")
    st.caption(
        "Ask a healthcare question, generate SQL, execute it, analyze the result, and visualize the outcome."
    )

    question = st.text_input(
        "Healthcare question",
        placeholder="Example: What is the average age by gender?",
    )

    if not st.button("Run analysis"):
        st.info("Enter a healthcare question and click Run analysis.")
        return

    if not question.strip():
        st.warning("Please provide a non-empty question.")
        return

    assistant, analyzer, visualizer = get_pipeline()

    try:
        context = assistant.build_context(question)
        sql, parameters = assistant.generate_sql(question, context)
        validation = assistant.validate_sql(sql)

        if not validation.is_valid:
            st.error("Generated SQL did not pass validation.")
            for issue in validation.issues:
                st.code(f"{issue.code}: {issue.message}")
            return

        query_result = assistant.execute_sql(sql, parameters)
        dataframe: pd.DataFrame = query_result.data
        analysis_result = analyzer.analyze(dataframe)
    except Exception as exc:  # pragma: no cover - UI-level error handling
        st.error(f"Unable to process this question: {exc}")
        return

    st.subheader("Generated SQL")
    st.code(sql, language="sql")

    st.subheader("Query result")
    if dataframe.empty:
        st.info("The query returned no rows.")
    else:
        st.dataframe(dataframe, use_container_width=True)  # pyright: ignore[reportUnknownMemberType]

    st.subheader("Insights")
    _render_insights(analysis_result.insights)

    st.subheader("Visualization")
    _render_chart(dataframe, analysis_result, visualizer)


if __name__ == "__main__":
    main()
