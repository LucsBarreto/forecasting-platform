"""
aplicação streamlit de inteligência comercial.

esta interface funciona como camada de consumo dos dados históricos
salvos no backend e dos artefatos de forecast disponibilizados em
outputs/runs/<run_id>.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import settings
from src.core.utils.datetime import coerce_excel_datetime
from src.data_sources.excel_source import resolve_excel_engine
from src.data_sources.factory import DataSourceFactory

st.set_page_config(
    page_title="Inteligência Comercial",
    page_icon="IC",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#0d9488"
DARK = "#16324f"
DATE_COLUMN = settings.data.date_column
VOLUME_COLUMN = "VOLUME"
VALUE_COLUMN = "VALOR"
MAX_AGGREGATION_ROWS = 150_000
REQUIRED_COLUMNS = list(settings.data.required_columns)
BASE_DIMENSIONS = [
    column
    for column in REQUIRED_COLUMNS
    if column not in {DATE_COLUMN, VOLUME_COLUMN, VALUE_COLUMN}
]
FILTER_DIMENSIONS = [
    "COD CLIENTE",
    "NOME CLIENTE",
    "COD ITEM",
    "PRODUTO",
    "FILIAL DESTINO",
    "FILIAL ORIGEM",
    "REGIONAL",
    "UF",
    "MARCA",
    "CATEGORIA",
    "SUBCANAL GTM",
    "CANAL GTM",
    "ATENDIMENTO",
]


@st.cache_data(show_spinner=False)
def load_dataframe(file_bytes: bytes | None, file_name: str) -> pd.DataFrame:
    """carrega a base histórica de vendas conforme o contrato do projeto."""

    if file_bytes is not None:
        stream = io.BytesIO(file_bytes)
        suffix = Path(file_name).suffix.lower()
        if suffix == ".csv":
            dataframe = pd.read_csv(stream)
        elif suffix == ".parquet":
            dataframe = pd.read_parquet(stream)
        else:
            dataframe = pd.read_excel(
                stream,
                engine=resolve_excel_engine(stream, suffix),
            )
    else:
        input_path = Path(settings.data.input_path)
        sources = (
            [input_path]
            if input_path.is_file()
            else sorted(input_path.glob(settings.data.file_pattern))
        )
        if not sources:
            raise FileNotFoundError(
                "Nenhuma base de vendas foi encontrada. Envie um arquivo ou configure a origem em configs/data.yaml."
            )
        dataframes = [
            DataSourceFactory.create(source).read(source) for source in sources
        ]
        dataframe = pd.concat(dataframes, ignore_index=True)

    dataframe.columns = (
        dataframe.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    missing = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing:
        raise ValueError(
            "Colunas ausentes na base: " + ", ".join(missing)
        )

    for column, dtype in settings.data.schema_config.dtypes.items():
        if column not in dataframe.columns:
            continue
        if dtype == "datetime64[ns]":
            dataframe[column] = coerce_excel_datetime(dataframe[column])
        elif dtype.startswith(("int", "float")):
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
        elif dtype == "string":
            dataframe[column] = dataframe[column].astype("string")

    dataframe = dataframe.dropna(subset=[DATE_COLUMN]).copy()
    return dataframe


@st.cache_data(show_spinner=False)
def list_run_directories() -> list[Path]:
    """lista as execuções públicas disponíveis em outputs/runs."""

    runs_root = Path(settings.data.output_path) / settings.data.runs_folder
    if not runs_root.exists():
        return []
    return sorted(runs_root.glob("RUN_*/"), reverse=True)


@st.cache_data(show_spinner=False)
def load_run_metadata(run_path: str | Path) -> dict[str, Any]:
    """carrega as metainformações da execução selecionada."""

    path = Path(run_path)
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return {}

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (TypeError, ValueError, OSError):
        return {}


@st.cache_data(show_spinner=False)
def load_run_metrics(run_path: str | Path) -> pd.DataFrame:
    """normaliza métricas de modelos em dataframe tabular."""

    path = Path(run_path)
    metrics_path = path / "metrics" / "metrics.json"
    if not metrics_path.exists():
        return pd.DataFrame(columns=["Modelo", "Métrica", "Resultado", "Ranking"])

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (TypeError, ValueError, OSError):
        return pd.DataFrame(columns=["Modelo", "Métrica", "Resultado", "Ranking"])

    metric_label = str(getattr(settings.models.evaluation, "metric", "mape")).upper()

    if isinstance(payload, dict):
        items = list(payload.items())
    elif isinstance(payload, list):
        items = []
        for entry in payload:
            if isinstance(entry, dict):
                if "model" in entry and "metric" in entry:
                    items.append((entry["model"], entry["metric"]))
                elif "modelo" in entry and "métrica" in entry:
                    items.append((entry["modelo"], entry["métrica"]))
                elif "name" in entry and "value" in entry:
                    items.append((entry["name"], entry["value"]))
    else:
        return pd.DataFrame(columns=["Modelo", "Métrica", "Resultado", "Ranking"])

    rows: list[dict[str, Any]] = []
    for name, value in items:
        if isinstance(value, dict):
            for metric_name, metric_value in value.items():
                rows.append(
                    {
                        "Modelo": str(name),
                        "Métrica": str(metric_name).upper(),
                        "Resultado": float(metric_value),
                    }
                )
        else:
            rows.append(
                {
                    "Modelo": str(name),
                    "Métrica": metric_label,
                    "Resultado": float(value),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Modelo", "Métrica", "Resultado", "Ranking"])

    metrics_df = pd.DataFrame(rows)
    metrics_df["Resultado"] = pd.to_numeric(metrics_df["Resultado"], errors="coerce")
    metrics_df = metrics_df.dropna(subset=["Resultado"]).copy()

    metric_name = metrics_df["Métrica"].iloc[0].upper()
    lower_is_better = metric_name in {
        "MAPE",
        "MAE",
        "RMSE",
        "MSE",
        "WAPE",
        "ERROR",
        "ERROR_RATE",
    }
    metrics_df = metrics_df.sort_values(
        by="Resultado",
        ascending=lower_is_better,
    ).reset_index(drop=True)
    metrics_df["Ranking"] = range(1, len(metrics_df) + 1)
    return metrics_df


@st.cache_data(show_spinner=False)
def discover_forecast_candidates(run_path: str | Path) -> list[dict[str, Any]]:
    """descobre os arquivos de previsão e as colunas modeláveis da execução."""

    run_dir = Path(run_path)
    forecasts_dir = run_dir / "forecasts"
    if not forecasts_dir.exists():
        return []

    candidates: list[dict[str, Any]] = []
    for csv_file in sorted(forecasts_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(csv_file)
        except (OSError, ValueError, pd.errors.EmptyDataError):
            continue

        if frame.empty or DATE_COLUMN not in frame.columns:
            continue

        for column in frame.columns:
            if column == DATE_COLUMN:
                continue
            model_name = str(column)
            candidates.append(
                {
                    "file": csv_file,
                    "model": model_name,
                    "target": _infer_target_name(csv_file.name, model_name),
                    "data": frame[[DATE_COLUMN, column]].copy(),
                }
            )

    return candidates


def _infer_target_name(filename: str, model_name: str) -> str:
    """inferir o target a partir do nome do arquivo/coluna quando disponível."""

    stem = Path(filename).stem.lower()
    if "volume" in stem:
        return "VOLUME"
    if "valor" in stem or "valor" in model_name.lower():
        return "VALOR"
    if "baseline" in model_name.lower():
        return "BASELINE"
    return model_name.upper()


def format_number(value: float) -> str:
    """formata números em padrão brasileiro para leitura em cards."""

    if pd.isna(value):
        return "0"
    return f"{value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_currency(value: float) -> str:
    """formata valores monetários em reais."""

    if pd.isna(value):
        return "R$ 0,00"
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def apply_filters(dataframe: pd.DataFrame) -> pd.DataFrame:
    """aplica filtros de orientação comercial e temporais na base histórica."""

    filtered = dataframe.copy()

    st.sidebar.markdown("## Filtros globais")
    st.sidebar.caption("Ajusta os dados históricos usados nos KPIs e gráficos.")

    start_date = filtered[DATE_COLUMN].min().date()
    end_date = filtered[DATE_COLUMN].max().date()
    period_options = {
        "Todo o período": (start_date, end_date),
        "Últimos 12 meses": (
            max(start_date, (pd.Timestamp(end_date) - pd.DateOffset(months=12)).date()),
            end_date,
        ),
        "Últimos 90 dias": (
            max(start_date, (pd.Timestamp(end_date) - pd.Timedelta(days=90)).date()),
            end_date,
        ),
        "Personalizado": None,
    }

    selected_period = st.sidebar.selectbox(
        "Período",
        list(period_options.keys()),
        help="Assegura que a análise histórica reflita o intervalo desejado.",
    )
    selected_dates = period_options[selected_period]
    if selected_dates is None:
        selected_dates = st.sidebar.date_input(
            "Intervalo personalizado",
            value=(start_date, end_date),
            min_value=start_date,
            max_value=end_date,
        )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        filtered = filtered[
            filtered[DATE_COLUMN].dt.date.between(selected_dates[0], selected_dates[1])
        ]

    for column in FILTER_DIMENSIONS:
        if column not in filtered.columns:
            continue
        values = sorted(filtered[column].dropna().astype(str).unique())
        selected = st.sidebar.multiselect(
            column,
            values,
            key=f"filter_{column}",
        )
        if selected:
            filtered = filtered[filtered[column].astype(str).isin(selected)]

    return filtered


def render_overview(dataframe: pd.DataFrame) -> None:
    """apresenta os cards globais do painel inicial."""

    volume = float(dataframe[VOLUME_COLUMN].sum())
    value = float(dataframe[VALUE_COLUMN].sum())
    customers = int(dataframe["COD CLIENTE"].nunique())
    average_price = value / volume if volume else 0.0

    cols = st.columns(4)
    cards = [
        ("Volume total", format_number(volume), "unidades"),
        ("Valor total", format_currency(value), "R$"),
        ("Preço médio", format_currency(average_price), "R$ / unidade"),
        ("Clientes", format_number(customers), "total"),
    ]

    for column, (label, metric, suffix) in zip(cols, cards, strict=True):
        column.metric(label, metric, suffix)


@st.cache_data(show_spinner=False)
def build_time_series(dataframe: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """agrega a evolução temporal por periodicidade escolhida."""

    frequency_map = {
        "Diária": "D",
        "Semanal": "W",
        "Mensal": "ME",
        "Trimestral": "QE",
        "Anual": "YE",
    }

    index = dataframe.set_index(DATE_COLUMN)
    series = index[[VOLUME_COLUMN, VALUE_COLUMN]].resample(frequency_map[frequency]).sum()
    series = series.reset_index()
    series[DATE_COLUMN] = pd.to_datetime(series[DATE_COLUMN])
    return series


@st.cache_data(show_spinner=False)
def build_historical_ranking(
    dataframe: pd.DataFrame,
    dimension: str,
    metric_choice: str,
) -> pd.DataFrame:
    """pré-agrega ranking histórico para reduzir recalculos de filtro e gráfico."""

    ranking = (
        dataframe.assign(**{dimension: dataframe[dimension].fillna("Não informado")})
        .groupby(dimension, as_index=False)
        .agg(Volume=(VOLUME_COLUMN, "sum"), Valor=(VALUE_COLUMN, "sum"))
    )
    if metric_choice == "Volume":
        ranking = ranking.sort_values("Volume", ascending=False)
        ranking["Metric"] = "Volume"
    elif metric_choice == "Valor":
        ranking = ranking.sort_values("Valor", ascending=False)
        ranking["Metric"] = "Valor"
    else:
        ranking["Volume + Valor"] = ranking["Volume"] + ranking["Valor"]
        ranking = ranking.sort_values("Volume + Valor", ascending=False)
        ranking["Metric"] = "Volume + Valor"
    return ranking


def render_historical_analysis(dataframe: pd.DataFrame) -> None:
    """renderiza análise histórica, ranking e evolução temporal."""

    if len(dataframe) > MAX_AGGREGATION_ROWS:
        st.info(
            "A base filtrada é grande. O dashboard está usando uma agregação resumida para manter a experiência responsiva."
        )

    st.subheader("Análise histórica")

    col1, col2, col3, col4 = st.columns(4)
    metric_choice = col1.selectbox(
        "Métrica",
        ["Volume", "Valor", "Volume + Valor"],
        key="historical_metric",
    )
    dimension = col2.selectbox(
        "Dimensão",
        BASE_DIMENSIONS,
        key="historical_dimension",
    )
    frequency = col3.selectbox(
        "Frequência",
        ["Diária", "Semanal", "Mensal", "Trimestral", "Anual"],
        key="historical_frequency",
    )
    chart_style = col4.selectbox(
        "Visualização",
        ["Barras", "Linha", "Área"],
        key="historical_style",
    )

    timeline = build_time_series(dataframe, frequency)
    if metric_choice == "Volume":
        timeline_metric = VOLUME_COLUMN
        timeline_label = "Volume"
    elif metric_choice == "Valor":
        timeline_metric = VALUE_COLUMN
        timeline_label = "Valor"
    else:
        timeline_metric = None
        timeline_label = "Volume + Valor"

    if metric_choice == "Volume + Valor":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=timeline[DATE_COLUMN],
                y=timeline[VOLUME_COLUMN],
                name="Volume",
                marker_color=ACCENT,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=timeline[DATE_COLUMN],
                y=timeline[VALUE_COLUMN],
                mode="lines+markers",
                name="Valor",
                line=dict(color=DARK, width=2),
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Volume", secondary_y=False)
        fig.update_yaxes(title_text="Valor", secondary_y=True)
        fig.update_layout(
            title="Volume e valor ao longo do tempo",
            height=390,
            margin=dict(l=10, r=10, t=55, b=10),
            hovermode="x unified",
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
        )
    else:
        chart_df = timeline[[DATE_COLUMN, timeline_metric]].rename(columns={timeline_metric: timeline_label})
        if chart_style == "Barras":
            chart = px.bar(
                chart_df,
                x=DATE_COLUMN,
                y=timeline_label,
                title=f"{timeline_label} por período",
                color_discrete_sequence=[ACCENT],
            )
        elif chart_style == "Área":
            chart = px.area(
                chart_df,
                x=DATE_COLUMN,
                y=timeline_label,
                title=f"{timeline_label} por período",
                color_discrete_sequence=[ACCENT],
            )
        else:
            chart = px.line(
                chart_df,
                x=DATE_COLUMN,
                y=timeline_label,
                markers=True,
                title=f"{timeline_label} por período",
                color_discrete_sequence=[ACCENT],
            )
        chart.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=55, b=10),
            hovermode="x unified",
        )
        st.plotly_chart(
            chart,
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
        )

    ranking = build_historical_ranking(dataframe, dimension, metric_choice)
    if metric_choice == "Volume":
        ranking_metric = "Volume"
    elif metric_choice == "Valor":
        ranking_metric = "Valor"
    else:
        ranking_metric = "Volume + Valor"

    limit = st.slider("Quantidade de itens no ranking", 5, 30, 12, key="historical_rank_limit")
    ranking = ranking.head(limit)

    if metric_choice == "Volume + Valor":
        chart_rank = px.bar(
            ranking.sort_values(ranking_metric),
            x=ranking_metric,
            y=dimension,
            orientation="h",
            title=f"Ranking por {dimension.lower()}",
            color=ranking_metric,
            color_continuous_scale=["#b7e4df", DARK],
        )
    else:
        chart_rank = px.bar(
            ranking.sort_values(ranking_metric),
            x=ranking_metric,
            y=dimension,
            orientation="h",
            title=f"Ranking por {dimension.lower()}",
            color=ranking_metric,
            color_continuous_scale=["#b7e4df", DARK],
        )

    chart_rank.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=55, b=10),
        showlegend=False,
    )
    st.plotly_chart(
        chart_rank,
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
    )

    csv_data = dataframe.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Baixar dados históricos filtrados",
        csv_data,
        file_name="dados_historicos_filtrados.csv",
        mime="text/csv",
    )


def render_run_overview(run_path: Path) -> None:
    """detalha a execução selecionada e os itens relevantes do run."""

    metadata = load_run_metadata(run_path)
    metrics_df = load_run_metrics(run_path)

    st.markdown("### Execução selecionada")
    title = run_path.name
    info_cols = st.columns(5)
    info = [
        ("Execução", title),
        ("Status", metadata.get("status", "UNKNOWN")),
        ("Início", metadata.get("started_at", "-")),
        ("Fim", metadata.get("finished_at", "-")),
        ("Pipeline", metadata.get("pipeline_version", "-")),
    ]
    for col, (label, value) in zip(info_cols, info, strict=True):
        col.caption(f"{label}: {value}")

    if metrics_df.empty:
        st.info("Não há métricas disponíveis para esta execução.")
    else:
        st.dataframe(
            metrics_df[["Modelo", "Métrica", "Resultado", "Ranking"]],
            hide_index=True,
            width="stretch",
        )

        csv_data = metrics_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Baixar métricas da execução",
            csv_data,
            file_name=f"{run_path.name}_metricas.csv",
            mime="text/csv",
        )


def render_forecasts_section() -> None:
    """apresenta as previsões da execução selecionada e o painel de comparação."""

    st.subheader("Previsões")
    runs = list_run_directories()
    if not runs:
        st.info("Nenhuma execução de forecast encontrada.")
        return

    selected_run = st.selectbox(
        "Execução",
        runs,
        format_func=lambda path: path.name,
        key="forecast_run",
    )

    render_run_overview(selected_run)

    candidates = discover_forecast_candidates(selected_run)
    if not candidates:
        st.warning("Não existem dados de previsão disponíveis para esta execução.")
        return

    model_names = sorted({candidate["model"] for candidate in candidates})
    selected_model = st.selectbox(
        "Modelo",
        model_names,
        key="forecast_model_selection",
    )

    selected_candidate = next(
        candidate for candidate in candidates if candidate["model"] == selected_model
    )
    forecast = selected_candidate["data"].copy()
    forecast[DATE_COLUMN] = pd.to_datetime(forecast[DATE_COLUMN], errors="coerce")
    forecast = forecast.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN)

    st.caption(
        f"Modelo: {selected_model} | Target: {selected_candidate['target']} | Arquivo: {selected_candidate['file'].name}"
    )

    if forecast.empty:
        st.warning("Os dados de previsão carregados para este modelo estão vazios.")
        return

    model_columns = [measurement for measurement in [selected_model] if measurement]
    if not model_columns:
        st.warning("Não foi possível identificar a coluna de previsão para a execução atual.")
        return

    fig = px.bar(
        forecast,
        x=DATE_COLUMN,
        y=selected_model,
        title=f"Previsão - {selected_model}",
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
    )

    st.dataframe(forecast, hide_index=True, width="stretch")
    csv_data = forecast.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Baixar dados de previsão",
        csv_data,
        file_name=f"{selected_run.name}_{selected_model}_previsao.csv",
        mime="text/csv",
    )


def render_model_comparison() -> None:
    """renderiza tabela de comparação de modelos da execução selecionada."""

    st.subheader("Comparação dos modelos")
    runs = list_run_directories()
    if not runs:
        st.info("Nenhuma execução de forecast encontrada para comparação.")
        return

    run = st.selectbox(
        "Execução para comparar",
        runs,
        format_func=lambda path: path.name,
        key="comparison_run",
    )

    metrics_df = load_run_metrics(run)
    if metrics_df.empty:
        st.info("Não há resultados de modelos disponíveis nesta execução.")
        return

    metrics_df = metrics_df[["Modelo", "Métrica", "Resultado", "Ranking"]].copy()
    metrics_df["Resultado"] = metrics_df["Resultado"].map(lambda value: f"{float(value):,.4f}" if pd.notna(value) else "-")
    st.dataframe(
        metrics_df,
        hide_index=True,
        width="stretch",
    )


def render_sidebar_file_uploader() -> None:
    """controla upload opcional de nova base no painel lateral."""

    st.sidebar.markdown("---")
    st.sidebar.subheader("Fonte de dados")
    uploaded = st.sidebar.file_uploader(
        "Base de vendas",
        type=["csv", "xlsx", "xlsb", "parquet"],
        help="Arquivo opcional para substituir a fonte configurada no backend.",
    )
    return uploaded


def main() -> None:
    """monta a aplicação principal do dashboard."""

    is_dark_mode = st.sidebar.toggle("Modo escuro", value=True)
    if is_dark_mode:
        palette = {
            "bg": "#0f172a",
            "panel": "#111827",
            "panel_text": "#e5eefb",
            "card": "#111827",
            "heading": "#f3f7ff",
            "text": "#e2e8f0",
            "muted": "#94a3b8",
            "metric_text": "#f8fafc",
            "border": "rgba(148, 163, 184, 0.18)",
            "shadow": "rgba(15, 23, 42, 0.36)",
        }
    else:
        palette = {
            "bg": "#f2f5f4",
            "panel": "#132f4c",
            "panel_text": "#f5fbfa",
            "card": "#ffffff",
            "heading": "#132f4c",
            "text": "#1f2937",
            "muted": "#52706e",
            "metric_text": "#132f4c",
            "border": "#dce7e4",
            "shadow": "rgba(19, 47, 76, 0.06)",
        }

    st.markdown(
        f"""
        <style>
        .stApp {{ background: {palette['bg']}; color: {palette['text']}; }}
        [data-testid="stSidebar"] {{ background: {palette['panel']}; }}
        [data-testid="stSidebar"] * {{ color: {palette['panel_text']}; }}
        .stApp h1, .stApp h2, .stApp h3 {{ color: {palette['heading']}; }}
        [data-testid="stMetric"] {{
            background: {palette['card']};
            border: 1px solid {palette['border']};
            border-radius: 10px;
            padding: 12px 16px;
            box-shadow: 0 2px 10px {palette['shadow']};
        }}
        [data-testid="stMetricLabel"] {{ color: {palette['muted']}; }}
        [data-testid="stMetricValue"] {{ color: {palette['metric_text']}; }}
        div[data-testid="stExpander"] {{
            background: {palette['card']};
            border: 1px solid {palette['border']};
            border-radius: 10px;
        }}
        .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
        .stDataFrame, .stDataFrame div {{ background: transparent; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Inteligência Comercial")
    st.caption("Dashboard operacional para exploração histórica e análise de previsões.")

    uploaded = render_sidebar_file_uploader()

    try:
        dataframe = load_dataframe(
            uploaded.getvalue() if uploaded else None,
            uploaded.name if uploaded else "configured_input",
        )
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        st.warning(str(exc))
        st.info("Envie uma base compatível com o contrato em configs/data.yaml ou utilize um arquivo de vendas válido.")
        return

    filtered = apply_filters(dataframe)
    if filtered.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
        return

    st.caption(
        f"{len(filtered):,} registros exibidos de {len(dataframe):,} | "
        f"{dataframe[DATE_COLUMN].min():%d/%m/%Y} a {dataframe[DATE_COLUMN].max():%d/%m/%Y}"
    )

    if len(filtered) > MAX_AGGREGATION_ROWS:
        st.caption(
            "Modo performance ativo: os gráficos e rankings foram resumidos para manter a navegação responsiva."
        )

    render_overview(filtered)
    render_historical_analysis(filtered)
    render_forecasts_section()
    render_model_comparison()


if __name__ == "__main__":
    main()
