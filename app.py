"""
aplicação streamlit de inteligência comercial.

este módulo fornece uma interface interativa para carregamento,
exploração, análise e visualização dos dados oficiais de vendas
e das previsões geradas pela plataforma de forecasting.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import settings
from src.core.utils.datetime import coerce_excel_datetime
from src.data_sources.excel_source import resolve_excel_engine
from src.data_sources.factory import DataSourceFactory

st.set_page_config(
    page_title="Inteligencia Comercial",
    page_icon="IC",
    layout="wide",
    initial_sidebar_state="expanded",
)


ACCENT = "#0d9488"
DARK = "#16324f"
REQUIRED_COLUMNS = settings.data.required_columns
DATE_COLUMN = settings.data.date_column
VOLUME_COLUMN = "VOLUME"
VALUE_COLUMN = "VALOR"
DIMENSION_COLUMNS = [
    column
    for column in REQUIRED_COLUMNS
    if column not in {DATE_COLUMN, VOLUME_COLUMN, VALUE_COLUMN}
]
FILTER_COLUMNS = [
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
    """
    carrega e tipa os dados conforme o contrato oficial de vendas.

    parameters
    ----------
    file_bytes
        conteúdo do arquivo enviado pelo usuário em bytes.
        se não informado, os arquivos configurados na aplicação
        serão utilizados como fonte de dados.

    file_name
        nome do arquivo utilizado para identificar o formato dos dados.

    returns
    -------
    pd.DataFrame
        dataframe contendo os dados carregados, com as colunas
        normalizadas e os tipos definidos pelo schema da aplicação.

    raises
    ------
    filenotfounderror
        se nenhum arquivo for encontrado no caminho configurado
        e nenhum arquivo for enviado pelo usuário.

    valueerror
        se o arquivo não possuir as colunas obrigatórias.
    """

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
                "Nenhuma base encontrada. Envie um arquivo pela barra lateral."
            )
        dataframes = [
            DataSourceFactory.create(source).read(source)
            for source in sources
        ]
        dataframe = pd.concat(dataframes, ignore_index=True)

    dataframe.columns = (
        dataframe.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    missing = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing:
        raise ValueError("Colunas ausentes no arquivo: " + ", ".join(missing))

    for column, dtype in settings.data.schema_config.dtypes.items():
        if column not in dataframe.columns:
            continue
        if dtype == "datetime64[ns]":
            dataframe[column] = coerce_excel_datetime(dataframe[column])
        elif dtype.startswith("int") or dtype.startswith("float"):
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
        elif dtype == "string":
            dataframe[column] = dataframe[column].astype("string")

    dataframe = dataframe.dropna(subset=[DATE_COLUMN]).copy()
    return dataframe


def format_number(value: float) -> str:
    """
    formata um valor numérico utilizando separadores no padrão brasileiro.

    parameters
    ----------
    value
        valor numérico que será formatado.

    returns
    -------
    str
        valor formatado sem casas decimais, utilizando ponto como
        separador de milhares.
    """

    return f"{value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")


def apply_filters(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    aplica os filtros de dimensões oficiais e retorna os dados filtrados.

    parameters
    ----------
    dataframe
        dataframe que será utilizado como base para aplicação dos filtros.

    returns
    -------
    pd.DataFrame
        dataframe contendo somente os registros selecionados pelos filtros.
    """

    filtered = dataframe.copy()
    st.sidebar.markdown("## Explorar dados")
    st.sidebar.caption("Os filtros atuam sobre toda a base carregada.")
    start_date = dataframe[DATE_COLUMN].min().date()
    end_date = dataframe[DATE_COLUMN].max().date()
    period_options = {
        "Todo o periodo": (start_date, end_date),
        "Ultimos 12 meses": (
            max(start_date, (pd.Timestamp(end_date) - pd.DateOffset(months=12)).date()),
            end_date,
        ),
        "Ultimos 90 dias": (
            max(start_date, (pd.Timestamp(end_date) - pd.Timedelta(days=90)).date()),
            end_date,
        ),
        "Personalizado": None,
    }
    selected_period = st.sidebar.selectbox(
        "Periodo",
        list(period_options),
        help="Todo o periodo usa a primeira e a ultima DATA da base carregada.",
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

    for column in FILTER_COLUMNS:
        if column not in dataframe.columns:
            continue
        values = sorted(dataframe[column].dropna().astype(str).unique())
        selected = st.sidebar.multiselect(column, values, key=f"filter_{column}")
        if selected:
            filtered = filtered[filtered[column].astype(str).isin(selected)]
    return filtered


def render_kpis(dataframe: pd.DataFrame) -> None:
    """
    renderiza os principais indicadores de desempenho dos dados filtrados.

    parameters
    ----------
    dataframe
        dataframe utilizado para cálculo e exibição dos indicadores.
    """

    volume = float(dataframe[VOLUME_COLUMN].sum())
    value = float(dataframe[VALUE_COLUMN].sum())
    customers = dataframe["COD CLIENTE"].nunique()
    products = dataframe["COD ITEM"].nunique()
    average = value / volume if volume else 0.0
    columns = st.columns(5)
    cards = [
        ("Volume vendido", format_number(volume)),
        ("Faturamento", f"R$ {value:,.2f}"),
        ("Clientes", format_number(customers)),
        ("Itens", format_number(products)),
        ("Valor por unidade", f"R$ {average:,.2f}"),
    ]
    for column, (label, metric) in zip(columns, cards, strict=True):
        column.metric(label, metric)


def render_analysis(dataframe: pd.DataFrame) -> None:
    """
    renderiza a análise interativa dos dados filtrados.

    parameters
    ----------
    dataframe
        dataframe utilizado para geração das séries temporais,
        rankings e dados para download.
    """

    st.subheader("Análise interativa")
    controls = st.columns([1, 1, 1, 1])
    metric_options = {"VOLUME": VOLUME_COLUMN, "VALOR": VALUE_COLUMN}
    metric_label = controls[0].selectbox("Métrica", list(metric_options))
    metric = metric_options[metric_label]
    chart_kind = controls[1].selectbox("Visualização", ["Linha", "Barras", "Área"])
    frequency = controls[2].selectbox(
        "Periodicidade",
        ["Diária", "Semanal", "Mensal", "Trimestral", "Anual"],
    )
    grouping = controls[3].selectbox(
        "Dimensão do ranking",
        ["PRODUTO", "MARCA", "CATEGORIA", "REGIONAL", "CANAL GTM", "FILIAL DESTINO"],
    )

    frequency_map = {
        "Diária": "D",
        "Semanal": "W",
        "Mensal": "ME",
        "Trimestral": "QE",
        "Anual": "YE",
    }
    timeline = (
        dataframe.set_index(DATE_COLUMN)[metric]
        .resample(frequency_map[frequency])
        .sum()
        .rename(metric_label)
        .reset_index()
    )
    if chart_kind == "Barras":
        chart = px.bar(
            timeline,
            x=DATE_COLUMN,
            y=metric_label,
            title=f"{metric_label} por periodo",
        )
    elif chart_kind == "Área":
        chart = px.area(
            timeline,
            x=DATE_COLUMN,
            y=metric_label,
            title=f"{metric_label} por periodo",
        )
    else:
        chart = px.line(
            timeline,
            x=DATE_COLUMN,
            y=metric_label,
            markers=True,
            title=f"{metric_label} por periodo",
        )
    if chart_kind in {"Linha", "Área"}:
        chart.update_traces(line_color=ACCENT)
    else:
        chart.update_traces(marker_color=ACCENT)
    chart.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(
        chart,
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToAdd": ["drawline", "drawrect"],
        },
    )

    rank_limit = st.slider("Quantidade de itens no ranking", 5, 30, 12)
    ranking = (
        dataframe.assign(**{grouping: dataframe[grouping].fillna("Não informado")})
        .groupby(grouping, as_index=False)[metric]
        .sum()
        .sort_values(metric, ascending=False)
        .head(rank_limit)
    )
    ranking_chart = px.bar(
        ranking.sort_values(metric),
        x=metric,
        y=grouping,
        orientation="h",
        title=f"{metric_label} por {grouping.lower()}",
        color=metric,
        color_continuous_scale=["#b7e4df", DARK],
    )
    ranking_chart.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=55, b=10),
        showlegend=False,
    )
    st.plotly_chart(
        ranking_chart,
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
    )

    csv_data = dataframe.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Baixar dados filtrados",
        csv_data,
        file_name="dados_filtrados.csv",
        mime="text/csv",
    )


def render_forecast() -> None:
    """
    renderiza as previsões e as métricas de desempenho dos modelos.

    o método carrega a execução de forecast mais recente disponível,
    apresenta as métricas registradas e permite visualizar as previsões
    dos modelos encontrados nos arquivos de saída.
    """

    st.subheader("Previsoes e desempenho do modelo")
    runs_path = Path(settings.data.output_path) / settings.data.runs_folder
    runs = sorted(runs_path.glob("RUN_*/"), reverse=True) if runs_path.exists() else []
    if not runs:
        st.info("Nenhuma execucao de forecast encontrada ainda.")
        return
    run = st.selectbox("Execucao", runs, format_func=lambda path: path.name)
    metrics_path = run / "metrics" / "metrics.json"
    forecast_files = sorted((run / "forecasts").glob("*.csv"))
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.dataframe(
            pd.DataFrame(
                [{"Modelo": name, "MAPE": value} for name, value in metrics.items()]
            ),
            hide_index=True,
            width="stretch",
        )
    if forecast_files:
        forecast = pd.read_csv(forecast_files[0])
        forecast_date = settings.data.date_column
        model_columns = [
            column
            for column in forecast.columns
            if column != forecast_date
        ]
        if model_columns and forecast_date in forecast.columns:
            model_name = st.selectbox(
                "Modelo para visualizar",
                model_columns,
                key="forecast_model",
            )
            forecast[forecast_date] = pd.to_datetime(
                forecast[forecast_date],
                errors="coerce",
            )
            forecast_chart = px.line(
                forecast.sort_values(forecast_date),
                x=forecast_date,
                y=model_name,
                markers=True,
                title=f"Previsão isolada - {model_name}",
                color_discrete_sequence=[ACCENT],
            )
            forecast_chart.update_layout(
                height=390,
                margin=dict(l=10, r=10, t=55, b=10),
                hovermode="x unified",
            )
            st.plotly_chart(
                forecast_chart,
                width="stretch",
                config={"displaylogo": False, "scrollZoom": True},
            )
        st.dataframe(forecast, hide_index=True, width="stretch")


def main() -> None:
    """
    executa a aplicação principal de inteligência comercial.

    responsabilidades
    ------------------
    - configurar o estilo e a apresentação da aplicação.
    - carregar a base de vendas configurada ou enviada pelo usuário.
    - aplicar os filtros selecionados.
    - renderizar indicadores, análises e previsões.
    - apresentar mensagens de erro e validação ao usuário.
    """

    st.markdown(
        """
        <style>
        .stApp { background: #f2f5f4; }
        [data-testid="stSidebar"] { background: #132f4c; }
        [data-testid="stSidebar"] * { color: #f5fbfa; }
        h1, h2, h3 { color: #132f4c; letter-spacing: 0; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dce7e4;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 2px 8px rgba(19, 47, 76, 0.06);
        }
        [data-testid="stMetricLabel"] { color: #52706e; }
        [data-testid="stMetricValue"] { color: #132f4c; }
        div[data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #dce7e4;
            border-radius: 8px;
        }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Inteligência Comercial")
    st.caption("Visão exploratória de vendas e desempenho do forecast")
    uploaded = st.sidebar.file_uploader(
        "Base de vendas",
        type=["csv", "xlsx", "xlsb", "parquet"],
    )
    try:
        dataframe = load_dataframe(
            uploaded.getvalue() if uploaded else None,
            uploaded.name if uploaded else "configured_input",
        )
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        st.warning(str(exc))
        st.info("Envie uma base com as 18 colunas definidas em configs/data.yaml.")
        return

    filtered = apply_filters(dataframe)
    st.caption(
        f"{len(filtered):,} registros exibidos de {len(dataframe):,} carregados "
        f"| {dataframe[DATE_COLUMN].min():%d/%m/%Y} a "
        f"{dataframe[DATE_COLUMN].max():%d/%m/%Y}"
    )
    if filtered.empty:
        st.warning("Os filtros atuais nao retornaram registros.")
        return
    render_kpis(filtered)
    render_analysis(filtered)
    render_forecast()
    with st.expander("Dados detalhados"):
        st.dataframe(filtered, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
