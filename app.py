from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sentinel import build_liquidity_sentinel
from sentinel.config import MODULE_LABELS
from sentinel.llm import answer_question, answer_question_local
from sentinel.utils import stress_color


st.set_page_config(page_title="RU Liquidity Sentinel", layout="wide", page_icon="PSB")


@st.cache_data(show_spinner=False)
def load_result(refresh_sources: bool = False):
    return build_liquidity_sentinel(refresh_sources=refresh_sources)


def metric_card(label: str, value: str, caption: str = ""):
    st.metric(label, value, caption)


def format_metric(value, suffix: str = "", decimals: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    if decimals == 0:
        return f"{int(value)}{suffix}"
    return f"{float(value):.{decimals}f}{suffix}"


def format_delta(latest_value, previous_value, suffix: str = "", decimals: int = 2) -> str | None:
    if pd.isna(latest_value) or pd.isna(previous_value):
        return None
    delta = float(latest_value) - float(previous_value)
    if abs(delta) < 1e-12:
        return f"0{suffix}"
    return f"{delta:+.{decimals}f}{suffix}"


def show_metric_grid(specs: list[tuple[str, str, str, int]], latest_row: pd.Series, previous_row: pd.Series):
    for chunk_start in range(0, len(specs), 4):
        cols = st.columns(4)
        for col, (label, field, suffix, decimals) in zip(cols, specs[chunk_start : chunk_start + 4]):
            with col:
                st.metric(
                    label,
                    format_metric(latest_row[field], suffix, decimals),
                    format_delta(latest_row[field], previous_row[field], suffix, decimals),
                )


def safe_answer_question(prompt: str, frame: pd.DataFrame, backtest: pd.DataFrame, messages: list[dict[str, str]]) -> str:
    try:
        return answer_question(prompt, frame, backtest, history=messages)
    except TypeError:
        try:
            return answer_question(prompt, frame, backtest)
        except Exception:
            return answer_question_local(prompt, frame, backtest)
    except Exception:
        try:
            fallback = answer_question_local(prompt, frame, backtest)
        except Exception:
            fallback = "Не удалось сформировать аналитический ответ. Попробуйте переформулировать вопрос."
        return fallback


ALL_METRIC_COLUMNS = [
    "lsi",
    "m1_actual_corr_accounts",
    "m1_required_reserves",
    "m1_reserve_spread",
    "m1_ruonia",
    "m1_mad_spread",
    "m1_mad_ruonia",
    "m1_end_period_flag",
    "m1_overbuffer_flag",
    "m2_repo_demand",
    "m2_repo_allotment",
    "m2_cover_ratio",
    "m2_key_rate",
    "m2_cutoff_rate",
    "m2_rate_spread",
    "m2_mad_cover",
    "m2_mad_rate_spread",
    "m2_demand_flag",
    "m3_offer",
    "m3_demand",
    "m3_placed",
    "m3_cover_ratio",
    "m3_weighted_yield",
    "m3_yield_spread",
    "m3_mad_cover",
    "m3_mad_yield_spread",
    "m3_nedospros_flag",
    "m3_perespros_flag",
    "tax_week_flag",
    "tax_due_core",
    "end_of_month_flag",
    "end_of_quarter_flag",
    "seasonal_factor",
    "m5_bank_treasury_balances",
    "m5_eks_deposits",
    "m5_bank_count",
    "m5_budget_drain",
    "m5_deposit_drain",
    "m5_mad_cbr",
    "m5_mad_roskazna",
    "m5_budget_drain_flag",
    "m1_contribution",
    "m2_contribution",
    "m3_contribution",
    "m4_contribution",
    "m5_contribution",
]


METRIC_GROUPS = {
    "М1 Усреднение резервов": [
        ("Корсчета, млрд руб.", "m1_actual_corr_accounts", "", 1),
        ("Обязательные резервы, млрд руб.", "m1_required_reserves", "", 1),
        ("Спред усреднения, млрд руб.", "m1_reserve_spread", "", 1),
        ("RUONIA", "m1_ruonia", "%", 2),
        ("MAD спреда", "m1_mad_spread", "", 2),
        ("MAD RUONIA", "m1_mad_ruonia", "", 2),
        ("Конец периода", "m1_end_period_flag", "", 0),
        ("Перебор нормы", "m1_overbuffer_flag", "", 0),
    ],
    "М2 Аукционы репо ЦБ": [
        ("Спрос repo, млрд руб.", "m2_repo_demand", "", 1),
        ("Размещение repo, млрд руб.", "m2_repo_allotment", "", 1),
        ("Cover repo", "m2_cover_ratio", "x", 2),
        ("Ключевая ставка", "m2_key_rate", "%", 2),
        ("Ставка отсечения", "m2_cutoff_rate", "%", 2),
        ("Спред ставки", "m2_rate_spread", " п.п.", 2),
        ("MAD cover repo", "m2_mad_cover", "", 2),
        ("Флаг спроса", "m2_demand_flag", "", 0),
    ],
    "М3 Размещение ОФЗ": [
        ("Предложение ОФЗ, млрд руб.", "m3_offer", "", 1),
        ("Спрос ОФЗ, млрд руб.", "m3_demand", "", 1),
        ("Размещено ОФЗ, млрд руб.", "m3_placed", "", 1),
        ("Cover ОФЗ", "m3_cover_ratio", "x", 2),
        ("Средняя доходность", "m3_weighted_yield", "%", 2),
        ("Спред доходности", "m3_yield_spread", " п.п.", 2),
        ("Недоспрос", "m3_nedospros_flag", "", 0),
        ("Переспрос", "m3_perespros_flag", "", 0),
    ],
    "М4 Налоги и сезонность": [
        ("Налоговая неделя", "tax_week_flag", "", 0),
        ("Ключевая дата налога", "tax_due_core", "", 0),
        ("Конец месяца", "end_of_month_flag", "", 0),
        ("Конец квартала", "end_of_quarter_flag", "", 0),
        ("Seasonal Factor", "seasonal_factor", "x", 2),
        ("Вклад М4", "m4_contribution", "", 2),
    ],
    "М5 Федеральное казначейство": [
        ("Остатки бюджета в банках, млрд руб.", "m5_bank_treasury_balances", "", 1),
        ("Депозиты ЕКС, млрд руб.", "m5_eks_deposits", "", 1),
        ("Банков-участников", "m5_bank_count", "", 0),
        ("Недельный отток, млрд руб.", "m5_budget_drain", "", 1),
        ("Месячный отток депозитов, млрд руб.", "m5_deposit_drain", "", 1),
        ("MAD ЦБ", "m5_mad_cbr", "", 2),
        ("MAD Росказна", "m5_mad_roskazna", "", 2),
        ("Флаг бюджетного оттока", "m5_budget_drain_flag", "", 0),
    ],
}


st.title("RU Liquidity Sentinel")
st.caption("Система раннего предупреждения стресса рублевой ликвидности")

with st.sidebar:
    st.header("Параметры")
    refresh = st.button("Обновить публичные источники")
    result = load_result(refresh)
    dates = result.aggregation.frame["date"]
    min_date = dates.min().date()
    max_date = dates.max().date()
    default_start = max(min_date, (dates.max() - pd.DateOffset(years=3)).date())
    selected_range = st.date_input(
        "Период",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start, end = selected_range
    else:
        start, end = default_start, max_date
    st.caption(f"История в выборке: {min_date:%d.%m.%Y} - {max_date:%d.%m.%Y}")
    if st.button("Показать всю историю"):
        start, end = min_date, max_date

frame = result.aggregation.frame
window = frame[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
latest = frame.iloc[-1]
previous = frame.iloc[-2] if len(frame) > 1 else latest

tab_dashboard, tab_modules, tab_backtest, tab_analyst, tab_sources = st.tabs(
    ["Дашборд", "Модули", "Backtest", "Аналитик", "Источники"]
)

with tab_dashboard:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("LSI", f"{latest['lsi']:.1f}", latest["status"])
    with c2:
        metric_card("RUONIA", f"{latest['m1_ruonia']:.2f}%")
    with c3:
        metric_card("Cover repo", f"{latest['m2_cover_ratio']:.2f}x")
    with c4:
        metric_card("Cover OFZ", f"{latest['m3_cover_ratio']:.2f}x")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["date"], y=window["lsi"], mode="lines", name="LSI", line=dict(color="#1f4e79", width=3)))
    fig.add_hrect(y0=40, y1=70, fillcolor="#f6c85f", opacity=0.18, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor="#d64545", opacity=0.16, line_width=0)
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=30, b=10), yaxis_range=[0, 100], title="Liquidity Stress Index")
    st.plotly_chart(fig, width="stretch")

    contrib_cols = [f"{module}_contribution" for module in MODULE_LABELS]
    latest_contrib = pd.DataFrame(
        {
            "module": [MODULE_LABELS[module] for module in MODULE_LABELS],
            "contribution": [latest[col] for col in contrib_cols],
        }
    )
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.subheader("Вклад модулей")
        st.plotly_chart(px.bar(latest_contrib, x="contribution", y="module", orientation="h", color="contribution", color_continuous_scale="Reds"), width="stretch")
    with cc2:
        st.subheader("Автокомментарий")
        st.info(result.commentary)
        st.markdown(f"**Цвет алерта:** <span style='color:{stress_color(latest['lsi'])}'>{latest['status']}</span>", unsafe_allow_html=True)

    st.subheader("Все текущие метрики")
    st.caption(f"Последняя дата расчёта: {latest['date'].date():%d.%m.%Y}; дельта показана к предыдущему рабочему дню.")
    for group, specs in METRIC_GROUPS.items():
        with st.expander(group, expanded=group.startswith("М1")):
            show_metric_grid(specs, latest, previous)

    with st.expander("История всех метрик за выбранный период", expanded=True):
        default_metrics = ["lsi", "m1_ruonia", "m2_cover_ratio", "m3_cover_ratio", "seasonal_factor", "m5_budget_drain"]
        selected_metrics = st.multiselect("Метрики на графике", ALL_METRIC_COLUMNS, default=default_metrics)
        if selected_metrics:
            history = window[["date", *selected_metrics]].copy()
            normalized = history.copy()
            for column in selected_metrics:
                col_min = normalized[column].min()
                col_max = normalized[column].max()
                if pd.notna(col_min) and pd.notna(col_max) and col_max != col_min:
                    normalized[column] = (normalized[column] - col_min) / (col_max - col_min) * 100
            fig = px.line(normalized, x="date", y=selected_metrics, title="Метрики, нормированные к 0-100 внутри выбранного периода")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(history.tail(250), width="stretch", hide_index=True)

with tab_modules:
    st.subheader("Сигналы пяти модулей")
    module_choice = st.selectbox("Модуль", list(MODULE_LABELS), format_func=lambda key: MODULE_LABELS[key])
    charts = {
        "m1": ("m1_reserve_spread", "m1_ruonia", "Спред резервов и RUONIA"),
        "m2": ("m2_cover_ratio", "m2_rate_spread", "Cover ratio и спред ставки repo"),
        "m3": ("m3_cover_ratio", "m3_yield_spread", "Cover ratio ОФЗ и спред доходности"),
        "m4": ("seasonal_factor", "tax_week_flag", "Сезонный фактор и налоговые недели"),
        "m5": ("m5_budget_drain", "m5_deposit_drain", "Приток/отток казначейства"),
    }
    y1, y2, title = charts[module_choice]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["date"], y=window[y1], name=y1, mode="lines"))
    fig.add_trace(go.Scatter(x=window["date"], y=window[y2], name=y2, mode="lines", yaxis="y2"))
    fig.update_layout(title=title, height=420, yaxis2=dict(overlaying="y", side="right"), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(window[["date", y1, y2, "lsi", "status"]].tail(60), width="stretch", hide_index=True)

with tab_backtest:
    st.subheader("Исторические стресс-эпизоды")
    st.dataframe(result.backtest, width="stretch", hide_index=True)
    st.subheader("Sensitivity ±20% к весам модулей")
    st.dataframe(result.sensitivity, width="stretch", hide_index=True)
    st.subheader("Интерпретируемая ML-модель")
    st.write(f"R2 train: {result.aggregation.model_quality['r2_train']:.3f}; holdout: {result.aggregation.model_quality['r2_holdout']:.3f}")
    st.dataframe(result.aggregation.coefficients, width="stretch", hide_index=True)

with tab_analyst:
    st.subheader("Аналитик")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    prompt = st.chat_input("Спросите: почему вырос LSI в августе 2023?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Аналитик думает..."):
            answer = safe_answer_question(prompt, frame, result.backtest, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

with tab_sources:
    st.subheader("Публичные источники")
    if result.source_results:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "key": item.key,
                        "source": item.source.name,
                        "url": item.source.url,
                        "cache": str(item.path),
                        "ok": item.ok,
                        "message": item.message,
                    }
                    for item in result.source_results
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.write("Нажмите кнопку обновления в сайдбаре, чтобы скачать свежие копии публичных страниц и Excel-файлов.")
