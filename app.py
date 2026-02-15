"""
Plota AI — Análise de dados tabulares com IA e visualização interativa.

Run:  streamlit run app.py
"""

from __future__ import annotations

import html
import os
import textwrap
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv

from autopilot import (
    PHASES,
    prompt_clean,
    prompt_dashboard,
    prompt_engineer,
    prompt_reflect,
)
from code_executor import execute_generated_code, extract_code_block
from data_loader import dataframe_summary, load_dataframe
from feature_engine import profile_dataframe, ColumnProfile
from gallery import (
    delete_plot,
    get_plot_code,
    get_plot_html,
    load_index,
    save_plot,
)
from llm_providers import GoogleAIProvider, KimiProvider, LLMMessage, LLMProvider

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Plota AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Custom CSS for a polished look
# ─────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Global ────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ───────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stTextArea label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #94a3b8 !important;
    font-weight: 500;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Chat messages ─────────────────────────── */
div[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 0.5rem;
}

/* ── Buttons ───────────────────────────────── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* ── Gallery card ──────────────────────────── */
.gallery-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    transition: box-shadow 0.2s ease;
}
.gallery-card:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.gallery-card h4 {
    margin: 0 0 0.5rem 0;
    color: #1e293b;
}
.gallery-card .meta {
    font-size: 0.8rem;
    color: #64748b;
}

/* ── Code block in chat ────────────────────── */
.generated-code-block {
    background: #1e293b;
    color: #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    overflow-x: auto;
    margin: 0.5rem 0;
}

/* ── Data preview table ────────────────────── */
div[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* ── Header area ───────────────────────────── */
.main-header {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
}
.main-header h1 {
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.main-header p {
    color: #64748b;
    font-size: 1rem;
}

/* ── Tab styling ───────────────────────────── */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
}

/* ── Auto-pilot stepper ────────────────────── */
.ap-stepper {
    display: flex;
    gap: 0.5rem;
    margin: 1rem 0 1.5rem 0;
}
.ap-step {
    flex: 1;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    background: #f1f5f9;
    border: 2px solid #e2e8f0;
    text-align: center;
    font-size: 0.85rem;
    font-weight: 500;
    color: #64748b;
    transition: all 0.3s ease;
}
.ap-step.active {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff;
    border-color: #6366f1;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35);
}
.ap-step.done {
    background: #ecfdf5;
    border-color: #10b981;
    color: #065f46;
}
.ap-step .ap-icon {
    font-size: 1.3rem;
    display: block;
    margin-bottom: 0.25rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "df": None,
    "file_name": None,
    "column_profiles": {},       # dict[str, ColumnProfile] from feature_engine
    "column_descriptions": {},
    "data_purpose": "",
    "chat_history": [],          # list[dict] with role, content, (code, figures)
    "llm_messages": [],          # raw LLMMessage list for API context
    "current_code": None,
    "current_figures": [],
    "gallery_view_id": None,
    "pending_save": None,
    "llm_provider": "kimi",
    "llm_model": "",  # resolved from provider's first model
    # Auto-pilot state
    "ap_running": False,
    "ap_phase": 0,               # 0-3 index into PHASES
    "ap_reflection": "",         # text output of the reflect phase
    "ap_log": [],                # list[dict] — phase results for display
    "ap_dashboard_code": None,   # code from the dashboard phase
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────
# LLM provider config & factory
# ─────────────────────────────────────────────────────────────────

PROVIDERS: list[dict[str, Any]] = [
    {
        "id": "kimi",
        "label": "Kimi (Moonshot)",
        "env": "MOONSHOT_API_KEY",
        "help": "https://platform.moonshot.ai/console/api-keys",
        "models": [
            {"id": "kimi-k2.5", "label": "Kimi K2.5"},
        ],
    },
    {
        "id": "google_ai",
        "label": "Google AI (Gemini)",
        "env": "GOOGLE_AI_API_KEY",
        "help": "https://aistudio.google.com/apikey",
        "models": [
            {"id": "gemini-3-flash-preview", "label": "Gemini 3 Flash Preview"},
            {"id": "gemini-2.0-flash", "label": "Gemini 2.0 Flash"},
        ],
    },
]


def get_provider() -> LLMProvider | None:
    """Build the configured LLM provider, or return None if not configured."""
    provider_id = st.session_state.get("llm_provider", "kimi")
    model_id = st.session_state.get("llm_model", "")
    api_key = st.session_state.get("api_key", "")

    provider = next((p for p in PROVIDERS if p["id"] == provider_id), None)
    if not provider:
        return None
    if not api_key:
        api_key = os.getenv(provider["env"], "")
    if not api_key:
        return None

    valid_model_ids = [m["id"] for m in provider["models"]]
    model = model_id if model_id in valid_model_ids else valid_model_ids[0]

    if provider_id == "kimi":
        return KimiProvider(api_key=api_key, model=model, thinking=False)
    if provider_id == "google_ai":
        return GoogleAIProvider(api_key=api_key, model=model)
    return None


def _apply_data_update(result: dict) -> bool:
    """If the executed code updated the DataFrame, persist it and re-profile.

    Returns True if the data was updated.
    """
    new_df = result.get("updated_df")
    if new_df is None:
        return False

    old_df = st.session_state.get("df")
    st.session_state["df"] = new_df

    # Re-profile columns
    st.session_state["column_profiles"] = profile_dataframe(new_df)

    # Merge column descriptions: keep existing ones for columns that
    # survived, add blanks for new ones, drop removed ones.
    old_descs = st.session_state.get("column_descriptions", {})
    new_descs = {col: old_descs.get(col, "") for col in new_df.columns}
    st.session_state["column_descriptions"] = new_descs

    return True


# ─────────────────────────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────────────────────────

def build_system_prompt(df: pd.DataFrame) -> str:
    profiles = st.session_state.get("column_profiles", {})
    summary = dataframe_summary(df, profiles=profiles or None)
    col_desc = st.session_state.get("column_descriptions", {})
    purpose = st.session_state.get("data_purpose", "")

    col_section = ""
    if col_desc:
        col_section = "\n\nDescrições das colunas (fornecidas pelo usuário):\n"
        for col, desc in col_desc.items():
            if desc.strip():
                col_section += f"  • {col}: {desc}\n"

    purpose_section = ""
    if purpose.strip():
        purpose_section = f"\n\nPropósito / contexto do dataset (do usuário):\n{purpose}\n"

    return textwrap.dedent(f"""\
        Você é o **Plota AI**, um analista de dados especialista e desenvolvedor
        Streamlit com fortes habilidades em engenharia de features.
        **Responda sempre em português brasileiro.**

        O usuário importou um dataset. Seu trabalho é:
        1. Ajudá-lo a entender seus dados.
        2. Sugerir análises e visualizações relevantes.
        3. **Interpretar corretamente o tipo semântico de cada coluna** antes de
           plotar ou analisar. Preste atenção à seção "Perfis semânticos das colunas"
           abaixo — ela indica quais colunas são booleanas, categóricas, ordinais,
           tags multi-valor, strings numéricas, strings de data, etc.
        4. Quando solicitado a plotar ou analisar, produza um **bloco de código Python
           auto-contido** que:
           - Usa a variável pré-carregada `df` (um pandas DataFrame).
           - Usa `plotly.express` (`px`) ou `plotly.graph_objects` (`go`) para gráficos interativos.
           - Chama `st.plotly_chart(fig, use_container_width=True)` para renderizar.
           - Pode também usar `st.metric`, `st.dataframe`, `st.markdown` ou `st.columns` para layout.
           - Para visualizações muito personalizadas, pode usar HTML/JS via
             `st.components.v1.html(html_string, height=…)`.
           - Deve ser totalmente executável sem input adicional do usuário.
           - Usa paletas de cores bonitas (ex: templates do Plotly como "plotly_white").
           - Adiciona títulos claros, rótulos de eixos, legendas e hover info.
           - Trata dados faltantes de forma elegante.
           - NUNCA chama `st.set_page_config`.
           - NUNCA lê arquivos — os dados já estão em `df`.

        ── Funções auxiliares de engenharia de features disponíveis no código ──
        As seguintes funções estão pré-carregadas no ambiente de execução:

        • `to_boolean(series, true_values=None) → pd.Series`
            Converte sim/não/Y/N/true/false/1/0/si/oui… para True/False.
            Auto-detecta mapeamento se *true_values* não for informado.

        • `explode_tags(df, column, sep=None, prefix=None) → pd.DataFrame`
            One-hot encode de coluna multi-valor/tags (ex: "esportes,música,arte").
            Auto-detecta separador. Retorna df com colunas indicadoras adicionadas.

        • `encode_categorical(series, method='onehot'|'label'|'frequency', prefix=None)`
            Codifica uma coluna categórica.

        • `parse_numeric_strings(series) → pd.Series`
            Remove "$", "€", "R$", "%", vírgulas etc. e retorna float.

        • `parse_dates(series, fmt=None) → pd.Series`
            Converte strings de data para datetime.

        • `extract_date_features(series, prefix=None) → pd.DataFrame`
            De datetime → colunas ano, mês, dia, dia_semana, hora.

        • `bin_numeric(series, bins=5, labels=None) → pd.Series`
            Agrupa coluna contínua em categorias de largura igual.

        • `normalize(series, method='minmax'|'zscore') → pd.Series`
            Escala uma coluna numérica.

        • `make_ordinal(series, order=None) → pd.Series`
            Converte para Categorical ordenado (auto-detecta escalas Likert, etc.).

        • O módulo completo `fe` também está disponível (`import feature_engine as fe`).

        ── Persistindo transformações nos dados ──
        Quando o usuário pedir para **transformar, limpar, refatorar, renomear, filtrar,
        mesclar colunas, corrigir cabeçalhos, reestruturar ou modificar o dataset**,
        você DEVE persistir as mudanças chamando:

            update_data(df_new)

        Isso substitui o dataset carregado para todas as análises subsequentes.
        O sistema irá automaticamente re-analisar as colunas e notificar o usuário.

        IMPORTANTE: Sempre chame `update_data(df_new)` como ÚLTIMA instrução
        quando o propósito do código é transformar os dados. Você também pode
        reatribuir `df = df_new` por conveniência — ambas abordagens funcionam.

        Padrões comuns de transformação:
        ─ Renomear colunas:       df_new = df.rename(columns={{...}})
        ─ Remover linhas:         df_new = df.iloc[1:].reset_index(drop=True)
        ─ Corrigir cabeçalhos:    montar mapeamento de colunas, renomear, remover linhas de cabeçalho
        ─ Converter tipos:        df_new["col"] = parse_dates(df["col"])
        ─ Adicionar derivadas:    df_new["col_bool"] = to_boolean(df["col"])
        ─ Explodir tags:          df_new = explode_tags(df, "tags_col")
        Após qualquer uma dessas, chame update_data(df_new).

        REGRAS IMPORTANTES para código gerado:
        ─ Quando uma coluna é "boolean", converta com `to_boolean()` antes de contar
          ou plotar proporções. NÃO trate strings "Sim"/"Não" brutas como categorias
          quando o usuário perguntar sobre proporções.
        ─ Quando uma coluna é "multi_value_tags", use `explode_tags()` para dividir
          em colunas indicadoras antes de agregar.
        ─ Quando uma coluna é "numeric_string" (ex: "R$1.200"), chame
          `parse_numeric_strings()` antes de fazer cálculos.
        ─ Quando uma coluna é "ordinal", use `make_ordinal()` para que os gráficos
          respeitem a ordem natural (ex: Discordo Totalmente < Discordo < Neutro < …).
        ─ Quando uma coluna é "date_string", converta com `parse_dates()` antes de
          análise temporal ou agrupamento por partes da data.
        ─ Quando uma coluna é "categorical", trate como fator — use value_counts,
          group-by ou `encode_categorical()` conforme apropriado.

        Sempre envolva o código gerado em um bloco ```python ... ```.
        Antes do código, forneça uma breve explicação do que a transformação ou
        gráfico faz e por quê.
        Após o código, ofereça sugestões de próximos passos.

        ── Resumo do dataset ──
        {summary}
        {col_section}
        {purpose_section}
    """)


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📁 Importar Dados")

    uploaded_file = st.file_uploader(
        "Escolha um arquivo",
        type=["csv", "tsv", "xlsx", "xls", "json", "xml"],
        help="Formatos: CSV, TSV, XLSX, XLS, JSON, XML",
    )

    if uploaded_file is not None:
        # Only reload if the file changed
        if st.session_state["file_name"] != uploaded_file.name:
            try:
                df = load_dataframe(uploaded_file)
                st.session_state["df"] = df
                st.session_state["file_name"] = uploaded_file.name
                # Auto-profile columns for semantic type detection
                st.session_state["column_profiles"] = profile_dataframe(df)
                # Reset column descriptions for new file
                st.session_state["column_descriptions"] = {
                    col: "" for col in df.columns
                }
                # Reset chat
                st.session_state["chat_history"] = []
                st.session_state["llm_messages"] = []
                st.session_state["current_code"] = None
                st.session_state["current_figures"] = []
                # Reset auto-pilot
                st.session_state["ap_running"] = False
                st.session_state["ap_phase"] = 0
                st.session_state["ap_reflection"] = ""
                st.session_state["ap_log"] = []
                st.session_state["ap_dashboard_code"] = None
                st.success(f"Carregado **{uploaded_file.name}** — {df.shape[0]} linhas × {df.shape[1]} colunas")
            except Exception as exc:
                st.error(f"Falha ao carregar arquivo: {exc}")

    # ── Column descriptions ──────────────────────────────────────
    if st.session_state["df"] is not None:
        st.markdown("---")
        st.markdown("### 🏷️ Descreva seus Dados")

        st.session_state["data_purpose"] = st.text_area(
            "Propósito / contexto geral",
            value=st.session_state.get("data_purpose", ""),
            height=80,
            placeholder="Ex: Respostas de pesquisa de 500 participantes sobre satisfação…",
        )

        with st.expander("Descrição das colunas", expanded=False):
            descs = st.session_state["column_descriptions"]
            for col in st.session_state["df"].columns:
                descs[col] = st.text_input(
                    f"`{col}`",
                    value=descs.get(col, ""),
                    key=f"col_desc_{col}",
                    placeholder=f"Descreva o que '{col}' representa…",
                )
            st.session_state["column_descriptions"] = descs

    # ── Navigation ───────────────────────────────────────────────
    st.markdown("---")
    page = st.radio(
        "Navegar",
        ["🤖 Auto-Pilot", "💬 Chat & Análise", "📊 Prévia dos Dados", "🖼️ Galeria"],
        index=0,
    )

    # ── Model / Provider (last) ───────────────────────────────────
    st.markdown("---")
    with st.expander("⚙️ Modelo / Provedor", expanded=False):
        # Provider selector
        provider_options = [p["label"] for p in PROVIDERS]
        provider_idx = next((i for i, p in enumerate(PROVIDERS) if p["id"] == st.session_state.get("llm_provider", "kimi")), 0)
        provider_label = st.selectbox(
            "Provedor de IA",
            provider_options,
            index=provider_idx,
            key="llm_provider_select",
        )
        provider = next(p for p in PROVIDERS if p["label"] == provider_label)
        st.session_state["llm_provider"] = provider["id"]

        # Model selector (filtered by provider)
        model_options = [m["label"] for m in provider["models"]]
        model_ids = [m["id"] for m in provider["models"]]
        current_model = st.session_state.get("llm_model", model_ids[0])
        model_idx = next((i for i, mid in enumerate(model_ids) if mid == current_model), 0)
        model_label = st.selectbox(
            "Modelo",
            model_options,
            index=model_idx,
            key="llm_model_select",
        )
        st.session_state["llm_model"] = model_ids[model_options.index(model_label)]

        # API key (per-provider, stored separately)
        api_key_env = provider["env"]
        api_key_key = f"api_key_{provider['id']}"
        api_key_default = st.session_state.get(api_key_key, "") or os.getenv(api_key_env, "")
        api_key_label = "Chave API Moonshot" if provider["id"] == "kimi" else "Chave API Google AI"
        api_key_input = st.text_input(
            api_key_label,
            type="password",
            value=api_key_default,
            help=f"Obtenha em {provider['help']}",
            key="api_key_input",
        )
        st.session_state[api_key_key] = api_key_input
        st.session_state["api_key"] = api_key_input  # current provider's key for get_provider

# ─────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="main-header">
    <h1>Plota AI</h1>
    <p>Importe dados · Descreva colunas · Converse com IA · Gere visualizações interativas</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────
# Page: Auto-Pilot
# ─────────────────────────────────────────────────────────────────

if page == "🤖 Auto-Pilot":
    if st.session_state["df"] is None:
        st.info("👈 Importe um arquivo na barra lateral para iniciar o Auto-Pilot.")
    elif not get_provider():
        st.warning("🔑 Insira sua chave API na barra lateral.")
    else:
        df_ap = st.session_state["df"]
        provider_ap = get_provider()
        profiles_ap: dict[str, ColumnProfile] = st.session_state.get("column_profiles", {})
        user_ctx = st.session_state.get("data_purpose", "")

        # ── Stepper bar ──────────────────────────────────────────
        current_phase = st.session_state.get("ap_phase", 0)
        ap_running = st.session_state.get("ap_running", False)

        def _stepper_html(current: int, running: bool) -> str:
            parts = []
            for i, ph in enumerate(PHASES):
                if i < current:
                    cls = "ap-step done"
                    icon = "✅"
                elif i == current and running:
                    cls = "ap-step active"
                    icon = ph["icon"]
                else:
                    cls = "ap-step"
                    icon = ph["icon"]
                parts.append(
                    f'<div class="{cls}">'
                    f'<span class="ap-icon">{icon}</span>'
                    f'{ph["label"]}'
                    f'</div>'
                )
            return '<div class="ap-stepper">' + "".join(parts) + '</div>'

        st.markdown(_stepper_html(current_phase, ap_running), unsafe_allow_html=True)

        # ── Context input (before launch) ────────────────────────
        if not ap_running and current_phase == 0 and not st.session_state.get("ap_log"):
            st.markdown("##### Forneça contexto opcional para guiar a análise")
            ap_context = st.text_area(
                "Descreva o dataset, as colunas ou o que deseja descobrir",
                value=user_ctx,
                height=100,
                placeholder="Ex: Esta é uma pesquisa sobre comunidades pesqueiras. "
                            "A primeira linha tem cabeçalhos de seções mescladas…",
                key="ap_user_context",
            )
            st.session_state["data_purpose"] = ap_context

            if st.button("🚀 Iniciar Auto-Pilot", type="primary", use_container_width=True):
                st.session_state["ap_running"] = True
                st.session_state["ap_phase"] = 0
                st.session_state["ap_log"] = []
                st.session_state["ap_reflection"] = ""
                st.session_state["ap_dashboard_code"] = None
                st.rerun()

        # ── Running pipeline ─────────────────────────────────────
        if ap_running:
            phase_idx = st.session_state["ap_phase"]

            # Helper to call LLM and stream into a placeholder
            def _llm_stream(system_prompt: str, max_tok: int = 16384) -> str:
                msgs = [LLMMessage(role="system", content=system_prompt)]
                full = ""
                ph = st.empty()
                try:
                    for chunk in provider_ap.chat_stream(msgs, max_tokens=max_tok):
                        full += chunk
                        ph.markdown(full + "▌")
                    ph.markdown(full)
                except Exception as exc:
                    full = f"⚠️ Error: {exc}"
                    ph.error(full)
                return full

            # ── Phase 0: Reflect ─────────────────────────────────
            if phase_idx == 0:
                st.markdown("### 🔍 Fase 1 — Analisando estrutura dos dados")
                with st.spinner("Pensando…"):
                    prompt = prompt_reflect(df_ap, profiles_ap, user_ctx)
                    reflection = _llm_stream(prompt)
                st.session_state["ap_reflection"] = reflection
                st.session_state["ap_log"].append({
                    "phase": "reflect", "text": reflection, "code": None,
                })
                st.session_state["ap_phase"] = 1
                st.rerun()

            # ── Phase 1: Clean ───────────────────────────────────
            elif phase_idx == 1:
                # Show previous reflection
                for entry in st.session_state["ap_log"]:
                    if entry["phase"] == "reflect":
                        with st.expander("🔍 Fase 1 — Análise dos dados", expanded=False):
                            st.markdown(entry["text"])

                st.markdown("### 🧹 Fase 2 — Limpeza e correção de cabeçalhos")
                reflection = st.session_state["ap_reflection"]
                prompt = prompt_clean(df_ap, profiles_ap, reflection, user_ctx)
                response = _llm_stream(prompt)

                code_str = None
                if "```python" in response or "```" in response:
                    try:
                        code_str = extract_code_block(response)
                    except (ValueError, IndexError):
                        code_str = None

                if code_str:
                    with st.spinner("Executando limpeza…"):
                        result = execute_generated_code(code_str, df_ap)
                    if result["success"]:
                        if _apply_data_update(result):
                            df_ap = st.session_state["df"]
                            profiles_ap = st.session_state["column_profiles"]
                            st.success(
                                f"✅ Dados limpos — {df_ap.shape[0]} linhas × {df_ap.shape[1]} colunas"
                            )
                    else:
                        st.warning("Código de limpeza teve erros, avançando para a próxima fase.")
                        with st.expander("Detalhes do erro"):
                            st.code(result["error"], language="text")

                st.session_state["ap_log"].append({
                    "phase": "clean", "text": response, "code": code_str,
                })
                st.session_state["ap_phase"] = 2
                st.rerun()

            # ── Phase 2: Feature engineering ─────────────────────
            elif phase_idx == 2:
                for entry in st.session_state["ap_log"]:
                    label = {"reflect": "🔍 Análise", "clean": "🧹 Limpeza"}.get(
                        entry["phase"], entry["phase"]
                    )
                    with st.expander(label, expanded=False):
                        st.markdown(entry["text"])

                st.markdown("### ⚙️ Fase 3 — Engenharia de features")
                # Refresh df/profiles in case clean phase updated them
                df_ap = st.session_state["df"]
                profiles_ap = st.session_state.get("column_profiles", {})
                reflection = st.session_state["ap_reflection"]

                prompt = prompt_engineer(df_ap, profiles_ap, reflection, user_ctx)
                response = _llm_stream(prompt)

                code_str = None
                if "```python" in response or "```" in response:
                    try:
                        code_str = extract_code_block(response)
                    except (ValueError, IndexError):
                        code_str = None

                if code_str:
                    with st.spinner("Executando engenharia de features…"):
                        result = execute_generated_code(code_str, df_ap)
                    if result["success"]:
                        if _apply_data_update(result):
                            df_ap = st.session_state["df"]
                            profiles_ap = st.session_state["column_profiles"]
                            st.success(
                                f"✅ Features processadas — {df_ap.shape[0]} linhas × {df_ap.shape[1]} colunas"
                            )
                    else:
                        st.warning("Engenharia de features teve erros, continuando para o dashboard.")
                        with st.expander("Detalhes do erro"):
                            st.code(result["error"], language="text")

                st.session_state["ap_log"].append({
                    "phase": "engineer", "text": response, "code": code_str,
                })
                st.session_state["ap_phase"] = 3
                st.rerun()

            # ── Phase 3: Dashboard ───────────────────────────────
            elif phase_idx == 3:
                for entry in st.session_state["ap_log"]:
                    label = {
                        "reflect": "🔍 Análise",
                        "clean": "🧹 Limpeza",
                        "engineer": "⚙️ Features",
                    }.get(entry["phase"], entry["phase"])
                    with st.expander(label, expanded=False):
                        st.markdown(entry["text"])

                st.markdown("### 📊 Fase 4 — Construindo dashboard")
                df_ap = st.session_state["df"]
                profiles_ap = st.session_state.get("column_profiles", {})
                reflection = st.session_state["ap_reflection"]

                prompt = prompt_dashboard(df_ap, profiles_ap, reflection, user_ctx)
                response = _llm_stream(prompt, max_tok=32000)

                code_str = None
                if "```python" in response or "```" in response:
                    try:
                        code_str = extract_code_block(response)
                    except (ValueError, IndexError):
                        code_str = None

                st.session_state["ap_log"].append({
                    "phase": "dashboard", "text": response, "code": code_str,
                })
                st.session_state["ap_dashboard_code"] = code_str
                st.session_state["ap_running"] = False
                st.session_state["ap_phase"] = 4  # done
                st.rerun()

        # ── Completed — show results ─────────────────────────────
        if not ap_running and st.session_state.get("ap_log"):
            # Show collapsed phase logs
            for entry in st.session_state["ap_log"]:
                if entry["phase"] == "dashboard":
                    continue  # dashboard renderizado separadamente abaixo
                label = {
                    "reflect": "🔍 Fase 1 — Análise dos dados",
                    "clean": "🧹 Fase 2 — Limpeza",
                    "engineer": "⚙️ Fase 3 — Engenharia de features",
                }.get(entry["phase"], entry["phase"])
                with st.expander(label, expanded=False):
                    st.markdown(entry["text"])
                    if entry.get("code"):
                        st.code(entry["code"], language="python")

            # Renderizar o dashboard
            dash_code = st.session_state.get("ap_dashboard_code")
            if dash_code:
                st.markdown("---")
                st.markdown("## 📊 Dashboard Gerado")
                dash_container = st.container()
                with dash_container:
                    result = execute_generated_code(
                        dash_code, st.session_state["df"], container=dash_container
                    )
                    if result["success"]:
                        for fig in result.get("show_only_figures", []):
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Erro ao renderizar dashboard:")
                        st.code(result["error"], language="text")

                with st.expander("Ver código do dashboard", expanded=False):
                    st.code(dash_code, language="python")

                # Salvar dashboard na galeria
                col_save, col_reset = st.columns(2)
                with col_save:
                    if st.button("💾 Salvar dashboard na galeria", use_container_width=True):
                        fig_jsons = []
                        plot_html_str = ""
                        if result.get("figures"):
                            import plotly.io as _pio
                            fig_jsons = [_pio.to_json(f) for f in result["figures"]]
                            plot_html_str = _pio.to_html(
                                result["figures"][0], full_html=True, include_plotlyjs="cdn"
                            )
                        save_plot(
                            title="Dashboard Auto-Pilot",
                            description="Dashboard gerado automaticamente pelo Auto-Pilot",
                            code=dash_code,
                            plot_html=plot_html_str,
                            user_query="Análise Auto-Pilot",
                            column_context=str(st.session_state.get("column_descriptions", "")),
                        )
                        st.success("Salvo na galeria!")
                with col_reset:
                    if st.button("🔄 Reiniciar Auto-Pilot", use_container_width=True, type="secondary"):
                        st.session_state["ap_running"] = False
                        st.session_state["ap_phase"] = 0
                        st.session_state["ap_log"] = []
                        st.session_state["ap_reflection"] = ""
                        st.session_state["ap_dashboard_code"] = None
                        st.rerun()

# ─────────────────────────────────────────────────────────────────
# Page: Data Preview
# ─────────────────────────────────────────────────────────────────

elif page == "📊 Prévia dos Dados":
    if st.session_state["df"] is None:
        st.info("👈 Importe um arquivo na barra lateral para começar.")
    else:
        df: pd.DataFrame = st.session_state["df"]
        profiles: dict[str, ColumnProfile] = st.session_state.get("column_profiles", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Linhas", f"{df.shape[0]:,}")
        col2.metric("Colunas", f"{df.shape[1]:,}")
        col3.metric("Memória", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        # Contar tipos semânticos
        type_counts: dict[str, int] = {}
        for p in profiles.values():
            type_counts[p.semantic_type] = type_counts.get(p.semantic_type, 0) + 1
        top_type = max(type_counts, key=type_counts.get) if type_counts else "—"  # type: ignore[arg-type]
        col4.metric("Tipo dominante", top_type)

        st.markdown("#### Prévia dos Dados")
        st.dataframe(df, use_container_width=True, height=400)

        st.markdown("#### Estatísticas")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        # ── Rich column profiles ─────────────────────────────────
        st.markdown("#### Perfis das Colunas")
        st.caption("Tipos semânticos são detectados automaticamente. "
                   "Sobrescreva descrevendo as colunas na barra lateral.")

        _SEMANTIC_BADGES: dict[str, str] = {
            "boolean": "🔘",
            "categorical": "🏷️",
            "ordinal": "📶",
            "multi_value_tags": "🔖",
            "numeric_string": "💲",
            "date_string": "📅",
            "free_text": "📝",
            "identifier": "🔑",
            "numeric": "🔢",
            "datetime": "📅",
            "unknown": "❓",
        }

        profile_rows = []
        for c in df.columns:
            p = profiles.get(c)
            if p:
                badge = _SEMANTIC_BADGES.get(p.semantic_type, "")
                detail = ""
                if p.semantic_type == "boolean" and p.boolean_map:
                    truthy = [k for k, v in p.boolean_map.items() if v]
                    falsy = [k for k, v in p.boolean_map.items() if not v]
                    detail = f"true={truthy}, false={falsy}"
                elif p.semantic_type == "multi_value_tags":
                    n_tags = len(p.tag_vocabulary or [])
                    sample_tags = (p.tag_vocabulary or [])[:6]
                    detail = f"sep='{p.tag_separator}', {n_tags} tags: {sample_tags}"
                elif p.semantic_type == "ordinal" and p.ordinal_order:
                    detail = " → ".join(p.ordinal_order)
                elif p.semantic_type == "categorical" and p.categories:
                    shown = p.categories[:8]
                    detail = ", ".join(shown)
                    if len(p.categories) > 8:
                        detail += f" … (+{len(p.categories) - 8} more)"
                elif p.semantic_type == "numeric_string":
                    detail = f"prefix='{p.numeric_prefix}' suffix='{p.numeric_suffix}'"
                elif p.semantic_type == "date_string":
                    detail = f"pattern: {p.date_format_hint}"

                profile_rows.append({
                    "Coluna": c,
                    "Tipo Pandas": p.pandas_dtype,
                    "Tipo Semântico": f"{badge} {p.semantic_type}",
                    "Únicos": p.n_unique,
                    "Nulo": "✓" if p.nullable else "",
                    "Detalhes": detail,
                    "Descrição": st.session_state["column_descriptions"].get(c, ""),
                })
            else:
                profile_rows.append({
                    "Coluna": c,
                    "Tipo Pandas": str(df[c].dtype),
                    "Tipo Semântico": "—",
                    "Únicos": int(df[c].nunique()),
                    "Nulo": "✓" if df[c].isna().any() else "",
                    "Detalhes": "",
                    "Descrição": st.session_state["column_descriptions"].get(c, ""),
                })

        st.dataframe(
            pd.DataFrame(profile_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Detalhes": st.column_config.TextColumn(width="large"),
            },
        )

        # ── Visual breakdown of detected types ───────────────────
        if type_counts:
            import plotly.express as _px
            type_df = pd.DataFrame([
                {"Tipo Semântico": k, "Contagem": v} for k, v in type_counts.items()
            ])
            fig = _px.bar(
                type_df, x="Tipo Semântico", y="Contagem",
                color="Tipo Semântico",
                template="plotly_white",
                title="Distribuição de tipos semânticos das colunas",
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# Page: Plot Gallery
# ─────────────────────────────────────────────────────────────────

elif page == "🖼️ Galeria":
    st.markdown("#### 🖼️ Galeria de Gráficos")
    index = load_index()

    if not index:
        st.info("Nenhum gráfico salvo ainda. Gere um gráfico no Chat e salve-o!")
    else:
        # ── Detail view ──────────────────────────────────────────
        view_id = st.session_state.get("gallery_view_id")
        if view_id:
            entry = None
            for e in index:
                if e["id"] == view_id:
                    entry = e
                    break
            if entry:
                col_back, col_title = st.columns([1, 5])
                with col_back:
                    if st.button("← Voltar à Galeria"):
                        st.session_state["gallery_view_id"] = None
                        st.rerun()
                with col_title:
                    st.markdown(f"### {entry['title']}")

                st.markdown(f"*{entry.get('description', '')}*")
                st.caption(f"Criado em: {entry['created_at']} · Consulta: {entry.get('user_query', '')}")

                plot_html = get_plot_html(view_id)
                if plot_html:
                    st.components.v1.html(plot_html, height=550, scrolling=True)

                with st.expander("Ver código gerado", expanded=False):
                    code = get_plot_code(view_id)
                    st.code(code, language="python")

                st.markdown("---")
                col_rerun, col_del = st.columns([1, 1])
                with col_rerun:
                    if st.button("🔄 Re-executar gráfico", use_container_width=True):
                        code = get_plot_code(view_id)
                        if code and st.session_state["df"] is not None:
                            result = execute_generated_code(code, st.session_state["df"])
                            if result["success"]:
                                for fig in result["figures"]:
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error(f"Erro ao re-executar código:\n{result['error']}")
                        else:
                            st.warning("Carregue o mesmo dataset primeiro para re-executar.")
                with col_del:
                    if st.button("🗑️ Excluir gráfico", use_container_width=True, type="secondary"):
                        delete_plot(view_id)
                        st.session_state["gallery_view_id"] = None
                        st.rerun()
            else:
                st.session_state["gallery_view_id"] = None
                st.rerun()
        else:
            # ── Grid view ────────────────────────────────────────
            cols = st.columns(3)
            for i, entry in enumerate(index):
                with cols[i % 3]:
                    st.markdown(
                        f"""<div class="gallery-card">
                            <h4>{html.escape(entry['title'])}</h4>
                            <p>{html.escape(entry.get('description', '')[:120])}</p>
                            <p class="meta">{entry['created_at']}</p>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    if st.button("Ver", key=f"view_{entry['id']}", use_container_width=True):
                        st.session_state["gallery_view_id"] = entry["id"]
                        st.rerun()

# ─────────────────────────────────────────────────────────────────
# Page: Chat & Analyze
# ─────────────────────────────────────────────────────────────────

elif page == "💬 Chat & Análise":
    if st.session_state["df"] is None:
        st.info("👈 Importe um arquivo na barra lateral para começar a conversar com seus dados.")
    elif not get_provider():
        st.warning("🔑 Insira sua chave API na barra lateral para ativar o assistente IA.")
    else:
        df = st.session_state["df"]
        provider = get_provider()

        # ── Quick-action suggestion chips ────────────────────────
        if not st.session_state["chat_history"]:
            st.markdown("##### 💡 Sugestões rápidas")

            # Montar sugestões contextuais baseadas nos tipos de colunas detectados
            profiles = st.session_state.get("column_profiles", {})
            type_set = {p.semantic_type for p in profiles.values()} if profiles else set()

            suggestions = ["Resuma este dataset e os tipos de suas colunas"]
            if "boolean" in type_set or "categorical" in type_set:
                suggestions.append("Mostre proporções de todas as colunas sim/não e categóricas")
            if "multi_value_tags" in type_set:
                suggestions.append("Exploda colunas de tags e mostre frequência de cada tag")
            if "ordinal" in type_set:
                suggestions.append("Mostre distribuições das colunas ordinais em ordem")
            if "numeric" in type_set or "numeric_string" in type_set:
                suggestions.append("Mostre distribuições e correlações das colunas numéricas")
            if len(suggestions) < 4:
                suggestions.append("Sugira análises interessantes para estes dados")
            suggestions = suggestions[:4]

            sugg_cols = st.columns(len(suggestions))
            for i, s in enumerate(suggestions):
                with sugg_cols[i]:
                    if st.button(s, key=f"sugg_{i}", use_container_width=True):
                        st.session_state["_pending_prompt"] = s
                        st.rerun()

        # ── Chat history display ─────────────────────────────────
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    # If assistant message had code + figures, re-render figures
                    if msg["role"] == "assistant" and msg.get("figures"):
                        for fig_json in msg["figures"]:
                            try:
                                fig = go.Figure(pio.from_json(fig_json))
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                pass
                    if msg["role"] == "assistant" and msg.get("code"):
                        with st.expander("Ver código", expanded=False):
                            st.code(msg["code"], language="python")
                        # Botão de salvar
                        save_key = f"save_{hash(msg['code'])}"
                        if st.button("💾 Salvar na galeria", key=save_key):
                            st.session_state["pending_save"] = msg
                            st.rerun()

        # ── Handle pending save ──────────────────────────────────
        if st.session_state.get("pending_save"):
            msg = st.session_state["pending_save"]
            with st.form("save_form"):
                st.markdown("#### Salvar gráfico na galeria")
                title = st.text_input("Título", value="")
                description = st.text_area("Descrição", value="")
                submitted = st.form_submit_button("Salvar")
                cancel = st.form_submit_button("Cancelar")
                if submitted:
                    # Generate HTML from first figure
                    plot_html_str = ""
                    if msg.get("figures"):
                        try:
                            fig = go.Figure(pio.from_json(msg["figures"][0]))
                            plot_html_str = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
                        except Exception:
                            plot_html_str = "<p>Não foi possível renderizar a prévia do gráfico.</p>"
                    else:
                        # Try executing code to capture HTML
                        code = msg.get("code", "")
                        if code and st.session_state["df"] is not None:
                            result = execute_generated_code(code, st.session_state["df"])
                            if result["figures"]:
                                plot_html_str = pio.to_html(result["figures"][0], full_html=True, include_plotlyjs="cdn")

                    save_plot(
                        title=title or "Gráfico sem título",
                        description=description,
                        code=msg.get("code", ""),
                        plot_html=plot_html_str,
                        user_query=msg.get("user_query", ""),
                        column_context=str(st.session_state.get("column_descriptions", "")),
                    )
                    st.session_state["pending_save"] = None
                    st.success("Salvo na galeria!")
                    st.rerun()
                if cancel:
                    st.session_state["pending_save"] = None
                    st.rerun()

        # ── Chat input ───────────────────────────────────────────
        pending = st.session_state.pop("_pending_prompt", None)
        user_input = st.chat_input("Pergunte sobre seus dados, peça um gráfico ou descreva o que precisa…")
        prompt = pending or user_input

        if prompt:
            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Build messages for LLM
            system_prompt = build_system_prompt(df)
            llm_messages: list[LLMMessage] = [LLMMessage(role="system", content=system_prompt)]

            # Add conversation history (keep last 20 exchanges for context window)
            for msg in st.session_state["llm_messages"][-40:]:
                llm_messages.append(msg)

            llm_messages.append(LLMMessage(role="user", content=prompt))

            # Stream response
            with chat_container:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""

                    try:
                        for chunk in provider.chat_stream(llm_messages, max_tokens=16384):
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)
                    except Exception as exc:
                        full_response = f"⚠️ Erro ao comunicar com o LLM: {exc}"
                        response_placeholder.error(full_response)

            # Extract and execute code if present
            code_str = None
            fig_jsons: list[str] = []
            rendered_in_exec = False

            if "```python" in full_response or "```" in full_response:
                try:
                    code_str = extract_code_block(full_response)
                except (ValueError, IndexError):
                    code_str = None

            if code_str:
                with chat_container:
                    with st.chat_message("assistant"):
                        exec_container = st.container()
                        with exec_container:
                            with st.spinner("Executando código gerado…"):
                                result = execute_generated_code(code_str, df, container=exec_container)

                            if result["success"]:
                                # Figures via st.plotly_chart are already displayed;
                                # figures via fig.show() need explicit rendering.
                                for fig in result.get("show_only_figures", []):
                                    st.plotly_chart(fig, use_container_width=True)
                                # Capture ALL figures for gallery saving
                                for fig in result["figures"]:
                                    fig_jsons.append(pio.to_json(fig))
                                rendered_in_exec = True
                                # Persist data transforms if the code updated df
                                if _apply_data_update(result):
                                    new_df = st.session_state["df"]
                                    st.info(
                                        f"📝 **Dados atualizados** — agora {new_df.shape[0]} linhas "
                                        f"× {new_df.shape[1]} colunas. "
                                        f"Colunas re-analisadas. Veja em Prévia dos Dados."
                                    )
                                    # Atualizar referência local do df para o resto desta interação
                                    df = new_df
                            else:
                                st.error("Erro na execução do código:")
                                st.code(result["error"], language="text")
                                # Try to self-heal: ask the LLM to fix the code
                                fix_messages = llm_messages + [
                                    LLMMessage(role="assistant", content=full_response),
                                    LLMMessage(
                                        role="user",
                                        content=f"O código acima produziu este erro:\n```\n{result['error']}\n```\nPor favor corrija o código e forneça uma versão corrigida.",
                                    ),
                                ]
                                try:
                                    fix_response = ""
                                    fix_placeholder = st.empty()
                                    for chunk in provider.chat_stream(fix_messages, max_tokens=16384):
                                        fix_response += chunk
                                        fix_placeholder.markdown(fix_response + "▌")
                                    fix_placeholder.markdown(fix_response)

                                    if "```python" in fix_response or "```" in fix_response:
                                        fixed_code = extract_code_block(fix_response)
                                        fix_result = execute_generated_code(fixed_code, df, container=exec_container)
                                        if fix_result["success"]:
                                            for fig in fix_result.get("show_only_figures", []):
                                                st.plotly_chart(fig, use_container_width=True)
                                            for fig in fix_result["figures"]:
                                                fig_jsons.append(pio.to_json(fig))
                                            code_str = fixed_code
                                            full_response += "\n\n---\n\n**[Auto-correção aplicada]**\n\n" + fix_response
                                            rendered_in_exec = True
                                            if _apply_data_update(fix_result):
                                                new_df = st.session_state["df"]
                                                st.info(
                                                    f"📝 **Dados atualizados** — agora {new_df.shape[0]} linhas "
                                                    f"× {new_df.shape[1]} colunas."
                                                )
                                                df = new_df
                                except Exception:
                                    pass

                        if code_str:
                            with st.expander("Ver código", expanded=False):
                                st.code(code_str, language="python")
                            save_key = f"save_new_{hash(code_str + prompt)}"
                            if st.button("💾 Salvar na galeria", key=save_key):
                                st.session_state["pending_save"] = {
                                    "code": code_str,
                                    "figures": fig_jsons,
                                    "user_query": prompt,
                                    "content": full_response,
                                }
                                st.rerun()

            # Update session state
            st.session_state["chat_history"].append({
                "role": "user",
                "content": prompt,
            })
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": full_response,
                "code": code_str,
                "figures": fig_jsons,
                "user_query": prompt,
            })
            st.session_state["llm_messages"].append(LLMMessage(role="user", content=prompt))
            st.session_state["llm_messages"].append(LLMMessage(role="assistant", content=full_response))
            st.session_state["current_code"] = code_str
