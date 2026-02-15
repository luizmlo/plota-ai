"""Auto-Pilot — pipeline automatizado de análise de dados em múltiplas fases.

Fases:
  1. Refletir    — LLM examina os dados brutos + perfis e descreve o que encontra
  2. Limpar      — LLM gera código para corrigir cabeçalhos, tipos, linhas inválidas
  3. Engenharia  — LLM gera código para processar features (booleanos, tags, datas…)
  4. Dashboard   — LLM gera um dashboard abrangente com múltiplos gráficos
"""

from __future__ import annotations

import textwrap

import pandas as pd

from data_loader import dataframe_summary
from feature_engine import ColumnProfile, profile_summary_text


# ─────────────────────────────────────────────────────────────────
# Definições das fases
# ─────────────────────────────────────────────────────────────────

PHASES = [
    {"key": "reflect",   "icon": "🔍", "label": "Refletindo sobre a estrutura dos dados"},
    {"key": "clean",     "icon": "🧹", "label": "Limpando e corrigindo cabeçalhos"},
    {"key": "engineer",  "icon": "⚙️",  "label": "Engenharia de features"},
    {"key": "dashboard", "icon": "📊", "label": "Construindo dashboard"},
]


# ─────────────────────────────────────────────────────────────────
# Bloco compartilhado sobre helpers disponíveis
# ─────────────────────────────────────────────────────────────────

_HELPERS_BLOCK = textwrap.dedent("""\
    Helpers disponíveis (pré-carregados, não é necessário importar):
      update_data(new_df)              — persiste um DataFrame transformado
      to_boolean(series)               — yes/no/Y/N → True/False
      explode_tags(df, col)            — one-hot para tags com múltiplos valores
      encode_categorical(series, method)
      parse_numeric_strings(series)    — remove $/%/, → float
      parse_dates(series, fmt=None)    — string → datetime
      extract_date_features(series)    — datetime → ano/mês/dia/dia_da_semana
      bin_numeric(series, bins=5)
      normalize(series, method)
      make_ordinal(series, order=None)
    Bibliotecas: pd, px (plotly.express), go (plotly.graph_objects),
                 pio (plotly.io), st (streamlit), fe (feature_engine).
""")


# ─────────────────────────────────────────────────────────────────
# Construtores de prompt — um por fase
# ─────────────────────────────────────────────────────────────────

def prompt_reflect(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    user_context: str = "",
) -> str:
    summary = dataframe_summary(df, profiles=profiles, max_rows=8)
    profile_text = profile_summary_text(profiles) if profiles else ""

    ctx = ""
    if user_context.strip():
        ctx = f"\n\nContexto fornecido pelo usuário:\n{user_context}\n"

    return textwrap.dedent(f"""\
        Você é o **Plota AI** realizando uma auditoria automática de dados.
        Responda sempre em português brasileiro.

        Examine o conjunto de dados abaixo e escreva uma análise clara e estruturada:

        1. **Sobre o que é este conjunto de dados?**  Infira o assunto, domínio e
           provável origem a partir dos nomes das colunas, valores e qualquer
           contexto fornecido pelo usuário.

        2. **Problemas de qualidade dos dados** — liste todos os problemas encontrados:
           - Cabeçalhos mesclados / sem nome / bagunçados (ex: "Unnamed: 3")
           - Linhas que na verdade são sub-cabeçalhos ou divisórias de seção
           - Tipos mistos em uma única coluna
           - Colunas do tipo booleano ainda armazenadas como strings
           - Datas armazenadas como strings
           - Números escondidos em strings (moeda, porcentagens)
           - Colunas com múltiplos valores / tags que precisam ser expandidas
           - Colunas que deveriam ser ordinais mas não estão ordenadas
           - Texto livre de alta cardinalidade vs. categorias verdadeiras
           - Padrões de dados ausentes

        3. **Plano de limpeza recomendado** — lista numerada de passos concretos.

        4. **Análises e visualizações sugeridas** — pelo menos 5 ideias de
           gráficos ou insights interessantes, considerando o significado dos dados.

        NÃO gere código ainda. Apenas forneça a análise textual.

        ── Conjunto de Dados ──
        {summary}

        {profile_text}
        {ctx}
    """)


def prompt_clean(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    reflection: str,
    user_context: str = "",
) -> str:
    summary = dataframe_summary(df, profiles=profiles, max_rows=5)
    profile_text = profile_summary_text(profiles) if profiles else ""

    ctx = ""
    if user_context.strip():
        ctx = f"\nContexto do usuário: {user_context}\n"

    return textwrap.dedent(f"""\
        Você é o **Plota AI**.  Com base na sua análise anterior, gere um
        único bloco de código Python que **limpe** este conjunto de dados.
        Responda sempre em português brasileiro.

        Sua análise anterior:
        ---
        {reflection}
        ---

        O que o código DEVE fazer (pule qualquer passo que não se aplique):
        • Corrigir cabeçalhos bagunçados — renomear colunas "Unnamed: N" para nomes
          significativos inferidos a partir do contexto dos dados. Se a(s) primeira(s)
          linha(s) contêm cabeçalhos de seção de células mescladas, use-os para
          construir nomes de colunas e depois descarte essas linhas.
        • Remover linhas lixo / divisórias.
        • Padronizar nomes de colunas (snake_case, manter idioma original se
          apropriado).
        • Descartar colunas / linhas completamente vazias.
        • No final, chamar `update_data(df_new)` para persistir as alterações.
        • Exibir um breve resumo do que foi alterado com st.markdown / st.success.

        REGRAS:
        • Produza exatamente UM bloco de código ```python```.
        • Os dados estão em `df`. NÃO leia arquivos.
        • NÃO chame st.set_page_config.
        • Seja conservador — não descarte dados que o usuário possa precisar.

        {_HELPERS_BLOCK}

        ── Conjunto de dados atual ──
        {summary}
        {profile_text}
        {ctx}
    """)


def prompt_engineer(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    reflection: str,
    user_context: str = "",
) -> str:
    summary = dataframe_summary(df, profiles=profiles, max_rows=5)
    profile_text = profile_summary_text(profiles) if profiles else ""

    ctx = ""
    if user_context.strip():
        ctx = f"\nContexto do usuário: {user_context}\n"

    return textwrap.dedent(f"""\
        Você é o **Plota AI**.  O conjunto de dados já foi limpo.
        Agora gere um único bloco de código Python que faça a **engenharia de features**.
        Responda sempre em português brasileiro.

        Sua análise anterior para referência:
        ---
        {reflection}
        ---

        O que o código DEVE fazer (pule qualquer passo que não se aplique):
        • Converter colunas do tipo booleano com `to_boolean()`.
        • Parsear colunas de data em formato string com `parse_dates()`.
        • Parsear colunas numéricas em formato string com `parse_numeric_strings()`.
        • Expandir colunas de tags com múltiplos valores com `explode_tags()`.
        • Converter colunas ordinais com `make_ordinal()`.
        • Opcionalmente adicionar colunas derivadas úteis (ex: partes de data, faixas etárias).
        • No final, chamar `update_data(df_new)` para persistir.
        • Mostrar um resumo das features criadas com st.markdown / st.success.

        REGRAS:
        • Produza exatamente UM bloco de código ```python```.
        • Os dados estão em `df`. NÃO leia arquivos.
        • NÃO chame st.set_page_config.
        • Processe apenas colunas que realmente precisam (verifique os perfis).

        {_HELPERS_BLOCK}

        ── Conjunto de dados atual (após limpeza) ──
        {summary}
        {profile_text}
        {ctx}
    """)


def prompt_dashboard(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    reflection: str,
    user_context: str = "",
) -> str:
    summary = dataframe_summary(df, profiles=profiles, max_rows=5)
    profile_text = profile_summary_text(profiles) if profiles else ""

    ctx = ""
    if user_context.strip():
        ctx = f"\nContexto do usuário: {user_context}\n"

    return textwrap.dedent(f"""\
        Você é o **Plota AI**.  O conjunto de dados está limpo e as features
        foram criadas.  Gere um único bloco de código Python que crie um
        **dashboard abrangente, bonito e interativo**.
        Responda sempre em português brasileiro.

        Sua análise anterior para referência:
        ---
        {reflection}
        ---

        Requisitos do dashboard:
        1. **Seção de cabeçalho** — título, subtítulo com descrição do conjunto
           de dados, métricas-chave em `st.columns` usando `st.metric`.

        2. **Pelo menos 5-6 gráficos** cobrindo diferentes aspectos dos dados:
           - Distribuição de colunas categóricas / booleanas principais (barra ou pizza)
           - Tabulações cruzadas / comparações agrupadas (barra agrupada, barra
             empilhada, mapa de calor)
           - Se existirem colunas numéricas: histogramas, box plots, gráficos de
             dispersão, mapas de calor de correlação
           - Se existirem colunas de data: tendências de séries temporais
           - Se colunas de tags foram expandidas: frequência de tags / co-ocorrência
           - Se existirem colunas ordinais: distribuição ordenada

        3. **Layout** — use `st.columns`, `st.tabs` ou cabeçalhos de seção
           para organizar os gráficos em um layout de dashboard limpo.
           Cada gráfico deve ter uma breve explicação em markdown acima dele.

        4. **Estilo** — use `template="plotly_white"`, paletas de cores bonitas
           (ex: px.colors.qualitative.Set2, Pastel, Bold).
           Dimensionamento consistente, bons títulos, rótulos de eixos, info ao passar o mouse.

        5. **Interatividade** — gráficos plotly já são interativos.
           Opcionalmente adicione um `st.selectbox` ou `st.multiselect` para
           filtragem se fizer sentido para os dados.

        REGRAS:
        • Produza exatamente UM bloco de código ```python```.
        • Os dados estão em `df`. NÃO leia arquivos.
        • NÃO chame st.set_page_config.
        • NÃO chame update_data() — esta é uma visualização somente leitura.
        • Trate dados ausentes de forma adequada (dropna, fillna conforme apropriado).
        • O código deve ser totalmente autocontido e executável.

        {_HELPERS_BLOCK}

        ── Conjunto de dados atual (limpo + features) ──
        {summary}
        {profile_text}
        {ctx}
    """)
