<p align="center">
  <h1 align="center">Plota AI</h1>
  <p align="center">
    Analise de dados tabulares com IA conversacional — gere dashboards interativos com um único clique.
  </p>
</p>

<p align="center">
  <a href="#funcionalidades">Funcionalidades</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#obtendo-uma-api-key">API Key</a> •
  <a href="#exemplo-de-uso">Exemplo</a> •
  <a href="#arquitetura">Arquitetura</a> •
  <a href="#adicionando-um-novo-provider-llm">Novo Provider</a> •
  <a href="#licença">Licença</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/LLM-Kimi%20K2.5-blueviolet" alt="Kimi K2.5">
  <img src="https://img.shields.io/badge/licença-MIT-green" alt="MIT License">
</p>

---

**Plota AI** é uma aplicação Streamlit que conecta seus dados tabulares a um agente de IA conversacional. Faça upload de um arquivo, descreva suas colunas e converse com a IA para explorar, limpar e visualizar os dados — tudo dentro do navegador.

O modo **Auto-Pilot** executa um pipeline completo de 4 fases com um único clique: auditoria dos dados, limpeza de cabeçalhos, engenharia de features e geração automática de dashboard.

---

## Funcionalidades

| Categoria | Descrição |
|---|---|
| **Auto-Pilot** | Pipeline automático de 4 fases — análise exploratória, limpeza, engenharia de features e geração de dashboard — tudo sem intervenção manual. |
| **Feature Engineering Inteligente** | Detecta automaticamente booleanos, tags multi-valor, ordinais, datas em strings, moedas e outros padrões semânticos, sugerindo transformações adequadas. |
| **Transformações Persistentes** | O agente pode limpar, renomear e reestruturar colunas — as alterações são aplicadas diretamente no DataFrame e persistem durante toda a sessão. |
| **Carregamento Multi-Formato** | Suporte nativo a CSV, TSV, XLSX, XLS, JSON e XML. |
| **Descrição de Colunas e Contexto** | Informe à IA o significado de cada coluna e o propósito geral do dataset para respostas mais precisas. |
| **Chat Conversacional** | Converse com o agente (Kimi K2.5 da Moonshot AI) para explorar dados, pedir análises e gerar gráficos. |
| **Geração de Código Auto-Contido** | A IA gera código Python executável usando Plotly + Streamlit, pronto para rodar. |
| **Auto-Execução com Auto-Correção** | O código gerado é executado imediatamente; se falhar, o agente corrige e tenta novamente automaticamente. |
| **Galeria de Gráficos** | Salve seus gráficos favoritos e revisite-os ou re-execute-os a qualquer momento. |
| **Providers de LLM Extensíveis** | Padrão abstrato de provider — acompanha a implementação do Kimi K2.5, mas é fácil adicionar OpenAI, Anthropic, modelos locais, etc. |

---

## Quick Start

```bash
# 1. Clone o repositório
git clone https://github.com/luizmlo/plota_ai.git
cd plota_ai

# 2. Crie e ative um ambiente virtual (recomendado)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure sua API key
cp .env.example .env
# Edite o arquivo .env com sua chave da Moonshot AI

# 5. Execute a aplicação
streamlit run app.py
```

> **Dica:** Você também pode informar a API key diretamente na barra lateral da aplicação, sem precisar do arquivo `.env`.
>
> A pasta `sample_data/` contém 10 datasets sintéticos para teste — importe qualquer um pelo upload na barra lateral.

---

## Obtendo uma API Key

O app suporta múltiplos provedores. Escolha na barra lateral:

| Provedor | API Key | Onde obter |
|----------|---------|------------|
| **Kimi (Moonshot)** | `MOONSHOT_API_KEY` | [Moonshot AI](https://platform.moonshot.ai/console/api-keys) |
| **Google AI (Gemini)** | `GOOGLE_AI_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) |

```env
# .env
MOONSHOT_API_KEY=sk-sua-chave-aqui
GOOGLE_AI_API_KEY=sua-chave-google-ai-aqui
```

---

## Exemplo de Uso

1. **Upload** — Faça upload de um CSV com respostas de pesquisa contendo colunas como `genero`, `pergunta_1_sim_nao`, `pergunta_2_sim_nao`.
2. **Descreva** — Informe o contexto: *"genero = Masculino/Feminino"*, *"pergunta_1_sim_nao = Apoia a política X?"*.
3. **Pergunte** — Digite no chat: *"Gere um gráfico de barras agrupado com as proporções de sim/não da pergunta 1, separado por gênero"*.
4. **Visualize** — O agente gera e executa o código, renderizando um gráfico interativo com Plotly diretamente na página.
5. **Salve** — Adicione o gráfico à galeria para consultar ou re-executar depois.

Ou simplesmente clique em **Auto-Pilot** e deixe a IA fazer tudo automaticamente.

---

## Arquitetura

```
plota_ai/
├── app.py                 # Aplicação principal Streamlit (4 páginas)
├── autopilot.py           # Pipeline Auto-Pilot multi-fase
├── feature_engine.py      # Detecção de tipos semânticos e transformações
├── data_loader.py         # Carregamento multi-formato
├── code_executor.py       # Execução sandboxed de código gerado
├── gallery.py             # Persistência da galeria de gráficos
├── llm_providers/
│   ├── base.py            # Interface abstrata do provider LLM
│   ├── kimi.py            # Kimi K2.5 (Moonshot AI)
│   └── google_ai.py       # Google AI (Gemini 3 Flash Preview, 2.0 Flash)
├── gallery_store/         # Gráficos salvos (auto-criado)
├── requirements.txt
└── .env.example
```

| Módulo | Responsabilidade |
|---|---|
| `app.py` | Orquestra a UI com 4 páginas: upload, descrição de colunas, chat com IA e galeria. |
| `autopilot.py` | Define e executa o pipeline Auto-Pilot (auditoria → limpeza → features → dashboard). |
| `feature_engine.py` | Analisa cada coluna para inferir tipos semânticos (booleano, tag, ordinal, data, moeda, etc.) e aplica transformações. |
| `data_loader.py` | Normaliza o carregamento de diferentes formatos tabulares em um único `pd.DataFrame`. |
| `code_executor.py` | Executa o código Python gerado pela IA em ambiente controlado, capturando saídas e erros. |
| `gallery.py` | Gerencia a persistência dos gráficos salvos (imagens + metadados) em `gallery_store/`. |
| `llm_providers/base.py` | Define a interface abstrata (`LLMProvider`) para integração com qualquer LLM. |
| `llm_providers/kimi.py` | Implementação concreta usando a API do Kimi K2.5 (Moonshot AI) via SDK OpenAI-compatible. |

---

## Adicionando um Novo Provider LLM

Crie um novo arquivo em `llm_providers/` e implemente a interface `LLMProvider`:

```python
from llm_providers.base import LLMProvider, LLMMessage, LLMResponse

class MeuProvider(LLMProvider):
    """Exemplo de provider customizado."""

    def chat(self, messages: list[LLMMessage], *, temperature=None, max_tokens=8192) -> LLMResponse:
        # Chame a API do seu modelo aqui
        resposta = chamar_minha_api(messages)
        return LLMResponse(content=resposta.texto, model="meu-modelo-v1")

    def chat_stream(self, messages: list[LLMMessage], *, temperature=None, max_tokens=8192):
        # Faça streaming da resposta, yielding chunks de texto
        for chunk in stream_minha_api(messages):
            yield chunk.texto

    def name(self) -> str:
        return "Meu Provider"
```

Depois, registre o provider em `app.py` e ele estará disponível na interface.

---

## Licença

Este projeto é distribuído sob a licença **MIT**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
