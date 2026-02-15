# Dados de Exemplo — Plota AI

Conjunto de 10 datasets sintéticos para demonstração e teste da aplicação.

## Datasets

| Arquivo | Contexto | Tipos semânticos exercitados |
|---------|----------|------------------------------|
| `01_pesquisa_satisfacao_produto.csv` | Pesquisa de satisfação de produto | ordinal (Likert), boolean, categorical |
| `02_vendas_ecommerce.csv` | Vendas de e-commerce | numeric, datetime, categorical, numeric_string (moeda) |
| `03_funcionarios_rh.xlsx` | Dados de funcionários (RH) | boolean, categorical, date_string, numeric |
| `04_consultas_medicas.csv` | Consultas médicas / prontuário | numeric, categorical, date_string |
| `05_pesquisa_pesca_comunitaria.csv` | Pesquisa com pescadores | boolean, multi_value_tags, date_string, categorical |
| `06_engajamento_redes_sociais.json` | Métricas de redes sociais | numeric, categorical, datetime |
| `07_avaliacoes_restaurantes.csv` | Avaliações de restaurantes | ordinal (rating), free_text, categorical |
| `08_estoque_loja.csv` | Inventário de loja | numeric_string (preço), categorical, numeric |
| `09_presenca_eventos.csv` | Presença em eventos | boolean, multi_value_tags, date_string |
| `10_tickets_suporte.csv` | Tickets de suporte ao cliente | categorical, ordinal (prioridade), free_text |

## Formatos

- **CSV**: 8 arquivos (encoding UTF-8 com BOM)
- **XLSX**: 1 arquivo
- **JSON**: 1 arquivo (orient=records)

## Regenerar

```bash
python scripts/generate_sample_data.py
```
