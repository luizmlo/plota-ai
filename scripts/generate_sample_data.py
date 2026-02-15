"""
Generate synthetic sample datasets for Plota AI demos.
Creates 10 datasets across different contexts and formats.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rand_choice(seq, n=None):
    return [random.choice(seq) for _ in range(n or 1)]


# ─────────────────────────────────────────────────────────────────
# 1. Survey — Product satisfaction (ordinal, boolean, categorical)
# ─────────────────────────────────────────────────────────────────
def gen_product_survey() -> pd.DataFrame:
    n = 80
    return pd.DataFrame({
        "respondent_id": range(1, n + 1),
        "age_group": _rand_choice(["18-24", "25-34", "35-44", "45-54", "55+"], n),
        "gender": _rand_choice(["Masculino", "Feminino", "Outro", "Prefiro não informar"], n),
        "satisfaction": _rand_choice(["Muito insatisfeito", "Insatisfeito", "Neutro", "Satisfeito", "Muito satisfeito"], n),
        "would_recommend": _rand_choice(["Sim", "Não"], n),
        "nps_score": [random.randint(0, 10) for _ in range(n)],
    })


# ─────────────────────────────────────────────────────────────────
# 2. E-commerce — Sales (numeric, datetime, categorical, currency)
# ─────────────────────────────────────────────────────────────────
def gen_ecommerce_sales() -> pd.DataFrame:
    n = 60
    products = ["Camiseta", "Calça", "Tênis", "Mochila", "Relógio", "Óculos", "Boné", "Jaqueta"]
    return pd.DataFrame({
        "order_id": [f"ORD-{1000 + i}" for i in range(n)],
        "date": pd.date_range("2024-01-01", periods=n, freq="D").tolist(),
        "product": _rand_choice(products, n),
        "quantity": [random.randint(1, 5) for _ in range(n)],
        "unit_price_brl": [f"R$ {random.randint(50, 500):,}".replace(",", ".") for _ in range(n)],
        "region": _rand_choice(["Sudeste", "Sul", "Nordeste", "Norte", "Centro-Oeste"], n),
    })


# ─────────────────────────────────────────────────────────────────
# 3. HR — Employee data (boolean, categorical, date_string, numeric)
# ─────────────────────────────────────────────────────────────────
def gen_hr_employees() -> pd.DataFrame:
    n = 50
    return pd.DataFrame({
        "emp_id": range(1001, 1001 + n),
        "name": [f"Funcionário {i}" for i in range(1, n + 1)],
        "department": _rand_choice(["TI", "RH", "Vendas", "Marketing", "Financeiro", "Operações"], n),
        "hire_date": [f"{random.randint(2018, 2024)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}" for _ in range(n)],
        "salary": [random.randint(4000, 18000) for _ in range(n)],
        "remote": _rand_choice(["Sim", "Não"], n),
        "performance": _rand_choice(["Ruim", "Regular", "Bom", "Excelente"], n),
    })


# ─────────────────────────────────────────────────────────────────
# 4. Health — Patient visits (numeric, categorical, date_string)
# ─────────────────────────────────────────────────────────────────
def gen_health_visits() -> pd.DataFrame:
    n = 70
    return pd.DataFrame({
        "patient_id": [f"PAC-{i:04d}" for i in range(1, n + 1)],
        "visit_date": [f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}" for _ in range(n)],
        "specialty": _rand_choice(["Cardiologia", "Clínico Geral", "Dermatologia", "Ortopedia", "Pediatria", "Neurologia"], n),
        "age": [random.randint(5, 85) for _ in range(n)],
        "blood_pressure": [f"{random.randint(110, 150)}/{random.randint(70, 95)}" for _ in range(n)],
        "weight_kg": [round(random.uniform(45, 120), 1) for _ in range(n)],
    })


# ─────────────────────────────────────────────────────────────────
# 5. Community survey (fishing) — merged-header style, multi-value tags
# ─────────────────────────────────────────────────────────────────
def gen_fishing_survey() -> pd.DataFrame:
    n = 40
    return pd.DataFrame({
        "numero": range(1, n + 1),
        "nome": [f"Entrevistado {i}" for i in range(1, n + 1)],
        "naturalidade": _rand_choice(["Santos", "Guarujá", "São Sebastião", "Cubatão", "Praia Grande"], n),
        "escolaridade": _rand_choice(["Ensino fundamental", "Ensino médio", "Graduação", "Pós-graduação"], n),
        "tipo_pesca": _rand_choice(["Artesanal", "Industrial", "Subsistência", "Recreativa"], n),
        "idade": [random.randint(18, 70) for _ in range(n)],
        "sexo": _rand_choice(["M", "F"], n),
        "data_entrevista": [f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/2024" for _ in range(n)],
        "viu_lixo": _rand_choice(["Sim", "Não"], n),
        "pescou_lixo": _rand_choice(["Sim", "Não"], n),
        "tipos_lixo_pescado": [random.choice(["plástico", "rede", "vidro", "metal", "plástico, rede", "vidro, metal", ""]) for _ in range(n)],
        "teve_prejuizo": _rand_choice(["Sim", "Não"], n),
        "tipos_prejuizo": [random.choice(["perda de rede", "motor danificado", "perda de pescado", "perda de rede, motor danificado", ""]) for _ in range(n)],
    })


# ─────────────────────────────────────────────────────────────────
# 6. Social media engagement (numeric, categorical, datetime)
# ─────────────────────────────────────────────────────────────────
def gen_social_media() -> pd.DataFrame:
    n = 55
    return pd.DataFrame({
        "post_id": [f"post_{i}" for i in range(1, n + 1)],
        "date": pd.date_range("2024-06-01", periods=n, freq="D").tolist(),
        "platform": _rand_choice(["Instagram", "Facebook", "LinkedIn", "Twitter/X", "TikTok"], n),
        "content_type": _rand_choice(["Post", "Story", "Reel", "Live"], n),
        "likes": [random.randint(10, 5000) for _ in range(n)],
        "comments": [random.randint(0, 200) for _ in range(n)],
        "shares": [random.randint(0, 150) for _ in range(n)],
    })


# ─────────────────────────────────────────────────────────────────
# 7. Restaurant reviews (ordinal, free_text, categorical)
# ─────────────────────────────────────────────────────────────────
def gen_restaurant_reviews() -> pd.DataFrame:
    n = 45
    ratings = ["Péssimo", "Ruim", "Regular", "Bom", "Excelente"]
    comments = [
        "Atendimento excelente, comida fresca.",
        "Demorou um pouco, mas valeu a pena.",
        "Preço alto para a qualidade oferecida.",
        "Melhor pizza da cidade! Recomendo.",
        "Ambiente agradável, voltarei em breve.",
        "Cardápio limitado.",
        "Serviço rápido e preços justos.",
    ]
    return pd.DataFrame({
        "review_id": range(1, n + 1),
        "restaurant": _rand_choice(["Pizzaria Bella", "Sushi House", "Churrascaria do Zé", "Café Central", "Hamburgueria Artesanal"], n),
        "rating": _rand_choice(ratings, n),
        "comment": _rand_choice(comments, n),
        "date": [f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}" for _ in range(n)],
    })


# ─────────────────────────────────────────────────────────────────
# 8. Store inventory (numeric_string, categorical, numeric)
# ─────────────────────────────────────────────────────────────────
def gen_store_inventory() -> pd.DataFrame:
    n = 35
    return pd.DataFrame({
        "sku": [f"SKU-{1000 + i}" for i in range(n)],
        "product": _rand_choice(["Notebook", "Mouse", "Teclado", "Monitor", "Webcam", "Headset", "Pen drive"], n),
        "category": _rand_choice(["Eletrônicos", "Periféricos", "Acessórios"], n),
        "price": [f"R$ {random.randint(30, 8000):,}".replace(",", ".") for _ in range(n)],
        "stock": [random.randint(0, 250) for _ in range(n)],
        "reorder_level": [random.randint(5, 30) for _ in range(n)],
    })


# ─────────────────────────────────────────────────────────────────
# 9. Event attendance (boolean, multi_value_tags, date_string)
# ─────────────────────────────────────────────────────────────────
def gen_event_attendance() -> pd.DataFrame:
    n = 65
    interests = [
        "tecnologia",
        "negócios",
        "networking",
        "tecnologia, negócios",
        "negócios, networking",
        "tecnologia, networking, startups",
        "marketing",
        "design",
    ]
    return pd.DataFrame({
        "attendee_id": range(1, n + 1),
        "event_date": [f"{random.randint(1, 28):02d}/0{random.randint(1, 9)}/2024" for _ in range(n)],
        "confirmed": _rand_choice(["Sim", "Não"], n),
        "checked_in": _rand_choice(["Sim", "Não"], n),
        "interests": _rand_choice(interests, n),
        "ticket_type": _rand_choice(["VIP", "Regular", "Estudante"], n),
    })


# ─────────────────────────────────────────────────────────────────
# 10. Support tickets (categorical, ordinal, date_string, free_text)
# ─────────────────────────────────────────────────────────────────
def gen_support_tickets() -> pd.DataFrame:
    n = 50
    return pd.DataFrame({
        "ticket_id": [f"TKT-{2000 + i}" for i in range(n)],
        "category": _rand_choice(["Billing", "Technical", "Account", "Product", "Other"], n),
        "priority": _rand_choice(["Baixa", "Média", "Alta", "Crítica"], n),
        "status": _rand_choice(["Aberto", "Em andamento", "Resolvido", "Fechado"], n),
        "created_at": [f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d} {random.randint(8, 18):02d}:00" for _ in range(n)],
        "resolution_time_hours": [round(random.uniform(0.5, 72), 1) for _ in range(n)],
        "feedback": _rand_choice([
            "Problema resolvido rapidamente, muito satisfeito.",
            "Demorou mais do que esperava.",
            "Atendimento prestativo e eficiente.",
            "Ainda estou com dúvidas sobre a cobrança.",
        ], n),
    })


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

DATASETS = [
    ("01_pesquisa_satisfacao_produto.csv", gen_product_survey, "csv"),
    ("02_vendas_ecommerce.csv", gen_ecommerce_sales, "csv"),
    ("03_funcionarios_rh.xlsx", gen_hr_employees, "xlsx"),
    ("04_consultas_medicas.csv", gen_health_visits, "csv"),
    ("05_pesquisa_pesca_comunitaria.csv", gen_fishing_survey, "csv"),
    ("06_engajamento_redes_sociais.json", gen_social_media, "json"),
    ("07_avaliacoes_restaurantes.csv", gen_restaurant_reviews, "csv"),
    ("08_estoque_loja.csv", gen_store_inventory, "csv"),
    ("09_presenca_eventos.csv", gen_event_attendance, "csv"),
    ("10_tickets_suporte.csv", gen_support_tickets, "csv"),
]


def main() -> None:
    for filename, gen_fn, fmt in DATASETS:
        df = gen_fn()
        path = OUT_DIR / filename
        if fmt == "csv":
            df.to_csv(path, index=False, encoding="utf-8-sig")
        elif fmt == "xlsx":
            df.to_excel(path, index=False, engine="openpyxl")
        elif fmt == "json":
            df.to_json(path, orient="records", indent=2, force_ascii=False)
        print(f"Created {path} ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"\n{len(DATASETS)} datasets created in {OUT_DIR}")


if __name__ == "__main__":
    main()
