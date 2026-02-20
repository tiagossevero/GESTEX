"""
================================================================================
GESTEX V013 — Dashboard DIME Análise Anual (2020–2024)
================================================================================
Sistema de análise de contribuintes do setor têxtil que utilizaram
benefícios fiscais (TTD 47, 372, 409, 410, 411).
Projeto: GESTEX — Grupo Especializado no Setor Têxtil
Autor: SEF/SC — NIAT / Chapecó
Data: Fevereiro/2026
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ssl
import warnings
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import io

# Configurações SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="GESTEX V013 — DIME Análise Anual",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Esconder sidebar por padrão
st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIGURAÇÃO DO BANCO DE DADOS
# =============================================================================
IMPALA_HOST = 'bdaworkernode02.sef.sc.gov.br'
IMPALA_PORT = 21050
IMPALA_USER = st.secrets["impala_credentials"]["user"]
IMPALA_PASSWORD = st.secrets["impala_credentials"]["password"]

@st.cache_resource
def get_impala_engine():
    """Cria e retorna engine Impala."""
    try:
        engine = create_engine(
            f'impala://{IMPALA_HOST}:{IMPALA_PORT}',
            connect_args={
                'user': IMPALA_USER,
                'password': IMPALA_PASSWORD,
                'auth_mechanism': 'LDAP',
                'use_ssl': True
            }
        )
        return engine
    except Exception as e:
        st.error(f"❌ Erro ao criar engine Impala: {e}")
        return None

def executar_query(query: str) -> pd.DataFrame:
    """Executa query no Impala e retorna DataFrame."""
    engine = get_impala_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Erro na query: {e}")
        return pd.DataFrame()

# =============================================================================
# FUNÇÕES AUXILIARES DE FORMATAÇÃO (PADRÃO BR)
# =============================================================================
def formatar_moeda(valor, abreviado=False) -> str:
    """Formata valor para moeda brasileira."""
    if pd.isna(valor) or valor is None:
        return "R$ 0"
    try:
        valor = float(valor)
        if abreviado:
            if abs(valor) >= 1_000_000_000:
                return f"R$ {valor/1_000_000_000:,.2f} Bi".replace(",", "X").replace(".", ",").replace("X", ".")
            elif abs(valor) >= 1_000_000:
                return f"R$ {valor/1_000_000:,.2f} Mi".replace(",", "X").replace(".", ",").replace("X", ".")
            elif abs(valor) >= 1_000:
                return f"R$ {valor/1_000:,.1f} mil".replace(",", "X").replace(".", ",").replace("X", ".")
            else:
                return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "R$ 0"

def fmt(valor):
    """Atalho para moeda abreviada."""
    return formatar_moeda(valor, abreviado=True)

def formatar_numero(valor, abreviado=False) -> str:
    """Formata número com separador de milhar."""
    if pd.isna(valor) or valor is None:
        return "0"
    try:
        valor = float(valor)
        if abreviado:
            if valor >= 1_000_000:
                return f"{valor/1_000_000:,.1f} Mi".replace(",", "X").replace(".", ",").replace("X", ".")
            elif valor >= 1_000:
                return f"{valor/1_000:,.1f} mil".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{int(valor):,}".replace(",", ".")
    except:
        return "0"

def formatar_pct(valor) -> str:
    """Formata percentual."""
    if pd.isna(valor) or valor is None:
        return "0,0%"
    try:
        return f"{float(valor):,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "0,0%"

def formatar_cnpj(cnpj: str) -> str:
    """Formata CNPJ: XX.XXX.XXX/XXXX-XX."""
    if not cnpj or len(str(cnpj).strip()) < 14:
        return str(cnpj) if cnpj else ""
    c = str(cnpj).strip().zfill(14)
    return f"{c[:2]}.{c[2:5]}.{c[5:8]}/{c[8:12]}-{c[12:14]}"

# =============================================================================
# ESTILOS CSS
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { font-family: 'Source Sans Pro', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 50%, #3182ce 100%);
        padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(26, 54, 93, 0.3);
    }
    .main-header h1 { color: white; font-size: 2rem; font-weight: 700; margin: 0; }
    .main-header p  { color: rgba(255,255,255,0.85); font-size: 1rem; margin: 0.5rem 0 0 0; }

    .section-header {
        background: #f7fafc; padding: 0.8rem 1.2rem; border-radius: 10px;
        margin: 1.2rem 0 0.8rem 0; border-left: 4px solid #2b6cb0;
    }
    .section-header h2 {
        color: #1a365d; font-size: 1.15rem; font-weight: 600; margin: 0;
    }

    .metric-card {
        background: white; border-radius: 12px; padding: 1.1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); border-left: 4px solid #2b6cb0;
        margin-bottom: 0.8rem; text-align: center;
    }
    .metric-card.green  { border-left-color: #38a169; }
    .metric-card.orange { border-left-color: #dd6b20; }
    .metric-card.purple { border-left-color: #805ad5; }
    .metric-card.red    { border-left-color: #e53e3e; }
    .metric-label {
        font-size: 0.75rem; color: #718096; text-transform: uppercase; font-weight: 600;
    }
    .metric-value {
        font-size: 1.4rem; font-weight: 700; color: #1a365d; margin-top: 0.2rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .cadastro-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #e2e8f0;
        margin-bottom: 0.75rem;
    }
    .cadastro-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-color: #2b6cb0; }

    .flag-ok     { background: #c6f6d5; color: #22543d; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; font-size: 0.8rem; display: inline-block; }
    .flag-warn   { background: #fefcbf; color: #744210; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; font-size: 0.8rem; display: inline-block; }
    .flag-danger { background: #fed7d7; color: #822727; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; font-size: 0.8rem; display: inline-block; }
    .flag-info   { background: #bee3f8; color: #2a4365; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; font-size: 0.8rem; display: inline-block; }

    .info-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
    .info-item label { font-size: 0.7rem; color: #718096; text-transform: uppercase; display: block; }
    .info-item span  { font-size: 0.95rem; color: #1a365d; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER PRINCIPAL
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>🏭 GESTEX V013 — DIME Análise Anual</h1>
    <p>Setor Têxtil | Benefícios Fiscais TTD 47 / 372 / 409 / 410 / 411 | 2020–2024</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# =============================================================================
@st.cache_data(ttl=3600)
def carregar_cadastro():
    """Carrega cadastro completo para busca."""
    return executar_query("""
        SELECT DISTINCT ie, cnpj, cnpj_raiz, razao_social, nome_fantasia,
               cnae, cnae_desc, regime_tributacao, enquadramento,
               nome_ges, municipio, ind_simples_nacional
        FROM teste.gestex_v013_cadastro
        ORDER BY razao_social
    """)

@st.cache_data(ttl=3600)
def buscar_contribuinte(termo: str) -> pd.DataFrame:
    """Busca contribuinte por CNPJ ou IE."""
    termo_limpo = termo.strip().replace(".", "").replace("/", "").replace("-", "")
    return executar_query(f"""
        SELECT DISTINCT ie, cnpj, cnpj_raiz, razao_social, nome_fantasia,
               cnae, cnae_desc, regime_tributacao, enquadramento,
               nome_ges, municipio, ind_simples_nacional
        FROM teste.gestex_v013_cadastro
        WHERE REPLACE(REPLACE(REPLACE(cnpj, '.', ''), '/', ''), '-', '') LIKE '%{termo_limpo}%'
           OR REPLACE(ie, '.', '') LIKE '%{termo_limpo}%'
           OR UPPER(razao_social) LIKE '%{termo_limpo.upper()}%'
        LIMIT 20
    """)

@st.cache_data(ttl=3600)
def carregar_dime_anual(cnpj_raiz: str) -> pd.DataFrame:
    return executar_query(f"""
        SELECT * FROM teste.gestex_v013_dime_anual
        WHERE cnpj_raiz = '{cnpj_raiz}' ORDER BY ano
    """)

@st.cache_data(ttl=3600)
def carregar_dime_mensal(cnpj_raiz: str) -> pd.DataFrame:
    return executar_query(f"""
        SELECT * FROM teste.gestex_v013_dime_dados_gerais
        WHERE cnpj_raiz = '{cnpj_raiz}' ORDER BY ano, mes
    """)

@st.cache_data(ttl=3600)
def carregar_efd_e110(cnpj_raiz: str) -> pd.DataFrame:
    return executar_query(f"""
        SELECT * FROM teste.gestex_v013_efd_e110
        WHERE cnpj_raiz = '{cnpj_raiz}' ORDER BY ano, mes
    """)

@st.cache_data(ttl=3600)
def carregar_dcip(cnpj_raiz: str) -> pd.DataFrame:
    return executar_query(f"""
        SELECT * FROM teste.gestex_v013_dcip
        WHERE cnpj_raiz = '{cnpj_raiz}' ORDER BY ano, mes
    """)

@st.cache_data(ttl=3600)
def carregar_dcip_anual(cnpj_raiz: str) -> pd.DataFrame:
    return executar_query(f"""
        SELECT * FROM teste.gestex_v013_dcip_anual
        WHERE cnpj_raiz = '{cnpj_raiz}' ORDER BY ano
    """)

@st.cache_data(ttl=3600)
def carregar_base_analitica(cnpj_raiz: str) -> pd.DataFrame:
    return executar_query(f"""
        SELECT * FROM teste.gestex_v013_base_analitica
        WHERE cnpj_raiz = '{cnpj_raiz}' ORDER BY ano
    """)

@st.cache_data(ttl=3600)
def carregar_analise(num: int, cnpj_raiz: str) -> pd.DataFrame:
    tabelas = {
        1: "teste.gestex_v013_analise1_icms_baixo",
        2: "teste.gestex_v013_analise2_entradas_comerciais",
        3: "teste.gestex_v013_analise3_cp_sem_producao",
        4: "teste.gestex_v013_analise4_cp_cumulativo",
        5: "teste.gestex_v013_analise5_q9070_q14031",
    }
    return executar_query(f"SELECT * FROM {tabelas[num]} WHERE cnpj_raiz = '{cnpj_raiz}'")

@st.cache_data(ttl=3600)
def carregar_cfop_sumarizada(cnpj_raiz: str) -> pd.DataFrame:
    return executar_query(f"""
        SELECT * FROM teste.gestex_v013_cfop_sumarizada
        WHERE cnpj_raiz = '{cnpj_raiz}' ORDER BY ano
    """)

# --- Dados agregados para visão setorial ---
@st.cache_data(ttl=3600)
def carregar_top_faturamento(ano: int, limite: int = 10) -> pd.DataFrame:
    return executar_query(f"""
        SELECT da.cnpj_raiz, c.razao_social, c.cnae_desc, c.nome_ges, c.municipio,
               da.faturamento_total, da.icms_recolher_total, da.cred_dcip_total
        FROM teste.gestex_v013_dime_anual da
        INNER JOIN (
            SELECT DISTINCT cnpj_raiz, razao_social, cnae_desc, nome_ges, municipio
            FROM teste.gestex_v013_cadastro
        ) c ON da.cnpj_raiz = c.cnpj_raiz
        WHERE da.ano = {ano}
        ORDER BY da.faturamento_total DESC
        LIMIT {limite}
    """)

@st.cache_data(ttl=3600)
def carregar_top_cp(ano: int, limite: int = 10) -> pd.DataFrame:
    return executar_query(f"""
        SELECT b.cnpj_raiz, b.razao_social, b.cnae_desc, b.tipo_cp,
               b.faturamento_total, b.vl_cp_total, b.icms_recolher_total
        FROM teste.gestex_v013_base_analitica b
        WHERE b.ano = {ano} AND b.vl_cp_total > 0
        ORDER BY b.vl_cp_total DESC
        LIMIT {limite}
    """)

@st.cache_data(ttl=3600)
def carregar_distribuicao_cnae() -> pd.DataFrame:
    return executar_query("""
        SELECT cnae, cnae_desc, COUNT(*) AS qtd
        FROM teste.gestex_v013_cadastro
        GROUP BY cnae, cnae_desc
        ORDER BY qtd DESC
        LIMIT 15
    """)

@st.cache_data(ttl=3600)
def carregar_distribuicao_ges() -> pd.DataFrame:
    return executar_query("""
        SELECT nome_ges, COUNT(*) AS qtd
        FROM teste.gestex_v013_cadastro
        GROUP BY nome_ges
        ORDER BY qtd DESC
    """)

@st.cache_data(ttl=3600)
def carregar_distribuicao_regime() -> pd.DataFrame:
    return executar_query("""
        SELECT regime_tributacao, COUNT(*) AS qtd
        FROM teste.gestex_v013_cadastro
        GROUP BY regime_tributacao
        ORDER BY qtd DESC
    """)

@st.cache_data(ttl=3600)
def carregar_resumo_flags() -> dict:
    """Conta registros em cada tabela de análise."""
    resultado = {}
    nomes = {
        1: ("ICMS ≤ 2% BC", "teste.gestex_v013_analise1_icms_baixo"),
        2: ("Entradas Comerciais", "teste.gestex_v013_analise2_entradas_comerciais"),
        3: ("CP sem Produção", "teste.gestex_v013_analise3_cp_sem_producao"),
        4: ("CP Cumulativo", "teste.gestex_v013_analise4_cp_cumulativo"),
        5: ("Q9070 / Q14031", "teste.gestex_v013_analise5_q9070_q14031"),
    }
    for num, (nome, tabela) in nomes.items():
        df = executar_query(f"SELECT COUNT(*) AS total FROM {tabela}")
        resultado[num] = {
            "nome": nome,
            "total": int(df.iloc[0]['total']) if not df.empty else 0
        }
    return resultado

# =============================================================================
# BARRA DE PESQUISA
# =============================================================================
st.markdown('<div class="section-header"><h2>🔍 Pesquisar Contribuinte</h2></div>', unsafe_allow_html=True)

col_busca1, col_busca2 = st.columns([3, 1])
with col_busca1:
    termo_busca = st.text_input(
        "Digite CNPJ, IE ou Razão Social:",
        placeholder="Ex: 12345678 ou 123456789 ou TEXTIL ABC...",
        key="busca_principal"
    )
with col_busca2:
    st.markdown("<br>", unsafe_allow_html=True)
    btn_buscar = st.button("🔍 Buscar", use_container_width=True)

# Estado da seleção
if "cnpj_raiz_selecionado" not in st.session_state:
    st.session_state.cnpj_raiz_selecionado = None
if "dados_cadastro_selecionado" not in st.session_state:
    st.session_state.dados_cadastro_selecionado = None

# Processar busca
if btn_buscar and termo_busca and len(termo_busca.strip()) >= 3:
    with st.spinner("Buscando..."):
        df_resultados = buscar_contribuinte(termo_busca)

    if df_resultados.empty:
        st.warning("Nenhum contribuinte encontrado.")
    elif len(df_resultados) == 1:
        # Seleção automática se único resultado
        row = df_resultados.iloc[0]
        st.session_state.cnpj_raiz_selecionado = row['cnpj_raiz']
        st.session_state.dados_cadastro_selecionado = row.to_dict()
        st.rerun()
    else:
        st.info(f"📋 {len(df_resultados)} resultado(s) encontrado(s). Selecione abaixo:")
        for idx, row in df_resultados.iterrows():
            label = f"{row['razao_social']} | CNPJ: {formatar_cnpj(row['cnpj'])} | IE: {row['ie']} | {row['municipio']}"
            if st.button(label, key=f"sel_{idx}"):
                st.session_state.cnpj_raiz_selecionado = row['cnpj_raiz']
                st.session_state.dados_cadastro_selecionado = row.to_dict()
                st.rerun()

# =============================================================================
# ABAS PRINCIPAIS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📋 Visão Geral",
    "📊 DIME Mensal",
    "📈 Apuração EFD",
    "💰 Crédito Presumido",
    "🚩 Análises (Flags)",
    "🏭 Visão Setorial",
    "🔬 Análise Exploratória",
    "🤖 Scoring ML"
])

cnpj_raiz = st.session_state.cnpj_raiz_selecionado
cad = st.session_state.dados_cadastro_selecionado

# =============================================================================
# ABA 1 — VISÃO GERAL DO CONTRIBUINTE
# =============================================================================
with tab1:
    if cnpj_raiz is None:
        st.info("🔍 Pesquise um contribuinte acima para visualizar os dados.")
    else:
        st.markdown('<div class="section-header"><h2>📋 Dados Cadastrais</h2></div>', unsafe_allow_html=True)

        # Card cadastral
        st.markdown(f"""
        <div class="cadastro-card">
            <div style="font-size: 1.3rem; font-weight: 700; color: #1a365d; margin-bottom: 0.5rem;">
                {cad.get('razao_social', '')}
            </div>
            <div class="info-row">
                <div class="info-item"><label>CNPJ</label><span>{formatar_cnpj(cad.get('cnpj', ''))}</span></div>
                <div class="info-item"><label>CNPJ Raiz</label><span>{cnpj_raiz}</span></div>
                <div class="info-item"><label>IE</label><span>{cad.get('ie', '')}</span></div>
                <div class="info-item"><label>CNAE</label><span>{cad.get('cnae', '')} — {cad.get('cnae_desc', '')}</span></div>
            </div>
            <div class="info-row">
                <div class="info-item"><label>Regime</label><span>{cad.get('regime_tributacao', '')}</span></div>
                <div class="info-item"><label>Enquadramento</label><span>{cad.get('enquadramento', '')}</span></div>
                <div class="info-item"><label>GES</label><span>{cad.get('nome_ges', '')}</span></div>
                <div class="info-item"><label>Município</label><span>{cad.get('municipio', '')}</span></div>
                <div class="info-item"><label>Simples Nacional</label><span>{'Sim' if cad.get('ind_simples_nacional') == 'S' else 'Não'}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Carregar dados anuais
        df_anual = carregar_dime_anual(cnpj_raiz)

        if not df_anual.empty:
            # KPIs de 5 anos
            fat_total = df_anual['faturamento_total'].sum()
            icms_total = df_anual['icms_recolher_total'].sum()
            cred_total = df_anual['creditos_total'].sum()
            dcip_total = df_anual['cred_dcip_total'].sum()
            estab_max = df_anual['qtd_estabelecimentos'].max()

            st.markdown('<div class="section-header"><h2>📊 Indicadores Consolidados (2020–2024)</h2></div>', unsafe_allow_html=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Faturamento Total</div><div class="metric-value">{fmt(fat_total)}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card green"><div class="metric-label">ICMS Recolhido</div><div class="metric-value">{fmt(icms_total)}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card orange"><div class="metric-label">Créditos Totais</div><div class="metric-value">{fmt(cred_total)}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card purple"><div class="metric-label">Crédito DCIP</div><div class="metric-value">{fmt(dcip_total)}</div></div>', unsafe_allow_html=True)
            with c5:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Estabelecimentos</div><div class="metric-value">{int(estab_max)}</div></div>', unsafe_allow_html=True)

            # Gráfico evolução anual
            st.markdown('<div class="section-header"><h2>📈 Evolução Anual</h2></div>', unsafe_allow_html=True)

            fig_anual = make_subplots(specs=[[{"secondary_y": True}]])
            fig_anual.add_trace(
                go.Bar(x=df_anual['ano'], y=df_anual['faturamento_total'],
                       name="Faturamento", marker_color='#2b6cb0',
                       text=[fmt(v) for v in df_anual['faturamento_total']],
                       textposition='outside'),
                secondary_y=False
            )
            fig_anual.add_trace(
                go.Scatter(x=df_anual['ano'], y=df_anual['icms_recolher_total'],
                           name="ICMS a Recolher", mode='lines+markers',
                           line=dict(color='#38a169', width=3),
                           marker=dict(size=8)),
                secondary_y=True
            )
            fig_anual.add_trace(
                go.Scatter(x=df_anual['ano'], y=df_anual['cred_dcip_total'],
                           name="Crédito DCIP", mode='lines+markers',
                           line=dict(color='#dd6b20', width=3, dash='dot'),
                           marker=dict(size=8)),
                secondary_y=True
            )
            fig_anual.update_layout(
                title="Faturamento × ICMS a Recolher × Crédito DCIP",
                height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis=dict(dtick=1)
            )
            fig_anual.update_yaxes(title_text="Faturamento (R$)", secondary_y=False, tickformat=",.0f")
            fig_anual.update_yaxes(title_text="ICMS / DCIP (R$)", secondary_y=True, tickformat=",.0f")
            st.plotly_chart(fig_anual, use_container_width=True)

            # Tabela detalhada
            with st.expander("📋 Ver tabela detalhada — DIME Anual"):
                df_display = df_anual.copy()
                colunas_moeda = ['faturamento_total', 'receita_bruta_total', 'bc_saidas_total',
                                 'bc_entradas_total', 'debitos_total', 'creditos_total',
                                 'icms_recolher_total', 'cred_dcip_total', 'saidas_internas',
                                 'saidas_interestaduais', 'exportacao', 'entradas_internas',
                                 'entradas_interestaduais', 'importacao']
                for col in colunas_moeda:
                    if col in df_display.columns:
                        df_display[col] = df_display[col].apply(lambda v: formatar_moeda(v))
                st.dataframe(df_display, use_container_width=True)
                csv = df_anual.to_csv(index=False, sep=';', decimal=',')
                st.download_button("⬇️ Exportar CSV", csv, f"gestex_v013_dime_anual_{cnpj_raiz}.csv", "text/csv")
        else:
            st.warning("⚠️ Nenhum dado DIME Anual encontrado para este CNPJ Raiz.")

# =============================================================================
# ABA 2 — DIME MENSAL
# =============================================================================
with tab2:
    if cnpj_raiz is None:
        st.info("🔍 Pesquise um contribuinte acima para visualizar os dados.")
    else:
        st.markdown('<div class="section-header"><h2>📊 DIME — Série Temporal Mensal</h2></div>', unsafe_allow_html=True)

        df_mensal = carregar_dime_mensal(cnpj_raiz)

        if not df_mensal.empty:
            # Filtro por ano
            anos_disp = sorted(df_mensal['ano'].unique().tolist())
            ano_sel = st.multiselect("Filtrar por ano:", anos_disp, default=anos_disp, key="filtro_ano_dime")
            df_filtrado = df_mensal[df_mensal['ano'].isin(ano_sel)] if ano_sel else df_mensal

            # Criar coluna período
            df_filtrado = df_filtrado.copy()
            df_filtrado['periodo'] = df_filtrado['ano'].astype(str) + '-' + df_filtrado['mes'].astype(str).str.zfill(2)

            # Gráfico 1: Faturamento mensal
            fig_fat = px.bar(
                df_filtrado.groupby('periodo', as_index=False).agg({'vl_faturamento': 'sum'}),
                x='periodo', y='vl_faturamento',
                title="Faturamento Mensal",
                color_discrete_sequence=['#2b6cb0']
            )
            fig_fat.update_layout(height=350, xaxis_title="Período", yaxis_title="R$", yaxis_tickformat=",.0f")
            st.plotly_chart(fig_fat, use_container_width=True)

            # Gráfico 2: Débitos × Créditos × ICMS a Recolher
            df_grupo = df_filtrado.groupby('periodo', as_index=False).agg({
                'vl_tot_debitos': 'sum', 'vl_tot_creditos': 'sum', 'vl_deb_recolher': 'sum'
            })

            fig_dcr = go.Figure()
            fig_dcr.add_trace(go.Scatter(x=df_grupo['periodo'], y=df_grupo['vl_tot_debitos'],
                                         name="Débitos", mode='lines', line=dict(color='#e53e3e', width=2)))
            fig_dcr.add_trace(go.Scatter(x=df_grupo['periodo'], y=df_grupo['vl_tot_creditos'],
                                         name="Créditos", mode='lines', line=dict(color='#38a169', width=2)))
            fig_dcr.add_trace(go.Bar(x=df_grupo['periodo'], y=df_grupo['vl_deb_recolher'],
                                      name="ICMS a Recolher", marker_color='rgba(43,108,176,0.4)'))
            fig_dcr.update_layout(
                title="Débitos × Créditos × ICMS a Recolher",
                height=380, xaxis_title="Período", yaxis_title="R$",
                yaxis_tickformat=",.0f",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_dcr, use_container_width=True)

            # Gráfico 3: Entradas × Saídas por destino/origem
            col1, col2 = st.columns(2)
            with col1:
                df_saidas = df_filtrado.groupby('periodo', as_index=False).agg({
                    'vl_saidas_internas': 'sum', 'vl_saidas_interestaduais': 'sum', 'vl_exportacao': 'sum'
                })
                fig_saidas = go.Figure()
                fig_saidas.add_trace(go.Bar(x=df_saidas['periodo'], y=df_saidas['vl_saidas_internas'], name="Internas", marker_color='#2b6cb0'))
                fig_saidas.add_trace(go.Bar(x=df_saidas['periodo'], y=df_saidas['vl_saidas_interestaduais'], name="Interestaduais", marker_color='#805ad5'))
                fig_saidas.add_trace(go.Bar(x=df_saidas['periodo'], y=df_saidas['vl_exportacao'], name="Exportação", marker_color='#dd6b20'))
                fig_saidas.update_layout(barmode='stack', title="Saídas por Destino", height=350, yaxis_tickformat=",.0f")
                st.plotly_chart(fig_saidas, use_container_width=True)

            with col2:
                df_entradas = df_filtrado.groupby('periodo', as_index=False).agg({
                    'vl_ent_internas': 'sum', 'vl_ent_interestaduais': 'sum', 'vl_importacao': 'sum'
                })
                fig_entradas = go.Figure()
                fig_entradas.add_trace(go.Bar(x=df_entradas['periodo'], y=df_entradas['vl_ent_internas'], name="Internas", marker_color='#38a169'))
                fig_entradas.add_trace(go.Bar(x=df_entradas['periodo'], y=df_entradas['vl_ent_interestaduais'], name="Interestaduais", marker_color='#3182ce'))
                fig_entradas.add_trace(go.Bar(x=df_entradas['periodo'], y=df_entradas['vl_importacao'], name="Importação", marker_color='#e53e3e'))
                fig_entradas.update_layout(barmode='stack', title="Entradas por Origem", height=350, yaxis_tickformat=",.0f")
                st.plotly_chart(fig_entradas, use_container_width=True)

            # Tabela detalhada
            with st.expander("📋 Ver tabela detalhada — DIME Mensal"):
                st.dataframe(df_filtrado, use_container_width=True)
                csv = df_filtrado.to_csv(index=False, sep=';', decimal=',')
                st.download_button("⬇️ Exportar CSV", csv, f"gestex_v013_dime_mensal_{cnpj_raiz}.csv", "text/csv")
        else:
            st.warning("⚠️ Nenhum dado DIME Mensal encontrado.")

# =============================================================================
# ABA 3 — APURAÇÃO EFD (E110)
# =============================================================================
with tab3:
    if cnpj_raiz is None:
        st.info("🔍 Pesquise um contribuinte acima para visualizar os dados.")
    else:
        st.markdown('<div class="section-header"><h2>📈 Apuração ICMS — EFD E110</h2></div>', unsafe_allow_html=True)

        df_efd = carregar_efd_e110(cnpj_raiz)

        if not df_efd.empty:
            df_efd = df_efd.copy()
            df_efd['periodo'] = df_efd['ano'].astype(str) + '-' + df_efd['mes'].astype(str).str.zfill(2)

            # Filtro por ano
            anos_efd = sorted(df_efd['ano'].unique().tolist())
            ano_sel_efd = st.multiselect("Filtrar por ano:", anos_efd, default=anos_efd, key="filtro_ano_efd")
            df_efd_f = df_efd[df_efd['ano'].isin(ano_sel_efd)] if ano_sel_efd else df_efd

            # Gráfico: Débitos × Créditos × Saldo
            df_efd_g = df_efd_f.groupby('periodo', as_index=False).agg({
                'vl_tot_debitos': 'sum', 'vl_tot_creditos': 'sum',
                'vl_icms_recolher': 'sum', 'vl_sld_credor_transportar': 'sum',
                'vl_tot_aj_creditos': 'sum', 'vl_tot_aj_debitos': 'sum'
            })

            fig_efd = go.Figure()
            fig_efd.add_trace(go.Scatter(x=df_efd_g['periodo'], y=df_efd_g['vl_tot_debitos'],
                                          name="Débitos", line=dict(color='#e53e3e', width=2)))
            fig_efd.add_trace(go.Scatter(x=df_efd_g['periodo'], y=df_efd_g['vl_tot_creditos'],
                                          name="Créditos", line=dict(color='#38a169', width=2)))
            fig_efd.add_trace(go.Bar(x=df_efd_g['periodo'], y=df_efd_g['vl_icms_recolher'],
                                      name="ICMS a Recolher", marker_color='rgba(43,108,176,0.5)'))
            fig_efd.add_trace(go.Scatter(x=df_efd_g['periodo'], y=df_efd_g['vl_sld_credor_transportar'],
                                          name="Saldo Credor", line=dict(color='#dd6b20', width=2, dash='dash')))
            fig_efd.update_layout(
                title="EFD E110 — Apuração ICMS Mensal",
                height=420, xaxis_title="Período", yaxis_title="R$", yaxis_tickformat=",.0f",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_efd, use_container_width=True)

            # Gráfico: Ajustes
            col1, col2 = st.columns(2)
            with col1:
                fig_aj = go.Figure()
                fig_aj.add_trace(go.Bar(x=df_efd_g['periodo'], y=df_efd_g['vl_tot_aj_debitos'],
                                         name="Aj. Débitos", marker_color='#e53e3e'))
                fig_aj.add_trace(go.Bar(x=df_efd_g['periodo'], y=df_efd_g['vl_tot_aj_creditos'],
                                         name="Aj. Créditos", marker_color='#38a169'))
                fig_aj.update_layout(title="Ajustes de Apuração", height=350, barmode='group', yaxis_tickformat=",.0f")
                st.plotly_chart(fig_aj, use_container_width=True)

            with col2:
                # Comparativo DIME × EFD (anual)
                df_dime_a = carregar_dime_anual(cnpj_raiz)
                df_efd_a = df_efd.groupby('ano', as_index=False).agg({
                    'vl_tot_debitos': 'sum', 'vl_tot_creditos': 'sum', 'vl_icms_recolher': 'sum'
                }).rename(columns={
                    'vl_tot_debitos': 'efd_debitos', 'vl_tot_creditos': 'efd_creditos',
                    'vl_icms_recolher': 'efd_icms'
                })

                if not df_dime_a.empty and not df_efd_a.empty:
                    df_comp = pd.merge(
                        df_dime_a[['ano', 'debitos_total', 'creditos_total', 'icms_recolher_total']],
                        df_efd_a, on='ano', how='outer'
                    )
                    df_comp['dif_debitos'] = df_comp['debitos_total'] - df_comp['efd_debitos']
                    df_comp['dif_creditos'] = df_comp['creditos_total'] - df_comp['efd_creditos']
                    df_comp['dif_icms'] = df_comp['icms_recolher_total'] - df_comp['efd_icms']

                    st.markdown("#### DIME × EFD — Diferenças Anuais")
                    st.dataframe(df_comp, use_container_width=True)

            # Tabela EFD detalhada
            with st.expander("📋 Ver tabela detalhada — EFD E110"):
                st.dataframe(df_efd_f, use_container_width=True)
                csv = df_efd_f.to_csv(index=False, sep=';', decimal=',')
                st.download_button("⬇️ Exportar CSV", csv, f"gestex_v013_efd_e110_{cnpj_raiz}.csv", "text/csv")
        else:
            st.warning("⚠️ Nenhum dado EFD E110 encontrado.")

# =============================================================================
# ABA 4 — CRÉDITO PRESUMIDO (DCIP)
# =============================================================================
with tab4:
    if cnpj_raiz is None:
        st.info("🔍 Pesquise um contribuinte acima para visualizar os dados.")
    else:
        st.markdown('<div class="section-header"><h2>💰 Crédito Presumido — DCIP</h2></div>', unsafe_allow_html=True)

        df_dcip = carregar_dcip(cnpj_raiz)
        df_dcip_a = carregar_dcip_anual(cnpj_raiz)
        df_base = carregar_base_analitica(cnpj_raiz)

        if not df_dcip_a.empty:
            # KPIs
            cp_total = df_dcip_a['vl_cp_total'].sum()
            tipos_usados = df_dcip_a['tipo_beneficio'].unique().tolist()
            teve_ext = df_dcip_a['teve_extempraneo'].max() if 'teve_extempraneo' in df_dcip_a.columns else 0

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card purple"><div class="metric-label">CP Total Utilizado</div><div class="metric-value">{fmt(cp_total)}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Tipos de TTD</div><div class="metric-value">{", ".join(tipos_usados)}</div></div>', unsafe_allow_html=True)
            with c3:
                # % CP sobre faturamento (se base_analitica disponível)
                if not df_base.empty:
                    fat_b = df_base['faturamento_total'].sum()
                    pct_cp = (cp_total / fat_b * 100) if fat_b > 0 else 0
                    st.markdown(f'<div class="metric-card orange"><div class="metric-label">CP / Faturamento</div><div class="metric-value">{formatar_pct(pct_cp)}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card orange"><div class="metric-label">CP / Faturamento</div><div class="metric-value">N/D</div></div>', unsafe_allow_html=True)
            with c4:
                cor_ext = "red" if teve_ext else "green"
                txt_ext = "SIM ⚠️" if teve_ext else "NÃO ✅"
                st.markdown(f'<div class="metric-card {cor_ext}"><div class="metric-label">Extemporâneo</div><div class="metric-value">{txt_ext}</div></div>', unsafe_allow_html=True)

            # Gráfico: CP por ano e tipo
            fig_cp = px.bar(
                df_dcip_a, x='ano', y='vl_cp_total', color='tipo_beneficio',
                barmode='group', title="Crédito Presumido por Ano e Tipo de TTD",
                color_discrete_sequence=['#2b6cb0', '#38a169', '#dd6b20', '#805ad5', '#e53e3e'],
                text_auto=True
            )
            fig_cp.update_layout(height=400, yaxis_tickformat=",.0f", xaxis=dict(dtick=1))
            st.plotly_chart(fig_cp, use_container_width=True)

            # Série mensal do DCIP
            if not df_dcip.empty:
                df_dcip_m = df_dcip.copy()
                df_dcip_m['periodo'] = df_dcip_m['ano'].astype(str) + '-' + df_dcip_m['mes'].astype(str).str.zfill(2)
                df_dcip_mg = df_dcip_m.groupby(['periodo', 'tipo_beneficio'], as_index=False)['vl_credito_presumido'].sum()

                fig_cp_m = px.line(
                    df_dcip_mg, x='periodo', y='vl_credito_presumido',
                    color='tipo_beneficio', title="Evolução Mensal do Crédito Presumido",
                    markers=True
                )
                fig_cp_m.update_layout(height=380, yaxis_tickformat=",.0f")
                st.plotly_chart(fig_cp_m, use_container_width=True)

            # Tabelas
            with st.expander("📋 DCIP Anual — Detalhamento"):
                st.dataframe(df_dcip_a, use_container_width=True)

            with st.expander("📋 DCIP Mensal — Detalhamento"):
                if not df_dcip.empty:
                    st.dataframe(df_dcip, use_container_width=True)
                    csv = df_dcip.to_csv(index=False, sep=';', decimal=',')
                    st.download_button("⬇️ Exportar CSV", csv, f"gestex_v013_dcip_{cnpj_raiz}.csv", "text/csv")
        else:
            st.info("ℹ️ Nenhum crédito presumido (DCIP) encontrado para este contribuinte.")

# =============================================================================
# ABA 5 — ANÁLISES AUTOMÁTICAS (FLAGS)
# =============================================================================
with tab5:
    if cnpj_raiz is None:
        st.info("🔍 Pesquise um contribuinte acima para visualizar os dados.")
    else:
        st.markdown('<div class="section-header"><h2>🚩 Análises Automáticas — Flags de Risco</h2></div>', unsafe_allow_html=True)

        analises_meta = {
            1: {"nome": "ICMS ≤ 2% da BC", "desc": "Contribuinte com carga tributária muito baixa (ICMS recolhido ≤ 2% da base de cálculo de saídas)", "icone": "📉"},
            2: {"nome": "Entradas Comerciais", "desc": "Possível uso indevido do CP em produtos revendidos (entradas comerciais elevadas)", "icone": "🔄"},
            3: {"nome": "CP sem Produção Própria", "desc": "Crédito presumido utilizado sem faturamento com produção própria", "icone": "🏭"},
            4: {"nome": "CP Cumulativo", "desc": "Uso de CP (TTD47/372) concomitante com Simples Nacional ou crédito extemporâneo", "icone": "⚠️"},
            5: {"nome": "Q9070 / Q14031", "desc": "Verificação: Q9070 + 3% × BC Devoluções = Q14031 (exclusiva TTD 47)", "icone": "🔍"},
        }

        total_flags = 0

        for num, meta in analises_meta.items():
            df_flag = carregar_analise(num, cnpj_raiz)
            encontrado = not df_flag.empty and len(df_flag) > 0

            if encontrado:
                total_flags += 1
                badge = '<span class="flag-danger">🔴 ENCONTRADO</span>'
            else:
                badge = '<span class="flag-ok">🟢 OK</span>'

            with st.expander(f"{meta['icone']} Análise {num}: {meta['nome']} — {'🔴 ENCONTRADO' if encontrado else '🟢 OK'}"):
                st.markdown(f"**Descrição:** {meta['desc']}")
                st.markdown(f"**Status:** {badge}", unsafe_allow_html=True)

                if encontrado:
                    st.dataframe(df_flag, use_container_width=True)
                else:
                    st.success("✅ Nenhuma ocorrência encontrada para este contribuinte.")

        # Resumo visual
        st.markdown("---")
        if total_flags == 0:
            st.markdown('<div class="flag-ok" style="font-size: 1rem; padding: 0.8rem 1.5rem;">✅ Nenhuma flag de risco identificada para este contribuinte.</div>', unsafe_allow_html=True)
        elif total_flags <= 2:
            st.markdown(f'<div class="flag-warn" style="font-size: 1rem; padding: 0.8rem 1.5rem;">⚠️ {total_flags} flag(s) de risco identificada(s). Recomenda-se análise detalhada.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="flag-danger" style="font-size: 1rem; padding: 0.8rem 1.5rem;">🔴 {total_flags} flags de risco identificadas! Contribuinte demanda atenção especial.</div>', unsafe_allow_html=True)

# =============================================================================
# ABA 6 — VISÃO SETORIAL (PANORÂMICA)
# =============================================================================
with tab6:
    st.markdown('<div class="section-header"><h2>🏭 Visão Setorial — Setor Têxtil SC</h2></div>', unsafe_allow_html=True)

    # Filtro de ano para rankings
    ano_setor = st.selectbox("Ano de referência:", [2024, 2023, 2022, 2021, 2020], key="ano_setor")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Top 10 — Faturamento")
        df_top_fat = carregar_top_faturamento(ano_setor)
        if not df_top_fat.empty:
            fig_top_fat = px.bar(
                df_top_fat, y='razao_social', x='faturamento_total',
                orientation='h', color='faturamento_total',
                color_continuous_scale='Blues',
                text=[fmt(v) for v in df_top_fat['faturamento_total']]
            )
            fig_top_fat.update_layout(
                yaxis={'categoryorder': 'total ascending'}, height=450,
                showlegend=False, coloraxis_showscale=False,
                xaxis_tickformat=",.0f", yaxis_title="", xaxis_title="Faturamento (R$)"
            )
            st.plotly_chart(fig_top_fat, use_container_width=True)
        else:
            st.info("Sem dados de faturamento para o ano selecionado.")

    with col2:
        st.markdown("#### 💰 Top 10 — Crédito Presumido")
        df_top_cp = carregar_top_cp(ano_setor)
        if not df_top_cp.empty:
            fig_top_cp = px.bar(
                df_top_cp, y='razao_social', x='vl_cp_total',
                orientation='h', color='tipo_cp',
                text=[fmt(v) for v in df_top_cp['vl_cp_total']]
            )
            fig_top_cp.update_layout(
                yaxis={'categoryorder': 'total ascending'}, height=450,
                xaxis_tickformat=",.0f", yaxis_title="", xaxis_title="CP Total (R$)"
            )
            st.plotly_chart(fig_top_cp, use_container_width=True)
        else:
            st.info("Sem dados de CP para o ano selecionado.")

    st.markdown("---")

    # Distribuições
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📊 Distribuição por CNAE")
        df_cnae = carregar_distribuicao_cnae()
        if not df_cnae.empty:
            fig_cnae = px.bar(
                df_cnae.head(10), x='qtd', y='cnae_desc',
                orientation='h', color='qtd', color_continuous_scale='Viridis'
            )
            fig_cnae.update_layout(
                yaxis={'categoryorder': 'total ascending'}, height=400,
                showlegend=False, coloraxis_showscale=False,
                yaxis_title="", xaxis_title="Qtd Contribuintes"
            )
            st.plotly_chart(fig_cnae, use_container_width=True)

    with col2:
        st.markdown("#### 🏢 Distribuição por GES")
        df_ges = carregar_distribuicao_ges()
        if not df_ges.empty:
            fig_ges = px.pie(
                df_ges.head(10), values='qtd', names='nome_ges',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_ges.update_layout(height=400)
            st.plotly_chart(fig_ges, use_container_width=True)

    with col3:
        st.markdown("#### 📋 Distribuição por Regime")
        df_regime = carregar_distribuicao_regime()
        if not df_regime.empty:
            fig_regime = px.pie(
                df_regime, values='qtd', names='regime_tributacao',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_regime.update_layout(height=400)
            st.plotly_chart(fig_regime, use_container_width=True)

    # Resumo de flags setorial
    st.markdown('<div class="section-header"><h2>🚩 Resumo de Flags — Visão Setorial</h2></div>', unsafe_allow_html=True)
    flags_resumo = carregar_resumo_flags()

    cols_flag = st.columns(5)
    for i, (num, info) in enumerate(flags_resumo.items()):
        with cols_flag[i]:
            cor = "red" if info['total'] > 0 else "green"
            st.markdown(f"""
            <div class="metric-card {cor}">
                <div class="metric-label">An. {num}: {info['nome']}</div>
                <div class="metric-value">{info['total']}</div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES DE DADOS AGREGADOS — ANÁLISE EXPLORATÓRIA (carregamento rápido)
# =============================================================================
@st.cache_data(ttl=3600)
def carregar_painel_setorial() -> pd.DataFrame:
    """Painel setorial: DIME anual + cadastro agregado por CNPJ Raiz."""
    return executar_query("""
        SELECT
            da.cnpj_raiz,
            da.ano,
            da.qtd_estabelecimentos,
            da.faturamento_total,
            da.receita_bruta_total,
            da.bc_saidas_total,
            da.bc_entradas_total,
            da.debitos_total,
            da.creditos_total,
            da.icms_recolher_total,
            da.cred_dcip_total,
            da.saidas_internas,
            da.saidas_interestaduais,
            da.exportacao,
            da.entradas_internas,
            da.entradas_interestaduais,
            da.importacao,
            c.razao_social,
            c.cnae,
            c.cnae_desc,
            c.regime_tributacao,
            c.nome_ges,
            c.municipio,
            c.ind_simples_nacional
        FROM teste.gestex_v013_dime_anual da
        INNER JOIN (
            SELECT cnpj_raiz,
                   MAX(razao_social) AS razao_social,
                   MAX(cnae) AS cnae,
                   MAX(cnae_desc) AS cnae_desc,
                   MAX(regime_tributacao) AS regime_tributacao,
                   MAX(nome_ges) AS nome_ges,
                   MAX(municipio) AS municipio,
                   MAX(ind_simples_nacional) AS ind_simples_nacional
            FROM teste.gestex_v013_cadastro
            GROUP BY cnpj_raiz
        ) c ON da.cnpj_raiz = c.cnpj_raiz
        ORDER BY da.cnpj_raiz, da.ano
    """)

@st.cache_data(ttl=3600)
def carregar_dcip_agregado() -> pd.DataFrame:
    """DCIP anual completa para todos os contribuintes."""
    return executar_query("""
        SELECT cnpj_raiz, ano, tipo_beneficio, vl_cp_total, qtd_registros, teve_extempraneo
        FROM teste.gestex_v013_dcip_anual
        ORDER BY cnpj_raiz, ano
    """)

@st.cache_data(ttl=3600)
def carregar_todas_flags() -> pd.DataFrame:
    """Consolida flags de todas as análises com contagem por contribuinte."""
    return executar_query("""
        SELECT cnpj_raiz, 'ICMS_BAIXO' AS flag, ano FROM teste.gestex_v013_analise1_icms_baixo
        UNION ALL
        SELECT cnpj_raiz, 'ENT_COMERCIAL' AS flag, ano FROM teste.gestex_v013_analise2_entradas_comerciais
        UNION ALL
        SELECT cnpj_raiz, 'CP_SEM_PROD' AS flag, ano FROM teste.gestex_v013_analise3_cp_sem_producao
        UNION ALL
        SELECT cnpj_raiz, 'CP_CUMULATIVO' AS flag, ano FROM teste.gestex_v013_analise4_cp_cumulativo
        UNION ALL
        SELECT cnpj_raiz, 'Q9070_Q14031' AS flag, ano FROM teste.gestex_v013_analise5_q9070_q14031
    """)

@st.cache_data(ttl=3600)
def carregar_evolucao_setorial() -> pd.DataFrame:
    """Evolução anual agregada de todo o setor."""
    return executar_query("""
        SELECT
            ano,
            COUNT(DISTINCT cnpj_raiz) AS qtd_contribuintes,
            SUM(faturamento_total) AS faturamento_setor,
            SUM(receita_bruta_total) AS receita_bruta_setor,
            SUM(debitos_total) AS debitos_setor,
            SUM(creditos_total) AS creditos_setor,
            SUM(icms_recolher_total) AS icms_setor,
            SUM(cred_dcip_total) AS dcip_setor,
            SUM(saidas_internas) AS saidas_internas_setor,
            SUM(saidas_interestaduais) AS saidas_ie_setor,
            SUM(exportacao) AS exportacao_setor,
            SUM(entradas_internas) AS entradas_internas_setor,
            SUM(entradas_interestaduais) AS entradas_ie_setor,
            SUM(importacao) AS importacao_setor,
            AVG(faturamento_total) AS faturamento_medio,
            AVG(icms_recolher_total) AS icms_medio
        FROM teste.gestex_v013_dime_anual
        GROUP BY ano
        ORDER BY ano
    """)

# =============================================================================
# FUNÇÕES AUXILIARES — FEATURE ENGINEERING & ML
# =============================================================================
def construir_features_ml(df_painel: pd.DataFrame, df_dcip: pd.DataFrame, df_flags: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói matriz de features por CNPJ Raiz para scoring ML.
    Agrega séries temporais em indicadores estáticos por contribuinte.
    """
    if df_painel.empty:
        return pd.DataFrame()

    # --- 1. Agregar DIME por contribuinte (todos os anos) ---
    agg = df_painel.groupby('cnpj_raiz').agg(
        razao_social=('razao_social', 'first'),
        cnae=('cnae', 'first'),
        cnae_desc=('cnae_desc', 'first'),
        regime=('regime_tributacao', 'first'),
        nome_ges=('nome_ges', 'first'),
        municipio=('municipio', 'first'),
        ind_simples=('ind_simples_nacional', 'first'),
        qtd_anos=('ano', 'nunique'),
        max_estabelecimentos=('qtd_estabelecimentos', 'max'),
        fat_total=('faturamento_total', 'sum'),
        fat_medio=('faturamento_total', 'mean'),
        fat_std=('faturamento_total', 'std'),
        receita_total=('receita_bruta_total', 'sum'),
        bc_saidas_total=('bc_saidas_total', 'sum'),
        bc_entradas_total=('bc_entradas_total', 'sum'),
        debitos_total=('debitos_total', 'sum'),
        creditos_total=('creditos_total', 'sum'),
        icms_total=('icms_recolher_total', 'sum'),
        icms_medio=('icms_recolher_total', 'mean'),
        dcip_total=('cred_dcip_total', 'sum'),
        saidas_int=('saidas_internas', 'sum'),
        saidas_ie=('saidas_interestaduais', 'sum'),
        exportacao=('exportacao', 'sum'),
        entradas_int=('entradas_internas', 'sum'),
        entradas_ie=('entradas_interestaduais', 'sum'),
        importacao=('importacao', 'sum'),
    ).reset_index()

    # Preencher NaN de std (contribuintes com 1 ano apenas)
    agg['fat_std'] = agg['fat_std'].fillna(0)

    # --- 2. Indicadores derivados ---
    agg['total_saidas'] = agg['saidas_int'] + agg['saidas_ie'] + agg['exportacao']
    agg['total_entradas'] = agg['entradas_int'] + agg['entradas_ie'] + agg['importacao']

    # Carga tributária efetiva
    agg['carga_tributaria'] = np.where(
        agg['bc_saidas_total'] > 0,
        agg['icms_total'] / agg['bc_saidas_total'] * 100, 0
    )

    # % DCIP sobre faturamento
    agg['pct_dcip_fat'] = np.where(
        agg['fat_total'] > 0,
        agg['dcip_total'] / agg['fat_total'] * 100, 0
    )

    # % Exportação sobre saídas totais
    agg['pct_exportacao'] = np.where(
        agg['total_saidas'] > 0,
        agg['exportacao'] / agg['total_saidas'] * 100, 0
    )

    # % Entradas interestaduais sobre total entradas
    agg['pct_entradas_ie'] = np.where(
        agg['total_entradas'] > 0,
        agg['entradas_ie'] / agg['total_entradas'] * 100, 0
    )

    # Razão créditos / débitos
    agg['razao_cred_deb'] = np.where(
        agg['debitos_total'] > 0,
        agg['creditos_total'] / agg['debitos_total'], 0
    )

    # Volatilidade do faturamento (coef. variação)
    agg['coef_var_fat'] = np.where(
        agg['fat_medio'] > 0,
        agg['fat_std'] / agg['fat_medio'] * 100, 0
    )

    # Variação faturamento (último ano vs primeiro)
    df_primeiro = df_painel.sort_values('ano').groupby('cnpj_raiz').first()[['faturamento_total']].rename(columns={'faturamento_total': 'fat_primeiro_ano'})
    df_ultimo = df_painel.sort_values('ano').groupby('cnpj_raiz').last()[['faturamento_total']].rename(columns={'faturamento_total': 'fat_ultimo_ano'})
    agg = agg.merge(df_primeiro, on='cnpj_raiz', how='left')
    agg = agg.merge(df_ultimo, on='cnpj_raiz', how='left')
    agg['variacao_fat_pct'] = np.where(
        agg['fat_primeiro_ano'] > 0,
        (agg['fat_ultimo_ano'] - agg['fat_primeiro_ano']) / agg['fat_primeiro_ano'] * 100, 0
    )

    # --- 3. Incorporar DCIP ---
    if not df_dcip.empty:
        dcip_pivot = df_dcip.groupby('cnpj_raiz').agg(
            cp_total_dcip=('vl_cp_total', 'sum'),
            qtd_tipos_ttd=('tipo_beneficio', 'nunique'),
            teve_extempraneo=('teve_extempraneo', 'max')
        ).reset_index()
        agg = agg.merge(dcip_pivot, on='cnpj_raiz', how='left')
    else:
        agg['cp_total_dcip'] = 0
        agg['qtd_tipos_ttd'] = 0
        agg['teve_extempraneo'] = 0

    agg['cp_total_dcip'] = agg['cp_total_dcip'].fillna(0)
    agg['qtd_tipos_ttd'] = agg['qtd_tipos_ttd'].fillna(0)
    agg['teve_extempraneo'] = agg['teve_extempraneo'].fillna(0)

    # --- 4. Incorporar flags ---
    if not df_flags.empty:
        flags_count = df_flags.groupby('cnpj_raiz')['flag'].nunique().reset_index().rename(columns={'flag': 'qtd_flags'})
        flags_pivot = df_flags.assign(val=1).pivot_table(
            index='cnpj_raiz', columns='flag', values='val', aggfunc='max', fill_value=0
        ).reset_index()
        agg = agg.merge(flags_count, on='cnpj_raiz', how='left')
        agg = agg.merge(flags_pivot, on='cnpj_raiz', how='left')
    else:
        agg['qtd_flags'] = 0

    agg['qtd_flags'] = agg['qtd_flags'].fillna(0)

    # Preencher colunas de flag individuais se não existirem
    for f in ['ICMS_BAIXO', 'ENT_COMERCIAL', 'CP_SEM_PROD', 'CP_CUMULATIVO', 'Q9070_Q14031']:
        if f not in agg.columns:
            agg[f] = 0
        agg[f] = agg[f].fillna(0)

    return agg


def executar_scoring(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Executa pipeline de scoring ML:
    1. Isolation Forest (detecção de anomalias)
    2. Score composto baseado em regras fiscais
    3. Clustering K-Means para perfis
    4. Ranking final combinado
    """
    if df_features.empty or len(df_features) < 5:
        return df_features

    # Colunas numéricas para ML
    cols_ml = [
        'fat_total', 'fat_medio', 'icms_total', 'icms_medio',
        'dcip_total', 'carga_tributaria', 'pct_dcip_fat',
        'pct_exportacao', 'pct_entradas_ie', 'razao_cred_deb',
        'coef_var_fat', 'variacao_fat_pct', 'cp_total_dcip',
        'qtd_tipos_ttd', 'teve_extempraneo', 'qtd_flags',
        'bc_saidas_total', 'bc_entradas_total', 'total_saidas', 'total_entradas'
    ]

    cols_disponiveis = [c for c in cols_ml if c in df_features.columns]
    X = df_features[cols_disponiveis].fillna(0).copy()

    # Tratar infinitos
    X = X.replace([np.inf, -np.inf], 0)

    # --- 1. ISOLATION FOREST ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)
    iso.fit(X_scaled)
    df_features = df_features.copy()
    df_features['anomalia_score'] = iso.decision_function(X_scaled)
    df_features['is_anomalia'] = (iso.predict(X_scaled) == -1).astype(int)

    # --- 2. SCORE BASEADO EM REGRAS FISCAIS ---
    normalizer = MinMaxScaler()
    X_norm = pd.DataFrame(normalizer.fit_transform(X), columns=cols_disponiveis, index=X.index)

    score = pd.Series(0.0, index=df_features.index)

    # Carga tributária baixa = maior risco
    if 'carga_tributaria' in X_norm.columns:
        score += (1 - X_norm['carga_tributaria']) * 20

    # % DCIP alto sobre faturamento = maior risco
    if 'pct_dcip_fat' in X_norm.columns:
        score += X_norm['pct_dcip_fat'] * 15

    # Muitas flags = risco
    if 'qtd_flags' in X_norm.columns:
        score += X_norm['qtd_flags'] * 25

    # Crédito extemporâneo = risco
    if 'teve_extempraneo' in df_features.columns:
        score += df_features['teve_extempraneo'].fillna(0) * 10

    # Razão crédito/débito alta = risco
    if 'razao_cred_deb' in X_norm.columns:
        score += X_norm['razao_cred_deb'] * 10

    # Anomalia no Isolation Forest = risco
    score += df_features['is_anomalia'] * 15

    # Variação de faturamento negativa abrupta = risco
    if 'variacao_fat_pct' in X_norm.columns:
        score += (1 - X_norm['variacao_fat_pct']) * 5

    df_features['score_fiscal'] = score

    # Normalizar score para 0-100
    smin, smax = score.min(), score.max()
    if smax > smin:
        df_features['score_fiscal'] = ((score - smin) / (smax - smin) * 100).round(2)
    else:
        df_features['score_fiscal'] = 50.0

    # --- 3. CLUSTERING K-MEANS ---
    n_clusters = min(5, max(2, len(df_features) // 10))
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_features['cluster'] = kmeans.fit_predict(X_scaled)

        # PCA para visualização 2D
        if X_scaled.shape[1] >= 2:
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)
            df_features['pca_x'] = coords[:, 0]
            df_features['pca_y'] = coords[:, 1]
            df_features['pca_var_explicada'] = round(sum(pca.explained_variance_ratio_) * 100, 1)
    except Exception:
        df_features['cluster'] = 0
        df_features['pca_x'] = 0
        df_features['pca_y'] = 0
        df_features['pca_var_explicada'] = 0

    # --- 4. CLASSIFICAÇÃO DE RISCO ---
    df_features['faixa_risco'] = pd.cut(
        df_features['score_fiscal'],
        bins=[-1, 25, 50, 75, 100],
        labels=['🟢 Baixo', '🟡 Moderado', '🟠 Alto', '🔴 Crítico']
    )

    # Ranking
    df_features['ranking'] = df_features['score_fiscal'].rank(ascending=False, method='min').astype(int)

    return df_features


# =============================================================================
# ABA 7 — ANÁLISE EXPLORATÓRIA
# =============================================================================
with tab7:
    st.markdown('<div class="section-header"><h2>🔬 Análise Exploratória — Setor Têxtil SC</h2></div>', unsafe_allow_html=True)

    # Carregar dados agregados (rápido)
    with st.spinner("Carregando dados agregados do setor..."):
        df_evolucao = carregar_evolucao_setorial()
        df_painel = carregar_painel_setorial()

    if df_painel.empty:
        st.error("❌ Não foi possível carregar os dados do painel setorial.")
    else:
        # ── Seção 1: KPIs Macro do Setor ──
        st.markdown('<div class="section-header"><h2>📊 Panorama Geral do Setor</h2></div>', unsafe_allow_html=True)

        qtd_contribuintes = df_painel['cnpj_raiz'].nunique()
        fat_total_setor = df_painel['faturamento_total'].sum()
        icms_total_setor = df_painel['icms_recolher_total'].sum()
        dcip_total_setor = df_painel['cred_dcip_total'].sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Contribuintes (CNPJ Raiz)</div><div class="metric-value">{formatar_numero(qtd_contribuintes)}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card green"><div class="metric-label">Faturamento Total (5 anos)</div><div class="metric-value">{fmt(fat_total_setor)}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card orange"><div class="metric-label">ICMS Recolhido (5 anos)</div><div class="metric-value">{fmt(icms_total_setor)}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card purple"><div class="metric-label">Crédito DCIP (5 anos)</div><div class="metric-value">{fmt(dcip_total_setor)}</div></div>', unsafe_allow_html=True)

        # ── Seção 2: Evolução Temporal do Setor ──
        st.markdown('<div class="section-header"><h2>📈 Evolução Temporal do Setor</h2></div>', unsafe_allow_html=True)

        if not df_evolucao.empty:
            col1, col2 = st.columns(2)

            with col1:
                fig_ev1 = make_subplots(specs=[[{"secondary_y": True}]])
                fig_ev1.add_trace(go.Bar(
                    x=df_evolucao['ano'], y=df_evolucao['faturamento_setor'],
                    name="Faturamento", marker_color='#2b6cb0',
                    text=[fmt(v) for v in df_evolucao['faturamento_setor']], textposition='outside'
                ), secondary_y=False)
                fig_ev1.add_trace(go.Scatter(
                    x=df_evolucao['ano'], y=df_evolucao['qtd_contribuintes'],
                    name="Nº Contribuintes", mode='lines+markers',
                    line=dict(color='#dd6b20', width=3)
                ), secondary_y=True)
                fig_ev1.update_layout(title="Faturamento e Nº de Contribuintes", height=380,
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02), xaxis=dict(dtick=1))
                fig_ev1.update_yaxes(title_text="Faturamento (R$)", secondary_y=False, tickformat=",.0f")
                fig_ev1.update_yaxes(title_text="Contribuintes", secondary_y=True)
                st.plotly_chart(fig_ev1, use_container_width=True)

            with col2:
                fig_ev2 = go.Figure()
                fig_ev2.add_trace(go.Bar(x=df_evolucao['ano'], y=df_evolucao['icms_setor'], name="ICMS Recolhido", marker_color='#38a169'))
                fig_ev2.add_trace(go.Bar(x=df_evolucao['ano'], y=df_evolucao['dcip_setor'], name="Crédito DCIP", marker_color='#805ad5'))
                fig_ev2.update_layout(title="ICMS Recolhido × DCIP", height=380, barmode='group',
                                      yaxis_tickformat=",.0f", xaxis=dict(dtick=1),
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig_ev2, use_container_width=True)

            # Composição das saídas
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=df_evolucao['ano'], y=df_evolucao['saidas_internas_setor'], name="Internas", marker_color='#2b6cb0'))
            fig_comp.add_trace(go.Bar(x=df_evolucao['ano'], y=df_evolucao['saidas_ie_setor'], name="Interestaduais", marker_color='#805ad5'))
            fig_comp.add_trace(go.Bar(x=df_evolucao['ano'], y=df_evolucao['exportacao_setor'], name="Exportação", marker_color='#dd6b20'))
            fig_comp.update_layout(title="Composição das Saídas do Setor", barmode='stack', height=380,
                                   yaxis_tickformat=",.0f", xaxis=dict(dtick=1),
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_comp, use_container_width=True)

        # ── Seção 3: Distribuições e Concentração ──
        st.markdown('<div class="section-header"><h2>📊 Distribuição e Concentração</h2></div>', unsafe_allow_html=True)

        ano_exp = st.selectbox("Ano de referência:", sorted(df_painel['ano'].unique(), reverse=True), key="ano_exploratoria")
        df_ano = df_painel[df_painel['ano'] == ano_exp].copy()

        if not df_ano.empty:
            col1, col2, col3 = st.columns(3)

            with col1:
                # Distribuição do faturamento (histograma)
                df_ano_pos = df_ano[df_ano['faturamento_total'] > 0]
                fig_hist = px.histogram(
                    df_ano_pos, x='faturamento_total', nbins=50,
                    title=f"Distribuição do Faturamento ({ano_exp})",
                    color_discrete_sequence=['#2b6cb0']
                )
                fig_hist.update_layout(height=350, xaxis_title="Faturamento (R$)", yaxis_title="Frequência", xaxis_tickformat=",.0f")
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                # Carga tributária
                df_ano['carga_trib'] = np.where(
                    df_ano['bc_saidas_total'] > 0,
                    df_ano['icms_recolher_total'] / df_ano['bc_saidas_total'] * 100, 0
                )
                df_carga = df_ano[df_ano['carga_trib'].between(0, 30)]
                fig_carga = px.histogram(
                    df_carga, x='carga_trib', nbins=50,
                    title=f"Distribuição da Carga Tributária ({ano_exp})",
                    color_discrete_sequence=['#38a169']
                )
                fig_carga.update_layout(height=350, xaxis_title="% ICMS / BC Saídas", yaxis_title="Frequência")
                st.plotly_chart(fig_carga, use_container_width=True)

            with col3:
                # Concentração: Top 10 vs Resto
                df_top10 = df_ano.nlargest(10, 'faturamento_total')
                fat_top10 = df_top10['faturamento_total'].sum()
                fat_resto = df_ano['faturamento_total'].sum() - fat_top10
                fig_conc = px.pie(
                    values=[fat_top10, fat_resto], names=['Top 10', 'Demais'],
                    title=f"Concentração de Faturamento ({ano_exp})",
                    color_discrete_sequence=['#e53e3e', '#bee3f8']
                )
                fig_conc.update_layout(height=350)
                st.plotly_chart(fig_conc, use_container_width=True)

            # Scatterplot: Faturamento × ICMS
            st.markdown("#### 🔎 Faturamento × ICMS a Recolher × DCIP")
            df_scatter = df_ano[df_ano['faturamento_total'] > 0].copy()
            df_scatter['dcip_positivo'] = df_scatter['cred_dcip_total'].clip(lower=0)

            fig_scatter = px.scatter(
                df_scatter, x='faturamento_total', y='icms_recolher_total',
                size='dcip_positivo', color='regime_tributacao',
                hover_name='razao_social',
                hover_data={'cnpj_raiz': True, 'municipio': True, 'faturamento_total': ':,.0f', 'icms_recolher_total': ':,.0f'},
                title=f"Faturamento × ICMS (tamanho = DCIP) — {ano_exp}",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_scatter.update_layout(height=500, xaxis_tickformat=",.0f", yaxis_tickformat=",.0f",
                                       xaxis_title="Faturamento (R$)", yaxis_title="ICMS a Recolher (R$)")
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Boxplots por GES
            st.markdown("#### 📦 Faturamento por GES — Boxplot")
            top_ges = df_ano.groupby('nome_ges')['faturamento_total'].sum().nlargest(10).index.tolist()
            df_box = df_ano[df_ano['nome_ges'].isin(top_ges)]
            fig_box = px.box(
                df_box, x='nome_ges', y='faturamento_total',
                color='nome_ges', title=f"Distribuição do Faturamento por GES — Top 10 ({ano_exp})"
            )
            fig_box.update_layout(height=420, showlegend=False, xaxis_title="", yaxis_title="Faturamento (R$)",
                                   yaxis_tickformat=",.0f")
            st.plotly_chart(fig_box, use_container_width=True)

            # Heatmap: Correlações
            st.markdown("#### 🌡️ Matriz de Correlação — Indicadores Fiscais")
            cols_corr = ['faturamento_total', 'bc_saidas_total', 'bc_entradas_total',
                         'debitos_total', 'creditos_total', 'icms_recolher_total',
                         'cred_dcip_total', 'saidas_internas', 'saidas_interestaduais',
                         'exportacao', 'entradas_internas', 'entradas_interestaduais', 'importacao']
            cols_corr = [c for c in cols_corr if c in df_ano.columns]
            corr = df_ano[cols_corr].corr()

            fig_heat = px.imshow(
                corr, text_auto='.2f', aspect='auto',
                color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                title="Correlação entre Indicadores Fiscais"
            )
            fig_heat.update_layout(height=550)
            st.plotly_chart(fig_heat, use_container_width=True)

            # Detalhes sob demanda
            with st.expander("📋 Tabela completa — Dados do ano selecionado"):
                st.dataframe(df_ano, use_container_width=True, height=400)
                csv = df_ano.to_csv(index=False, sep=';', decimal=',')
                st.download_button("⬇️ Exportar CSV", csv, f"gestex_v013_exploratoria_{ano_exp}.csv", "text/csv")


# =============================================================================
# ABA 8 — SCORING ML
# =============================================================================
with tab8:
    st.markdown('<div class="section-header"><h2>🤖 Machine Learning — Scoring de Risco Fiscal</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="cadastro-card">
        <strong>Metodologia do Score de Risco</strong><br>
        <span style="font-size: 0.9rem; color: #4a5568;">
        O modelo combina múltiplas técnicas para priorizar contribuintes para fiscalização:<br>
        <b>1. Isolation Forest</b> — Detecta anomalias estatísticas nos padrões fiscais (15% do score).<br>
        <b>2. Score Baseado em Regras</b> — Pondera indicadores fiscais: carga tributária baixa, DCIP/faturamento alto, flags de análise, créditos extemporâneos, razão crédito/débito elevada (60% do score).<br>
        <b>3. K-Means Clustering</b> — Agrupa contribuintes em perfis comportamentais para análise comparativa.<br>
        <b>4. Ranking Combinado</b> — Score final 0-100, onde 100 = maior risco de irregularidade.<br>
        <em>Features: 20+ indicadores derivados de DIME, EFD, DCIP e flags de análise.</em>
        </span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Executar Modelo de Scoring", use_container_width=True, type="primary"):
        with st.spinner("Carregando dados e construindo features..."):
            df_painel_ml = carregar_painel_setorial()
            df_dcip_ml = carregar_dcip_agregado()
            df_flags_ml = carregar_todas_flags()

        if df_painel_ml.empty:
            st.error("❌ Sem dados para executar o modelo.")
        else:
            with st.spinner("Construindo features (feature engineering)..."):
                df_features = construir_features_ml(df_painel_ml, df_dcip_ml, df_flags_ml)

            with st.spinner("Executando modelos de ML (Isolation Forest + K-Means + Score)..."):
                df_scored = executar_scoring(df_features)

            # Salvar no session_state
            st.session_state['df_scored'] = df_scored
            st.success(f"✅ Scoring concluído! {len(df_scored)} contribuintes analisados.")

    # Exibir resultados se disponíveis
    if 'df_scored' in st.session_state and not st.session_state['df_scored'].empty:
        df_scored = st.session_state['df_scored']

        # ── KPIs do Scoring ──
        st.markdown('<div class="section-header"><h2>📊 Resultados do Scoring</h2></div>', unsafe_allow_html=True)

        total_analisados = len(df_scored)
        total_anomalias = df_scored['is_anomalia'].sum()
        score_medio = df_scored['score_fiscal'].mean()
        total_critico = len(df_scored[df_scored['faixa_risco'] == '🔴 Crítico'])
        total_alto = len(df_scored[df_scored['faixa_risco'] == '🟠 Alto'])

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Contribuintes Analisados</div><div class="metric-value">{formatar_numero(total_analisados)}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card red"><div class="metric-label">Anomalias Detectadas</div><div class="metric-value">{formatar_numero(total_anomalias)}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card orange"><div class="metric-label">Score Médio</div><div class="metric-value">{score_medio:.1f}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card red"><div class="metric-label">Risco Crítico</div><div class="metric-value">{total_critico}</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="metric-card orange"><div class="metric-label">Risco Alto</div><div class="metric-value">{total_alto}</div></div>', unsafe_allow_html=True)

        # ── Distribuição de Risco ──
        col1, col2 = st.columns(2)

        with col1:
            faixa_counts = df_scored['faixa_risco'].value_counts().reset_index()
            faixa_counts.columns = ['faixa', 'qtd']
            cores_faixa = {'🟢 Baixo': '#38a169', '🟡 Moderado': '#ecc94b', '🟠 Alto': '#dd6b20', '🔴 Crítico': '#e53e3e'}
            fig_faixa = px.bar(
                faixa_counts, x='faixa', y='qtd', color='faixa',
                color_discrete_map=cores_faixa,
                title="Distribuição por Faixa de Risco",
                text='qtd'
            )
            fig_faixa.update_layout(height=380, showlegend=False, xaxis_title="", yaxis_title="Qtd Contribuintes")
            st.plotly_chart(fig_faixa, use_container_width=True)

        with col2:
            fig_hist_score = px.histogram(
                df_scored, x='score_fiscal', nbins=40,
                title="Distribuição do Score de Risco",
                color_discrete_sequence=['#2b6cb0']
            )
            fig_hist_score.update_layout(height=380, xaxis_title="Score (0-100)", yaxis_title="Frequência")
            st.plotly_chart(fig_hist_score, use_container_width=True)

        # ── Clustering PCA ──
        if 'pca_x' in df_scored.columns and df_scored['pca_x'].abs().sum() > 0:
            st.markdown('<div class="section-header"><h2>🔵 Clusters — Perfis de Contribuintes (PCA 2D)</h2></div>', unsafe_allow_html=True)

            var_exp = df_scored['pca_var_explicada'].iloc[0] if 'pca_var_explicada' in df_scored.columns else 0

            fig_pca = px.scatter(
                df_scored, x='pca_x', y='pca_y',
                color='faixa_risco', symbol='cluster',
                hover_name='razao_social',
                hover_data={'cnpj_raiz': True, 'score_fiscal': ':.1f', 'carga_tributaria': ':.2f',
                            'pct_dcip_fat': ':.2f', 'qtd_flags': True},
                title=f"Mapa de Clusters (PCA — {var_exp}% variância explicada)",
                color_discrete_map=cores_faixa
            )
            fig_pca.update_layout(height=550, xaxis_title="Componente Principal 1", yaxis_title="Componente Principal 2")
            st.plotly_chart(fig_pca, use_container_width=True)

            # Resumo por cluster
            with st.expander("📊 Perfil médio por Cluster"):
                perfil_cols = ['cluster', 'fat_total', 'icms_total', 'dcip_total', 'carga_tributaria',
                               'pct_dcip_fat', 'razao_cred_deb', 'qtd_flags', 'score_fiscal']
                perfil_cols = [c for c in perfil_cols if c in df_scored.columns]
                perfil = df_scored.groupby('cluster')[perfil_cols[1:]].mean().round(2)
                perfil['qtd_empresas'] = df_scored.groupby('cluster').size()
                st.dataframe(perfil, use_container_width=True)

        # ── TOP N para Fiscalização ──
        st.markdown('<div class="section-header"><h2>🎯 Ranking — Prioridade para Fiscalização</h2></div>', unsafe_allow_html=True)

        col_filtro1, col_filtro2, col_filtro3 = st.columns(3)
        with col_filtro1:
            top_n = st.slider("Exibir Top N:", 10, 100, 30, 5, key="top_n_ml")
        with col_filtro2:
            filtro_ges_ml = st.multiselect(
                "Filtrar por GES:",
                sorted(df_scored['nome_ges'].dropna().unique().tolist()),
                key="filtro_ges_ml"
            )
        with col_filtro3:
            filtro_faixa = st.multiselect(
                "Filtrar por Faixa:",
                ['🔴 Crítico', '🟠 Alto', '🟡 Moderado', '🟢 Baixo'],
                default=['🔴 Crítico', '🟠 Alto'],
                key="filtro_faixa_ml"
            )

        df_ranking = df_scored.sort_values('score_fiscal', ascending=False).copy()
        if filtro_ges_ml:
            df_ranking = df_ranking[df_ranking['nome_ges'].isin(filtro_ges_ml)]
        if filtro_faixa:
            df_ranking = df_ranking[df_ranking['faixa_risco'].isin(filtro_faixa)]

        df_ranking_top = df_ranking.head(top_n)

        # Gráfico ranking horizontal
        fig_rank = px.bar(
            df_ranking_top.head(20), y='razao_social', x='score_fiscal',
            orientation='h', color='faixa_risco',
            color_discrete_map=cores_faixa,
            hover_data={'cnpj_raiz': True, 'carga_tributaria': ':.2f', 'pct_dcip_fat': ':.2f', 'qtd_flags': True},
            title=f"Top {min(20, len(df_ranking_top))} — Score de Risco Fiscal",
            text='score_fiscal'
        )
        fig_rank.update_layout(
            height=max(400, min(20, len(df_ranking_top)) * 35),
            yaxis={'categoryorder': 'total ascending'}, yaxis_title="",
            xaxis_title="Score de Risco (0–100)"
        )
        fig_rank.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig_rank, use_container_width=True)

        # Tabela detalhada do ranking
        colunas_exibir = [
            'ranking', 'faixa_risco', 'cnpj_raiz', 'razao_social', 'nome_ges', 'municipio',
            'regime', 'score_fiscal', 'is_anomalia', 'carga_tributaria', 'pct_dcip_fat',
            'fat_total', 'icms_total', 'dcip_total', 'qtd_flags', 'teve_extempraneo',
            'razao_cred_deb', 'variacao_fat_pct', 'cluster'
        ]
        colunas_exibir = [c for c in colunas_exibir if c in df_ranking_top.columns]
        st.dataframe(df_ranking_top[colunas_exibir], use_container_width=True, height=500)

        csv_rank = df_ranking[colunas_exibir].to_csv(index=False, sep=';', decimal=',')
        st.download_button("⬇️ Exportar Ranking Completo (CSV)", csv_rank,
                           "gestex_v013_ranking_ml.csv", "text/csv")

        # ── Detalhe sob demanda: selecionar empresa do ranking ──
        st.markdown('<div class="section-header"><h2>🔎 Detalhe do Contribuinte Selecionado</h2></div>', unsafe_allow_html=True)

        empresas_ranking = df_ranking_top[['cnpj_raiz', 'razao_social', 'score_fiscal', 'faixa_risco']].copy()
        empresas_ranking['label'] = empresas_ranking.apply(
            lambda r: f"[{r['faixa_risco']}] Score {r['score_fiscal']:.1f} — {r['razao_social']} ({r['cnpj_raiz']})", axis=1
        )

        empresa_sel = st.selectbox("Selecione uma empresa para detalhar:", empresas_ranking['label'].tolist(), key="detalhe_ml")

        if empresa_sel:
            cnpj_detalhe = empresas_ranking[empresas_ranking['label'] == empresa_sel]['cnpj_raiz'].iloc[0]
            row_detalhe = df_scored[df_scored['cnpj_raiz'] == cnpj_detalhe].iloc[0]

            # Card resumo
            st.markdown(f"""
            <div class="cadastro-card">
                <div style="font-size: 1.2rem; font-weight: 700; color: #1a365d;">{row_detalhe.get('razao_social', '')}</div>
                <div class="info-row" style="margin-top: 0.5rem;">
                    <div class="info-item"><label>CNPJ Raiz</label><span>{cnpj_detalhe}</span></div>
                    <div class="info-item"><label>GES</label><span>{row_detalhe.get('nome_ges', '')}</span></div>
                    <div class="info-item"><label>Município</label><span>{row_detalhe.get('municipio', '')}</span></div>
                    <div class="info-item"><label>Regime</label><span>{row_detalhe.get('regime', '')}</span></div>
                    <div class="info-item"><label>CNAE</label><span>{row_detalhe.get('cnae_desc', '')}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # KPIs do contribuinte
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.markdown(f'<div class="metric-card red"><div class="metric-label">Score Risco</div><div class="metric-value">{row_detalhe["score_fiscal"]:.1f}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Carga Tributária</div><div class="metric-value">{formatar_pct(row_detalhe.get("carga_tributaria", 0))}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card purple"><div class="metric-label">% DCIP/Fat</div><div class="metric-value">{formatar_pct(row_detalhe.get("pct_dcip_fat", 0))}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card orange"><div class="metric-label">Flags</div><div class="metric-value">{int(row_detalhe.get("qtd_flags", 0))}</div></div>', unsafe_allow_html=True)
            with c5:
                anom_txt = "SIM ⚠️" if row_detalhe.get('is_anomalia', 0) else "NÃO ✅"
                cor_anom = "red" if row_detalhe.get('is_anomalia', 0) else "green"
                st.markdown(f'<div class="metric-card {cor_anom}"><div class="metric-label">Anomalia</div><div class="metric-value">{anom_txt}</div></div>', unsafe_allow_html=True)
            with c6:
                ext_txt = "SIM ⚠️" if row_detalhe.get('teve_extempraneo', 0) else "NÃO ✅"
                cor_ext = "red" if row_detalhe.get('teve_extempraneo', 0) else "green"
                st.markdown(f'<div class="metric-card {cor_ext}"><div class="metric-label">Extemporâneo</div><div class="metric-value">{ext_txt}</div></div>', unsafe_allow_html=True)

            # Radar chart do perfil de risco
            st.markdown("#### 🕸️ Perfil de Risco (Radar)")

            radar_cols = {
                'carga_tributaria': 'Carga Tributária',
                'pct_dcip_fat': '% DCIP/Fat',
                'razao_cred_deb': 'Razão Créd/Déb',
                'coef_var_fat': 'Volatilidade Fat.',
                'pct_entradas_ie': '% Ent. Interestaduais',
                'pct_exportacao': '% Exportação'
            }
            radar_cols_disp = {k: v for k, v in radar_cols.items() if k in df_scored.columns}

            if radar_cols_disp:
                # Normalizar para escala 0-1 (usando min-max do dataset)
                vals_empresa = []
                vals_media = []
                categorias = []
                for col, nome in radar_cols_disp.items():
                    v_emp = row_detalhe.get(col, 0)
                    v_med = df_scored[col].mean()
                    v_max = df_scored[col].max()
                    v_min = df_scored[col].min()
                    if v_max > v_min:
                        vals_empresa.append((v_emp - v_min) / (v_max - v_min))
                        vals_media.append((v_med - v_min) / (v_max - v_min))
                    else:
                        vals_empresa.append(0)
                        vals_media.append(0)
                    categorias.append(nome)

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_empresa + [vals_empresa[0]], theta=categorias + [categorias[0]],
                    fill='toself', name='Contribuinte', line_color='#e53e3e', fillcolor='rgba(229,62,62,0.2)'
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_media + [vals_media[0]], theta=categorias + [categorias[0]],
                    fill='toself', name='Média Setor', line_color='#2b6cb0', fillcolor='rgba(43,108,176,0.1)'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    height=420, title="Perfil do Contribuinte vs Média do Setor"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Série temporal sob demanda
            with st.expander("📈 Evolução Anual Detalhada (DIME)"):
                df_det_anual = carregar_dime_anual(cnpj_detalhe)
                if not df_det_anual.empty:
                    fig_det = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_det.add_trace(go.Bar(
                        x=df_det_anual['ano'], y=df_det_anual['faturamento_total'],
                        name="Faturamento", marker_color='#2b6cb0'
                    ), secondary_y=False)
                    fig_det.add_trace(go.Scatter(
                        x=df_det_anual['ano'], y=df_det_anual['icms_recolher_total'],
                        name="ICMS", mode='lines+markers', line=dict(color='#38a169', width=3)
                    ), secondary_y=True)
                    fig_det.add_trace(go.Scatter(
                        x=df_det_anual['ano'], y=df_det_anual['cred_dcip_total'],
                        name="DCIP", mode='lines+markers', line=dict(color='#dd6b20', width=3, dash='dot')
                    ), secondary_y=True)
                    fig_det.update_layout(title="Evolução Anual", height=380, xaxis=dict(dtick=1))
                    fig_det.update_yaxes(title_text="Faturamento", secondary_y=False, tickformat=",.0f")
                    fig_det.update_yaxes(title_text="ICMS / DCIP", secondary_y=True, tickformat=",.0f")
                    st.plotly_chart(fig_det, use_container_width=True)
                    st.dataframe(df_det_anual, use_container_width=True)
                else:
                    st.info("Sem dados DIME para este contribuinte.")

            with st.expander("💰 Crédito Presumido Detalhado (DCIP)"):
                df_det_dcip = carregar_dcip_anual(cnpj_detalhe)
                if not df_det_dcip.empty:
                    fig_det_cp = px.bar(
                        df_det_dcip, x='ano', y='vl_cp_total', color='tipo_beneficio',
                        barmode='group', title="Crédito Presumido por Ano/TTD"
                    )
                    fig_det_cp.update_layout(height=350, yaxis_tickformat=",.0f", xaxis=dict(dtick=1))
                    st.plotly_chart(fig_det_cp, use_container_width=True)
                    st.dataframe(df_det_dcip, use_container_width=True)
                else:
                    st.info("Sem dados DCIP para este contribuinte.")

            with st.expander("🚩 Flags de Análise"):
                for num in range(1, 6):
                    df_f = carregar_analise(num, cnpj_detalhe)
                    nomes_flag = {1: "ICMS Baixo", 2: "Ent. Comerciais", 3: "CP sem Produção",
                                  4: "CP Cumulativo", 5: "Q9070/Q14031"}
                    if not df_f.empty:
                        st.markdown(f'<span class="flag-danger">🔴 Análise {num}: {nomes_flag[num]}</span>', unsafe_allow_html=True)
                        st.dataframe(df_f, use_container_width=True)
                    else:
                        st.markdown(f'<span class="flag-ok">🟢 Análise {num}: {nomes_flag[num]} — OK</span>', unsafe_allow_html=True)

            # Botão para selecionar esta empresa na aba 1
            if st.button(f"📋 Ver visão completa deste contribuinte (Aba 1)", key="ir_aba1_ml"):
                # Buscar dados cadastrais
                df_cad = buscar_contribuinte(cnpj_detalhe)
                if not df_cad.empty:
                    st.session_state.cnpj_raiz_selecionado = cnpj_detalhe
                    st.session_state.dados_cadastro_selecionado = df_cad.iloc[0].to_dict()
                    st.rerun()

    else:
        st.info("👆 Clique no botão acima para executar o modelo de scoring. O processamento leva alguns segundos.")
        st.markdown("""
        **O que o modelo analisa:**
        - Carga tributária efetiva (ICMS / BC Saídas)
        - Proporção do crédito presumido sobre o faturamento
        - Presença em tabelas de flags (5 análises automáticas)
        - Uso de créditos extemporâneos
        - Razão créditos / débitos (distorção na apuração)
        - Volatilidade e tendência do faturamento
        - Anomalias estatísticas via Isolation Forest
        - Agrupamento em clusters de perfis semelhantes
        """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 1rem;'>
    <p><strong>GESTEX V013</strong> — DIME Análise Anual (2020–2024)</p>
    <p>SEF/SC — Grupo Especializado no Setor Têxtil | NIAT</p>
    <p style='font-size: 0.8rem;'>Dados: Big Data Impala/Hue | Schema: teste.gestex_v013_*</p>
</div>
""", unsafe_allow_html=True)