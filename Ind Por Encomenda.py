"""
================================================================================
GESTEX V014 - Dashboard de Industrialização por Encomenda
================================================================================
Sistema de análise de operações de industrialização por encomenda (ICMS)
Consome 18 tabelas analíticas no schema teste.gestex_v014_*
Autor: SEF/SC - Secretaria de Estado da Fazenda de Santa Catarina
Data: Fevereiro/2026
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ssl
import warnings
from sqlalchemy import create_engine
from io import BytesIO

# ML imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
    page_title="GESTEX V014 - Industrialização",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Esconder sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
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
# FUNÇÕES AUXILIARES DE FORMATAÇÃO
# =============================================================================
def formatar_moeda(valor, abreviado=False) -> str:
    """Formata valor para moeda brasileira."""
    if pd.isna(valor) or valor is None:
        return "R$ 0"
    try:
        valor = float(valor)
        if abreviado:
            if valor >= 1_000_000_000:
                return f"R$ {valor/1_000_000_000:,.2f} Bi".replace(",", "X").replace(".", ",").replace("X", ".")
            elif valor >= 1_000_000:
                return f"R$ {valor/1_000_000:,.2f} Mi".replace(",", "X").replace(".", ",").replace("X", ".")
            elif valor >= 1_000:
                return f"R$ {valor/1_000:,.1f} mil".replace(",", "X").replace(".", ",").replace("X", ".")
            else:
                return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "R$ 0"

def formatar_moeda_abrev(valor) -> str:
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
            else:
                return f"{int(valor):,}".replace(",", ".")
        else:
            return f"{int(valor):,}".replace(",", ".")
    except:
        return "0"

def formatar_numero_abrev(valor) -> str:
    return formatar_numero(valor, abreviado=True)

def formatar_percentual(valor) -> str:
    if pd.isna(valor) or valor is None:
        return "0,0%"
    try:
        return f"{float(valor):,.1f}%".replace(".", ",")
    except:
        return "0,0%"

def formatar_qtd(valor) -> str:
    """Formata quantidade com 2 decimais no padrão BR."""
    if pd.isna(valor) or valor is None:
        return "0"
    try:
        valor = float(valor)
        if valor == int(valor):
            return f"{int(valor):,}".replace(",", ".")
        else:
            return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "0"

def to_excel(df: pd.DataFrame) -> bytes:
    """Converte DataFrame para bytes Excel para download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# =============================================================================
# CFOP LABELS
# =============================================================================
CFOP_LABELS = {
    5901: "5901 - Remessa p/ Industrialização (interna)",
    6901: "6901 - Remessa p/ Industrialização (interestadual)",
    5902: "5902 - Retorno de Industrialização (interna)",
    6902: "6902 - Retorno de Industrialização (interestadual)",
    5903: "5903 - Retorno de remessa não aplicada (interna)",
    6903: "6903 - Retorno de remessa não aplicada (interestadual)",
    5124: "5124 - Cobrança Industrialização (interna)",
    6124: "6124 - Cobrança Industrialização (interestadual)",
    5125: "5125 - Industrialização s/ trânsito (interna)",
    6125: "6125 - Industrialização s/ trânsito (interestadual)",
    5924: "5924 - Remessa p/ Industrialização (interna - outra)",
    6924: "6924 - Remessa p/ Industrialização (interest. - outra)",
    5925: "5925 - Retorno de Industrialização (interna - outra)",
    6925: "6925 - Retorno de Industrialização (interest. - outra)",
}

def cfop_label(cfop_val):
    """Retorna label do CFOP ou o próprio valor."""
    try:
        return CFOP_LABELS.get(int(cfop_val), str(cfop_val))
    except:
        return str(cfop_val)

# =============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# =============================================================================

# --- Tabelas LEVES (carregar completas) ---
@st.cache_data(ttl=3600)
def carregar_fluxo_completo():
    return executar_query("SELECT * FROM teste.gestex_v014_fluxo_completo")

@st.cache_data(ttl=3600)
def carregar_painel_consolidado():
    return executar_query("SELECT * FROM teste.gestex_v014_painel_consolidado")

@st.cache_data(ttl=3600)
def carregar_anomalia_sem_retorno():
    return executar_query("SELECT * FROM teste.gestex_v014_anomalia_sem_retorno")

@st.cache_data(ttl=3600)
def carregar_remessas_enviadas():
    return executar_query("SELECT * FROM teste.gestex_v014_remessas_enviadas")

@st.cache_data(ttl=3600)
def carregar_retornos():
    return executar_query("SELECT * FROM teste.gestex_v014_retornos")

@st.cache_data(ttl=3600)
def carregar_cobranca():
    return executar_query("SELECT * FROM teste.gestex_v014_cobranca_industrializacao")

@st.cache_data(ttl=3600)
def carregar_inventario_resumo():
    return executar_query("SELECT * FROM teste.gestex_v014_inventario_resumo")

@st.cache_data(ttl=3600)
def carregar_estoque_evolucao():
    return executar_query("SELECT * FROM teste.gestex_v014_estoque_evolucao")

# --- Busca de identificadorarquivo a partir de CNPJ ---
@st.cache_data(ttl=3600)
def buscar_identificadores(cnpj: str) -> pd.DataFrame:
    cnpj_limpo = cnpj.replace(".", "").replace("/", "").replace("-", "").strip()
    query = f"""
        SELECT DISTINCT identificadorarquivo
        FROM teste.gestex_v014_mapa_contribuintes
        WHERE codigocnpj = '{cnpj_limpo}'
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def buscar_dados_cadastrais(cnpj: str) -> pd.DataFrame:
    cnpj_limpo = cnpj.replace(".", "").replace("/", "").replace("-", "").strip()
    query = f"""
        SELECT
            codigocnpj,
            nomeempresarial,
            ie AS inscricaoestadual,
            cod_municipio,
            nome_fantasia AS nomefantasia
        FROM teste.gestex_v014_mapa_contribuintes
        WHERE codigocnpj = '{cnpj_limpo}'
        ORDER BY mes_referencia DESC
        LIMIT 1
    """
    return executar_query(query)

# --- Tabelas PESADAS (filtrar por identificadorarquivo) ---
@st.cache_data(ttl=3600)
def carregar_producao_k230(id_arquivo: str):
    query = f"""
        SELECT * FROM teste.gestex_v014_producao_k230
        WHERE identificadorarquivo = {id_arquivo}
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_producao_descrita(id_arquivo: str):
    query = f"""
        SELECT * FROM teste.gestex_v014_producao_descrita
        WHERE identificadorarquivo = {id_arquivo}
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_ficha_tecnica(id_arquivo: str):
    query = f"""
        SELECT * FROM teste.gestex_v014_ficha_tecnica
        WHERE identificadorarquivo = {id_arquivo}
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_inventario(id_arquivo: str):
    query = f"""
        SELECT * FROM teste.gestex_v014_inventario
        WHERE identificadorarquivo = {id_arquivo}
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_estoque_terceiros(id_arquivo: str):
    query = f"""
        SELECT * FROM teste.gestex_v014_estoque_terceiros
        WHERE identificadorarquivo = {id_arquivo}
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_industrializacao_terceiros(id_arquivo: str):
    query = f"""
        SELECT * FROM teste.gestex_v014_industrializacao_terceiros
        WHERE identificadorarquivo = {id_arquivo}
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_anomalia_producao_sem_remessa(id_arquivo: str):
    query = f"""
        SELECT * FROM teste.gestex_v014_anomalia_producao_sem_remessa
        WHERE identificadorarquivo = {id_arquivo}
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_ajustes_e111(cnpj: str):
    cnpj_limpo = cnpj.replace(".", "").replace("/", "").replace("-", "").strip()
    query = f"""
        SELECT * FROM teste.gestex_v014_ajustes_e111
        WHERE codigocnpj = '{cnpj_limpo}'
        ORDER BY ano_referencia, mes_referencia
    """
    return executar_query(query)

# =============================================================================
# FUNÇÕES DE CARREGAMENTO — ANÁLISE EXPLORATÓRIA (AGREGADAS / LEVES)
# =============================================================================

@st.cache_data(ttl=3600)
def carregar_panorama_contribuintes():
    """Panorama geral: 1 linha por contribuinte com métricas agregadas do fluxo."""
    query = """
        SELECT
            f.identificadorarquivo,
            m.codigocnpj,
            m.nomeempresarial,
            m.ie,
            m.cod_municipio,
            m.nome_fantasia,
            COUNT(*) AS total_operacoes,
            COUNT(DISTINCT f.chv_nfe) AS total_notas,
            COUNT(DISTINCT f.cod_part) AS total_parceiros,
            COUNT(DISTINCT f.cod_item) AS total_itens,
            SUM(CASE WHEN f.cfop IN (5901,6901) THEN f.vl_item ELSE 0 END) AS vl_remessas,
            SUM(CASE WHEN f.cfop IN (5902,6902,5903,6903) THEN f.vl_item ELSE 0 END) AS vl_retornos,
            SUM(CASE WHEN f.cfop IN (5124,6124,5125,6125) THEN f.vl_item ELSE 0 END) AS vl_industrializacao,
            SUM(CASE WHEN f.cfop IN (5901,6901) THEN f.qtd ELSE 0 END) AS qtd_remessas,
            SUM(CASE WHEN f.cfop IN (5902,6902,5903,6903) THEN f.qtd ELSE 0 END) AS qtd_retornos,
            SUM(f.vl_icms) AS total_icms,
            COUNT(DISTINCT f.mes_referencia) AS meses_ativos,
            MIN(f.dt_doc) AS primeira_operacao,
            MAX(f.dt_doc) AS ultima_operacao
        FROM teste.gestex_v014_fluxo_completo f
        LEFT JOIN (
            SELECT codigocnpj, nomeempresarial, ie, cod_municipio, nome_fantasia, identificadorarquivo
            FROM teste.gestex_v014_mapa_contribuintes
            GROUP BY codigocnpj, nomeempresarial, ie, cod_municipio, nome_fantasia, identificadorarquivo
        ) m ON CAST(f.identificadorarquivo AS BIGINT) = CAST(m.identificadorarquivo AS BIGINT)
        GROUP BY f.identificadorarquivo, m.codigocnpj, m.nomeempresarial, m.ie, m.cod_municipio, m.nome_fantasia
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_panorama_participantes():
    """Panorama por participante (industrializador ou encomendante)."""
    query = """
        SELECT
            cnpj_participante,
            nome_participante,
            municipio_participante,
            COUNT(*) AS total_operacoes,
            COUNT(DISTINCT identificadorarquivo) AS total_declarantes,
            COUNT(DISTINCT chv_nfe) AS total_notas,
            COUNT(DISTINCT cod_item) AS total_itens,
            SUM(CASE WHEN cfop IN (5901,6901) THEN vl_item ELSE 0 END) AS vl_remessas,
            SUM(CASE WHEN cfop IN (5902,6902,5903,6903) THEN vl_item ELSE 0 END) AS vl_retornos,
            SUM(CASE WHEN cfop IN (5124,6124,5125,6125) THEN vl_item ELSE 0 END) AS vl_industrializacao,
            SUM(vl_icms) AS total_icms,
            COUNT(DISTINCT mes_referencia) AS meses_ativos
        FROM teste.gestex_v014_fluxo_completo
        WHERE cnpj_participante IS NOT NULL
        GROUP BY cnpj_participante, nome_participante, municipio_participante
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_resumo_cfop_mensal():
    """Evolução mensal por grupo CFOP."""
    query = """
        SELECT
            mes_referencia,
            CASE
                WHEN cfop IN (5901,6901) THEN 'Remessa'
                WHEN cfop IN (5902,6902,5903,6903) THEN 'Retorno'
                WHEN cfop IN (5124,6124,5125,6125) THEN 'Industrialização'
                ELSE 'Outros'
            END AS tipo_operacao,
            COUNT(*) AS qtd_operacoes,
            SUM(vl_item) AS valor_total,
            SUM(qtd) AS qtd_total,
            SUM(vl_icms) AS icms_total,
            COUNT(DISTINCT identificadorarquivo) AS contribuintes,
            COUNT(DISTINCT chv_nfe) AS notas
        FROM teste.gestex_v014_fluxo_completo
        GROUP BY mes_referencia,
            CASE
                WHEN cfop IN (5901,6901) THEN 'Remessa'
                WHEN cfop IN (5902,6902,5903,6903) THEN 'Retorno'
                WHEN cfop IN (5124,6124,5125,6125) THEN 'Industrialização'
                ELSE 'Outros'
            END
        ORDER BY mes_referencia
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_resumo_anomalias_por_contribuinte():
    """Anomalias sem retorno agregadas por parceiro."""
    query = """
        SELECT
            cod_part,
            COUNT(*) AS qtd_anomalias,
            SUM(qtd_remetida) AS total_qtd_remetida,
            SUM(vl_remetido) AS total_vl_remetido,
            SUM(qtd_pendente) AS total_qtd_pendente,
            COUNT(DISTINCT cod_item) AS itens_pendentes,
            COUNT(DISTINCT chv_nfe) AS notas_pendentes,
            MIN(dt_doc) AS primeira_remessa,
            MAX(dt_doc) AS ultima_remessa
        FROM teste.gestex_v014_anomalia_sem_retorno
        GROUP BY cod_part
        ORDER BY total_vl_remetido DESC
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_resumo_painel_por_contribuinte():
    """Painel consolidado agregado por contribuinte."""
    query = """
        SELECT
            identificadorarquivo,
            COUNT(DISTINCT cod_part) AS parceiros,
            COUNT(DISTINCT cod_item) AS itens,
            SUM(COALESCE(qtd_remetida, 0)) AS total_qtd_remetida,
            SUM(COALESCE(vl_remetido, 0)) AS total_vl_remetido,
            SUM(COALESCE(qtd_retornada, 0)) AS total_qtd_retornada,
            SUM(COALESCE(vl_retornado, 0)) AS total_vl_retornado,
            SUM(COALESCE(vl_industrializacao, 0)) AS total_vl_industrializacao,
            SUM(COALESCE(saldo_qtd, 0)) AS total_saldo_qtd,
            SUM(COALESCE(saldo_valor, 0)) AS total_saldo_valor,
            SUM(CASE WHEN saldo_qtd > 0 THEN 1 ELSE 0 END) AS itens_pendentes,
            SUM(CASE WHEN saldo_qtd = 0 THEN 1 ELSE 0 END) AS itens_ok,
            SUM(CASE WHEN saldo_qtd < 0 THEN 1 ELSE 0 END) AS itens_excesso
        FROM teste.gestex_v014_painel_consolidado
        GROUP BY identificadorarquivo
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_top_ncm():
    """Top NCMs movimentados."""
    query = """
        SELECT
            cod_ncm,
            descr_item,
            COUNT(*) AS operacoes,
            SUM(vl_item) AS valor_total,
            SUM(qtd) AS qtd_total,
            COUNT(DISTINCT identificadorarquivo) AS contribuintes,
            COUNT(DISTINCT cod_part) AS parceiros
        FROM teste.gestex_v014_fluxo_completo
        WHERE cod_ncm IS NOT NULL AND cod_ncm != ''
        GROUP BY cod_ncm, descr_item
        ORDER BY valor_total DESC
        LIMIT 50
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_inventario_resumo_geral():
    """Resumo geral do inventário por propriedade."""
    query = """
        SELECT
            identificadorarquivo,
            propriedade,
            SUM(qtd_itens) AS qtd_itens,
            SUM(vl_total) AS vl_total,
            SUM(qtd_parceiros) AS qtd_parceiros
        FROM teste.gestex_v014_inventario_resumo
        GROUP BY identificadorarquivo, propriedade
    """
    return executar_query(query)

@st.cache_data(ttl=3600)
def carregar_estoque_evolucao_geral():
    """Evolução estoque terceiros geral."""
    query = """
        SELECT
            mes_referencia,
            SUM(qtd_itens) AS qtd_itens,
            SUM(qtd_total) AS qtd_total,
            SUM(qtd_parceiros) AS qtd_parceiros,
            COUNT(DISTINCT identificadorarquivo) AS contribuintes
        FROM teste.gestex_v014_estoque_evolucao
        GROUP BY mes_referencia
        ORDER BY mes_referencia
    """
    return executar_query(query)

# =============================================================================
# FUNÇÕES DE ML — SCORING DE FISCALIZAÇÃO
# =============================================================================

def construir_features_fiscalizacao(df_panorama, df_painel_agg, df_anomalias_agg):
    """Constrói matriz de features para scoring de fiscalização."""

    df = df_panorama.copy()
    df['identificadorarquivo'] = df['identificadorarquivo'].astype(str)

    # Calcular indicadores derivados
    df['saldo_remessa_retorno'] = df['vl_remessas'].fillna(0) - df['vl_retornos'].fillna(0)
    df['pct_retorno'] = np.where(
        df['vl_remessas'] > 0,
        df['vl_retornos'] / df['vl_remessas'] * 100,
        0
    )
    df['valor_medio_operacao'] = np.where(
        df['total_operacoes'] > 0,
        (df['vl_remessas'].fillna(0) + df['vl_retornos'].fillna(0)) / df['total_operacoes'],
        0
    )
    df['icms_por_operacao'] = np.where(
        df['total_operacoes'] > 0,
        df['total_icms'].fillna(0) / df['total_operacoes'],
        0
    )
    df['concentracao_parceiros'] = np.where(
        df['total_operacoes'] > 0,
        df['total_operacoes'] / df['total_parceiros'].clip(lower=1),
        0
    )

    # Merge com painel consolidado
    if df_painel_agg is not None and not df_painel_agg.empty:
        df_p = df_painel_agg.copy()
        df_p['identificadorarquivo'] = df_p['identificadorarquivo'].astype(str)
        df = df.merge(
            df_p[['identificadorarquivo', 'total_saldo_valor', 'itens_pendentes', 'itens_ok', 'itens_excesso']],
            on='identificadorarquivo', how='left'
        )
    else:
        df['total_saldo_valor'] = 0
        df['itens_pendentes'] = 0
        df['itens_ok'] = 0
        df['itens_excesso'] = 0

    # Merge com anomalias (por cod_part — aproximação)
    if df_anomalias_agg is not None and not df_anomalias_agg.empty:
        df['qtd_anomalias'] = 0
        df['vl_anomalias'] = 0
    else:
        df['qtd_anomalias'] = 0
        df['vl_anomalias'] = 0

    df = df.fillna(0)
    return df

def executar_isolation_forest(df_features, feature_cols, contamination=0.15):
    """Executa Isolation Forest para detecção de outliers."""
    X = df_features[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    df_features['anomaly_score'] = iso_forest.fit_predict(X_scaled)
    df_features['anomaly_score_raw'] = iso_forest.decision_function(X_scaled)
    # Normalizar score: quanto MENOR (mais negativo), MAIS anômalo
    # Inverter para: quanto MAIOR, mais prioridade de fiscalização
    scaler_score = MinMaxScaler(feature_range=(0, 100))
    df_features['risk_score'] = scaler_score.fit_transform(
        -df_features[['anomaly_score_raw']]
    )

    return df_features, iso_forest, scaler

def executar_clustering(df_features, feature_cols, n_clusters=5):
    """Executa K-Means para agrupar contribuintes por perfil."""
    X = df_features[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_features['cluster'] = kmeans.fit_predict(X_scaled)

    # PCA para visualização 2D
    if X_scaled.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        df_features['pca_x'] = coords[:, 0]
        df_features['pca_y'] = coords[:, 1]
    else:
        df_features['pca_x'] = X_scaled[:, 0]
        df_features['pca_y'] = X_scaled[:, 1] if X_scaled.shape[1] > 1 else 0

    return df_features, kmeans, scaler

def calcular_score_regras(df):
    """Score complementar baseado em regras de negócio fiscais."""
    score = pd.Series(0.0, index=df.index)

    # 1. Saldo de remessa sem retorno elevado (peso 25)
    if 'saldo_remessa_retorno' in df.columns:
        q90 = df['saldo_remessa_retorno'].quantile(0.90)
        if q90 > 0:
            score += np.where(df['saldo_remessa_retorno'] > q90, 25, 0)
            score += np.where(df['saldo_remessa_retorno'] > q90 * 0.5, 10, 0)

    # 2. Taxa de retorno muito baixa (peso 20)
    if 'pct_retorno' in df.columns:
        score += np.where(
            (df['vl_remessas'] > 0) & (df['pct_retorno'] < 50), 20, 0
        )
        score += np.where(
            (df['vl_remessas'] > 0) & (df['pct_retorno'] < 20), 10, 0
        )

    # 3. Volume elevado de operações (peso 15) — materialidade
    if 'vl_remessas' in df.columns:
        q80 = df['vl_remessas'].quantile(0.80)
        if q80 > 0:
            score += np.where(df['vl_remessas'] > q80, 15, 0)

    # 4. Alta concentração em poucos parceiros (peso 10)
    if 'concentracao_parceiros' in df.columns:
        q90 = df['concentracao_parceiros'].quantile(0.90)
        if q90 > 0:
            score += np.where(df['concentracao_parceiros'] > q90, 10, 0)

    # 5. Itens pendentes no painel (peso 15)
    if 'itens_pendentes' in df.columns:
        score += np.where(df['itens_pendentes'] > 0, 15, 0)
        score += np.where(df['itens_pendentes'] > 5, 10, 0)

    # 6. ICMS baixo relativo ao volume (peso 10)
    if 'icms_por_operacao' in df.columns and 'valor_medio_operacao' in df.columns:
        ratio = np.where(
            df['valor_medio_operacao'] > 0,
            df['icms_por_operacao'] / df['valor_medio_operacao'],
            0
        )
        score += np.where(ratio < 0.01, 10, 0)

    # Normalizar 0-100
    max_score = score.max()
    if max_score > 0:
        score = score / max_score * 100

    return score

# =============================================================================
# ESTILOS CSS
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp { font-family: 'Source Sans Pro', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 50%, #3182ce 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(26, 54, 93, 0.3);
    }
    .main-header h1 { color: white; font-size: 2rem; font-weight: 700; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.85); font-size: 1rem; margin: 0.5rem 0 0 0; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #1e6e3c;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-card.blue { border-left-color: #3182ce; }
    .metric-card.orange { border-left-color: #dd6b20; }
    .metric-card.purple { border-left-color: #805ad5; }
    .metric-card.red { border-left-color: #e53e3e; }
    .metric-card.teal { border-left-color: #319795; }

    .metric-label {
        font-size: 0.8rem;
        color: #718096;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a365d;
        margin-top: 0.25rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .section-header {
        background: #f7fafc;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #1e6e3c;
    }
    .section-header h2 {
        color: #1a365d;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0;
    }

    .info-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 0.75rem;
        border: 1px solid #e2e8f0;
    }
    .info-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-color: #3182ce;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-success { background: #c6f6d5; color: #22543d; }
    .badge-warning { background: #fefcbf; color: #744210; }
    .badge-danger { background: #fed7d7; color: #742a2a; }
    .badge-info { background: #bee3f8; color: #2a4365; }

    .contribuinte-box {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #90cdf4;
        margin-bottom: 1rem;
    }

    .fluxo-box {
        background: #f7fafc;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.8;
    }

    .status-ok { color: #22543d; background: #c6f6d5; padding: 2px 8px; border-radius: 4px; }
    .status-pendente { color: #742a2a; background: #fed7d7; padding: 2px 8px; border-radius: 4px; }
    .status-excesso { color: #744210; background: #fefcbf; padding: 2px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
if "cnpj_selecionado" not in st.session_state:
    st.session_state.cnpj_selecionado = ""
if "id_arquivo_selecionado" not in st.session_state:
    st.session_state.id_arquivo_selecionado = ""
if "dados_cadastrais" not in st.session_state:
    st.session_state.dados_cadastrais = None

# =============================================================================
# HEADER PRINCIPAL
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>🏭 GESTEX V014 — Industrialização por Encomenda</h1>
    <p>Análise de Remessas, Retornos, Produção e Anomalias | SEF/SC — 2025</p>
</div>
""", unsafe_allow_html=True)

# Diagrama do fluxo de industrialização (colapsado)
with st.expander("📖 Fluxo de Industrialização por Encomenda — Referência"):
    st.markdown("""
    <div class="fluxo-box">
    <strong>ENCOMENDANTE</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>INDUSTRIALIZADOR</strong><br>
    &nbsp;&nbsp;&nbsp; │ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │<br>
    &nbsp;&nbsp;&nbsp; │── CFOP 5901/6901 (Remessa insumos) ──────►│ &nbsp;Recebe insumos<br>
    &nbsp;&nbsp;&nbsp; │ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; │ &nbsp;Processa (K230/K250)<br>
    &nbsp;&nbsp;&nbsp; │◄── CFOP 5902/6902 (Retorno produto) ──────│ &nbsp;Devolve produto<br>
    &nbsp;&nbsp;&nbsp; │◄── CFOP 5124/6124 (Cobrança serviço) ─────│ &nbsp;Cobra industrialização<br>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Remessas:** 5901 / 6901")
        st.markdown("**Retornos:** 5902 / 6902 / 5903 / 6903")
    with col2:
        st.markdown("**Cobrança:** 5124 / 6124 / 5125 / 6125")
        st.markdown("**Produção própria:** K230 + K235")
    with col3:
        st.markdown("**Produção terceiros:** K250 + K255")
        st.markdown("**Estoque terceiros:** K200")

# =============================================================================
# ABAS PRINCIPAIS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍 Consulta por Contribuinte",
    "📊 Fluxo de Industrialização",
    "⚖️ Remessa vs Retorno",
    "🚨 Anomalias",
    "🏭 Produção e Estoques",
    "📈 Análise Exploratória",
    "🤖 Fiscalização Inteligente"
])

# =============================================================================
# ABA 1: CONSULTA POR CONTRIBUINTE
# =============================================================================
with tab1:
    st.markdown('<div class="section-header"><h2>🔍 Consulta por Contribuinte</h2></div>', unsafe_allow_html=True)

    col_input1, col_input2 = st.columns([2, 1])
    with col_input1:
        cnpj_input = st.text_input(
            "Informe o CNPJ (14 dígitos) ou IE (9 dígitos):",
            value=st.session_state.cnpj_selecionado,
            placeholder="00.000.000/0000-00 ou 000000000",
            key="cnpj_ie_input"
        )
    with col_input2:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_buscar = st.button("🔍 Buscar Contribuinte", use_container_width=True, type="primary")

    if btn_buscar and cnpj_input:
        cnpj_limpo = cnpj_input.replace(".", "").replace("/", "").replace("-", "").strip()
        st.session_state.cnpj_selecionado = cnpj_limpo

        with st.spinner("Buscando dados do contribuinte..."):
            # Buscar identificadorarquivo
            df_ids = buscar_identificadores(cnpj_limpo)
            if not df_ids.empty:
                id_arquivo = str(df_ids.iloc[0]['identificadorarquivo'])
                st.session_state.id_arquivo_selecionado = id_arquivo

                # Buscar dados cadastrais
                df_cad = buscar_dados_cadastrais(cnpj_limpo)
                if not df_cad.empty:
                    st.session_state.dados_cadastrais = df_cad.iloc[0]
            else:
                st.warning("⚠️ Nenhum registro encontrado para este CNPJ/IE no ano 2025.")
                st.session_state.id_arquivo_selecionado = ""
                st.session_state.dados_cadastrais = None

    # Exibir dados se contribuinte selecionado
    if st.session_state.id_arquivo_selecionado:
        cad = st.session_state.dados_cadastrais
        id_arq = st.session_state.id_arquivo_selecionado

        if cad is not None:
            razao = cad.get('nomeempresarial', 'N/D')
            fantasia = cad.get('nomefantasia', '')
            cnpj_display = cad.get('codigocnpj', 'N/D')
            ie_display = cad.get('inscricaoestadual', 'N/D')
            cod_mun = cad.get('cod_municipio', 'N/D')

            st.markdown(f"""
            <div class="contribuinte-box">
                <h3 style="margin:0; color:#1a365d;">{razao}</h3>
                {'<p style="margin:0; color:#4a5568; font-style:italic;">' + fantasia + '</p>' if fantasia else ''}
                <p style="margin:0.25rem 0; color:#4a5568;">
                    <strong>CNPJ:</strong> {cnpj_display} &nbsp;|&nbsp;
                    <strong>IE:</strong> {ie_display} &nbsp;|&nbsp;
                    <strong>Cód. Município:</strong> {cod_mun}
                </p>
                <p style="margin:0; font-size:0.8rem; color:#718096;">
                    Identificador arquivo: {id_arq}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Cards com métricas rápidas do contribuinte
        st.markdown('<div class="section-header"><h2>📊 Métricas Rápidas</h2></div>', unsafe_allow_html=True)

        with st.spinner("Carregando métricas..."):
            df_fluxo = carregar_fluxo_completo()
            df_painel = carregar_painel_consolidado()
            df_anomalia = carregar_anomalia_sem_retorno()

            cnpj_limpo = st.session_state.cnpj_selecionado

            # Filtrar por contribuinte: como DECLARANTE (identificadorarquivo) OU como PARTICIPANTE (cnpj_participante)
            if not df_fluxo.empty:
                mask_declarante = df_fluxo['identificadorarquivo'].astype(str) == id_arq
                mask_participante = df_fluxo['cnpj_participante'].astype(str) == cnpj_limpo
                df_fluxo_contrib = df_fluxo[mask_declarante | mask_participante].copy()
                # Guardar no session_state para uso nas outras abas
                st.session_state['df_fluxo_contrib'] = df_fluxo_contrib
            else:
                df_fluxo_contrib = pd.DataFrame()
                st.session_state['df_fluxo_contrib'] = df_fluxo_contrib

            if not df_painel.empty:
                mask_p_decl = df_painel['identificadorarquivo'].astype(str) == id_arq
                mask_p_part = df_painel['cod_part'].astype(str) == cnpj_limpo if 'cod_part' in df_painel.columns else pd.Series(False, index=df_painel.index)
                df_painel_contrib = df_painel[mask_p_decl | mask_p_part].copy()
            else:
                df_painel_contrib = pd.DataFrame()

            # Métricas
            total_operacoes = len(df_fluxo_contrib) if not df_fluxo_contrib.empty else 0

            vl_remessas = 0
            vl_retornos = 0
            if not df_fluxo_contrib.empty:
                remessa_mask = df_fluxo_contrib['cfop'].isin([5901, 6901])
                retorno_mask = df_fluxo_contrib['cfop'].isin([5902, 6902, 5903, 6903])
                vl_remessas = df_fluxo_contrib.loc[remessa_mask, 'vl_item'].sum()
                vl_retornos = df_fluxo_contrib.loc[retorno_mask, 'vl_item'].sum()

            saldo_pendente = float(df_painel_contrib['saldo_valor'].sum()) if not df_painel_contrib.empty else 0

            # Anomalias
            qtd_anomalias = 0
            if not df_anomalia.empty and not df_fluxo_contrib.empty:
                chaves_contrib = set(df_fluxo_contrib['chv_nfe'].dropna().unique()) if 'chv_nfe' in df_fluxo_contrib.columns else set()
                if chaves_contrib:
                    qtd_anomalias = len(df_anomalia[df_anomalia['chv_nfe'].isin(chaves_contrib)])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Operações</div>
                <div class="metric-value">{formatar_numero(total_operacoes)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="metric-label">Valor Remessas</div>
                <div class="metric-value">{formatar_moeda_abrev(vl_remessas)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card teal">
                <div class="metric-label">Valor Retornos</div>
                <div class="metric-value">{formatar_moeda_abrev(vl_retornos)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            cor_saldo = "red" if saldo_pendente > 0 else ""
            st.markdown(f"""
            <div class="metric-card {cor_saldo}">
                <div class="metric-label">Saldo Pendente</div>
                <div class="metric-value">{formatar_moeda_abrev(saldo_pendente)}</div>
            </div>
            """, unsafe_allow_html=True)

        # Segunda linha de métricas
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            qtd_parceiros = df_fluxo_contrib['cod_part'].nunique() if not df_fluxo_contrib.empty else 0
            st.markdown(f"""
            <div class="metric-card purple">
                <div class="metric-label">Parceiros Comerciais</div>
                <div class="metric-value">{formatar_numero(qtd_parceiros)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col6:
            qtd_itens = df_fluxo_contrib['cod_item'].nunique() if not df_fluxo_contrib.empty else 0
            st.markdown(f"""
            <div class="metric-card orange">
                <div class="metric-label">Itens Distintos</div>
                <div class="metric-value">{formatar_numero(qtd_itens)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col7:
            qtd_notas = df_fluxo_contrib['chv_nfe'].nunique() if not df_fluxo_contrib.empty else 0
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="metric-label">Notas Fiscais</div>
                <div class="metric-value">{formatar_numero(qtd_notas)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col8:
            st.markdown(f"""
            <div class="metric-card {'red' if qtd_anomalias > 0 else ''}">
                <div class="metric-label">Anomalias Detectadas</div>
                <div class="metric-value">{formatar_numero(qtd_anomalias)}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("ℹ️ Informe um CNPJ ou IE e clique em **Buscar** para visualizar os dados do contribuinte.")

# =============================================================================
# ABA 2: FLUXO DE INDUSTRIALIZAÇÃO
# =============================================================================
with tab2:
    st.markdown('<div class="section-header"><h2>📊 Fluxo de Industrialização</h2></div>', unsafe_allow_html=True)

    df_fluxo = carregar_fluxo_completo()

    if df_fluxo.empty:
        st.warning("⚠️ Não foi possível carregar o fluxo completo.")
    else:
        # Filtros
        col_f1, col_f2, col_f3 = st.columns(3)

        # Filtrar por contribuinte se selecionado
        id_arq = st.session_state.id_arquivo_selecionado
        cnpj_limpo = st.session_state.cnpj_selecionado
        if id_arq:
            mask_decl = df_fluxo['identificadorarquivo'].astype(str) == id_arq
            mask_part = df_fluxo['cnpj_participante'].astype(str) == cnpj_limpo if cnpj_limpo else pd.Series(False, index=df_fluxo.index)
            df_fluxo_filtro = df_fluxo[mask_decl | mask_part].copy()
            st.info(f"📌 Filtrado pelo contribuinte selecionado — como declarante (ID: {id_arq}) e como participante (CNPJ: {cnpj_limpo})")
        else:
            df_fluxo_filtro = df_fluxo.copy()
            st.info("💡 Selecione um contribuinte na **Aba 1** para filtrar, ou visualize todos os dados.")

        with col_f1:
            cfops_disponiveis = sorted(df_fluxo_filtro['cfop'].dropna().unique().tolist())
            cfop_sel = st.multiselect(
                "Filtrar por CFOP:",
                options=cfops_disponiveis,
                format_func=cfop_label,
                key="fluxo_cfop"
            )
        with col_f2:
            parceiros_disp = sorted(df_fluxo_filtro['nome_participante'].dropna().unique().tolist())
            parceiro_sel = st.multiselect("Filtrar por Participante:", parceiros_disp, key="fluxo_parceiro")
        with col_f3:
            meses_disp = sorted(df_fluxo_filtro['mes_referencia'].dropna().unique().tolist())
            mes_sel = st.multiselect("Filtrar por Mês:", meses_disp, key="fluxo_mes")

        # Aplicar filtros
        df_view = df_fluxo_filtro.copy()
        if cfop_sel:
            df_view = df_view[df_view['cfop'].isin(cfop_sel)]
        if parceiro_sel:
            df_view = df_view[df_view['nome_participante'].isin(parceiro_sel)]
        if mes_sel:
            df_view = df_view[df_view['mes_referencia'].isin(mes_sel)]

        st.markdown(f"**{formatar_numero(len(df_view))} registros encontrados**")

        # Gráficos
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            # Valor por CFOP
            if not df_view.empty:
                df_cfop_agg = df_view.groupby('cfop').agg(
                    valor_total=('vl_item', 'sum'),
                    qtd_registros=('vl_item', 'count')
                ).reset_index()
                df_cfop_agg['cfop_label'] = df_cfop_agg['cfop'].apply(cfop_label)

                fig_cfop = px.bar(
                    df_cfop_agg.sort_values('valor_total', ascending=True),
                    y='cfop_label',
                    x='valor_total',
                    orientation='h',
                    color='valor_total',
                    color_continuous_scale='Blues',
                    title="Valor Total por CFOP"
                )
                fig_cfop.update_layout(
                    yaxis_title="", xaxis_title="Valor (R$)",
                    showlegend=False, coloraxis_showscale=False,
                    height=400, xaxis_tickformat=",.0f"
                )
                st.plotly_chart(fig_cfop, use_container_width=True)

        with col_g2:
            # Evolução mensal
            if not df_view.empty and 'mes_referencia' in df_view.columns:
                df_mensal = df_view.groupby('mes_referencia').agg(
                    valor_total=('vl_item', 'sum'),
                    qtd_operacoes=('vl_item', 'count')
                ).reset_index().sort_values('mes_referencia')

                fig_mensal = go.Figure()
                fig_mensal.add_trace(go.Bar(
                    x=df_mensal['mes_referencia'],
                    y=df_mensal['valor_total'],
                    name='Valor',
                    marker_color='#3182ce',
                    text=[formatar_moeda_abrev(v) for v in df_mensal['valor_total']],
                    textposition='outside'
                ))
                fig_mensal.update_layout(
                    title="Evolução Mensal — Valor",
                    xaxis_title="Mês", yaxis_title="Valor (R$)",
                    height=400, xaxis_tickformat="d", yaxis_tickformat=",.0f"
                )
                st.plotly_chart(fig_mensal, use_container_width=True)

        # Top participantes
        if not df_view.empty:
            st.markdown('<div class="section-header"><h2>🤝 Top Participantes (Industrializadores / Encomendantes)</h2></div>', unsafe_allow_html=True)

            df_parceiro_agg = df_view.groupby(['cod_part', 'nome_participante', 'cnpj_participante']).agg(
                valor_total=('vl_item', 'sum'),
                qtd_operacoes=('vl_item', 'count'),
                qtd_itens=('cod_item', 'nunique')
            ).reset_index().sort_values('valor_total', ascending=False)

            fig_parceiros = px.bar(
                df_parceiro_agg.head(15),
                y='nome_participante',
                x='valor_total',
                orientation='h',
                color='qtd_operacoes',
                color_continuous_scale='Viridis',
                title="Top 15 Participantes por Valor"
            )
            fig_parceiros.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Valor Total (R$)", yaxis_title="",
                height=450, xaxis_tickformat=",.0f",
                coloraxis_colorbar_title="Operações"
            )
            st.plotly_chart(fig_parceiros, use_container_width=True)

        # Tabela detalhada
        with st.expander("📋 Ver tabela completa do fluxo"):
            cols_display = ['dt_doc', 'cfop', 'ind_oper', 'nome_participante', 'cnpj_participante',
                           'cod_item', 'descr_item', 'qtd', 'vl_item', 'aliq_icms', 'vl_icms']
            cols_exist = [c for c in cols_display if c in df_view.columns]
            st.dataframe(df_view[cols_exist].head(500), use_container_width=True)

            st.download_button(
                "📥 Exportar Fluxo (Excel)",
                data=to_excel(df_view[cols_exist]),
                file_name="gestex_v014_fluxo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# =============================================================================
# ABA 3: PAINEL REMESSA vs RETORNO
# =============================================================================
with tab3:
    st.markdown('<div class="section-header"><h2>⚖️ Painel Consolidado — Remessa vs Retorno</h2></div>', unsafe_allow_html=True)

    df_painel = carregar_painel_consolidado()

    if df_painel.empty:
        st.warning("⚠️ Não foi possível carregar o painel consolidado.")
    else:
        # Filtrar por contribuinte se selecionado
        id_arq = st.session_state.id_arquivo_selecionado
        cnpj_limpo = st.session_state.cnpj_selecionado
        if id_arq:
            mask_decl = df_painel['identificadorarquivo'].astype(str) == id_arq
            # cod_part pode ser código interno, não CNPJ — manter filtro por identificadorarquivo
            df_painel_view = df_painel[mask_decl].copy()
            # Se não achou como declarante, tentar tabela inteira (contribuinte só como participante)
            if df_painel_view.empty:
                df_painel_view = df_painel.copy()
                st.info("💡 Contribuinte não encontrado como declarante no painel. Exibindo todos os registros.")
            else:
                st.info(f"📌 Filtrado pelo contribuinte (ID: {id_arq})")
        else:
            df_painel_view = df_painel.copy()

        if df_painel_view.empty:
            st.info("Nenhum dado encontrado para este contribuinte no painel consolidado.")
        else:
            # Métricas resumo
            total_remetido = df_painel_view['vl_remetido'].sum()
            total_retornado = df_painel_view['vl_retornado'].sum()
            total_industrializacao = df_painel_view['vl_industrializacao'].sum()
            saldo_total = df_painel_view['saldo_valor'].sum()

            qtd_ok = len(df_painel_view[df_painel_view['saldo_qtd'] == 0])
            qtd_pendente = len(df_painel_view[df_painel_view['saldo_qtd'] > 0])
            qtd_excesso = len(df_painel_view[df_painel_view['saldo_qtd'] < 0])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card blue">
                    <div class="metric-label">Total Remetido</div>
                    <div class="metric-value">{formatar_moeda_abrev(total_remetido)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card teal">
                    <div class="metric-label">Total Retornado</div>
                    <div class="metric-value">{formatar_moeda_abrev(total_retornado)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card orange">
                    <div class="metric-label">Industrialização</div>
                    <div class="metric-value">{formatar_moeda_abrev(total_industrializacao)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                cor = "red" if saldo_total > 0 else ""
                st.markdown(f"""
                <div class="metric-card {cor}">
                    <div class="metric-label">Saldo Pendente</div>
                    <div class="metric-value">{formatar_moeda_abrev(saldo_total)}</div>
                </div>
                """, unsafe_allow_html=True)

            # Status badges
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.markdown(f'<span class="badge badge-success">✅ {qtd_ok} itens OK (saldo = 0)</span>', unsafe_allow_html=True)
            with col_b2:
                st.markdown(f'<span class="badge badge-danger">🔴 {qtd_pendente} itens pendentes (saldo > 0)</span>', unsafe_allow_html=True)
            with col_b3:
                st.markdown(f'<span class="badge badge-warning">⚠️ {qtd_excesso} itens c/ excesso (saldo < 0)</span>', unsafe_allow_html=True)

            st.markdown("---")

            # Gráfico Remessa vs Retorno por parceiro
            df_parceiro_painel = df_painel_view.groupby('cod_part').agg(
                remetido=('vl_remetido', 'sum'),
                retornado=('vl_retornado', 'sum'),
                industrializado=('vl_industrializacao', 'sum'),
                saldo=('saldo_valor', 'sum')
            ).reset_index().sort_values('remetido', ascending=False).head(15)

            if not df_parceiro_painel.empty:
                fig_painel = go.Figure()
                fig_painel.add_trace(go.Bar(
                    name='Remetido', y=df_parceiro_painel['cod_part'], x=df_parceiro_painel['remetido'],
                    orientation='h', marker_color='#3182ce'
                ))
                fig_painel.add_trace(go.Bar(
                    name='Retornado', y=df_parceiro_painel['cod_part'], x=df_parceiro_painel['retornado'],
                    orientation='h', marker_color='#38a169'
                ))
                fig_painel.add_trace(go.Bar(
                    name='Industrialização', y=df_parceiro_painel['cod_part'], x=df_parceiro_painel['industrializado'],
                    orientation='h', marker_color='#dd6b20'
                ))
                fig_painel.update_layout(
                    barmode='group',
                    title="Top 15 Parceiros — Remessa vs Retorno vs Industrialização",
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis_title="Valor (R$)", yaxis_title="Parceiro",
                    height=500, xaxis_tickformat=",.0f"
                )
                st.plotly_chart(fig_painel, use_container_width=True)

            # Tabela com cores condicionais
            with st.expander("📋 Tabela Detalhada — Painel Consolidado"):
                df_display = df_painel_view.copy()

                def status_saldo(row):
                    s = row.get('saldo_qtd', 0)
                    if pd.isna(s):
                        return "N/D"
                    if s == 0:
                        return "✅ OK"
                    elif s > 0:
                        return "🔴 Pendente"
                    else:
                        return "⚠️ Excesso"

                df_display['status'] = df_display.apply(status_saldo, axis=1)

                cols_painel = ['cod_part', 'cod_item', 'qtd_remetida', 'vl_remetido',
                              'qtd_retornada', 'vl_retornado', 'qtd_industrializada',
                              'vl_industrializacao', 'saldo_qtd', 'saldo_valor', 'status']
                cols_exist = [c for c in cols_painel if c in df_display.columns]
                st.dataframe(df_display[cols_exist], use_container_width=True)

                st.download_button(
                    "📥 Exportar Painel (Excel)",
                    data=to_excel(df_display[cols_exist]),
                    file_name="gestex_v014_painel.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# =============================================================================
# ABA 4: ANOMALIAS
# =============================================================================
with tab4:
    st.markdown('<div class="section-header"><h2>🚨 Detecção de Anomalias</h2></div>', unsafe_allow_html=True)

    sub_tab_4_1, sub_tab_4_2 = st.tabs([
        "📦 Remessas sem Retorno",
        "🔧 Produção sem Remessa"
    ])

    # --- Subtab 4.1: Remessas sem Retorno ---
    with sub_tab_4_1:
        df_anomalia = carregar_anomalia_sem_retorno()

        if df_anomalia.empty:
            st.info("Nenhuma anomalia de remessa sem retorno encontrada.")
        else:
            # Filtros
            col_af1, col_af2, col_af3 = st.columns(3)
            with col_af1:
                parceiros_anom = sorted(df_anomalia['cod_part'].dropna().unique().tolist())
                parceiro_anom_sel = st.multiselect("Filtrar por Parceiro:", parceiros_anom, key="anom_parceiro")
            with col_af2:
                vl_min = st.number_input("Valor mínimo pendente (R$):", min_value=0.0, value=0.0, step=100.0, key="anom_vl_min")
            with col_af3:
                qtd_min = st.number_input("Qtd mínima pendente:", min_value=0.0, value=0.0, step=1.0, key="anom_qtd_min")

            df_anom_view = df_anomalia.copy()
            if parceiro_anom_sel:
                df_anom_view = df_anom_view[df_anom_view['cod_part'].isin(parceiro_anom_sel)]
            if vl_min > 0:
                df_anom_view = df_anom_view[df_anom_view['vl_remetido'] >= vl_min]
            if qtd_min > 0:
                df_anom_view = df_anom_view[df_anom_view['qtd_pendente'] >= qtd_min]

            # Métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card red">
                    <div class="metric-label">Total Pendente</div>
                    <div class="metric-value">{formatar_moeda_abrev(df_anom_view['vl_remetido'].sum())}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card orange">
                    <div class="metric-label">Itens Pendentes</div>
                    <div class="metric-value">{formatar_numero(len(df_anom_view))}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card purple">
                    <div class="metric-label">Parceiros Envolvidos</div>
                    <div class="metric-value">{formatar_numero(df_anom_view['cod_part'].nunique())}</div>
                </div>
                """, unsafe_allow_html=True)

            # Tabela
            st.dataframe(df_anom_view.head(500), use_container_width=True)

            st.download_button(
                "📥 Exportar Anomalias (Excel)",
                data=to_excel(df_anom_view),
                file_name="gestex_v014_anomalia_sem_retorno.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # --- Subtab 4.2: Produção sem Remessa (PESADA) ---
    with sub_tab_4_2:
        id_arq = st.session_state.id_arquivo_selecionado
        if not id_arq:
            st.warning("⚠️ Selecione um contribuinte na **Aba 1** para carregar dados de produção sem remessa (tabela pesada).")
        else:
            st.info(f"📌 Contribuinte selecionado — ID: {id_arq}")

            if st.button("🔍 Carregar Produção sem Remessa", key="btn_prod_sem_remessa"):
                with st.spinner("Carregando dados (tabela pesada, pode demorar)..."):
                    df_prod_anom = carregar_anomalia_producao_sem_remessa(id_arq)

                if df_prod_anom.empty:
                    st.success("✅ Nenhuma anomalia de produção sem remessa detectada para este contribuinte.")
                else:
                    # Filtrar por status
                    status_vals = df_prod_anom['status'].dropna().unique().tolist() if 'status' in df_prod_anom.columns else []
                    status_sel = st.multiselect("Filtrar por Status:", status_vals, default=status_vals, key="prod_anom_status")

                    df_prod_view = df_prod_anom.copy()
                    if status_sel:
                        df_prod_view = df_prod_view[df_prod_view['status'].isin(status_sel)]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Registros", formatar_numero(len(df_prod_view)))
                    with col2:
                        st.metric("Insumos distintos", formatar_numero(df_prod_view['cod_insumo'].nunique()) if 'cod_insumo' in df_prod_view.columns else "0")

                    st.dataframe(df_prod_view.head(500), use_container_width=True)

                    st.download_button(
                        "📥 Exportar (Excel)",
                        data=to_excel(df_prod_view),
                        file_name="gestex_v014_producao_sem_remessa.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

# =============================================================================
# ABA 5: PRODUÇÃO E ESTOQUES (PESADAS - precisa contribuinte)
# =============================================================================
with tab5:
    st.markdown('<div class="section-header"><h2>🏭 Produção e Estoques</h2></div>', unsafe_allow_html=True)

    id_arq = st.session_state.id_arquivo_selecionado
    cnpj_sel = st.session_state.cnpj_selecionado

    if not id_arq:
        st.warning("⚠️ Selecione um contribuinte na **Aba 1** para acessar dados de produção e estoques (tabelas pesadas com milhões de registros).")
    else:
        st.info(f"📌 Contribuinte selecionado — ID: {id_arq}")

        sub5_1, sub5_2, sub5_3, sub5_4, sub5_5 = st.tabs([
            "📋 Produção K230",
            "📐 Ficha Técnica",
            "📦 Inventário H010",
            "📈 Estoque K200",
            "🧾 Ajustes E111"
        ])

        # --- Subtab 5.1: Produção K230 ---
        with sub5_1:
            if st.button("🔍 Carregar Produção K230", key="btn_k230"):
                with st.spinner("Carregando produção K230..."):
                    df_k230 = carregar_producao_descrita(id_arq)

                if df_k230.empty:
                    st.info("Nenhum registro de produção K230 encontrado.")
                else:
                    st.success(f"✅ {formatar_numero(len(df_k230))} registros carregados")

                    # Resumo
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        qtd_ordens = df_k230['cod_doc_op'].nunique() if 'cod_doc_op' in df_k230.columns else 0
                        st.metric("Ordens de Produção", formatar_numero(qtd_ordens))
                    with col2:
                        qtd_prod = df_k230['cod_produzido'].nunique() if 'cod_produzido' in df_k230.columns else 0
                        st.metric("Produtos Distintos", formatar_numero(qtd_prod))
                    with col3:
                        qtd_ins = df_k230['cod_insumo'].nunique() if 'cod_insumo' in df_k230.columns else 0
                        st.metric("Insumos Distintos", formatar_numero(qtd_ins))

                    # Top produtos
                    if 'desc_produzido' in df_k230.columns and 'qtd_produzida' in df_k230.columns:
                        df_top_prod = df_k230.groupby(['cod_produzido', 'desc_produzido']).agg(
                            total_produzido=('qtd_produzida', 'sum')
                        ).reset_index().sort_values('total_produzido', ascending=False).head(15)

                        fig_prod = px.bar(
                            df_top_prod,
                            y='desc_produzido', x='total_produzido',
                            orientation='h', color='total_produzido',
                            color_continuous_scale='Greens',
                            title="Top 15 Produtos por Quantidade Produzida"
                        )
                        fig_prod.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            yaxis_title="", xaxis_title="Qtd Produzida",
                            height=450, coloraxis_showscale=False
                        )
                        st.plotly_chart(fig_prod, use_container_width=True)

                    with st.expander("📋 Tabela completa K230"):
                        st.dataframe(df_k230.head(500), use_container_width=True)
                        st.download_button(
                            "📥 Exportar K230 (Excel)", data=to_excel(df_k230),
                            file_name="gestex_v014_producao_k230.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_k230"
                        )

        # --- Subtab 5.2: Ficha Técnica ---
        with sub5_2:
            if st.button("🔍 Carregar Ficha Técnica", key="btn_ficha"):
                with st.spinner("Carregando ficha técnica implícita..."):
                    df_ficha = carregar_ficha_tecnica(id_arq)

                if df_ficha.empty:
                    st.info("Nenhuma ficha técnica encontrada.")
                else:
                    st.success(f"✅ {formatar_numero(len(df_ficha))} combinações produto-insumo")

                    # Filtro por produto
                    if 'desc_produzido' in df_ficha.columns:
                        prods_ficha = sorted(df_ficha['desc_produzido'].dropna().unique().tolist())
                        prod_ficha_sel = st.selectbox("Selecione o produto:", ["Todos"] + prods_ficha, key="ficha_prod")

                        df_ficha_view = df_ficha.copy()
                        if prod_ficha_sel != "Todos":
                            df_ficha_view = df_ficha_view[df_ficha_view['desc_produzido'] == prod_ficha_sel]

                        # Gráfico de razão insumo/produto
                        if 'razao_insumo_por_produto' in df_ficha_view.columns:
                            df_ficha_top = df_ficha_view.sort_values('total_insumo_consumido', ascending=False).head(20)

                            fig_ficha = px.bar(
                                df_ficha_top,
                                y='desc_insumo' if 'desc_insumo' in df_ficha_top.columns else 'cod_insumo',
                                x='razao_insumo_por_produto',
                                orientation='h',
                                color='razao_insumo_por_produto',
                                color_continuous_scale='Oranges',
                                title="Razão Insumo/Produto (Top 20)"
                            )
                            fig_ficha.update_layout(
                                yaxis={'categoryorder': 'total ascending'},
                                yaxis_title="Insumo", xaxis_title="Razão (Insumo / Produto)",
                                height=500, coloraxis_showscale=False
                            )
                            st.plotly_chart(fig_ficha, use_container_width=True)

                        st.dataframe(df_ficha_view, use_container_width=True)
                        st.download_button(
                            "📥 Exportar Ficha (Excel)", data=to_excel(df_ficha_view),
                            file_name="gestex_v014_ficha_tecnica.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_ficha"
                        )
                    else:
                        st.dataframe(df_ficha, use_container_width=True)

        # --- Subtab 5.3: Inventário H010 ---
        with sub5_3:
            if st.button("🔍 Carregar Inventário H010", key="btn_inv"):
                with st.spinner("Carregando inventário..."):
                    df_inv = carregar_inventario(id_arq)

                if df_inv.empty:
                    st.info("Nenhum inventário encontrado.")
                else:
                    st.success(f"✅ {formatar_numero(len(df_inv))} itens no inventário")

                    # Resumo por propriedade
                    if 'propriedade_desc' in df_inv.columns and 'vl_item' in df_inv.columns:
                        df_inv_resumo = df_inv.groupby('propriedade_desc').agg(
                            qtd_itens=('cod_item', 'nunique'),
                            valor_total=('vl_item', 'sum'),
                            qtd_registros=('vl_item', 'count')
                        ).reset_index()

                        col1, col2 = st.columns(2)
                        with col1:
                            fig_inv_prop = px.pie(
                                df_inv_resumo, values='valor_total', names='propriedade_desc',
                                color_discrete_sequence=['#38a169', '#3182ce', '#dd6b20'],
                                title="Valor do Inventário por Propriedade", hole=0.4
                            )
                            fig_inv_prop.update_layout(height=350)
                            st.plotly_chart(fig_inv_prop, use_container_width=True)

                        with col2:
                            for _, row in df_inv_resumo.iterrows():
                                st.markdown(f"""
                                <div class="info-card">
                                    <strong>{row['propriedade_desc']}</strong><br>
                                    <span style="font-size:1.2rem; color:#1a365d; font-weight:700;">
                                        {formatar_moeda_abrev(row['valor_total'])}
                                    </span><br>
                                    <span style="font-size:0.85rem; color:#718096;">
                                        {formatar_numero(row['qtd_itens'])} itens distintos |
                                        {formatar_numero(row['qtd_registros'])} registros
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)

                    with st.expander("📋 Tabela completa do Inventário"):
                        st.dataframe(df_inv.head(500), use_container_width=True)
                        st.download_button(
                            "📥 Exportar Inventário (Excel)", data=to_excel(df_inv),
                            file_name="gestex_v014_inventario.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_inv"
                        )

        # --- Subtab 5.4: Estoque K200 ---
        with sub5_4:
            if st.button("🔍 Carregar Estoque Terceiros K200", key="btn_k200"):
                with st.spinner("Carregando estoque de terceiros..."):
                    df_est = carregar_estoque_terceiros(id_arq)

                if df_est.empty:
                    st.info("Nenhum estoque de terceiros encontrado.")
                else:
                    st.success(f"✅ {formatar_numero(len(df_est))} registros de estoque")

                    # Evolução mensal
                    if 'mes_referencia' in df_est.columns and 'qtd' in df_est.columns:
                        df_est_mensal = df_est.groupby('mes_referencia').agg(
                            qtd_total=('qtd', 'sum'),
                            itens_distintos=('cod_item', 'nunique'),
                            parceiros=('cod_part', 'nunique')
                        ).reset_index().sort_values('mes_referencia')

                        fig_est = go.Figure()
                        fig_est.add_trace(go.Scatter(
                            x=df_est_mensal['mes_referencia'], y=df_est_mensal['qtd_total'],
                            mode='lines+markers', name='Qtd Total',
                            line=dict(color='#3182ce', width=3),
                            marker=dict(size=8)
                        ))
                        fig_est.update_layout(
                            title="Evolução Mensal — Estoque de Terceiros",
                            xaxis_title="Mês", yaxis_title="Quantidade Total",
                            height=400, xaxis_tickformat="d"
                        )
                        st.plotly_chart(fig_est, use_container_width=True)

                    # Top proprietários
                    if 'nome_proprietario' in df_est.columns:
                        df_est_prop = df_est.groupby(['cod_part', 'nome_proprietario']).agg(
                            qtd_total=('qtd', 'sum'),
                            itens=('cod_item', 'nunique')
                        ).reset_index().sort_values('qtd_total', ascending=False).head(10)

                        st.markdown("#### 👤 Top Proprietários")
                        st.dataframe(df_est_prop, use_container_width=True)

                    with st.expander("📋 Tabela completa K200"):
                        st.dataframe(df_est.head(500), use_container_width=True)
                        st.download_button(
                            "📥 Exportar K200 (Excel)", data=to_excel(df_est),
                            file_name="gestex_v014_estoque_terceiros.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_k200"
                        )

        # --- Subtab 5.5: Ajustes E111 ---
        with sub5_5:
            if not cnpj_sel:
                st.warning("⚠️ Selecione um contribuinte na Aba 1.")
            else:
                if st.button("🔍 Carregar Ajustes E111", key="btn_e111"):
                    with st.spinner("Carregando ajustes de apuração..."):
                        df_e111 = carregar_ajustes_e111(cnpj_sel)

                    if df_e111.empty:
                        st.info("Nenhum ajuste E111 encontrado.")
                    else:
                        st.success(f"✅ {formatar_numero(len(df_e111))} registros de ajustes")

                        # Resumo apuração por mês
                        cols_apur = ['mes_referencia', 'vl_debitos', 'vl_creditos',
                                    'vl_aj_debitos', 'vl_aj_creditos', 'vl_recolher', 'vl_sld_credor']
                        cols_exist = [c for c in cols_apur if c in df_e111.columns]

                        if 'mes_referencia' in df_e111.columns and 'vl_recolher' in df_e111.columns:
                            df_apur = df_e111.drop_duplicates(subset=['mes_referencia', 'vl_recolher']).sort_values('mes_referencia')

                            # Gráfico débitos vs créditos
                            if 'vl_debitos' in df_apur.columns and 'vl_creditos' in df_apur.columns:
                                fig_apur = go.Figure()
                                fig_apur.add_trace(go.Bar(
                                    x=df_apur['mes_referencia'], y=df_apur['vl_debitos'],
                                    name='Débitos', marker_color='#e53e3e'
                                ))
                                fig_apur.add_trace(go.Bar(
                                    x=df_apur['mes_referencia'], y=df_apur['vl_creditos'],
                                    name='Créditos', marker_color='#38a169'
                                ))
                                fig_apur.add_trace(go.Scatter(
                                    x=df_apur['mes_referencia'], y=df_apur['vl_recolher'],
                                    name='ICMS a Recolher', mode='lines+markers',
                                    line=dict(color='#805ad5', width=3)
                                ))
                                fig_apur.update_layout(
                                    barmode='group',
                                    title="Apuração ICMS Mensal — Débitos vs Créditos",
                                    xaxis_title="Mês", yaxis_title="Valor (R$)",
                                    height=450, xaxis_tickformat="d", yaxis_tickformat=",.0f"
                                )
                                st.plotly_chart(fig_apur, use_container_width=True)

                        # Detalhamento dos ajustes
                        st.markdown("#### 📝 Detalhamento dos Ajustes")
                        cols_aj = ['mes_referencia', 'cod_aj_apur', 'descr_compl_aj', 'vl_aj_apur']
                        cols_exist_aj = [c for c in cols_aj if c in df_e111.columns]

                        if cols_exist_aj:
                            df_aj_view = df_e111[cols_exist_aj].dropna(subset=['cod_aj_apur'] if 'cod_aj_apur' in cols_exist_aj else [])
                            if not df_aj_view.empty:
                                # Top ajustes por valor
                                df_top_aj = df_aj_view.groupby(
                                    [c for c in ['cod_aj_apur', 'descr_compl_aj'] if c in df_aj_view.columns]
                                ).agg(
                                    valor_total=('vl_aj_apur', 'sum'),
                                    ocorrencias=('vl_aj_apur', 'count')
                                ).reset_index().sort_values('valor_total', ascending=False)

                                st.dataframe(df_top_aj.head(50), use_container_width=True)

                        with st.expander("📋 Tabela completa E111"):
                            st.dataframe(df_e111.head(500), use_container_width=True)
                            st.download_button(
                                "📥 Exportar E111 (Excel)", data=to_excel(df_e111),
                                file_name="gestex_v014_ajustes_e111.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="dl_e111"
                            )

# =============================================================================
# ABA 6: ANÁLISE EXPLORATÓRIA
# =============================================================================
with tab6:
    st.markdown('<div class="section-header"><h2>📈 Análise Exploratória — Visão Geral do Setor</h2></div>', unsafe_allow_html=True)
    st.caption("Dados agregados das tabelas `teste.gestex_v014_*` — carregamento rápido, sem dados individuais pesados.")

    sub6_1, sub6_2, sub6_3, sub6_4, sub6_5 = st.tabs([
        "🌐 Panorama Geral",
        "📅 Evolução Temporal",
        "🏢 Mapa de Participantes",
        "📦 NCMs e Produtos",
        "📊 Estoque e Inventário"
    ])

    # ----- Subtab 6.1: Panorama Geral -----
    with sub6_1:
        with st.spinner("Carregando panorama..."):
            df_panorama = carregar_panorama_contribuintes()
            df_fluxo_geral = carregar_fluxo_completo()

        if df_panorama.empty:
            st.warning("⚠️ Sem dados no panorama.")
        else:
            # KPIs Gerais
            st.markdown('<div class="section-header"><h2>📊 Indicadores Gerais — Industrialização por Encomenda 2025</h2></div>', unsafe_allow_html=True)

            total_contribuintes = df_panorama['identificadorarquivo'].nunique()
            total_operacoes_geral = df_panorama['total_operacoes'].sum()
            vl_total_remessas = df_panorama['vl_remessas'].sum()
            vl_total_retornos = df_panorama['vl_retornos'].sum()
            vl_total_industrializacao = df_panorama['vl_industrializacao'].sum()
            total_icms = df_panorama['total_icms'].sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Contribuintes Declarantes</div>
                    <div class="metric-value">{formatar_numero(total_contribuintes)}</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card blue">
                    <div class="metric-label">Total de Operações</div>
                    <div class="metric-value">{formatar_numero(total_operacoes_geral)}</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card purple">
                    <div class="metric-label">ICMS Total</div>
                    <div class="metric-value">{formatar_moeda_abrev(total_icms)}</div>
                </div>""", unsafe_allow_html=True)

            col4, col5, col6 = st.columns(3)
            with col4:
                st.markdown(f"""
                <div class="metric-card blue">
                    <div class="metric-label">Valor Remessas</div>
                    <div class="metric-value">{formatar_moeda_abrev(vl_total_remessas)}</div>
                </div>""", unsafe_allow_html=True)
            with col5:
                st.markdown(f"""
                <div class="metric-card teal">
                    <div class="metric-label">Valor Retornos</div>
                    <div class="metric-value">{formatar_moeda_abrev(vl_total_retornos)}</div>
                </div>""", unsafe_allow_html=True)
            with col6:
                st.markdown(f"""
                <div class="metric-card orange">
                    <div class="metric-label">Cobrança Industrialização</div>
                    <div class="metric-value">{formatar_moeda_abrev(vl_total_industrializacao)}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # Distribuição de valores — Histograma
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_dist = px.histogram(
                    df_panorama[df_panorama['vl_remessas'] > 0],
                    x='vl_remessas', nbins=30,
                    title="Distribuição — Valor de Remessas por Contribuinte",
                    color_discrete_sequence=['#3182ce']
                )
                fig_dist.update_layout(
                    xaxis_title="Valor Remessas (R$)", yaxis_title="Frequência",
                    height=380, xaxis_tickformat=",.0f"
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            with col_g2:
                # Top 15 contribuintes por valor
                df_top15 = df_panorama.nlargest(15, 'vl_remessas')
                label_col = 'nomeempresarial' if 'nomeempresarial' in df_top15.columns else 'identificadorarquivo'
                fig_top = px.bar(
                    df_top15, y=label_col, x='vl_remessas',
                    orientation='h', color='vl_retornos',
                    color_continuous_scale='Teal',
                    title="Top 15 Declarantes — Valor de Remessas"
                )
                fig_top.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    yaxis_title="", xaxis_title="Valor Remessas (R$)",
                    height=380, coloraxis_colorbar_title="Retornos",
                    xaxis_tickformat=",.0f"
                )
                st.plotly_chart(fig_top, use_container_width=True)

            # Scatter: Remessa vs Retorno
            st.markdown('<div class="section-header"><h2>🔄 Remessa vs Retorno — Equilíbrio por Contribuinte</h2></div>', unsafe_allow_html=True)

            df_scatter = df_panorama[(df_panorama['vl_remessas'] > 0) | (df_panorama['vl_retornos'] > 0)].copy()
            if not df_scatter.empty:
                fig_scatter = px.scatter(
                    df_scatter,
                    x='vl_remessas', y='vl_retornos',
                    size='total_operacoes',
                    color='total_parceiros',
                    hover_name='nomeempresarial' if 'nomeempresarial' in df_scatter.columns else 'identificadorarquivo',
                    color_continuous_scale='Viridis',
                    title="Cada ponto = 1 contribuinte declarante"
                )
                # Linha de equilíbrio (remessa = retorno)
                max_val = max(df_scatter['vl_remessas'].max(), df_scatter['vl_retornos'].max())
                fig_scatter.add_trace(go.Scatter(
                    x=[0, max_val], y=[0, max_val],
                    mode='lines', name='Equilíbrio (1:1)',
                    line=dict(dash='dash', color='red', width=1)
                ))
                fig_scatter.update_layout(
                    xaxis_title="Valor Remessas (R$)", yaxis_title="Valor Retornos (R$)",
                    height=500, xaxis_tickformat=",.0f", yaxis_tickformat=",.0f",
                    coloraxis_colorbar_title="Parceiros"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.caption("Pontos **abaixo** da linha vermelha = remessas > retornos (possível pendência).")

            # Tabela completa
            with st.expander("📋 Tabela — Todos os Contribuintes Declarantes"):
                cols_show = [c for c in ['nomeempresarial', 'codigocnpj', 'ie', 'cod_municipio',
                             'total_operacoes', 'total_notas', 'total_parceiros',
                             'vl_remessas', 'vl_retornos', 'vl_industrializacao', 'total_icms',
                             'meses_ativos'] if c in df_panorama.columns]
                st.dataframe(df_panorama[cols_show].sort_values('vl_remessas', ascending=False), use_container_width=True)
                st.download_button(
                    "📥 Exportar Panorama (Excel)", data=to_excel(df_panorama[cols_show]),
                    file_name="gestex_panorama_contribuintes.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_panorama"
                )

    # ----- Subtab 6.2: Evolução Temporal -----
    with sub6_2:
        with st.spinner("Carregando evolução temporal..."):
            df_temporal = carregar_resumo_cfop_mensal()

        if df_temporal.empty:
            st.warning("⚠️ Sem dados temporais.")
        else:
            st.markdown('<div class="section-header"><h2>📅 Evolução Mensal por Tipo de Operação</h2></div>', unsafe_allow_html=True)

            # Gráfico de linhas — Valor por tipo
            fig_tempo = px.line(
                df_temporal, x='mes_referencia', y='valor_total',
                color='tipo_operacao', markers=True,
                color_discrete_map={
                    'Remessa': '#3182ce', 'Retorno': '#38a169',
                    'Industrialização': '#dd6b20', 'Outros': '#805ad5'
                },
                title="Valor Mensal por Tipo de Operação"
            )
            fig_tempo.update_layout(
                xaxis_title="Mês", yaxis_title="Valor (R$)",
                height=450, xaxis_tickformat="d", yaxis_tickformat=",.0f",
                legend_title="Tipo"
            )
            st.plotly_chart(fig_tempo, use_container_width=True)

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                # Qtd operações
                fig_qtd = px.bar(
                    df_temporal, x='mes_referencia', y='qtd_operacoes',
                    color='tipo_operacao', barmode='group',
                    title="Quantidade de Operações por Mês",
                    color_discrete_map={
                        'Remessa': '#3182ce', 'Retorno': '#38a169',
                        'Industrialização': '#dd6b20', 'Outros': '#805ad5'
                    }
                )
                fig_qtd.update_layout(
                    xaxis_title="Mês", yaxis_title="Qtd",
                    height=380, xaxis_tickformat="d", legend_title="Tipo"
                )
                st.plotly_chart(fig_qtd, use_container_width=True)

            with col_t2:
                # Contribuintes ativos por mês
                fig_contrib = px.bar(
                    df_temporal, x='mes_referencia', y='contribuintes',
                    color='tipo_operacao', barmode='stack',
                    title="Contribuintes Ativos por Mês",
                    color_discrete_map={
                        'Remessa': '#3182ce', 'Retorno': '#38a169',
                        'Industrialização': '#dd6b20', 'Outros': '#805ad5'
                    }
                )
                fig_contrib.update_layout(
                    xaxis_title="Mês", yaxis_title="Contribuintes",
                    height=380, xaxis_tickformat="d", legend_title="Tipo"
                )
                st.plotly_chart(fig_contrib, use_container_width=True)

            # ICMS mensal
            df_icms_mes = df_temporal.groupby('mes_referencia')['icms_total'].sum().reset_index()
            fig_icms = go.Figure()
            fig_icms.add_trace(go.Bar(
                x=df_icms_mes['mes_referencia'], y=df_icms_mes['icms_total'],
                marker_color='#805ad5',
                text=[formatar_moeda_abrev(v) for v in df_icms_mes['icms_total']],
                textposition='outside'
            ))
            fig_icms.update_layout(
                title="ICMS Total Mensal — Operações de Industrialização",
                xaxis_title="Mês", yaxis_title="ICMS (R$)",
                height=380, xaxis_tickformat="d", yaxis_tickformat=",.0f"
            )
            st.plotly_chart(fig_icms, use_container_width=True)

    # ----- Subtab 6.3: Mapa de Participantes -----
    with sub6_3:
        with st.spinner("Carregando participantes..."):
            df_part = carregar_panorama_participantes()

        if df_part.empty:
            st.warning("⚠️ Sem dados de participantes.")
        else:
            st.markdown('<div class="section-header"><h2>🏢 Rede de Participantes — Industrializadores e Encomendantes</h2></div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Participantes</div>
                    <div class="metric-value">{formatar_numero(len(df_part))}</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card blue">
                    <div class="metric-label">Valor Movimentado</div>
                    <div class="metric-value">{formatar_moeda_abrev(df_part['vl_remessas'].sum() + df_part['vl_retornos'].sum())}</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card orange">
                    <div class="metric-label">Média Declarantes/Participante</div>
                    <div class="metric-value">{formatar_qtd(df_part['total_declarantes'].mean())}</div>
                </div>""", unsafe_allow_html=True)

            # Top participantes
            df_part_sorted = df_part.sort_values('vl_remessas', ascending=False)
            label_col_p = 'nome_participante' if 'nome_participante' in df_part_sorted.columns else 'cnpj_participante'

            fig_part = px.bar(
                df_part_sorted.head(20),
                y=label_col_p, x='vl_remessas',
                orientation='h', color='total_declarantes',
                color_continuous_scale='Oranges',
                title="Top 20 Participantes — Valor de Remessas Recebidas/Enviadas"
            )
            fig_part.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                yaxis_title="", xaxis_title="Valor (R$)",
                height=550, xaxis_tickformat=",.0f",
                coloraxis_colorbar_title="Declarantes"
            )
            st.plotly_chart(fig_part, use_container_width=True)

            # Concentração — Top participantes por qtd de declarantes
            st.markdown('<div class="section-header"><h2>🔗 Concentração — Participantes com Mais Declarantes Vinculados</h2></div>', unsafe_allow_html=True)
            df_concentracao = df_part.nlargest(15, 'total_declarantes')
            fig_conc = px.bar(
                df_concentracao,
                y=label_col_p, x='total_declarantes',
                orientation='h', color='vl_remessas',
                color_continuous_scale='Blues',
                title="Participantes com Maior Número de Declarantes"
            )
            fig_conc.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                yaxis_title="", xaxis_title="Nº Declarantes",
                height=450, coloraxis_colorbar_title="Valor (R$)"
            )
            st.plotly_chart(fig_conc, use_container_width=True)

            with st.expander("📋 Tabela Completa de Participantes"):
                st.dataframe(df_part_sorted, use_container_width=True)
                st.download_button(
                    "📥 Exportar Participantes (Excel)", data=to_excel(df_part_sorted),
                    file_name="gestex_participantes.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_participantes"
                )

    # ----- Subtab 6.4: NCMs e Produtos -----
    with sub6_4:
        with st.spinner("Carregando NCMs..."):
            df_ncm = carregar_top_ncm()

        if df_ncm.empty:
            st.warning("⚠️ Sem dados de NCMs.")
        else:
            st.markdown('<div class="section-header"><h2>📦 Top NCMs Movimentados na Industrialização</h2></div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                fig_ncm = px.bar(
                    df_ncm.head(20),
                    y='cod_ncm', x='valor_total',
                    orientation='h', color='contribuintes',
                    color_continuous_scale='Viridis',
                    title="Top 20 NCMs por Valor Total",
                    hover_data=['descr_item', 'operacoes', 'parceiros']
                )
                fig_ncm.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    yaxis_title="NCM", xaxis_title="Valor Total (R$)",
                    height=550, xaxis_tickformat=",.0f",
                    coloraxis_colorbar_title="Contribuintes"
                )
                st.plotly_chart(fig_ncm, use_container_width=True)

            with col2:
                fig_ncm_treemap = px.treemap(
                    df_ncm.head(30),
                    path=['cod_ncm'],
                    values='valor_total',
                    color='operacoes',
                    color_continuous_scale='Blues',
                    title="Mapa de Calor — NCMs por Valor",
                    hover_data=['descr_item']
                )
                fig_ncm_treemap.update_layout(height=550)
                st.plotly_chart(fig_ncm_treemap, use_container_width=True)

            with st.expander("📋 Tabela Completa — NCMs"):
                st.dataframe(df_ncm, use_container_width=True)

    # ----- Subtab 6.5: Estoque e Inventário -----
    with sub6_5:
        with st.spinner("Carregando dados de estoque e inventário..."):
            df_inv_geral = carregar_inventario_resumo_geral()
            df_est_evo = carregar_estoque_evolucao_geral()

        st.markdown('<div class="section-header"><h2>📊 Estoque de Terceiros e Inventário — Visão Geral</h2></div>', unsafe_allow_html=True)

        if not df_est_evo.empty:
            fig_est = go.Figure()
            fig_est.add_trace(go.Scatter(
                x=df_est_evo['mes_referencia'], y=df_est_evo['qtd_total'],
                mode='lines+markers', name='Qtd Total Estoque',
                line=dict(color='#3182ce', width=3), marker=dict(size=8)
            ))
            fig_est.add_trace(go.Bar(
                x=df_est_evo['mes_referencia'], y=df_est_evo['contribuintes'],
                name='Contribuintes', marker_color='rgba(56,161,105,0.4)',
                yaxis='y2'
            ))
            fig_est.update_layout(
                title="Evolução Mensal — Estoque de Terceiros (K200)",
                xaxis_title="Mês", yaxis_title="Quantidade Total",
                yaxis2=dict(title="Contribuintes", overlaying='y', side='right'),
                height=420, xaxis_tickformat="d"
            )
            st.plotly_chart(fig_est, use_container_width=True)

        if not df_inv_geral.empty:
            col1, col2 = st.columns(2)
            with col1:
                df_inv_prop = df_inv_geral.groupby('propriedade').agg(
                    valor_total=('vl_total', 'sum'),
                    contribuintes=('identificadorarquivo', 'nunique')
                ).reset_index()
                fig_inv = px.pie(
                    df_inv_prop, values='valor_total', names='propriedade',
                    color_discrete_sequence=['#38a169', '#3182ce', '#dd6b20'],
                    title="Inventário (H010) — Valor por Tipo de Propriedade",
                    hole=0.4
                )
                fig_inv.update_layout(height=380)
                st.plotly_chart(fig_inv, use_container_width=True)

            with col2:
                for _, row in df_inv_prop.iterrows():
                    st.markdown(f"""
                    <div class="info-card">
                        <strong>{row['propriedade']}</strong><br>
                        <span style="font-size:1.2rem; color:#1a365d; font-weight:700;">
                            {formatar_moeda_abrev(row['valor_total'])}
                        </span><br>
                        <span style="font-size:0.85rem; color:#718096;">
                            {formatar_numero(row['contribuintes'])} contribuintes
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

# =============================================================================
# ABA 7: FISCALIZAÇÃO INTELIGENTE (ML)
# =============================================================================
with tab7:
    st.markdown('<div class="section-header"><h2>🤖 Fiscalização Inteligente — Priorização por Machine Learning</h2></div>', unsafe_allow_html=True)
    st.caption("Combina Isolation Forest (detecção de anomalias), K-Means (perfis de contribuinte) e regras fiscais para gerar um ranking de fiscalização.")

    sub7_1, sub7_2, sub7_3 = st.tabs([
        "🎯 Ranking de Fiscalização",
        "🔬 Análise de Clusters",
        "📋 Detalhes da Empresa"
    ])

    # Carregar dados base para ML
    with st.spinner("Carregando dados base para análise de ML..."):
        df_panorama_ml = carregar_panorama_contribuintes()
        df_painel_agg = carregar_resumo_painel_por_contribuinte()
        df_anomalias_agg = carregar_resumo_anomalias_por_contribuinte()

    if df_panorama_ml.empty:
        st.warning("⚠️ Sem dados suficientes para executar o modelo.")
    else:
        # Construir features
        df_ml = construir_features_fiscalizacao(df_panorama_ml, df_painel_agg, df_anomalias_agg)

        # Features numéricas para ML
        FEATURE_COLS = [
            'total_operacoes', 'total_notas', 'total_parceiros', 'total_itens',
            'vl_remessas', 'vl_retornos', 'vl_industrializacao', 'total_icms',
            'meses_ativos', 'saldo_remessa_retorno', 'pct_retorno',
            'valor_medio_operacao', 'icms_por_operacao', 'concentracao_parceiros',
            'total_saldo_valor', 'itens_pendentes'
        ]

        # Filtrar colunas que existem
        feature_cols_exist = [c for c in FEATURE_COLS if c in df_ml.columns]

        # Filtrar apenas contribuintes com atividade relevante
        df_ml_filtered = df_ml[df_ml['total_operacoes'] >= 2].copy()

        if len(df_ml_filtered) < 5:
            st.warning("⚠️ Poucos contribuintes com operações suficientes para análise de ML.")
        else:
            # ----- Subtab 7.1: Ranking de Fiscalização -----
            with sub7_1:
                st.markdown('<div class="section-header"><h2>🎯 Ranking de Priorização — Top Empresas para Fiscalização</h2></div>', unsafe_allow_html=True)

                col_cfg1, col_cfg2 = st.columns(2)
                with col_cfg1:
                    contamination = st.slider(
                        "Sensibilidade do Isolation Forest (% anomalias esperadas):",
                        min_value=5, max_value=40, value=15, step=5,
                        key="iso_contamination"
                    ) / 100
                with col_cfg2:
                    peso_ml = st.slider(
                        "Peso ML vs Regras Fiscais:",
                        min_value=0, max_value=100, value=60, step=10,
                        help="0 = só regras fiscais, 100 = só ML",
                        key="peso_ml"
                    ) / 100

                if st.button("🚀 Executar Modelo de Fiscalização", type="primary", key="btn_ml"):
                    with st.spinner("Executando Isolation Forest + Scoring por Regras..."):
                        # Isolation Forest
                        df_result, iso_model, iso_scaler = executar_isolation_forest(
                            df_ml_filtered.copy(), feature_cols_exist, contamination
                        )

                        # Score por regras fiscais
                        df_result['score_regras'] = calcular_score_regras(df_result)

                        # Score combinado
                        df_result['score_final'] = (
                            peso_ml * df_result['risk_score'] +
                            (1 - peso_ml) * df_result['score_regras']
                        )

                        # Classificação
                        df_result['prioridade'] = pd.cut(
                            df_result['score_final'],
                            bins=[-1, 25, 50, 75, 100],
                            labels=['🟢 Baixa', '🟡 Moderada', '🟠 Alta', '🔴 Crítica']
                        )

                        st.session_state['df_ml_result'] = df_result

                # Exibir resultados se existirem
                if 'df_ml_result' in st.session_state:
                    df_result = st.session_state['df_ml_result']

                    # Métricas do modelo
                    n_critica = len(df_result[df_result['prioridade'] == '🔴 Crítica'])
                    n_alta = len(df_result[df_result['prioridade'] == '🟠 Alta'])
                    n_moderada = len(df_result[df_result['prioridade'] == '🟡 Moderada'])
                    n_baixa = len(df_result[df_result['prioridade'] == '🟢 Baixa'])

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card red">
                            <div class="metric-label">🔴 Crítica</div>
                            <div class="metric-value">{n_critica}</div>
                        </div>""", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card orange">
                            <div class="metric-label">🟠 Alta</div>
                            <div class="metric-value">{n_alta}</div>
                        </div>""", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color:#ecc94b;">
                            <div class="metric-label">🟡 Moderada</div>
                            <div class="metric-value">{n_moderada}</div>
                        </div>""", unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">🟢 Baixa</div>
                            <div class="metric-value">{n_baixa}</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("---")

                    # Gráfico de distribuição de scores
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig_score_dist = px.histogram(
                            df_result, x='score_final', nbins=25,
                            color='prioridade',
                            color_discrete_map={
                                '🔴 Crítica': '#e53e3e', '🟠 Alta': '#dd6b20',
                                '🟡 Moderada': '#ecc94b', '🟢 Baixa': '#38a169'
                            },
                            title="Distribuição do Score Final"
                        )
                        fig_score_dist.update_layout(
                            xaxis_title="Score (0-100)", yaxis_title="Frequência",
                            height=400, legend_title="Prioridade"
                        )
                        st.plotly_chart(fig_score_dist, use_container_width=True)

                    with col_g2:
                        fig_scatter_ml = px.scatter(
                            df_result, x='risk_score', y='score_regras',
                            color='prioridade', size='vl_remessas',
                            hover_name='nomeempresarial' if 'nomeempresarial' in df_result.columns else 'identificadorarquivo',
                            color_discrete_map={
                                '🔴 Crítica': '#e53e3e', '🟠 Alta': '#dd6b20',
                                '🟡 Moderada': '#ecc94b', '🟢 Baixa': '#38a169'
                            },
                            title="Score ML vs Score Regras"
                        )
                        fig_scatter_ml.update_layout(
                            xaxis_title="Score ML (Isolation Forest)", yaxis_title="Score Regras Fiscais",
                            height=400, legend_title="Prioridade"
                        )
                        st.plotly_chart(fig_scatter_ml, use_container_width=True)

                    # Ranking tabela
                    st.markdown('<div class="section-header"><h2>📋 Ranking Detalhado — Priorização de Fiscalização</h2></div>', unsafe_allow_html=True)

                    df_ranking = df_result.sort_values('score_final', ascending=False)
                    cols_ranking = [c for c in [
                        'prioridade', 'score_final', 'risk_score', 'score_regras',
                        'nomeempresarial', 'codigocnpj', 'ie',
                        'total_operacoes', 'vl_remessas', 'vl_retornos',
                        'pct_retorno', 'saldo_remessa_retorno', 'total_icms',
                        'total_parceiros', 'itens_pendentes'
                    ] if c in df_ranking.columns]

                    st.dataframe(
                        df_ranking[cols_ranking].head(50).style.format({
                            'score_final': '{:.1f}',
                            'risk_score': '{:.1f}',
                            'score_regras': '{:.1f}',
                            'pct_retorno': '{:.1f}%'
                        }),
                        use_container_width=True
                    )

                    st.download_button(
                        "📥 Exportar Ranking Completo (Excel)",
                        data=to_excel(df_ranking[cols_ranking]),
                        file_name="gestex_ranking_fiscalizacao.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_ranking"
                    )

                    # Feature importance (proxy via correlação com score)
                    with st.expander("🔍 Importância das Variáveis no Score"):
                        correlations = df_result[feature_cols_exist + ['score_final']].corr()['score_final'].drop('score_final')
                        df_corr = correlations.abs().sort_values(ascending=True).reset_index()
                        df_corr.columns = ['variavel', 'importancia']

                        fig_imp = px.bar(
                            df_corr, y='variavel', x='importancia',
                            orientation='h', color='importancia',
                            color_continuous_scale='Reds',
                            title="Correlação das Variáveis com o Score Final"
                        )
                        fig_imp.update_layout(
                            yaxis_title="", xaxis_title="Correlação (abs)",
                            height=500, coloraxis_showscale=False
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

            # ----- Subtab 7.2: Análise de Clusters -----
            with sub7_2:
                st.markdown('<div class="section-header"><h2>🔬 Análise de Clusters — Perfis de Contribuintes</h2></div>', unsafe_allow_html=True)

                n_clusters = st.slider("Número de clusters:", min_value=2, max_value=8, value=4, key="n_clusters")

                if st.button("🧪 Executar Clusterização (K-Means)", type="primary", key="btn_cluster"):
                    with st.spinner("Executando K-Means + PCA..."):
                        df_cluster, kmeans_model, km_scaler = executar_clustering(
                            df_ml_filtered.copy(), feature_cols_exist, n_clusters
                        )
                        st.session_state['df_cluster'] = df_cluster

                if 'df_cluster' in st.session_state:
                    df_cluster = st.session_state['df_cluster']

                    # Scatter PCA
                    fig_pca = px.scatter(
                        df_cluster, x='pca_x', y='pca_y',
                        color='cluster', size='vl_remessas',
                        hover_name='nomeempresarial' if 'nomeempresarial' in df_cluster.columns else 'identificadorarquivo',
                        title="Mapa de Clusters (PCA 2D)",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_pca.update_layout(
                        xaxis_title="Componente Principal 1",
                        yaxis_title="Componente Principal 2",
                        height=500
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)

                    # Perfil de cada cluster
                    st.markdown('<div class="section-header"><h2>📊 Perfil Médio por Cluster</h2></div>', unsafe_allow_html=True)

                    metrics_cluster = ['total_operacoes', 'vl_remessas', 'vl_retornos',
                                      'total_icms', 'total_parceiros', 'pct_retorno',
                                      'saldo_remessa_retorno', 'meses_ativos']
                    metrics_exist = [c for c in metrics_cluster if c in df_cluster.columns]

                    df_perfil = df_cluster.groupby('cluster')[metrics_exist].mean().round(2)

                    # Exibir cada cluster como card
                    n_cl = df_cluster['cluster'].nunique()
                    cols_cl = st.columns(min(n_cl, 4))
                    for i, (cl_id, row) in enumerate(df_perfil.iterrows()):
                        with cols_cl[i % len(cols_cl)]:
                            n_membros = len(df_cluster[df_cluster['cluster'] == cl_id])
                            st.markdown(f"""
                            <div class="info-card">
                                <strong style="font-size:1.1rem;">Cluster {cl_id}</strong>
                                <span class="badge badge-info">{n_membros} empresas</span><br><br>
                                <span style="font-size:0.85rem;">
                                    Operações: {formatar_numero(row.get('total_operacoes', 0))}<br>
                                    Remessas: {formatar_moeda_abrev(row.get('vl_remessas', 0))}<br>
                                    Retornos: {formatar_moeda_abrev(row.get('vl_retornos', 0))}<br>
                                    Taxa Retorno: {formatar_percentual(row.get('pct_retorno', 0))}<br>
                                    ICMS: {formatar_moeda_abrev(row.get('total_icms', 0))}<br>
                                    Parceiros: {formatar_numero(row.get('total_parceiros', 0))}<br>
                                    Meses ativos: {formatar_qtd(row.get('meses_ativos', 0))}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)

                    # Radar chart comparativo dos clusters
                    st.markdown("---")
                    fig_radar = go.Figure()
                    for cl_id, row in df_perfil.iterrows():
                        # Normalizar cada métrica entre 0 e 1 para o radar
                        values_norm = []
                        for col in metrics_exist:
                            col_max = df_perfil[col].max()
                            values_norm.append(row[col] / col_max if col_max > 0 else 0)
                        values_norm.append(values_norm[0])  # fechar polígono

                        fig_radar.add_trace(go.Scatterpolar(
                            r=values_norm,
                            theta=metrics_exist + [metrics_exist[0]],
                            name=f'Cluster {cl_id}'
                        ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title="Comparativo de Perfis — Radar",
                        height=500, showlegend=True
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                    with st.expander("📋 Tabela Completa com Clusters"):
                        cols_cl_show = [c for c in [
                            'cluster', 'nomeempresarial', 'codigocnpj',
                            'total_operacoes', 'vl_remessas', 'vl_retornos',
                            'pct_retorno', 'total_icms', 'total_parceiros'
                        ] if c in df_cluster.columns]
                        st.dataframe(df_cluster[cols_cl_show].sort_values(['cluster', 'vl_remessas'], ascending=[True, False]), use_container_width=True)

            # ----- Subtab 7.3: Detalhes da Empresa -----
            with sub7_3:
                st.markdown('<div class="section-header"><h2>📋 Detalhes da Empresa Selecionada</h2></div>', unsafe_allow_html=True)
                st.caption("Selecione uma empresa do ranking para ver seu perfil detalhado e justificativas do score.")

                if 'df_ml_result' in st.session_state:
                    df_result = st.session_state['df_ml_result']
                    df_sorted = df_result.sort_values('score_final', ascending=False)

                    # Selectbox com empresa
                    label_col = 'nomeempresarial' if 'nomeempresarial' in df_sorted.columns else 'identificadorarquivo'
                    opcoes = df_sorted.head(50).apply(
                        lambda r: f"{r.get('prioridade', '')} | Score {r.get('score_final', 0):.0f} | {r.get(label_col, 'N/D')} ({r.get('codigocnpj', '')})",
                        axis=1
                    ).tolist()

                    empresa_sel = st.selectbox("Selecione a empresa:", opcoes, key="empresa_detalhe")

                    if empresa_sel:
                        idx_sel = opcoes.index(empresa_sel)
                        emp = df_sorted.iloc[idx_sel]

                        # Card da empresa
                        st.markdown(f"""
                        <div class="contribuinte-box">
                            <h3 style="margin:0; color:#1a365d;">{emp.get('nomeempresarial', 'N/D')}</h3>
                            <p style="margin:0.25rem 0; color:#4a5568;">
                                <strong>CNPJ:</strong> {emp.get('codigocnpj', 'N/D')} &nbsp;|&nbsp;
                                <strong>IE:</strong> {emp.get('ie', 'N/D')} &nbsp;|&nbsp;
                                <strong>Município:</strong> {emp.get('cod_municipio', 'N/D')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Scores
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card red">
                                <div class="metric-label">Score Final</div>
                                <div class="metric-value">{emp.get('score_final', 0):.1f}</div>
                            </div>""", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card orange">
                                <div class="metric-label">Score ML</div>
                                <div class="metric-value">{emp.get('risk_score', 0):.1f}</div>
                            </div>""", unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card purple">
                                <div class="metric-label">Score Regras</div>
                                <div class="metric-value">{emp.get('score_regras', 0):.1f}</div>
                            </div>""", unsafe_allow_html=True)
                        with col4:
                            st.markdown(f"""
                            <div class="metric-card blue">
                                <div class="metric-label">Prioridade</div>
                                <div class="metric-value">{emp.get('prioridade', 'N/D')}</div>
                            </div>""", unsafe_allow_html=True)

                        # Indicadores detalhados
                        st.markdown('<div class="section-header"><h2>📊 Indicadores Detalhados</h2></div>', unsafe_allow_html=True)

                        col_d1, col_d2, col_d3 = st.columns(3)
                        with col_d1:
                            st.markdown("**Volumes**")
                            st.metric("Total Operações", formatar_numero(emp.get('total_operacoes', 0)))
                            st.metric("Notas Fiscais", formatar_numero(emp.get('total_notas', 0)))
                            st.metric("Itens Distintos", formatar_numero(emp.get('total_itens', 0)))
                            st.metric("Meses Ativos", formatar_numero(emp.get('meses_ativos', 0)))
                        with col_d2:
                            st.markdown("**Valores**")
                            st.metric("Remessas", formatar_moeda_abrev(emp.get('vl_remessas', 0)))
                            st.metric("Retornos", formatar_moeda_abrev(emp.get('vl_retornos', 0)))
                            st.metric("Industrialização", formatar_moeda_abrev(emp.get('vl_industrializacao', 0)))
                            st.metric("ICMS Total", formatar_moeda_abrev(emp.get('total_icms', 0)))
                        with col_d3:
                            st.markdown("**Indicadores de Risco**")
                            pct_ret = emp.get('pct_retorno', 0)
                            st.metric("Taxa de Retorno", formatar_percentual(pct_ret),
                                     delta=f"{'⚠️ Baixa' if pct_ret < 50 else '✅ Normal'}")
                            st.metric("Saldo Pendente", formatar_moeda_abrev(emp.get('saldo_remessa_retorno', 0)))
                            st.metric("Parceiros", formatar_numero(emp.get('total_parceiros', 0)))
                            st.metric("Itens Pendentes", formatar_numero(emp.get('itens_pendentes', 0)))

                        # Justificativas do score
                        st.markdown('<div class="section-header"><h2>🔍 Justificativas do Score</h2></div>', unsafe_allow_html=True)

                        justificativas = []
                        if emp.get('pct_retorno', 100) < 50 and emp.get('vl_remessas', 0) > 0:
                            justificativas.append(f"🔴 **Taxa de retorno muito baixa**: {formatar_percentual(emp.get('pct_retorno', 0))} — menos de 50% das remessas foram retornadas")
                        if emp.get('saldo_remessa_retorno', 0) > 0:
                            justificativas.append(f"🟠 **Saldo pendente positivo**: {formatar_moeda(emp.get('saldo_remessa_retorno', 0))} em remessas sem retorno correspondente")
                        if emp.get('itens_pendentes', 0) > 0:
                            justificativas.append(f"🟠 **{int(emp.get('itens_pendentes', 0))} itens pendentes** no painel consolidado")
                        if emp.get('concentracao_parceiros', 0) > df_ml_filtered['concentracao_parceiros'].quantile(0.90):
                            justificativas.append(f"🟡 **Alta concentração**: muitas operações com poucos parceiros")
                        if emp.get('risk_score', 0) > 75:
                            justificativas.append(f"🔴 **Anomalia detectada pelo Isolation Forest** (score ML: {emp.get('risk_score', 0):.1f})")

                        if justificativas:
                            for j in justificativas:
                                st.markdown(j)
                        else:
                            st.success("✅ Nenhum fator de risco significativo identificado pelas regras de negócio.")

                        # Comparação com a média
                        st.markdown('<div class="section-header"><h2>📊 Comparação com a Média do Setor</h2></div>', unsafe_allow_html=True)

                        metrics_comp = ['total_operacoes', 'vl_remessas', 'vl_retornos', 'pct_retorno', 'total_icms', 'total_parceiros']
                        metrics_comp_exist = [c for c in metrics_comp if c in df_ml_filtered.columns]

                        emp_vals = [emp.get(c, 0) for c in metrics_comp_exist]
                        mean_vals = [df_ml_filtered[c].mean() for c in metrics_comp_exist]

                        fig_comp = go.Figure()
                        fig_comp.add_trace(go.Bar(name='Empresa', x=metrics_comp_exist, y=emp_vals, marker_color='#e53e3e'))
                        fig_comp.add_trace(go.Bar(name='Média Setor', x=metrics_comp_exist, y=mean_vals, marker_color='#3182ce'))
                        fig_comp.update_layout(
                            barmode='group', title="Empresa vs Média do Setor",
                            height=400, yaxis_tickformat=",.0f"
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)

                else:
                    st.info("ℹ️ Execute o modelo na aba **Ranking de Fiscalização** primeiro.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 1rem;'>
    <p><strong>GESTEX V014</strong> — Industrialização por Encomenda</p>
    <p>SEF/SC — Secretaria de Estado da Fazenda de Santa Catarina | 2025</p>
    <p style='font-size: 0.8rem;'>Dados extraídos do Big Data Impala — Schema teste.gestex_v014_*</p>
</div>
""", unsafe_allow_html=True)