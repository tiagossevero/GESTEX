-- ============================================================
-- GESTEX V013 — STEP 2 de 3
-- RECRIAÇÃO: gestex_v013_dime_anual
-- ============================================================
-- MUDANÇA NESTE ARQUIVO: nenhuma — o SQL é idêntico ao original.
--
-- ⚠️  Mas esta tabela PRECISA ser recriada após o step 1,
--     pois é um CREATE TABLE AS SELECT (tabela materializada).
--     Sem recriar, os dados ainda refletem o estado anterior
--     de gestex_v013_dime_dados_gerais (sem o filtro de
--     cnpj_raiz GESTEX / sem o consolidador incluído).
--
-- Após executar, execute obrigatoriamente o step 3.
-- ============================================================

DROP TABLE IF EXISTS teste.gestex_v013_dime_anual;

CREATE TABLE teste.gestex_v013_dime_anual AS
SELECT
    cnpj_raiz,
    ano,
    COUNT(DISTINCT ie)              AS qtd_estabelecimentos,
    SUM(vl_faturamento)             AS faturamento_total,
    SUM(vl_receita_bruta)           AS receita_bruta_total,
    SUM(vl_tot_sai_bc)              AS bc_saidas_total,
    SUM(vl_tot_ent_bc)              AS bc_entradas_total,
    SUM(vl_tot_debitos)             AS debitos_total,
    SUM(vl_tot_creditos)            AS creditos_total,
    SUM(vl_deb_recolher)            AS icms_recolher_total,
    SUM(vl_cred_dcip)               AS cred_dcip_total,
    SUM(vl_saidas_internas)         AS saidas_internas,
    SUM(vl_saidas_interestaduais)   AS saidas_interestaduais,
    SUM(vl_exportacao)              AS exportacao,
    SUM(vl_ent_internas)            AS entradas_internas,
    SUM(vl_ent_interestaduais)      AS entradas_interestaduais,
    SUM(vl_importacao)              AS importacao
FROM teste.gestex_v013_dime_dados_gerais
GROUP BY cnpj_raiz, ano
;
