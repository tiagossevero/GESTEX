-- ============================================================
-- GESTEX V013 — STEP 1 de 3
-- RECRIAÇÃO: gestex_v013_dime_dados_gerais
-- ============================================================
-- MUDANÇA: adicionado filtro por CNPJ raiz GESTEX no WHERE.
--
-- Antes: incluía TODAS as declarações DIME do SC (sem filtro
--        de contribuinte) — tabela enorme e sem escopo definido.
--
-- Depois: inclui TODOS os estabelecimentos SC cujo CNPJ raiz
--         pertença a algum contribuinte GESTEX cadastrado,
--         inclusive estabelecimentos com CNAE diferente
--         (ex: consolidador de apuração) que não constam no
--         gestex_v013_cadastro mas fazem parte do mesmo grupo.
--
-- Isso garante a consolidação por CNPJ raiz exigida na
--         análise anual DIME (todos os estab. SC do grupo).
--
-- ⚠️  Após executar este script, execute obrigatoriamente
--     os steps 2 e 3 para recriar as tabelas dependentes.
-- ============================================================

DROP TABLE IF EXISTS teste.gestex_v013_dime_dados_gerais;

CREATE TABLE teste.gestex_v013_dime_dados_gerais AS
SELECT
    d.nu_ie                                     AS ie,
    d.nu_cnpj                                   AS cnpj,
    SUBSTR(d.nu_cnpj, 1, 8)                    AS cnpj_raiz,
    CAST(d.nu_ano_ref AS INT)                   AS ano,
    d.nu_per_ref                                AS mes,
    -- Faturamento
    COALESCE(d.vl_faturamento, 0)               AS vl_faturamento,
    COALESCE(d.vl_receita_bruta, 0)             AS vl_receita_bruta,
    -- Totais Entradas
    COALESCE(d.vl_tot_ent_contabil, 0)          AS vl_tot_ent_contabil,
    COALESCE(d.vl_tot_ent_bc, 0)                AS vl_tot_ent_bc,
    COALESCE(d.vl_tot_ent_imposto_creditado, 0) AS vl_tot_ent_imposto_creditado,
    COALESCE(d.vl_tot_ent_isentas_nao_trib, 0)  AS vl_tot_ent_isentas_nao_trib,
    COALESCE(d.vl_tot_ent_outras, 0)            AS vl_tot_ent_outras,
    -- Totais Saídas
    COALESCE(d.vl_tot_sai_contabil, 0)          AS vl_tot_sai_contabil,
    COALESCE(d.vl_tot_sai_bc, 0)                AS vl_tot_sai_bc,
    COALESCE(d.vl_tot_sai_imposto_creditado, 0) AS vl_tot_sai_imposto_creditado,
    COALESCE(d.vl_tot_sai_isentas_nao_trib, 0)  AS vl_tot_sai_isentas_nao_trib,
    COALESCE(d.vl_tot_sai_outras, 0)            AS vl_tot_sai_outras,
    -- Créditos e Débitos
    COALESCE(d.vl_tot_cred, 0)                  AS vl_tot_creditos,
    COALESCE(d.vl_tot_deb, 0)                   AS vl_tot_debitos,
    COALESCE(d.vl_cred_mes_anterior, 0)         AS vl_cred_mes_anterior,
    COALESCE(d.vl_cred_dcip, 0)                 AS vl_cred_dcip,
    COALESCE(d.vl_cred_mes_seguinte, 0)         AS vl_cred_mes_seguinte,
    COALESCE(d.vl_deb_recolher, 0)              AS vl_deb_recolher,
    -- Saídas por destino
    COALESCE(d.vl_saidas_internas, 0)           AS vl_saidas_internas,
    COALESCE(d.vl_saidas_interestaduais, 0)     AS vl_saidas_interestaduais,
    COALESCE(d.vl_exportacao, 0)                AS vl_exportacao,
    -- Entradas por origem
    COALESCE(d.vl_ent_int, 0)                   AS vl_ent_internas,
    COALESCE(d.vl_ent_ies, 0)                   AS vl_ent_interestaduais,
    COALESCE(d.vl_importacao, 0)                AS vl_importacao,
    -- Substituição Tributária
    COALESCE(d.vl_tot_ent_bc_imposto_retido, 0) AS vl_ent_bc_st,
    COALESCE(d.vl_tot_ent_imposto_retido, 0)    AS vl_ent_icms_st,
    COALESCE(d.vl_tot_sai_bc_imposto_retido, 0) AS vl_sai_bc_st,
    COALESCE(d.vl_tot_sai_imposto_retido, 0)    AS vl_sai_icms_st,
    -- Funcionários
    d.qt_funcionarios
FROM usr_sat_ods.vw_ods_decl_dime d
WHERE d.sn_cancelada = 0
  AND CAST(d.nu_ano_ref AS INT) IN (2020, 2021, 2022, 2023, 2024)
  -- ✅ MUDANÇA: filtra pelo cnpj_raiz dos grupos GESTEX,
  --    incluindo TODOS os seus estabelecimentos em SC
  --    (não apenas os têxteis do cadastro).
  AND SUBSTR(d.nu_cnpj, 1, 8) IN (
      SELECT DISTINCT cnpj_raiz
      FROM teste.gestex_v013_cadastro
  )
;
