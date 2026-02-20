-- ============================================================
-- GESTEX V013 — STEP 3 de 3
-- RECRIAÇÃO: gestex_v013_base_analitica
-- ============================================================
-- MUDANÇA: substituído o LEFT JOIN direto com gestex_v013_cadastro
--          por um subquery com ROW_NUMBER().
--
-- Antes: LEFT JOIN gestex_v013_cadastro cad ON da.cnpj_raiz = cad.cnpj_raiz
--        → se o CNPJ raiz tiver N estabelecimentos GESTEX no cadastro,
--          a base_analitica gerava N linhas com os MESMOS totais
--          financeiros (duplicação silenciosa), causando flags e
--          scoring duplicados nas análises downstream.
--
-- Depois: LEFT JOIN com subquery que usa ROW_NUMBER() para selecionar
--         apenas 1 representante por cnpj_raiz (menor IE) →
--         exatamente 1 linha por (cnpj_raiz, ano).
-- ============================================================

DROP TABLE IF EXISTS teste.gestex_v013_base_analitica;

CREATE TABLE teste.gestex_v013_base_analitica AS
SELECT
    cad.ie,
    cad.cnpj,
    cad.cnpj_raiz,
    cad.razao_social,
    cad.cnae,
    cad.cnae_desc,
    cad.regime_tributacao,
    da.ano,
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
    -- Crédito Presumido
    dc47.vl_cp_total                                                    AS vl_cp_ttd47,
    dc372.vl_cp_total                                                   AS vl_cp_ttd372,
    COALESCE(dc47.vl_cp_total, 0) + COALESCE(dc372.vl_cp_total, 0)    AS vl_cp_total,
    CASE
        WHEN dc47.vl_cp_total  IS NOT NULL AND dc372.vl_cp_total IS NOT NULL THEN 'TTD47+TTD372'
        WHEN dc47.vl_cp_total  IS NOT NULL                                   THEN 'TTD47'
        WHEN dc372.vl_cp_total IS NOT NULL                                   THEN 'TTD372'
        ELSE 'NENHUM'
    END                                                                 AS tipo_cp,
    -- TTDs adicionais
    dc409.vl_cp_total                                                   AS vl_cp_ttd409,
    dc410.vl_cp_total                                                   AS vl_cp_ttd410,
    dc411.vl_cp_total                                                   AS vl_cp_ttd411,
    -- Extemporâneo
    COALESCE(dc47.teve_extempraneo,  0)                                 AS extempraneo_47,
    COALESCE(dc372.teve_extempraneo, 0)                                 AS extempraneo_372

FROM teste.gestex_v013_dime_anual da

-- Filtra apenas cnpj_raiz que constam no cadastro GESTEX
INNER JOIN (
    SELECT DISTINCT cnpj_raiz
    FROM teste.gestex_v013_cadastro
) cad_raiz ON da.cnpj_raiz = cad_raiz.cnpj_raiz

-- ✅ MUDANÇA: usa ROW_NUMBER() para garantir 1 representante
--    por cnpj_raiz (menor IE), evitando linhas duplicadas
--    quando há múltiplos estabelecimentos GESTEX no mesmo grupo.
LEFT JOIN (
    SELECT
        cnpj_raiz,
        ie,
        cnpj,
        razao_social,
        cnae,
        cnae_desc,
        regime_tributacao,
        ROW_NUMBER() OVER (PARTITION BY cnpj_raiz ORDER BY ie) AS rn
    FROM teste.gestex_v013_cadastro
) cad ON da.cnpj_raiz = cad.cnpj_raiz AND cad.rn = 1

LEFT JOIN teste.gestex_v013_dcip_anual dc47
    ON da.cnpj_raiz = dc47.cnpj_raiz
   AND da.ano       = dc47.ano
   AND dc47.tipo_beneficio = 'TTD47'

LEFT JOIN teste.gestex_v013_dcip_anual dc372
    ON da.cnpj_raiz = dc372.cnpj_raiz
   AND da.ano       = dc372.ano
   AND dc372.tipo_beneficio = 'TTD372'

LEFT JOIN teste.gestex_v013_dcip_anual dc409
    ON da.cnpj_raiz = dc409.cnpj_raiz
   AND da.ano       = dc409.ano
   AND dc409.tipo_beneficio = 'TTD409'

LEFT JOIN teste.gestex_v013_dcip_anual dc410
    ON da.cnpj_raiz = dc410.cnpj_raiz
   AND da.ano       = dc410.ano
   AND dc410.tipo_beneficio = 'TTD410'

LEFT JOIN teste.gestex_v013_dcip_anual dc411
    ON da.cnpj_raiz = dc411.cnpj_raiz
   AND da.ano       = dc411.ano
   AND dc411.tipo_beneficio = 'TTD411'

WHERE (dc47.vl_cp_total IS NOT NULL OR dc372.vl_cp_total IS NOT NULL)
;
