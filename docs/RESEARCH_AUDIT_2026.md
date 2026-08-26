# B3 Portfolio Lab — auditoria da degradação pós-2021 e protocolo corrigido

## Status desta entrega

Esta revisão separa duas perguntas que estavam misturadas no resultado anterior:

1. **o sinal realmente perdeu capacidade preditiva após 2021?**
2. **o backtest e o pseudo-live eram metodologicamente válidos?**

Os números fornecidos no prompt são evidência suficiente de uma quebra de regime agregada — Rank IC médio aproximado de `+0,119` em 2014–2021 e `-0,054` em 2022–2026 — e da hipótese de preservação parcial da cauda superior, porque `top_minus_bottom` permaneceu positivo na maior parte do período recente. Contudo, a auditoria do código encontrou problemas de relógio que precisam ser corrigidos **antes** de interpretar CAGR, Sharpe ou 2026 como evidência econômica definitiva.

O repositório não versiona o painel Parquet de 1.018.057 linhas nem `models/signal_model.joblib`. Por isso esta entrega não inventa PSI, IC por feature, novos CAGRs ou resultados de ablation que não puderam ser recalculados. Em vez disso, corrige as invariantes demonstravelmente erradas e adiciona os executáveis que geram todos esses relatórios com os dados locais oficiais.

## Conclusões principais

### 1. O CAGR histórico de 37,82% precisa ser revalidado

O engine anterior instalava a nova carteira no fechamento da própria data usada para formar o sinal e, em seguida, creditava o retorno `close(t) -> close(t+1)` à carteira nova. Como as features de `t` usam o fechamento de `t`, isso constitui **same-close execution leakage**.

A correção faz o seguinte:

- sinal é formado após o fechamento de `t`;
- `t -> t+1` pertence à carteira antiga/caixa;
- a nova carteira é executada no próximo fechamento disponível, `t+1`;
- o primeiro retorno capturado pela nova carteira é `t+1 -> t+2`.

Consequência: nenhum CAGR histórico anterior deve ser usado como conclusão final sem rerun completo.

### 2. Os labels de 63 pregões atravessavam a fronteira de validação

O split anterior verificava apenas:

`max(train.trade_date) < min(validation.trade_date)`

Isso não basta para um target forward de 63 barras. Uma observação de outubro/dezembro pode ter `trade_date` no treino e um label que termina durante o ano de validação.

Agora cada amostra carrega:

- `target_start_date`;
- `target_end_date`;
- `target_horizon_bars`;
- `execution_lag_bars`.

Antes de cada fold, qualquer amostra cujo `target_end_date` toque a validação é purgada. Há também suporte explícito a embargo conservador.

### 3. `trained_until=2025-12-30` não significava “informação disponível até 2025-12-30”

O artefato anterior era treinado com features de 2025, mas o target das últimas ~63 sessões de 2025 dependia de preços de 2026. Portanto, usar apenas a data da feature como `trained_until` não provava que o modelo estivesse limpo para um pseudo-live de 2026.

A nova definição usa **knowledge cutoff**:

- para o diagnóstico de 2026, `knowledge_cutoff = 2025-12-31`;
- uma linha só entra no treino se `target_end_date <= 2025-12-31`;
- `training_target_end_max` é persistido no metadata;
- um artefato legado sem `knowledge_cutoff` é recusado pelo loader seguro.

Isso provavelmente fará `trained_until_feature_date` recuar para aproximadamente setembro/outubro de 2025, dependendo do calendário de negociação. Esse recuo é correto: labels posteriores ainda não seriam conhecidos no fechamento de 2025.

### 4. O painel mensal misturava datas distintas dentro do mesmo cross-section

`monthly_snapshots()` selecionava a última observação de **cada ticker** no mês. Em ações menos líquidas, isso podia misturar, por exemplo, uma feature de 27/01 com outra de 31/01 e depois tratá-las como um único ranking mensal.

Agora o mês possui uma única data de mercado comum. Apenas observações efetivamente disponíveis nessa data compõem o cross-section; dados stale não são forward-filled silenciosamente.

### 5. A correlação usada pelo QKP/API não tinha `as_of`

O helper da API usava a cauda dos dados presentes em disco. Em uma execução histórica/replay, isso permitia que a matriz de correlação fosse estimada com cotações posteriores ao snapshot.

Agora:

- `historical_correlation(..., as_of=...)` é obrigatório;
- são usados apenas preços `trade_date <= as_of`;
- adjusted total-return prices são preferidos;
- Ledoit–Wolf foi mantido;
- há teste que altera drasticamente todos os preços futuros e exige que a covariância `as_of` permaneça idêntica.

## Pontos auditados que já estavam corretos

Nem todo componente precisava ser reescrito.

- **CVM point-in-time:** o join usa `DT_RECEB` com `join_asof(..., strategy="backward")` e valida `DT_RECEB <= trade_date`.
- **Imputer/scaler:** `SimpleImputer`, `VarianceThreshold` e `StandardScaler` são fitados apenas no conjunto de treino de cada fold.
- **Feature availability:** features técnicas são backward-looking; ranks são contemporâneos ao mesmo cross-section.
- **Total return:** o pipeline possui `adjusted_close` baseado em corporate actions e mantém COTAHIST raw imutável.
- **Covariance shrinkage:** Ledoit–Wolf já existia e foi preservado.
- **Universe:** não foi identificado filtro explícito “somente empresas listadas hoje” no parser COTAHIST; ainda assim, cobertura histórica do issuer bridge deve continuar sendo monitorada.

### Corporate actions e “future action leakage”

Uma série backward-adjusted usa fatores conhecidos ex post para reconstruir uma série contínua. Para features baseadas em **razões/retornos**, uma ação futura multiplica todas as cotações anteriores ao evento pelo mesmo fator e portanto cancela no retorno de uma janela totalmente anterior ao evento. Foi adicionado teste de invariância para esse caso. Janelas/targets que cruzam o evento devem incorporar o retorno econômico da ação, o que é desejado em uma série total-return.

## Evidência de concept drift já disponível

Os resultados fornecidos antes desta correção mostram:

| Período | Rank IC médio aproximado |
|---|---:|
| 2014–2021 | +0,119 |
| 2022–2026 | -0,054 |

Além disso, em 2022–2025/2026 o `top_minus_bottom` permaneceu frequentemente positivo. A interpretação correta neste momento é:

- há evidência de **global ranking failure** pós-2021;
- ainda é plausível existir **tail-selection preservation**;
- não é possível afirmar quais das 40 features causaram a quebra sem executar os relatórios feature-level no painel local;
- o rerun após eliminar leakage é obrigatório, porque parte da performance histórica pode mudar materialmente.

## Mapa do código e alterações

| Arquivo | Função/classe | Responsabilidade | Alteração |
|---|---|---|---|
| `features/technical.py` | `build_technical_features` | features e target | target inicia após a execução possível; persiste `target_end_date` |
| `features/technical.py` | `monthly_snapshots` | painel mensal | uma data comum por cross-section |
| `data_quality/checks.py` | `purge_overlapping_labels` | leakage de labels | purge explícito por fim do label |
| `data_quality/checks.py` | `filter_labels_known_by` | conhecimento disponível | impede label que termina após cutoff |
| `ml/walk_forward.py` | `TrainingPolicy` | janela/recency/target | rolling e half-life development-only |
| `ml/walk_forward.py` | `train_once` | fold OOS | purge antes do fit e sample weights opcionais |
| `ml/walk_forward.py` | `fit_final_model` | artefato final | fit apenas em labels conhecidos pelo cutoff |
| `ml/model.py` | `EnsembleRegressor.fit` | LGBM/Cat/Ridge | `sample_weight` consistente nos componentes |
| `backtest/engine.py` | `run_monthly_topk_backtest*` | simulação | next-close execution, detalhamento de posições/contribuições |
| `services/api/app/research.py` | `historical_correlation` | risco PIT | `as_of` obrigatório, adjusted prices, Ledoit–Wolf |
| `services/api/app/routers/portfolio.py` | `optimize` | QKP + allocation | passa o `as_of` do snapshot à matriz de risco |
| `ml/diagnostics.py` | drift/IC/Top-K/importância | investigação | relatórios por regime e ano |
| `research/statistics.py` | moving-block bootstrap | inferência | CIs preservando dependência temporal |
| `research/gates.py` | `evaluate_acceptance_gates` | promoção | gates separados e detalhados, sem 2026 |
| `research/experiments.py` | experiment registry | auditabilidade | grava config/folds/commit/métricas |
| `backtest/weighted.py` | risk benchmarks | portfolio construction | minvar/inv-vol/HRP, sample vs Ledoit–Wolf, PIT |
| `scripts/qkp_ablation.py` | ablation | ML vs QKP vs allocation | usa parâmetros atuais, nunca 2026 |

## Antes / depois / justificativa / evidência / risco de overfitting

### Same-close execution

**Antes:** sinal em `t`, carteira nova recebe `t -> t+1`.

**Depois:** sinal em `t`, execução em `t+1`, carteira nova recebe a partir de `t+1 -> t+2`.

**Justificativa:** preço de execução deve ocorrer depois da disponibilidade do sinal.

**Evidência:** ordem das operações no engine anterior instalava `target` antes de calcular `daily_ret[index-1]`.

**Risco de overfitting:** nenhum; é uma correção de causalidade temporal e deve ser aplicada mesmo se piorar todos os resultados.

### Purging de labels

**Antes:** split baseado apenas em `trade_date`.

**Depois:** `target_end_date < validation_start` é uma invariante.

**Justificativa:** um label de 63 barras pode conter preços da validação.

**Evidência:** o target anterior era construído com `shift(-63)` e o validator não conhecia o fim do label.

**Risco de overfitting:** nenhum; reduz informação disponível e tende a tornar a avaliação mais conservadora.

### Knowledge cutoff do artefato

**Antes:** “trained until” = maior data de feature.

**Depois:** metadata distingue `trained_until_feature_date`, `training_target_end_max` e `knowledge_cutoff`.

**Justificativa:** o que importa para pseudo-live é quando o label completo ficou conhecido.

**Evidência:** feature de dezembro/2025 com target 63 barras pode usar março/2026.

**Risco de overfitting:** nenhum; é prevenção direta de leakage.

### Cross-section mensal

**Antes:** última data por ticker.

**Depois:** mesma data de mercado para todos os nomes do cross-section.

**Justificativa:** rankings cross-sectionais exigem informação contemporânea comparável.

**Evidência:** o antigo `.group_by([ticker, month]).tail(1)` permitia datas distintas.

**Risco de overfitting:** baixo; reduz a amostra de nomes stale e pode alterar a composição do universo.

### Rolling / recency / targets / regime features

**Antes:** expanding window e target excess fixos, sem grade governada.

**Depois:** o código consegue comparar expanding, rolling 3/5/7/10, half-lives 1/2/3/5 e targets raw/excess/rank em folds 2019–2025.

**Justificativa:** testar concept drift sem apagar histórico por suposição.

**Evidência:** Rank IC fornecido deteriora fortemente após 2021.

**Risco de overfitting:** alto se o melhor resultado for selecionado após olhar 2026. Por isso o script nunca usa 2026 e registra todos os experimentos.

## Feature drift e Feature IC

`make diagnose-model-degradation` gera:

- `feature_drift_regime_a_vs_b.csv` com mean, median, std, q05/q25/q75/q95, missingness, PSI, KS e Wasserstein;
- `feature_ic_by_year.csv` com IC médio, mediano, DP, ICIR, SE e t-stat;
- `feature_ic_regime_comparison.csv` com `IC_2014_2021`, `IC_2022_2025`, `delta_ic`, inversão de sinal e classificação;
- Top-K por ano para `K={5,10,20,30,50}`;
- importância LightGBM gain/split, CatBoost PredictionValuesChange e Ridge por fold;
- comparação das políticas de treino;
- comparação dos targets.

Nenhuma dessas tabelas utiliza 2026 na seleção.

## Top-K e hipótese de preservação da cauda

Para cada ano, o novo diagnóstico calcula:

- Precision@K;
- Recall@K;
- NDCG@K;
- mean/median future return do Top-K;
- hit rate;
- Top-K menos universo;
- Top-K menos bottom-K;
- Top-K menos equal-weight universe.

Isso permite decidir se o Rank IC global negativo é incompatível ou não com uma boa shortlist para o QKP.

## Decomposição da performance

O backtest detalhado persiste `selections.parquet` e `contributions.parquet`. Em seguida:

`make decompose-performance`

gera performance anual e mensal, melhores/piores 10 meses, contribuição por ticker e setor, melhores/piores 10 tickers e estatísticas de concentração. Para evitar uma falsa precisão de atribuição multiplicativa, a fração dos “top 5 years” é calculada em log-wealth positivo e a concentração de posições usa contribuição diária agregada; os nomes das métricas deixam essa definição explícita.

## Benchmarks

### Implementados

- `universe_1n` — obrigatório;
- top-liquidity equal weight;
- momentum;
- multifactor;
- ML;
- research_v2;
- minimum variance;
- inverse volatility;
- HRP;
- sample covariance vs Ledoit–Wolf.

### Ibovespa / IBrX-100

Não foi criado um proxy artificial nem usado um feed não governado. O repositório atual não possui uma série oficial PIT/total-return desses índices. Esses benchmarks devem ser ligados quando houver um dataset governado com datas e metodologia verificáveis. IBrX-100 permanece condicionado à disponibilidade correta, exatamente como pedido. Para Ibovespa, a ausência é registrada como gap de dados — não substituída silenciosamente.

A decisão segue DeMiguel, Garlappi & Uppal: o 1/N é tratado como benchmark econômico obrigatório, não como decoração.

## Rolling vs expanding e recency weighting

A grade development-only contém:

- expanding;
- rolling 3 anos;
- rolling 5 anos;
- rolling 7 anos;
- rolling 10 anos;
- expanding com half-life 1, 2, 3 e 5 anos.

O código não presume janela curta melhor. A decisão deve considerar média, mediana, pior fold, dispersão de Rank IC, Precision@10/NDCG@10 e, após rerun, CAGR/Sharpe/MDD/turnover líquidos.

## Regime features

Foi criado um painel **candidato**, não habilitado automaticamente, com um conjunto pequeno e interpretável:

- retorno de mercado 21d/63d;
- realized volatility do mercado;
- breadth;
- cross-sectional dispersion;
- market drawdown;
- SELIC target;
- mudança da SELIC;
- momentum × market volatility;
- volatility × market volatility.

SELIC é anexada com atraso conservador de um dia para não presumir disponibilidade antes do fechamento da B3.

**IPCA não foi adicionado ao modelo.** O materializador SGS atual preserva a data da observação, mas não uma release timestamp auditável do mês de referência. Juntar IPCA diretamente pela data de referência poderia introduzir look-ahead. Essa omissão é deliberada e mais segura do que “cumprir a lista” com leakage.

Câmbio e commodities também não são habilitados sem um relógio de disponibilidade governado.

A lógica segue Gu, Kelly & Xiu: interações não lineares podem ser úteis, mas precisam provar ganho OOS; o simples fato de uma variável ser macro não justifica sua inclusão.

## Modelos

O ensemble mantém LightGBM, CatBoost e Ridge como componentes comparáveis. Os relatórios de importância e previsões por componente permitem verificar se a degradação está concentrada nas árvores, no modelo linear ou no ensemble.

Deep learning continua fora de escopo nesta etapa.

ElasticNet permanece como baseline experimental a ser acrescentado somente após o primeiro rerun de corretude; não foi inserido no ensemble de produção sem evidência. Esse é um dos poucos itens opcionais do prompt que deliberadamente não altera o modelo antes de haver diagnóstico numérico corrigido.

## Target

O painel passa a expor três definições para comparação development-only:

1. `future_return` — total return executável;
2. `target_excess_return` — retorno futuro menos média cross-sectional do mercado;
3. `target_cross_sectional_rank` — rank relativo do retorno futuro.

Todos começam depois do signal close. Nenhum target é escolhido usando 2026.

## Transaction costs e turnover

O backtest development gera sensitivity para:

- 5 bps;
- 10 bps;
- 25 bps;
- 50 bps;
- 100 bps;

com gross CAGR/Sharpe (0 bps), net CAGR/Sharpe e turnover.

O rank buffer/hysteresis existente foi preservado. Não foram ativados minimum holding period, turnover penalty ou QKP turnover penalty por padrão porque ainda não existe evidência pós-correção mostrando que o benefício compensa a distorção de sinal. Esses itens podem ser adicionados em uma segunda rodada development-only se o rerun mostrar que o buffer é insuficiente.

## Research V2 e QKP ablation

O diagnóstico separa:

1. ML Top-K equal weight;
2. candidate screening;
3. QKP com os parâmetros atuais da API (`candidate_count=18`, `risk_aversion=0,7`, `uncertainty_penalty=0,5`, cardinalidade 6–10);
4. mesmos selecionados com HRP.

Isso evita creditar ao ML um ganho que veio do QKP ou da alocação. Os parâmetros do QKP não são retunados em 2026.

A composição do `research_v2` também permanece explícita: ML percentile, multifactor e uncertainty. Uma “ablation de risco” não é inventada como se fizesse parte desse score; risco é avaliado na camada de portfolio construction.

## Acceptance gate corrigido

O antigo `accepted` misturava poucos booleanos e não mostrava limiar/valor real. Agora existem três níveis:

### `model_signal_accepted`

- `recent_rank_ic >= 0,02`;
- `top10_spread > 0`.

### `portfolio_strategy_accepted`

- net Sharpe `>= 0,75`;
- MDD `>= -35%`;
- turnover `<= 5x/ano`;
- CAGR maior que 1/N.

### `production_ready`

Além dos dois anteriores:

- leakage tests precisam passar;
- o uso de 2026 como diagnóstico precisa estar explicitamente reconhecido.

Cada gate grava `gate_name`, operador, threshold, actual e pass/fail. Os thresholds não são alterados em resposta ao resultado de 2026.

## Incerteza estatística

`make bootstrap-performance` usa **moving block bootstrap**, não IID, para CIs de:

- CAGR;
- Sharpe;
- Rank IC;
- Top-10 spread.

Também grava:

- IC standard error;
- IC t-stat;
- ICIR;
- Top-10 hit rate.

## Multiple testing

Todo experimento pode ser persistido em `data/gold/experiments/` com:

- `experiment_id`;
- timestamp;
- git commit;
- features;
- parâmetros;
- janela de treino;
- folds de validação;
- métricas;
- notas.

A grade de políticas escreve todos os resultados, inclusive os ruins. Não existe código que execute centenas de configurações e guarde apenas a melhor silenciosamente.

## 2026

2026 é diagnostic-only em toda a nova trilha.

O loader do modelo exige:

- `knowledge_cutoff < 2026-01-01`;
- `target_end_max < 2026-01-01`.

`make backtest USE_TRAINED_MODEL=1` produz um relatório de 2026 com `accepted: null`. Portanto, um resultado bom ou ruim em 2026 não pode alterar o gate.

Mensagem persistida nos outputs:

> 2026 has already been observed during research and is no longer a pristine holdout.

O próximo holdout realmente novo é futuro.

## Testes adicionados

Foram adicionadas verificações para:

- target começa depois do signal close;
- purge de labels sobrepostos;
- knowledge cutoff respeita fim do label;
- mês usa uma data comum;
- recency weights determinísticos/monótonos;
- next-close execution;
- transaction cost no momento de execução;
- mixed-date cross-section rejeitado;
- Top-K metrics;
- ensemble determinístico com mesma seed/dados;
- future corporate action não altera trailing return totalmente pré-evento;
- PIT covariance invariável a alterações em preços futuros.

Os testes CVM PIT existentes continuam sendo parte da suíte.

## O que precisa ser rerodado localmente

Depois de aplicar esta correção, **não reutilize** o painel mensal nem o artefato antigo. A semântica de target e month-end mudou.

Sequência recomendada:

```bash
make refresh-research-data START_YEAR=2011 END_YEAR=2026
make test
make diagnose-model-degradation
make risk-benchmarks
make tune TRIALS=25
make train
make backtest
make bootstrap-performance
make decompose-performance
make qkp-ablation
make backtest USE_TRAINED_MODEL=1
```

A última linha é a única etapa que reporta 2026 e deve ser executada **depois** de congelar a configuração usando apenas 2019–2025.

## Critério de sucesso

Não é sucesso “consertar” o CAGR de 2026. O rerun deve privilegiar:

- estabilidade entre folds;
- redução da dependência de 2014–2021;
- Rank IC ou Top-K mais estáveis;
- MDD menor;
- turnover menor;
- retorno líquido positivo em vários folds;
- vantagem consistente sobre 1/N e multifator;
- comportamento recente menos frágil.

Um CAGR menor que 37,82% pode ser uma melhoria científica se vier acompanhado de Sharpe, drawdown, estabilidade e evidência OOS superiores.

## Literatura e decisões concretas

### Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*

Usado para justificar avaliação cross-sectional rigorosa, comparação de modelos não lineares e candidatos de interação característica × regime. Não justifica adicionar dezenas de features automaticamente.

### DeMiguel, Garlappi & Uppal (2009), *Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?*

Usado para tornar 1/N benchmark obrigatório e para não tratar otimização sofisticada como economicamente relevante sem superar um baseline ingênuo OOS.

### Ledoit & Wolf, *Honey, I Shrunk the Sample Covariance Matrix*

Usado para manter shrinkage covariance e compará-la explicitamente com sample covariance. A matriz é sempre point-in-time.

### Jegadeesh & Titman

Usado para manter momentum 12–1 como baseline econômico, não como justificativa para escolher seu peso olhando 2026.

### Novy-Marx, *The Other Side of Value: The Gross Profitability Premium*

Usado para manter gross profitability como componente de qualidade point-in-time quando a cobertura CVM existe.

## Checklist do prompt

| Parte | Status nesta correção |
|---|---|
| 1. Auditoria de corretude | Implementada; 5 problemas concretos identificados/corrigidos |
| 2. Regimes A/B | Harness PSI/KS/Wasserstein implementado |
| 3. Feature IC | Implementado por mês/ano/regime, ICIR/SE/t-stat |
| 4. Importância por fold | LGB gain/split, CatBoost PVC, Ridge implementados |
| 5. Top-K | K 5/10/20/30/50 implementado |
| 6. Decomposição CAGR | outputs detalhados + script de decomposição |
| 7. Benchmarks | 1/N, liquidez, fatores, ML e risk benchmarks; índices oficiais aguardam fonte governada |
| 8. Rolling vs expanding | 3/5/7/10 + expanding em development-only |
| 9. Recency weighting | half-lives 1/2/3/5 implementados |
| 10. Regime macro | conjunto pequeno candidato; IPCA/FX não ativados sem availability clock confiável |
| 11. Modelos | LGB/Cat/Ridge/ensemble comparáveis; DL não adicionado; ElasticNet não promovido sem rerun |
| 12. Target | raw/excess/rank implementados |
| 13. Purging/embargo | implementados |
| 14. Custos | 5/10/25/50/100 bps implementados |
| 15. Turnover | buffer preservado; novas penalidades condicionadas ao rerun |
| 16. Risk control | risk benchmarks + decomposição por camada; não inventa componente inexistente |
| 17. QKP | ML vs QKP vs allocation separado |
| 18. Covariância | sample vs Ledoit–Wolf e `as_of` obrigatório |
| 19. Regime detection | nenhum HMM; candidatos simples primeiro |
| 20. Acceptance gate | três gates, threshold/actual/pass-fail |
| 21. Estatística | moving-block bootstrap + IC stats |
| 22. Multiple testing | experiment registry implementado |
| 23. 2026 diagnóstico | hard-coded como diagnostic-only no protocolo padrão |
| 24. Arquivos | mapa real documentado; nenhuma path inventada para código existente |
| 25. Alterações | somente correções demonstradas ativadas; hipóteses ficam candidates |
| 26. Testes | invariantes novas adicionadas; suíte completa deve rodar no Docker local |
| 27. Relatório final | este documento + outputs gerados pelos scripts |
| 28. Critério de sucesso | gate foca robustez, custos, 1/N e estabilidade |

## Limitação de verificação desta entrega

O ambiente usado para preparar o pacote não contém Docker nem Polars e o GitHub não armazena os Parquets de pesquisa/model artifact. Foi possível executar `compileall`/`py_compile` de todos os arquivos Python e validar a sintaxe do Makefile, mas **não** executar a suíte Polars nem reproduzir os backtests numéricos aqui. A validação empírica final precisa ocorrer no Docker do próprio projeto após reconstruir os dados.

Essa limitação é intencionalmente registrada em vez de declarar testes ou resultados que não foram executados.
