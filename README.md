# B3 Portfolio Lab

Protótipo local de pesquisa quantitativa para construir carteiras de ações da B3 combinando **dados históricos oficiais**, machine learning, modelos quantitativos de risco e **seleção exata via Quadratic Knapsack Problem (QKP)**.

> **Software de pesquisa. Não é recomendação de investimento.** Os valores do modo demo são sintéticos. O COTAHIST bruto da B3 continua imutável e não é total-return; a versão parcial v2 acrescenta uma camada analítica ajustada a partir de eventos corporativos públicos da B3. Eventos complexos/incompletos continuam sendo tratados como limitação de pesquisa, não como retorno inventado.


## Partial v2 research upgrade (August 2026)

The P2 economic-inference and capacity protocol is documented in
[`docs/QUANT_ML_P2_IMPLEMENTATION.md`](docs/QUANT_ML_P2_IMPLEMENTATION.md).

This package includes a partial research-grade upgrade focused on the weaknesses found in the
first real backtest. The main changes are:

- official B3 corporate-action ingestion (cash dividends/JCP, splits, reverse splits and bonuses);
- an immutable `b3_prices_adjusted` analytical layer and quarantine of invalid raw prices;
- automatic historical ticker -> CNPJ -> `CD_CVM` mapping from CVM FCA + CAD, including delisted
  names when present in FCA;
- point-in-time quality/profitability features, including gross profitability;
- a covariance-like QKP pair-risk term using both correlation and asset volatility;
- rank-buffer turnover control;
- mandatory OOS comparison of ML, momentum, multifactor and `research_v2` strategies.

For the most automated data build currently available:

```bash
make refresh-research-data START_YEAR=2011 END_YEAR=2026
make backtest
```

The corporate-action stage uses a persistent per-issuer cache and six concurrent HTTP requests by
default. This `balanced` profile limits Polars to four threads so the machine remains interactive.
The first run downloads missing issuers; later runs validate and reuse every successful or valid
empty response. `FORCE=1` explicitly refreshes all issuers.

```bash
# Conservative profile for a notebook in active use (3 HTTP requests, 2 CPU threads)
PIPELINE_PROFILE=low-impact make refresh-research-data START_YEAR=2011 END_YEAR=2026

# Fine-grained overrides
B3_HTTP_CONCURRENCY=4 PIPELINE_CPU_THREADS=3 \
  make refresh-research-data START_YEAR=2011 END_YEAR=2026

# Re-fetch all B3 corporate-action responses
make refresh-research-data START_YEAR=2011 END_YEAR=2026 FORCE=1
```

Available profiles are `balanced` (6 HTTP/4 CPU, default), `low-impact` (3/2), and `fast`
(10/6). Network controls can also be set independently with `B3_HTTP_TIMEOUT_CONNECT` (default
10 seconds), `B3_HTTP_TIMEOUT_READ` (30 seconds), and `B3_HTTP_RETRIES` (3 retries after the first
attempt). Retries apply only to connection/timeout errors and HTTP 429/500/502/503/504, with
exponential backoff, jitter, and `Retry-After` support.

Raw responses are stored under `data/raw/b3_corporate_actions/issuer=<root>/` with response and
metadata files (SHA-256, status, timestamp, source, HTTP status, retry count). Writes use a `.part`
file and atomic rename, so an interrupted run resumes from already validated issuers. The normalized
Parquet remains sorted and deduplicated by its natural event key before publication.

The comparison is written to:

```text
data/gold/backtests/comparison.json
```

See `docs/RESEARCH_BASIS_V2.md` for the academic/repository basis and remaining limitations. The
research gate reports PASS/FAIL; it never changes parameters merely to force a positive holdout.

## O que está implementado

- Downloader e parser do **COTAHIST oficial da B3** (layout fixo de 245 bytes).
- Camadas de dados imutáveis com SHA-256, Parquet + ZSTD e separação raw/silver/gold.
- Ingestão de **CVM ITR/DFP** preservando `DT_RECEB` para joins point-in-time.
- Ingestão de séries macro do **BCB SGS**.
- Feature engineering sem uso intencional de dados futuros.
- Ensemble de ML com **LightGBM + CatBoost + Ridge**, avaliação cross-sectional por data, validação walk-forward e tuning temporal opcional com Optuna.
- Modelos quantitativos: momentum, low-volatility, Ledoit-Wolf, HRP, inverse volatility, minimum variance e Black-Litterman.
- Seleção exata de ativos por QKP:
  - **SCIP/PySCIPOpt** como backend principal, com linearização exata dos termos quadráticos;
  - branch-and-bound exato independente em Python e C++23 para verificação/fallback em instâncias pequenas.
- Backtester mensal com turnover e custos de transação.
- Backend **FastAPI**.
- **PostgreSQL** como banco de dados/system of record da aplicação; Valkey somente para cache/rate limiting efêmero.
- Autenticação com Argon2id, sessões server-side, cookies HttpOnly, CSRF e security headers.
- Front-end Next.js/TypeScript responsivo, limpo e orientado à pesquisa.
- Ambiente local por Docker Compose ou Podman Compose.

## 1. Rodar o protótipo em modo demo

Pré-requisitos:

- Docker Engine + Docker Compose; ou
- Podman + `podman compose`.

Na raiz do projeto:

```bash
cp .env.example .env
```

Antes de qualquer uso fora da sua própria máquina, altere `SECRET_KEY` no `.env`.

Depois:

```bash
docker compose up -d --build
```

Abra:

- Front-end: http://localhost:3000
- Swagger/OpenAPI: http://localhost:8000/docs
- Health check: http://localhost:8000/api/v1/system/health

Com Podman:

```bash
COMPOSE='podman compose' make up
```

O modo demo é ativado por `DEMO_MODE=true` e cria um snapshot sintético para testar o fluxo completo:

```text
front-end -> API -> PostgreSQL -> QKP exato -> alocação HRP -> resposta
                      │
                    Valkey (rate limiting)
```

Com `DEMO_MODE=false`, o endpoint de otimização exige sessão autenticada e CSRF; no modo demo esse gate é dispensado para facilitar a avaliação local.

## 2. Baixar e processar histórico oficial da B3

O endpoint anual usado pelo projeto segue o padrão público do COTAHIST:

```text
https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{AAAA}.ZIP
```

Por exemplo:

```bash
make ingest-b3 YEAR=2025
make ingest-b3 YEAR=2024
make ingest-b3 YEAR=2023
```

Os arquivos são armazenados em:

```text
data/raw/b3/<ano>/
```

Após o parsing:

```text
data/silver/b3_prices/year=<ano>/part-000.parquet
```

Com pelo menos dois anos disponíveis, o pipeline também recria:

```text
data/gold/features/monthly_features.parquet
```

### Se a B3 bloquear o download automatizado

O endpoint legado pode apresentar proteção anti-bot em alguns acessos. Baixe manualmente o ZIP pela página oficial de Cotações Históricas e salve, por exemplo, em:

```text
data/raw/b3/2025/COTAHIST_A2025.ZIP
```

Depois execute:

```bash
make ingest-b3 YEAR=2025 NO_DOWNLOAD=1
```

## 3. Ingerir dados macro do Banco Central

```bash
make ingest-macro START=2011-01-01
```

O protótipo já possui configuração inicial para SELIC, IPCA e USD/BRL pelo SGS.

## 4. Ingerir ITR/DFP da CVM

```bash
make ingest-cvm YEAR=2025 DOC=ITR
make ingest-cvm YEAR=2025 DOC=DFP
```

O pipeline mantém `DT_RECEB`, pois um balanço só pode virar feature após ter sido recebido/publicado pela CVM.

O caminho preferencial da versão parcial v2 **não usa fuzzy matching nem exige manutenção manual do bridge**. Ele baixa o CAD e os FCA anuais oficiais da CVM e constrói historicamente:

```text
ticker -> CNPJ -> CD_CVM -> intervalo de vigência
```

Execute:

```bash
make ingest-cvm-registry START_YEAR=2010 END_YEAR=2026
make build-fundamentals
```

O FCA fornece `Codigo_Negociacao` e datas de negociação, permitindo recuperar também emissores/tickers históricos quando presentes no arquivo. O CAD fornece a ligação por CNPJ ao `CD_CVM`. O join de fundamentos continua point-in-time: uma demonstração só pode entrar quando `DT_RECEB <= trade_date`.

`make generate-issuer-bridge` continua disponível apenas como fallback/revisão para nomes que a fonte oficial não resolveu. Ele não deve substituir silenciosamente um identificador oficial.

O arquivo `monthly_features_with_fundamentals.parquet` passa a ser preferido automaticamente por tuning, treino e backtest.

## 5. Ajustar hiperparâmetros e treinar o modelo

Com pelo menos cinco anos de histórico, opcionalmente execute:

```bash
make tune TRIALS=25
```

O Optuna usa somente anos anteriores ao último ano do painel, que permanece como holdout intocado. Depois de carregar histórico suficiente:

```bash
make train
```

O treinamento (`make train`):

1. carrega os snapshots mensais e prefere o painel com fundamentos quando disponível;
2. executa validação temporal walk-forward;
3. aplica imputação/escala apenas usando o período de treino;
4. treina LightGBM, CatBoost e Ridge;
5. forma um ensemble;
6. mede Rank IC, top-minus-bottom e hit rate;
7. salva o modelo em `models/`;
8. publica o snapshot mais recente no PostgreSQL, quando o banco está disponível.

O modelo não é considerado "bom" apenas por ter erro baixo in-sample. A arquitetura foi feita para rejeitar modelos que não sustentam sinal fora da amostra.

## 6. Rodar backtest walk-forward

```bash
make backtest
```

Saídas principais:

```text
data/gold/backtests/ml_top10/
data/gold/backtests/momentum_top10/
data/gold/backtests/multifactor_top10/
data/gold/backtests/research_v2/
data/gold/backtests/comparison.json
```

As quatro estratégias são comparadas nos mesmos períodos OOS. `research_v2` combina o ranking do ML com um prior multifator e incerteza, sem retunar automaticamente o mesmo holdout caso o gate falhe. O script treina apenas com anos anteriores ao ano de teste e cobra custo configurável sobre turnover. O turnover reportado é de mão única, incluindo caixa:

```text
0.5 * (sum(abs(w_target - w_before)) + abs(cash_target - cash_before))
```

Os custos usam a mesma base. Os sinais são consolidados em um único rebalanceamento por mês; a curva começa quando existe o primeiro sinal acionável. Como o COTAHIST é bruto, transições identificadas por mudança de `distribution_number` são excluídas e contadas explicitamente. Retornos extremos sem esse marcador interrompem o backtest.

## 7. Como funciona o QKP

A seleção de ativos resolve:

```text
max  sum_i p_i x_i + sum_{i<j} q_ij x_i x_j
```

sujeito a orçamento, cardinalidade e limites opcionais por setor.

`p_i` representa utilidade individual estimada: alpha previsto, uma contribuição controlada do score quantitativo (momentum + baixa volatilidade + qualidade/profitabilidade point-in-time), liquidez e penalidade de incerteza.

`q_ij` representa interação entre os ativos. Na versão parcial v2, a penalização usa uma escala covariance-like: correlação combinada à volatilidade dos dois ativos, em vez de tratar toda correlação igual independentemente do risco individual.

O backend principal usa SCIP. O produto `x_i*x_j` é substituído por uma variável binária `y_ij` com:

```text
y_ij <= x_i
y_ij <= x_j
y_ij >= x_i + x_j - 1
```

Isso mantém o problema matematicamente equivalente e permite coeficientes quadráticos negativos, necessários quando correlação/covariância representa penalidade de risco.

A resposta só recebe `exact=true` quando o solver prova a otimalidade.

## 8. Seleção e alocação são etapas diferentes

O QKP responde:

> quais ativos devem entrar?

Depois HRP/minimum variance/inverse volatility responde:

> quanto do capital colocar em cada ativo selecionado?

Essa separação permite comparar cientificamente:

- ML + equal weight;
- ML + QKP + equal weight;
- ML + QKP + HRP;
- fatores quantitativos + QKP;
- momentum simples + equal weight.

Assim é possível saber se o QKP realmente adicionou valor.

## 9. Testar o solver nativo exato

```bash
make native
./native/qkp/build/qkp_tests
```

Durante a geração deste projeto, o solver C++ foi compilado e o teste passou. A implementação Python também foi comparada contra brute force em instâncias aleatórias pequenas com coeficientes positivos e negativos.

## 10. Estrutura

```text
apps/web/                    front-end Next.js
services/api/                FastAPI + PostgreSQL
packages/portfolio_core/     dados, ML, quant, QKP e backtest
native/qkp/                  solver exato independente em C++23
scripts/                     ingestão, treino e backtest
infra/postgres/              inicialização do PostgreSQL
docs/                        relatório e decisões técnicas
data/                        datasets locais (não versionados)
models/                      modelos treinados (não versionados)
```

## 11. PostgreSQL

PostgreSQL é o **único armazenamento persistente transacional e system of record da aplicação**. Identidade, sessões, snapshots consumidos pelo produto e portfólios ficam nele. O Valkey guarda apenas contadores efêmeros de rate limiting.

Parquet é usado somente como formato de arquivo analítico imutável para grandes históricos. Isso evita empurrar décadas de dados brutos para o banco transacional sem necessidade e mantém o sistema de registro da aplicação no PostgreSQL, conforme definido no projeto.

## 12. Limitações que precisam ser resolvidas antes de comercializar

- Raw COTAHIST remains unadjusted; partial v2 can build a B3 corporate-action-adjusted analytical layer, but complex reorganizations and endpoint coverage for delisted issuers still require audit.
- The bridge can now be generated deterministically from CVM FCA + CAD; unresolved/overlapping historical identities still fail or require audit.
- O demo usa correlações sintéticas somente quando não há histórico local suficiente; com COTAHIST disponível, a API estima a correlação a partir de retornos usando Ledoit-Wolf.
- É necessário validar direito de uso comercial/derivação/redistribuição dos dados da B3 para o produto pretendido.
- É necessário enquadramento jurídico/regulatório antes de vender recomendações individualizadas ou análise automatizada de valores mobiliários.
- Nenhum modelo de ML pode ter lucro futuro garantido; o objetivo da arquitetura é tornar essa hipótese testável e auditável.

Leia `docs/RELATORIO_TECNICO.md` para a justificativa completa das decisões.
