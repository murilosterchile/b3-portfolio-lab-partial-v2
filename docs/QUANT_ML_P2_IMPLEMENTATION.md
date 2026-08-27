# Quant/ML P2 — protocolo operacional

Esta revisão mantém `2026` exclusivamente diagnóstico. Nenhum threshold, custo,
hiperparâmetro, universo, K, peso ou método de alocação é escolhido com dados cujo
`knowledge_cutoff` ultrapasse `2025-12-31`.

## Contratos de dados externos

- CDI oficial: `data/silver/bcb/series=cdi_daily/part-000.parquet`, gerado da série
  SGS 12 (`% ao dia`) por `make ingest-macro START=2011-01-01`; o backtest converte
  percentuais para retornos decimais. Alternativamente aceita
  `data/silver/benchmarks/cdi.parquet` com `trade_date,daily_return`.
- Ibovespa total return: `data/silver/benchmarks/ibov_total_return.parquet`, com
  `trade_date` e `adjusted_close` (ou `daily_return`). A série deve ser governada e
  identificada como índice de retorno total. Ausência/cobertura parcial é reportada;
  não há preenchimento por proxy.

O Sharpe principal usa retorno diário líquido menos CDI e a volatilidade desse
excesso. `raw_sharpe` permanece secundário. Caixa não investido acumula CDI.

## Custos e capacidade

`research_v2` usa, por regra registrada, taxa de 3 bps, spread por faixas de ADV,
impacto monotônico proporcional à raiz da participação e cap de 2% do ADV dos 21
pregões anteriores à execução. O bid/ask EOD só é usado quando explicitamente habilitado; por padrão são
usadas faixas conservadoras de ADV. `costs.parquet` separa fees, spread e impact.

O relatório de development também executa o baseline fixo de 5/10/25/50/100 bps e
a curva de capacidade para capitais de R$100 mil/R$1 milhão/R$10 milhões, nos caps
de 1%/2%/5% do ADV. Esses valores são sensitivity pré-definida, não calibração.

## Seleção, inferência e gates

- `make tune` executa nested walk-forward: Optuna só vê inner folds; cada outer year
  é avaliado uma vez; todos os trials/configurações são persistidos.
- IC e spread entram no gate pelo limite inferior do moving-block bootstrap.
- performance entra pelo excesso contra CDI e Ibovespa, DSR e número de tentativas.
- `production_ready` exige um manifesto atual da suíte registrada. O manifesto só é
  produzido por `make research-tests` e é invalidado por qualquer mudança no código
  ou nos testes.
- `make bootstrap-performance` calcula diferenciais alinhados contra benchmarks,
  DSR e PBO. O bloco de três meses excede conservadoramente o overlap do target de
  21 barras.

## Atribuição

`make decompose-performance` produz rolling excess 12/24/36 meses, alpha/beta anual
contra IBOV, episódios/duração/recuperação de drawdown, regimes causais de mercado,
volatilidade e juros, contribuição por ticker/setor/emissor, concentração e
decomposição de turnover/custos. Regimes usam somente janelas anteriores ao mês
classificado.

Um resultado negativo continua válido. Ausência de CDI/IBOV, CI inferior não
positivo, DSR insuficiente ou testes temporais sem manifesto impedem promoção; o
pipeline não relaxa essas condições.

Ordem recomendada: `make ingest-macro START=2011-01-01`,
`make diagnose-model-degradation`, `make tune TRIALS=25`, `make research-tests`,
`make backtest`, `make bootstrap-performance` e `make decompose-performance`.
Somente depois de congelar tudo até 2025 deve-se executar
`make backtest USE_TRAINED_MODEL=1` para o diagnóstico de 2026.
