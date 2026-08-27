"use client";

import { useEffect, useMemo, useState } from "react";
import { Activity, ArrowUpRight, BrainCircuit, Database, ShieldCheck, Sparkles, Target } from "lucide-react";
import { Asset, Optimization, loadRanking, loadSystemStatus, optimizePortfolio } from "../lib/api";
import { ScoreRing } from "./ScoreRing";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const brl = (x: number) => x.toLocaleString("pt-BR", { style: "currency", currency: "BRL" });

export default function Dashboard() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [demoMode, setDemoMode] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  const [result, setResult] = useState<Optimization | null>(null);
  const [budget, setBudget] = useState(50000);
  const [risk, setRisk] = useState(0.7);
  const [positions, setPositions] = useState(10);

  useEffect(() => {
    Promise.all([loadRanking(), loadSystemStatus()])
      .then(([ranking, system]) => {
        setAssets(ranking);
        setDemoMode(system.demo_mode);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const averageScore = useMemo(
    () => assets.length ? assets.slice(0, 10).reduce((s, a) => s + a.ml_score, 0) / Math.min(10, assets.length) : 0,
    [assets]
  );

  async function runOptimization() {
    setOptimizing(true);
    setError("");
    try {
      const response = await optimizePortfolio({
        budget,
        min_positions: Math.max(4, positions - 3),
        max_positions: positions,
        candidate_count: 30,
        risk_aversion: risk,
        uncertainty_penalty: 0.5,
        allocation: "hrp",
      });
      setResult(response);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Falha na otimização");
    } finally {
      setOptimizing(false);
    }
  }

  return (
    <main>
      <header className="topbar">
        <div className="brand"><div className="brandMark">B3</div><div><strong>Portfolio Lab</strong><span>Quant Research Platform</span></div></div>
        <div className="statusPill"><span className="liveDot" /> research environment</div>
      </header>

      <section className="hero">
        <div>
          <div className="eyebrow"><Sparkles size={14} /> ML signals + exact optimization</div>
          <h1>Decisões quantitativas,<br/><em>auditáveis por construção.</em></h1>
          <p>Ranking de ações da B3, sinais de machine learning, risco robusto e seleção exata via Quadratic Knapsack.</p>
        </div>
        <div className="heroMetric">
          <span>Top-10 model score</span><strong>{averageScore.toFixed(1)}</strong><small>demo / latest snapshot</small>
        </div>
      </section>

      {error && <div className="alert">{error}</div>}
      {demoMode && (
        <div className="demoNotice">
          <strong>Modo demo:</strong> preços, scores e previsões desta tela são sintéticos.
          Ingira B3/CVM/BCB e treine o modelo para publicar um snapshot de pesquisa real.
        </div>
      )}

      <section className="metricsGrid">
        <div className="metricCard"><BrainCircuit/><span>Signal engine</span><strong>Ensemble ML</strong><small>LightGBM · CatBoost · Ridge</small></div>
        <div className="metricCard"><Target/><span>Optimizer</span><strong>Exact QKP</strong><small>SCIP + verified B&amp;B fallback</small></div>
        <div className="metricCard"><ShieldCheck/><span>Risk</span><strong>Ledoit-Wolf</strong><small>HRP / minimum variance allocation</small></div>
        <div className="metricCard"><Database/><span>Data</span><strong>B3 + CVM + BCB</strong><small>point-in-time architecture</small></div>
      </section>

      <section className="contentGrid">
        <div className="panel rankingPanel">
          <div className="panelHead"><div><span className="kicker">DISCOVER</span><h2>Ranking quantitativo</h2></div><div className="smallBadge"><Activity size={14}/> latest model snapshot</div></div>
          <div className="tableWrap">
            <table>
              <thead><tr><th>#</th><th>Ativo</th><th>ML</th><th>Alpha esperado</th><th>Vol.</th><th>Setor</th></tr></thead>
              <tbody>
                {loading && <tr><td colSpan={6} className="empty">Carregando ranking...</td></tr>}
                {assets.slice(0, 12).map((asset, index) => (
                  <tr key={asset.ticker} title={asset.explanation}>
                    <td className="rank">{String(index + 1).padStart(2, "0")}</td>
                    <td><div className="ticker"><strong>{asset.ticker}</strong><span>{asset.company}</span></div></td>
                    <td><ScoreRing score={asset.ml_score}/></td>
                    <td className={asset.predicted_excess_return >= 0 ? "positive" : "negative"}>{pct(asset.predicted_excess_return)} <ArrowUpRight size={14}/></td>
                    <td>{pct(asset.volatility_annual)}</td><td><span className="sectorTag">{asset.sector}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <aside className="panel builder">
          <div className="panelHead"><div><span className="kicker">OPTIMIZE</span><h2>Portfolio Builder</h2></div></div>
          <label>Capital disponível <strong>{brl(budget)}</strong></label>
          <input type="range" min="5000" max="250000" step="5000" value={budget} onChange={(e) => setBudget(Number(e.target.value))}/>
          <label>Número máximo de ações <strong>{positions}</strong></label>
          <input type="range" min="6" max="15" value={positions} onChange={(e) => setPositions(Number(e.target.value))}/>
          <label>Aversão ao risco <strong>{risk.toFixed(1)}</strong></label>
          <input type="range" min="0" max="2" step="0.1" value={risk} onChange={(e) => setRisk(Number(e.target.value))}/>
          <div className="riskScale"><span>Retorno</span><span>Equilíbrio</span><span>Defensivo</span></div>
          <button onClick={runOptimization} disabled={optimizing || assets.length === 0}>{optimizing ? "Resolvendo QKP..." : "Construir portfólio"}</button>
          <p className="finePrint">A seleção é exata. Os pesos são calculados separadamente com HRP para não misturar seleção combinatória e sizing.</p>
        </aside>
      </section>

      {result && (
        <section className="resultSection">
          <div className="resultHeader"><div><span className="kicker">PORTFOLIO</span><h2>Carteira otimizada</h2></div><div className="solverBadge">{result.solver} · {result.exact ? "ótimo provado" : result.status}</div></div>
          <div className="resultMetrics">
            <div><span>Alpha esperado</span><strong>{pct(result.expected_excess_return)}</strong></div>
            <div><span>Volatilidade</span><strong>{pct(result.expected_volatility)}</strong></div>
            <div><span>Posições</span><strong>{result.positions.length}</strong></div>
            <div><span>Objetivo QKP</span><strong>{result.objective.toFixed(1)}</strong></div>
          </div>
          <div className="positionsGrid">
            {result.positions.map((p) => (
              <div className="positionCard" key={p.ticker}>
                <div><strong>{p.ticker}</strong><span>{p.company}</span></div>
                <div className="weight">{pct(p.weight)}</div>
                <div className="bar"><span style={{width: `${Math.min(100, p.weight * 500)}%`}} /></div>
                <small>{brl(p.amount)} · alpha {pct(p.expected_excess_return)}</small>
              </div>
            ))}
          </div>
          <div className="notes">{result.notes.map((n) => <p key={n}>{n}</p>)}</div>
        </section>
      )}

      <footer><span>B3 Portfolio Lab · research prototype</span><span>Não constitui recomendação de investimento.</span></footer>
    </main>
  );
}
