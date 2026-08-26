export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Asset = {
  ticker: string;
  company: string;
  sector: string;
  price: number;
  predicted_excess_return: number;
  prediction_uncertainty: number;
  volatility_annual: number;
  ml_score: number;
  quant_score: number;
  liquidity_score: number;
  explanation: string;
};

export type Position = {
  ticker: string;
  company: string;
  sector: string;
  weight: number;
  amount: number;
  price: number;
  expected_excess_return: number;
  volatility_annual: number;
};

export type Optimization = {
  status: string;
  solver: string;
  exact: boolean;
  objective: number;
  expected_excess_return: number;
  expected_volatility: number;
  positions: Position[];
  notes: string[];
};

function csrfToken(): string | undefined {
  if (typeof document === "undefined") return undefined;
  const match = document.cookie
    .split("; ")
    .find((entry) => entry.startsWith("csrf_token="));
  return match ? decodeURIComponent(match.slice("csrf_token=".length)) : undefined;
}


export type SystemStatus = {
  status: string;
  database: string;
  demo_mode: boolean;
};

export async function loadSystemStatus(): Promise<SystemStatus> {
  const response = await fetch(`${API_URL}/api/v1/system/health`, { cache: "no-store" });
  if (!response.ok) throw new Error("API indisponível");
  return response.json();
}

export async function loadRanking(): Promise<Asset[]> {
  const response = await fetch(`${API_URL}/api/v1/market/ranking?limit=30`, {
    cache: "no-store",
    credentials: "include",
  });
  if (!response.ok) throw new Error("API indisponível");
  return response.json();
}

export async function optimizePortfolio(payload: Record<string, number | string>): Promise<Optimization> {
  const csrf = csrfToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (csrf) headers["X-CSRF-Token"] = csrf;
  const response = await fetch(`${API_URL}/api/v1/portfolio/optimize`, {
    method: "POST",
    headers,
    credentials: "include",
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: "Falha na otimização" }));
    throw new Error(body.detail ?? "Falha na otimização");
  }
  return response.json();
}
