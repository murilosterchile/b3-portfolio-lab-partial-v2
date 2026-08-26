export function ScoreRing({ score }: { score: number }) {
  const value = Math.max(0, Math.min(100, score));
  return (
    <div className="scoreRing" style={{ "--score": `${value * 3.6}deg` } as React.CSSProperties}>
      <span>{Math.round(value)}</span>
    </div>
  );
}
