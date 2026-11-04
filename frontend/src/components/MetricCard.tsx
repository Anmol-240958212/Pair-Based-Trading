interface Props {
  label: string;
  value: string;
  note?: string;
  tone?: "default" | "good" | "warn";
}

export function MetricCard({ label, value, note, tone = "default" }: Props) {
  return (
    <article className={`metric-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </article>
  );
}
