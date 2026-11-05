import PlotlyChart from "react-plotly.js";
import type { Data, Layout } from "plotly.js";

interface Props {
  data: Data[];
  title: string;
  description?: string;
  layout?: Partial<Layout>;
}

export function Plot({ data, title, description, layout = {} }: Props) {
  return (
    <div className="chart-card">
      <PlotlyChart
        data={data}
        layout={{
          title: { text: title, font: { color: "#e8f0fb", size: 16 } },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "#9fb0c5", family: "Inter, system-ui, sans-serif" },
          margin: { l: 55, r: 24, t: 55, b: 48 },
          xaxis: { gridcolor: "#1f3045", zerolinecolor: "#31445c" },
          yaxis: { gridcolor: "#1f3045", zerolinecolor: "#31445c" },
          autosize: true,
          ...layout,
        }}
        config={{ displaylogo: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
      {description && <p className="chart-explanation">{description}</p>}
    </div>
  );
}
