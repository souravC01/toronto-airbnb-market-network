import { useState, useEffect } from "react";

import generated from "./data/generated.json";

/**
 * Every number rendered by this page comes from `./data/generated.json`, which
 * `scripts/export_portfolio_data.py` derives from `results/tables/*.csv`. CI runs
 * that script with `--check` and fails the build if the JSON and the CSVs
 * disagree, so the page cannot silently drift away from the pipeline again.
 *
 * Only editorial prose is written by hand below. If you want to change a number,
 * change the pipeline and re-run the exporter.
 */

type GraphVariant = (typeof generated.graphs)[number] & { takeaway: string };
type ValidationScheme = (typeof generated.validationSchemes)[number] & { note: string };
type SensitivitySetting = (typeof generated.sensitivitySettings)[number] & { note: string };

const headline = generated.headline;

const graphNotes: Record<string, string> = {
  A: "The geographic baseline reconstructs neighbourhood-like local clusters most closely, but leaves 100 isolated components.",
  B: "Sparse ownership links bridge local clusters without overwhelming the graph with dense host cliques.",
  C: "Similarity links connect the entire city into one component and a small number of broad market segments that cross administrative lines.",
};

const algorithmNotes: Record<string, string> = {
  Louvain:
    "Primary interpretation method: recovers one large citywide background market plus a set of structured local and boutique segments.",
  Leiden:
    "Robustness check: near-identical modularity from a different partition with a larger dominant segment, which is itself evidence that the exact partition is not uniquely determined.",
};

const validationNotes: Record<string, string> = {
  random:
    "Listing-level shuffled folds reproduce a small raw R² gain that is positive in every fold. Read this alongside the permutation null rather than the complexity-adjusted column.",
  host:
    "Every host stays strictly in train or test (zero host overlap, verified per fold), so the association is not host leakage.",
  spatial:
    "Holding out whole geographic regions lowers R² because within-block price variance is lower, not because absolute error is worse — MAE is in fact the lowest of the three schemes. The gain is inconsistent across blocks.",
};

const sensitivityNotes: Record<string, string> = {
  baseline: "Standard setup: 500 m spatial radius, attribute k = 5, and 0.60/0.25/0.15 weights.",
  "radius-300-m": "A tighter radius fragments the market and shifts exact membership most strongly.",
  "radius-700-m": "A wider radius merges segments and increases agreement with official neighbourhoods.",
  "attribute-k-3": "Fewer similarity neighbours produces the closest alternative partition to the baseline.",
  "attribute-k-10": "More similarity bridges keep the community count stable while shifting cluster boundaries.",
  "spatial-heavy": "Emphasizing location over attributes restores more neighbourhood-like boundaries.",
  "attribute-heavy": "Emphasizing listing similarity produces broader, less administrative citywide segments.",
};

const graphVariants: GraphVariant[] = generated.graphs.map((graph) => ({
  ...graph,
  takeaway: graphNotes[graph.id] ?? "",
}));

const algorithms = Object.fromEntries(
  Object.entries(generated.algorithms).map(([name, data]) => [
    name,
    { ...data, note: algorithmNotes[name] ?? "" },
  ]),
) as Record<string, (typeof generated.algorithms)["Louvain"] & { note: string }>;

const topCommunitiesSummary = generated.topCommunities;

const validationSchemes: ValidationScheme[] = generated.validationSchemes.map((scheme) => ({
  ...scheme,
  note: validationNotes[scheme.id] ?? "",
}));

const sensitivitySettings: SensitivitySetting[] = generated.sensitivitySettings.map((setting) => ({
  ...setting,
  note: sensitivityNotes[setting.id] ?? "",
}));

function Arrow() {
  return <span aria-hidden="true">↗</span>;
}

function ScoreBar({ label, value, tone }: { label: string; value: number; tone: "base" | "expanded" }) {
  return (
    <div className="score-row">
      <div className="score-meta">
        <span>{label}</span>
        <strong>{value.toFixed(4)}</strong>
      </div>
      <div className="score-track" aria-label={`${label} R squared ${value.toFixed(4)}`}>
        <div className={`score-fill ${tone}`} style={{ width: `${Math.min(100, value * 145)}%` }} />
      </div>
    </div>
  );
}

export function PortfolioExperience() {
  const [graphIndex, setGraphIndex] = useState(2);
  const [algorithm, setAlgorithm] = useState<keyof typeof algorithms>("Louvain");
  const [validationIndex, setValidationIndex] = useState(1);
  const [sensitivityIndex, setSensitivityIndex] = useState(0);
  const [isDark, setIsDark] = useState(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem("theme");
      return stored === "dark";
    }
    return false;
  });
  const [showScrollTop, setShowScrollTop] = useState(false);

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isDark]);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 400) {
        setShowScrollTop(true);
      } else {
        setShowScrollTop(false);
      }
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const toggleTheme = () => {
    setIsDark((prev) => !prev);
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const deltaValues = validationSchemes.map((scheme) => Number(scheme.delta));
  const deltaRange = `${Math.min(...deltaValues) >= 0 ? "+" : ""}${Math.min(
    ...deltaValues,
  ).toFixed(4)} to ${Math.max(...deltaValues) >= 0 ? "+" : ""}${Math.max(
    ...deltaValues,
  ).toFixed(4)}`;
  const graph = graphVariants[graphIndex];
  const algorithmData = algorithms[algorithm];
  const validation = validationSchemes[validationIndex];
  const sensitivity = sensitivitySettings[sensitivityIndex];

  return (
    <div className="site-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>

      {/* Top Navigation */}
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Toronto Airbnb Network home">
          <span className="brand-mark">TN</span>
          <span className="brand-text">Toronto Airbnb Network</span>
        </a>
        <nav className="nav-links" aria-label="Case study sections">
          <a href="#network">Network</a>
          <a href="#communities">Communities</a>
          <a href="#robustness">Robustness</a>
          <a href="#sensitivity">Sensitivity</a>
          <a href="#method">Method</a>
        </nav>
        <div className="topbar-actions">
          <button
            type="button"
            className="theme-toggle-btn"
            onClick={toggleTheme}
            aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
            title={isDark ? "Switch to light theme" : "Switch to dark theme"}
          >
            <span aria-hidden="true">{isDark ? "☀️" : "🌙"}</span>
          </button>
          <a
            className="btn-pill btn-secondary btn-pill-sm"
            href="/report/EECS4414-Airbnb-Network-Analysis-Final-Report.pdf"
            target="_blank"
            rel="noreferrer"
            aria-label="Open the original report in a new tab"
          >
            Report <Arrow />
          </a>
          <a
            className="btn-pill btn-primary btn-pill-sm"
            href="https://github.com/souravC01/toronto-airbnb-market-network"
            target="_blank"
            rel="noreferrer"
          >
            GitHub <Arrow />
          </a>
        </div>
      </header>

      {/* Marquee Utility Ribbon */}
      <aside className="marquee-strip" aria-label="Key project metrics ticker">
        <div className="marquee-content">
          <div className="marquee-item">
            <span>{headline.listings} LISTINGS</span> <span className="marquee-dot">·</span>
            <span>{headline.officialAreas} OFFICIAL AREAS</span> <span className="marquee-dot">·</span>
            <span>17 MARKET COMMUNITIES</span> <span className="marquee-dot">·</span>
            <span>1 CONNECTED COMPONENT</span> <span className="marquee-dot">·</span>
            <span>TRANSDUCTIVE 5-FOLD VALIDATION</span> <span className="marquee-dot">·</span>
            <span>REPRODUCIBLE RESEARCH PIPELINE</span> <span className="marquee-dot">·</span>
          </div>
          <div className="marquee-item">
            <span>{headline.listings} LISTINGS</span> <span className="marquee-dot">·</span>
            <span>{headline.officialAreas} OFFICIAL AREAS</span> <span className="marquee-dot">·</span>
            <span>17 MARKET COMMUNITIES</span> <span className="marquee-dot">·</span>
            <span>1 CONNECTED COMPONENT</span> <span className="marquee-dot">·</span>
            <span>TRANSDUCTIVE 5-FOLD VALIDATION</span> <span className="marquee-dot">·</span>
            <span>REPRODUCIBLE RESEARCH PIPELINE</span> <span className="marquee-dot">·</span>
          </div>
        </div>
      </aside>

      <main id="main-content">
        {/* Hero Section (Pure White Canvas) */}
        <section className="hero" id="top">
          <div className="hero-grid">
            <div className="hero-copy">
              <p className="eyebrow">TORONTO AIRBNB SNAPSHOT</p>
              <h1 className="display-xl">Toronto’s Airbnb market doesn’t stop at neighbourhood lines.</h1>
              <p className="subhead">
                {headline.listings} listings. One connected market. See how proximity, ownership, and similarity reshape the city beyond its official boundaries.
              </p>
              <div className="hero-actions">
                <a className="btn-pill btn-primary" href="#network">
                  Explore the network <span aria-hidden="true">↓</span>
                </a>
                <a
                  className="btn-pill btn-secondary"
                  href="/report/EECS4414-Airbnb-Network-Analysis-Final-Report.pdf"
                  target="_blank"
                  rel="noreferrer"
                  aria-label="Open the original EECS 4414 final report in a new tab"
                >
                  View report <Arrow />
                </a>
              </div>
              <dl className="hero-stats-row">
                <div className="hero-stat-item">
                  <dt>Listings</dt>
                  <dd>{headline.listings}</dd>
                </div>
                <div className="hero-stat-item">
                  <dt>Official Areas</dt>
                  <dd>140</dd>
                </div>
                <div className="hero-stat-item">
                  <dt>Communities</dt>
                  <dd>17</dd>
                </div>
                <div className="hero-stat-item">
                  <dt>Components</dt>
                  <dd>1</dd>
                </div>
              </dl>
            </div>

            <div className="hero-figure-card">
              <img
                src="/figures/graph_c_louvain_community_map_labelled.png"
                alt="Map of Toronto Airbnb market communities with a citywide background and highlighted local segments"
                width={2250}
                height={2100}
                fetchPriority="high"
              />
              <div className="hero-figure-footer">
                <div>
                  <strong>Graph C · Louvain Communities</strong>
                </div>
                <div className="font-mono-tag" style={{ fontSize: "0.75rem" }}>
                  1 Single Connected Component
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Story Section 01: LIME PASTEL COLOR BLOCK (#dceeb1) */}
        <section className="color-block-wrapper" id="network">
          <div className="color-block color-block-lime">
            <p className="eyebrow">01 · BUILD THE NETWORK</p>
            <h2 className="display-lg">Three views of the same market</h2>
            <p className="subhead">
              Relationships are added one layer at a time to isolate how geography, host ownership, and listing similarity reshape the network.
            </p>

            {/* Interactive Graph Variant Pill Selector */}
            <div className="pill-tab-container" role="tablist" aria-label="Graph layer variants">
              {graphVariants.map((item, index) => (
                <button
                  key={item.id}
                  type="button"
                  role="tab"
                  className="pill-tab"
                  aria-selected={graphIndex === index}
                  onClick={() => setGraphIndex(index)}
                >
                  <strong>Graph {item.id}:</strong> {item.name}
                </button>
              ))}
            </div>

            {/* Live Layer Metrics Cards */}
            <div className="metric-grid" role="region" aria-live="polite">
              <div className="metric-card">
                <div className="metric-card-label">Network Edges</div>
                <div className="metric-card-value">{graph.edges}</div>
                <div className="metric-card-sub">Nodes: {graph.nodes}</div>
              </div>
              <div className="metric-card">
                <div className="metric-card-label">Connected Components</div>
                <div className="metric-card-value">{graph.components}</div>
                <div className="metric-card-sub">Largest: {graph.largestComp}</div>
              </div>
              <div className="metric-card">
                <div className="metric-card-label">Louvain Communities</div>
                <div className="metric-card-value">{graph.communities}</div>
                <div className="metric-card-sub">Modularity: {graph.modularity}</div>
              </div>
              <div className="metric-card">
                <div className="metric-card-label">Neighbourhood NMI</div>
                <div className="metric-card-value">{graph.nmi.toFixed(4)}</div>
                <div className="metric-card-sub">VI: {graph.vi}</div>
              </div>
            </div>

            {/* Layer Explanation Box */}
            <div className="white-panel" style={{ marginTop: "1.5rem" }}>
              <div className="eyebrow" style={{ fontSize: "0.75rem", marginBottom: "0.5rem" }}>
                Active Layer Properties
              </div>
              <p style={{ margin: 0, fontWeight: 480, fontSize: "1.05rem" }}>{graph.takeaway}</p>
              <div style={{ marginTop: "0.75rem", display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                {graph.relationships.map((rel) => (
                  <span key={rel} className="relationship-badge">
                    {rel}
                  </span>
                ))}
              </div>
            </div>

            {/* Canonical Graph Statistics Table */}
            <div className="table-container">
              <table className="editorial-table" aria-label="Canonical graph statistics comparison table">
                <thead>
                  <tr>
                    <th>Graph Variant</th>
                    <th className="num-cell">Nodes</th>
                    <th className="num-cell">Edges</th>
                    <th className="num-cell">Components</th>
                    <th className="num-cell">Largest Comp</th>
                    <th className="num-cell">Avg Degree</th>
                    <th className="num-cell">Modularity</th>
                    <th className="num-cell">Neighbourhood NMI</th>
                  </tr>
                </thead>
                <tbody>
                  {graphVariants.map((g) => (
                    <tr key={g.id} style={g.id === graph.id ? { background: "rgba(0,0,0,0.04)", fontWeight: 540 } : {}}>
                      <td>
                        <strong>Graph {g.id}:</strong> {g.name}
                      </td>
                      <td className="num-cell">{g.nodes}</td>
                      <td className="num-cell">{g.edges}</td>
                      <td className="num-cell">{g.components}</td>
                      <td className="num-cell">{g.largestComp}</td>
                      <td className="num-cell">{g.avgDegree}</td>
                      <td className="num-cell">{g.modularity}</td>
                      <td className="num-cell">{g.nmi.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Publication Figure: Alignment Comparison */}
            <div className="white-figure-tile">
              <img
                src="/figures/alignment_nmi_comparison.png"
                alt="Grouped bar chart comparing Normalized Mutual Information between detected communities and official neighbourhoods across Graph A, B, and C"
                width={2400}
                height={1500}
                loading="lazy"
              />
              <div className="figure-caption">
                Figure 1: Normalized Mutual Information (NMI) of detected Louvain and Leiden communities against official Toronto neighbourhoods.
              </div>
            </div>

            {/* Key Shift Callout */}
            <div className="callout-box">
              <span className="callout-tag">Key Shift</span>
              <p>
                <strong>100 components become one.</strong> A relatively small set of non-spatial attribute-similarity links bridges disparate geographic pockets into a single unified market network.
              </p>
            </div>
          </div>
        </section>

        {/* Story Section 02: LILAC PASTEL COLOR BLOCK (#c5b0f4) */}
        <section className="color-block-wrapper" id="communities">
          <div className="color-block color-block-lilac">
            <p className="eyebrow">02 · DETECT COMMUNITIES</p>
            <h2 className="display-lg">One market, multiple defensible partitions</h2>
            <p className="subhead">
              Louvain and Leiden agree on strong modular structure, but community IDs and exact membership remain algorithm-dependent.
            </p>

            {/* Algorithm Pill Switcher */}
            <div className="pill-tab-container" role="tablist" aria-label="Community detection algorithm">
              {(Object.keys(algorithms) as Array<keyof typeof algorithms>).map((name) => (
                <button
                  key={name}
                  type="button"
                  role="tab"
                  className="pill-tab"
                  aria-selected={algorithm === name}
                  onClick={() => setAlgorithm(name)}
                >
                  <strong>{name} Algorithm</strong>
                </button>
              ))}
            </div>

            {/* Algorithm Metrics & Map Display */}
            <div className="two-col-grid">
              <div className="white-figure-tile">
                <img
                  src={algorithmData.image}
                  alt={`${algorithm} detected communities mapped across Toronto Airbnb listings`}
                  width={2100}
                  height={1800}
                  loading="lazy"
                />
                <div className="figure-caption">
                  {algorithm} partition: {algorithmData.communities} detected communities across {headline.listings} listings.
                </div>
              </div>

              <div className="col-stack">
                <div className="white-panel">
                  <div className="eyebrow" style={{ fontSize: "0.75rem", marginBottom: "0.5rem" }}>
                    {algorithm} Algorithm Performance
                  </div>
                  <div className="metric-grid" style={{ margin: "0.5rem 0" }}>
                    <div className="metric-card" style={{ padding: "0.85rem 1rem" }}>
                      <div className="metric-card-label">Communities</div>
                      <div className="metric-card-value">{algorithmData.communities}</div>
                    </div>
                    <div className="metric-card" style={{ padding: "0.85rem 1rem" }}>
                      <div className="metric-card-label">Modularity (Q)</div>
                      <div className="metric-card-value">{algorithmData.modularity.toFixed(4)}</div>
                    </div>
                    <div className="metric-card" style={{ padding: "0.85rem 1rem" }}>
                      <div className="metric-card-label">Largest Segment</div>
                      <div className="metric-card-value">{algorithmData.largest}</div>
                    </div>
                    <div className="metric-card" style={{ padding: "0.85rem 1rem" }}>
                      <div className="metric-card-label">Neighbourhood NMI</div>
                      <div className="metric-card-value">{algorithmData.nmi.toFixed(4)}</div>
                    </div>
                  </div>
                  <p style={{ margin: "0.75rem 0 0", fontSize: "0.92rem", lineHeight: "1.5" }}>{algorithmData.note}</p>
                </div>

                <div className="white-panel">
                  <div className="eyebrow" style={{ fontSize: "0.75rem", marginBottom: "0.5rem" }}>
                    Market Segment Archetypes
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                    <div className="archetype-card">
                      <div className="archetype-card-header">
                        <span>Citywide Background</span>
                        <span>5,857 listings · $125 median</span>
                      </div>
                      <p>
                        Lower-priced entire homes spread across {headline.widestCommunitySpan} official neighbourhoods.
                      </p>
                    </div>
                    <div className="archetype-card">
                      <div className="archetype-card-header">
                        <span>Premium Downtown Waterfront</span>
                        <span>Up to $201 median</span>
                      </div>
                      <p>
                        High-density waterfront luxury condos with high local concentration.
                      </p>
                    </div>
                    <div className="archetype-card">
                      <div className="archetype-card-header">
                        <span>Budget Student & Suburban Rooms</span>
                        <span>$46–$51 median</span>
                      </div>
                      <p>
                        Dispersed private room clusters near campuses and peripheral zones.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Top Communities Summary Table */}
            <div className="table-container">
              <table className="editorial-table" aria-label="Detected communities summary table">
                <thead>
                  <tr>
                    <th>Community ID</th>
                    <th className="num-cell">Listings</th>
                    <th className="num-cell">Median Price</th>
                    <th>Dominant Neighbourhood</th>
                    <th>Dominant Room Type</th>
                    <th className="num-cell">Neighbourhoods Spanned</th>
                  </tr>
                </thead>
                <tbody>
                  {topCommunitiesSummary.map((comm) => (
                    <tr key={comm.id}>
                      <td>
                        <strong>{comm.id}</strong>
                      </td>
                      <td className="num-cell">{comm.size}</td>
                      <td className="num-cell">{comm.price}</td>
                      <td>{comm.dominantNeighbourhood}</td>
                      <td>{comm.roomType}</td>
                      <td className="num-cell">{comm.span}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Publication Figures: Characterisation & Profile Bubble */}
            <div className="two-col-grid">
              <div className="white-figure-tile">
                <img
                  src="/figures/community_characterisation.png"
                  alt="Detected communities ranked by median price, dominant neighbourhood, room type, and size"
                  width={3000}
                  height={2100}
                  loading="lazy"
                />
                <div className="figure-caption">
                  Figure 2: Profile of detected market segments ranked by median nightly price, dominant neighbourhood, and room type.
                </div>
              </div>

              <div className="white-figure-tile">
                <img
                  src="/figures/community_profile_bubble.png"
                  alt="Scatter bubble chart of community size versus median price coloured by room type"
                  width={3000}
                  height={1950}
                  loading="lazy"
                />
                <div className="figure-caption">
                  Figure 3: Community profile bubble landscape: geographic spread vs median price (bubble size = listings).
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Story Section 03: CREAM PASTEL COLOR BLOCK (#f4ecd6) */}
        <section className="color-block-wrapper" id="robustness">
          <div className="color-block color-block-cream">
            <p className="eyebrow">03 · TEST THE CLAIM</p>
            <h2 className="display-lg">Does the network improve price prediction?</h2>
            <p className="subhead">
              Paired 5-fold cross-validation separates a small raw association from a genuinely useful predictive gain.
            </p>

            {/* Validation Scheme Pill Selector */}
            <div className="pill-tab-container" role="tablist" aria-label="Validation scheme">
              {validationSchemes.map((item, index) => (
                <button
                  key={item.id}
                  type="button"
                  role="tab"
                  className="pill-tab"
                  aria-selected={validationIndex === index}
                  onClick={() => setValidationIndex(index)}
                >
                  {item.label}
                </button>
              ))}
            </div>

            {/* Interactive Model Comparison Scores */}
            <div className="white-panel" style={{ marginTop: "1.5rem" }} role="region" aria-live="polite">
              <div className="eyebrow" style={{ fontSize: "0.75rem", marginBottom: "0.5rem" }}>
                {validation.fullName} · Mean Test R² Across 5 Folds
              </div>
              <ScoreBar label="Baseline Model (Listing + Official Neighbourhood)" value={validation.baseline} tone="base" />
              <ScoreBar label="Expanded Model (+ Network Community Feature)" value={validation.expanded} tone="expanded" />

              <div className="delta-grid">
                <div className="delta-item">
                  <span>Raw R² Change</span>
                  <strong className="gain">{validation.delta}</strong>
                </div>
                <div className="delta-item">
                  <span>Adjusted R² Change</span>
                  <strong className="caution">{validation.adjustedDelta}</strong>
                </div>
                <div className="delta-item">
                  <span>Baseline Dollar MAE</span>
                  <strong>{validation.baseMae}</strong>
                </div>
                <div className="delta-item">
                  <span>Expanded Dollar MAE</span>
                  <strong>{validation.expandedMae}</strong>
                </div>
                <div className="delta-item">
                  <span>MAE Difference</span>
                  <strong className="gain">{validation.maeDelta}</strong>
                </div>
                <div className="delta-item">
                  <span>Raw R² Fold Wins</span>
                  <strong>{validation.wins}</strong>
                </div>
              </div>

              <p style={{ margin: "1rem 0 0", fontSize: "0.92rem", lineHeight: "1.5" }}>{validation.note}</p>
            </div>

            {/* 5-Fold Price Model CV Summary Table */}
            <div className="table-container">
              <table className="editorial-table" aria-label="5-fold cross validation summary table">
                <thead>
                  <tr>
                    <th>Validation Scheme</th>
                    <th className="num-cell">Baseline R²</th>
                    <th className="num-cell">Expanded R²</th>
                    <th className="num-cell">Mean ΔR²</th>
                    <th className="num-cell">Mean Δ Adjusted R²</th>
                    <th className="num-cell">Baseline MAE ($)</th>
                    <th className="num-cell">Expanded MAE ($)</th>
                    <th className="num-cell">Mean ΔMAE ($)</th>
                  </tr>
                </thead>
                <tbody>
                  {validationSchemes.map((scheme) => (
                    <tr
                      key={scheme.id}
                      style={scheme.id === validation.id ? { background: "rgba(0,0,0,0.04)", fontWeight: 540 } : {}}
                    >
                      <td>
                        <strong>{scheme.fullName}</strong>
                      </td>
                      <td className="num-cell">{scheme.baseline.toFixed(4)}</td>
                      <td className="num-cell">{scheme.expanded.toFixed(4)}</td>
                      <td className="num-cell" style={{ color: "var(--semantic-success)" }}>
                        {scheme.delta}
                      </td>
                      <td className="num-cell" style={{ color: "#b91c1c" }}>
                        {scheme.adjustedDelta}
                      </td>
                      <td className="num-cell">{scheme.baseMae}</td>
                      <td className="num-cell">{scheme.expandedMae}</td>
                      <td className="num-cell" style={{ color: "var(--semantic-success)" }}>
                        {scheme.maeDelta}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Publication Figures: CV Comparison & Spatial Blocks */}
            <div className="two-col-grid">
              <div className="white-figure-tile">
                <img
                  src="/figures/price_model_cv_comparison.png"
                  alt="Price model cross-validation score and delta comparison across random, host-grouped, and spatial splits"
                  width={3600}
                  height={2400}
                  loading="lazy"
                />
                <div className="figure-caption">
                  Figure 4: Price-model cross-validation comparison: mean test R², dollar MAE, and paired fold changes.
                </div>
              </div>

              <div className="white-figure-tile">
                <img
                  src="/figures/price_model_cv_spatial_blocks.png"
                  alt="Map of the five compact spatial blocks used in spatial cross-validation"
                  width={2100}
                  height={1800}
                  loading="lazy"
                />
                <div className="figure-caption">
                  Figure 5: Five deterministic geographic blocks used for spatial cross-validation holdout folds.
                </div>
              </div>
            </div>

            {/* Evidence-Based Verdict Card */}
            <div className="callout-box">
              <span className="callout-tag">Evidence-Based Verdict</span>
              <p>
                <strong>A real signal, but not a material pricing boost.</strong> Across random, host-grouped, and spatial-block five-fold validation, adding the network community label moves out-of-sample R² by {deltaRange}. The gain clears a size-matched random-partition null, so it is not just the effect of adding degrees of freedom — but the effect size is small enough that the network’s value here is explanatory rather than predictive.
              </p>
              <p className="callout-footnote">
                The earlier version of this verdict rested on a decline in complexity-adjusted R². That statistic applies an in-sample parameter penalty to an out-of-sample score, so any 17-level categorical — including random labels — produces the same decline. It has been replaced by a permutation test and paired fold-level intervals.
              </p>
            </div>
          </div>
        </section>

        {/* Story Section 04: CORAL PASTEL COLOR BLOCK (#f3c9b6) */}
        <section className="color-block-wrapper" id="sensitivity">
          <div className="color-block color-block-coral">
            <p className="eyebrow">04 · PARAMETER SENSITIVITY</p>
            <h2 className="display-lg">The broad structure persists; exact boundaries move.</h2>
            <p className="subhead">
              Seven reasonable Graph C parameter settings demonstrate that while modular citywide segments always emerge, exact community assignments are parameter-dependent.
            </p>

            {/* Sensitivity Configuration Pill Selector */}
            <div className="pill-tab-container" role="tablist" aria-label="Parameter sensitivity configuration">
              {sensitivitySettings.map((item, index) => (
                <button
                  key={item.id}
                  type="button"
                  role="tab"
                  className="pill-tab"
                  aria-selected={sensitivityIndex === index}
                  onClick={() => setSensitivityIndex(index)}
                >
                  {item.label}
                </button>
              ))}
            </div>

            {/* Active Sensitivity Configuration Card */}
            <div className="white-panel" style={{ marginTop: "1.5rem" }} role="region" aria-live="polite">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: "0.5rem" }}>
                <h3 className="headline" style={{ margin: 0 }}>
                  Configuration: {sensitivity.label}
                </h3>
                <span className="font-mono-tag" style={{ fontSize: "0.85rem", fontWeight: 700 }}>
                  {sensitivity.communities} Communities Detected
                </span>
              </div>

              <div className="metric-grid">
                <div className="metric-card">
                  <div className="metric-card-label">Spatial Radius</div>
                  <div className="metric-card-value">{sensitivity.radius}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-card-label">Attribute k-NN</div>
                  <div className="metric-card-value">k = {sensitivity.k}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-card-label">Modularity (Q)</div>
                  <div className="metric-card-value">{sensitivity.modularity.toFixed(4)}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-card-label">NMI vs Baseline</div>
                  <div className="metric-card-value">{sensitivity.stability.toFixed(4)}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-card-label">Neighbourhood NMI</div>
                  <div className="metric-card-value">{sensitivity.neighbourhood.toFixed(4)}</div>
                </div>
              </div>

              <p style={{ margin: 0, fontSize: "0.92rem", lineHeight: "1.5" }}>{sensitivity.note}</p>
            </div>

            {/* Parameter Sensitivity Results Table */}
            <div className="table-container">
              <table className="editorial-table" aria-label="Parameter sensitivity results table">
                <thead>
                  <tr>
                    <th>Configuration</th>
                    <th className="num-cell">Radius</th>
                    <th className="num-cell">Attribute k</th>
                    <th>Edge Weights (Spatial / Host / Attr)</th>
                    <th className="num-cell">Communities</th>
                    <th className="num-cell">Modularity</th>
                    <th className="num-cell">NMI vs Baseline</th>
                    <th className="num-cell">Neighbourhood NMI</th>
                  </tr>
                </thead>
                <tbody>
                  {sensitivitySettings.map((setting) => (
                    <tr
                      key={setting.id}
                      style={setting.id === sensitivity.id ? { background: "rgba(0,0,0,0.04)", fontWeight: 540 } : {}}
                    >
                      <td>
                        <strong>{setting.label}</strong>
                      </td>
                      <td className="num-cell">{setting.radius}</td>
                      <td className="num-cell">{setting.k}</td>
                      <td>{setting.weights}</td>
                      <td className="num-cell">{setting.communities}</td>
                      <td className="num-cell">{setting.modularity.toFixed(4)}</td>
                      <td className="num-cell">{setting.stability.toFixed(4)}</td>
                      <td className="num-cell">{setting.neighbourhood.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Publication Figure: Parameter Sensitivity */}
            <div className="white-figure-tile">
              <img
                src="/figures/parameter_sensitivity.png"
                alt="Bar charts of communities, modularity, and NMI across seven Graph C parameter configurations"
                width={3900}
                height={2400}
                loading="lazy"
              />
              <div className="figure-caption">
                Figure 6: One-at-a-time parameter sensitivity sweep across spatial radius, attribute neighbours, and weight profiles.
              </div>
            </div>
          </div>
        </section>

        {/* Story Section 05: MINT PASTEL COLOR BLOCK (#c8e6cd) */}
        <section className="color-block-wrapper" id="method">
          <div className="color-block color-block-mint">
            <p className="eyebrow">05 · RESEARCH INTEGRITY</p>
            <h2 className="display-lg">Reproducible by design</h2>
            <p className="subhead">
              The portfolio refresh turns a course submission into an auditable case study with locked dependencies, automated unit tests, generated canonical artifacts, and honest scope limitations.
            </p>

            <div className="four-col-grid">
              <div className="white-panel">
                <div className="font-mono-tag" style={{ fontSize: "0.75rem", marginBottom: "0.5rem" }}>
                  01 · Public Snapshot
                </div>
                <h3 className="headline" style={{ fontSize: "1.15rem", marginBottom: "0.5rem" }}>
                  Inside Airbnb Snapshot
                </h3>
                <p style={{ margin: 0, fontSize: "0.88rem" }}>
                  November 2025 public Toronto data: {headline.listings} cleaned listings across {headline.officialAreas} official neighbourhoods.
                </p>
              </div>

              <div className="white-panel">
                <div className="font-mono-tag" style={{ fontSize: "0.75rem", marginBottom: "0.5rem" }}>
                  02 · Price-Free Graph
                </div>
                <h3 className="headline" style={{ fontSize: "1.15rem", marginBottom: "0.5rem" }}>
                  No Data Leakage
                </h3>
                <p style={{ margin: 0, fontSize: "0.88rem" }}>
                  Graph edges are constructed strictly from spatial proximity, host IDs, and listing attributes. Price never enters graph construction.
                </p>
              </div>

              <div className="white-panel">
                <div className="font-mono-tag" style={{ fontSize: "0.75rem", marginBottom: "0.5rem" }}>
                  03 · Paired Evaluation
                </div>
                <h3 className="headline" style={{ fontSize: "1.15rem", marginBottom: "0.5rem" }}>
                  Strict 5-Fold Folds
                </h3>
                <p style={{ margin: 0, fontSize: "0.88rem" }}>
                  Baseline and expanded models are evaluated on identical random, host-grouped, and spatial-block folds.
                </p>
              </div>

              <div className="white-panel">
                <div className="font-mono-tag" style={{ fontSize: "0.75rem", marginBottom: "0.5rem" }}>
                  04 · Honest Scope
                </div>
                <h3 className="headline" style={{ fontSize: "1.15rem", marginBottom: "0.5rem" }}>
                  Transductive Validation
                </h3>
                <p style={{ margin: 0, fontSize: "0.88rem" }}>
                  Evaluation is transductive and based on a single temporal snapshot, so the results describe observed network structure rather than causal or deployment-level effects.
                </p>
              </div>
            </div>

            {/* Expandable Technical Methodology */}
            <details className="method-accordion">
              <summary>Open the technical methodology</summary>
              <div className="method-accordion-content">
                <p>
                  <strong>Network Construction:</strong> Weighted undirected graphs built with BallTree haversine distance (500 m radius), sparse nearest same-host connections (k=5), and cosine-similarity nearest neighbours (k=5) over standardized numerical and one-hot categorical listing features.
                </p>
                <p>
                  <strong>Community Detection:</strong> Seeded Louvain and Leiden modularity optimization, evaluated using modularity (Q), Normalized Mutual Information (NMI), and Variation of Information (VI).
                </p>
                <p>
                  <strong>Price Modelling:</strong> Ridge regression on log-transformed winsorized nightly price, with room types, property types, and official administrative neighbourhoods as the baseline.
                </p>
              </div>
            </details>

            {/* Portfolio Attribution Card */}
            <div className="white-panel" style={{ marginTop: "1.5rem" }}>
              <div className="eyebrow" style={{ fontSize: "0.75rem", marginBottom: "0.5rem" }}>
                Portfolio attribution
              </div>
              <h3 className="headline" style={{ fontSize: "1.25rem", margin: "0 0 0.5rem" }}>
                Research, analysis, and portfolio presentation by Sourav Chandhok.
              </h3>
              <p style={{ margin: 0, fontSize: "0.92rem", lineHeight: "1.5" }}>
                Developed from a research project and extended with reproducible analysis, grouped and spatial validation, parameter-sensitivity testing, and this interactive presentation.
              </p>
            </div>
          </div>
        </section>

        {/* Closing Section (White Canvas) */}
        <section className="hero" style={{ paddingTop: "0", textAlign: "center" }}>
          <p className="eyebrow" style={{ justifyContent: "center" }}>
            THE TAKEAWAY
          </p>
          <h2 className="display-lg" style={{ maxWidth: "900px", margin: "0 auto 1.5rem" }}>
            A market can be geographically local and structurally citywide at the same time.
          </h2>
          <p className="subhead" style={{ maxWidth: "700px", margin: "0 auto 2.5rem" }}>
            The value of network analysis lies not only in what it reveals, but in knowing what the evidence cannot support.
          </p>
          <div className="hero-actions" style={{ justifyContent: "center", marginBottom: "0" }}>
            <a
              className="btn-pill btn-primary"
              href="https://github.com/souravC01/toronto-airbnb-market-network"
              target="_blank"
              rel="noreferrer"
            >
              View repository <Arrow />
            </a>
            <a
              className="btn-pill btn-secondary"
              href="/report/EECS4414-Airbnb-Network-Analysis-Final-Report.pdf"
              target="_blank"
              rel="noreferrer"
              aria-label="Open the original EECS 4414 final report in a new tab"
            >
              View report <Arrow />
            </a>
          </div>
        </section>
      </main>

      {/* Footer (Clean White Canvas) */}
      <footer className="footer">
        <div className="footer-grid">
          <div className="footer-brand">
            <h4>Toronto Airbnb Market Network</h4>
            <p>A network-science case study of {headline.listings} Toronto Airbnb listings, community structure, robustness, and price influence.</p>
          </div>

          <div className="footer-col">
            <h5>Navigation</h5>
            <ul>
              <li><a href="#network">Network Layers</a></li>
              <li><a href="#communities">Community Detection</a></li>
              <li><a href="#robustness">Price Validation</a></li>
              <li><a href="#sensitivity">Parameter Sensitivity</a></li>
              <li><a href="#method">Methodology</a></li>
            </ul>
          </div>

          <div className="footer-col">
            <h5>Artifacts</h5>
            <ul>
              <li>
                <a href="/report/EECS4414-Airbnb-Network-Analysis-Final-Report.pdf" target="_blank" rel="noreferrer">
                  Final Report PDF <Arrow />
                </a>
              </li>
              <li>
                <a href="https://github.com/souravC01/toronto-airbnb-market-network" target="_blank" rel="noreferrer">
                  GitHub Repository <Arrow />
                </a>
              </li>
              <li>
                <a href="https://insideairbnb.com/get-the-data/" target="_blank" rel="noreferrer">
                  Inside Airbnb Data <Arrow />
                </a>
              </li>
            </ul>
          </div>

          <div className="footer-col">
            <h5>Attribution</h5>
            <ul>
              <li>Sourav Chandhok</li>
              <li>York University</li>
              <li>November 2025 Snapshot</li>
            </ul>
          </div>
        </div>

        <div className="footer-bottom">
          <span>Toronto Airbnb Market Network · Network science · reproducibility · honest evaluation</span>
          <span>York University ·</span>
        </div>
      </footer>

      {/* Floating Scroll to Top Button */}
      <button
        type="button"
        className={`scroll-top-btn ${showScrollTop ? "visible" : ""}`}
        onClick={scrollToTop}
        aria-label="Scroll to top of page"
        title="Scroll to top"
      >
        <span aria-hidden="true">↑</span>
      </button>
    </div>
  );
}
