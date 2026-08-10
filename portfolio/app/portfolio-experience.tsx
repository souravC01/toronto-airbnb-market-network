"use client";

import Image from "next/image";
import { useState } from "react";

const graphVariants = [
  {
    id: "A",
    name: "Spatial only",
    relationships: ["Within 500 m"],
    edges: "1.55M",
    components: 100,
    communities: 134,
    nmi: 0.7494,
    takeaway:
      "The geographic baseline reconstructs neighbourhood-like local clusters most closely.",
  },
  {
    id: "B",
    name: "Spatial + shared host",
    relationships: ["Within 500 m", "Same-host neighbours"],
    edges: "1.56M",
    components: 53,
    communities: 94,
    nmi: 0.703,
    takeaway:
      "Sparse ownership links bridge local clusters without overwhelming the graph with host cliques.",
  },
  {
    id: "C",
    name: "Full market network",
    relationships: ["Within 500 m", "Same-host neighbours", "Listing similarity"],
    edges: "1.61M",
    components: 1,
    communities: 17,
    nmi: 0.5182,
    takeaway:
      "Similarity links connect the city into broader market segments that cross administrative lines.",
  },
];

const algorithms = {
  Louvain: {
    communities: 17,
    modularity: 0.7926,
    largest: "5,857",
    nmi: 0.5182,
    image: "/figures/graph_c_louvain_community_map.png",
    note: "Primary interpretation method; recovers a citywide background market and 16 structured segments.",
  },
  Leiden: {
    communities: 16,
    modularity: 0.7936,
    largest: "7,078",
    nmi: 0.4941,
    image: "/figures/graph_c_leiden_community_map.png",
    note: "Robustness check; nearly identical modularity with a different partition and larger dominant segment.",
  },
};

const validationSchemes = [
  {
    id: "random",
    label: "Random folds",
    baseline: 0.6341,
    expanded: 0.6358,
    delta: "+0.0017",
    adjustedDelta: "−0.0004",
    maeDelta: "−$0.18",
    wins: "5 / 5",
    note: "Listing-level folds reproduce the small raw improvement, but complexity-adjusted performance declines.",
  },
  {
    id: "host",
    label: "Host-grouped",
    baseline: 0.6246,
    expanded: 0.6262,
    delta: "+0.0016",
    adjustedDelta: "−0.0007",
    maeDelta: "−$0.14",
    wins: "5 / 5",
    note: "Every host stays entirely in train or test. The result is not explained by the same host leaking across folds.",
  },
  {
    id: "spatial",
    label: "Spatial blocks",
    baseline: 0.529,
    expanded: 0.5314,
    delta: "+0.0024",
    adjustedDelta: "−0.0010",
    maeDelta: "−$0.43",
    wins: "3 / 5",
    note: "Holding out whole geographic regions is much harder and more variable; the raw gain is not consistent across blocks.",
  },
];

const sensitivitySettings = [
  { id: "baseline", label: "Baseline", communities: 17, modularity: 0.7926, stability: 1, neighbourhood: 0.5182, note: "500 m radius, attribute k = 5, and 0.60 / 0.25 / 0.15 edge weights." },
  { id: "r300", label: "Radius 300 m", communities: 21, modularity: 0.8025, stability: 0.6945, neighbourhood: 0.434, note: "A tighter radius fragments the market into more communities and shifts exact membership most strongly." },
  { id: "r700", label: "Radius 700 m", communities: 14, modularity: 0.743, stability: 0.7438, neighbourhood: 0.5631, note: "A wider radius merges segments and increases agreement with official neighbourhoods." },
  { id: "k3", label: "Attribute k = 3", communities: 19, modularity: 0.7981, stability: 0.8631, neighbourhood: 0.5506, note: "Fewer similarity neighbours produces the closest alternative partition to the baseline." },
  { id: "k10", label: "Attribute k = 10", communities: 17, modularity: 0.7819, stability: 0.8136, neighbourhood: 0.4668, note: "More similarity bridges retain 17 communities but change their composition." },
  { id: "spatial-heavy", label: "Spatial-heavy", communities: 19, modularity: 0.7983, stability: 0.8541, neighbourhood: 0.555, note: "Emphasizing location restores more neighbourhood-like structure." },
  { id: "attribute-heavy", label: "Attribute-heavy", communities: 14, modularity: 0.7702, stability: 0.7429, neighbourhood: 0.4366, note: "Emphasizing listing similarity produces broader, less administrative market segments." },
];

function Arrow() {
  return <span aria-hidden="true">↗</span>;
}

function ScoreBar({ label, value, tone }: { label: string; value: number; tone: "base" | "expanded" }) {
  return (
    <div className="score-row">
      <div className="score-label">
        <span>{label}</span>
        <strong>{value.toFixed(4)}</strong>
      </div>
      <div className="score-track" aria-label={`${label} R squared ${value.toFixed(4)}`}>
        <span className={`score-fill ${tone}`} style={{ width: `${Math.min(100, value * 145)}%` }} />
      </div>
    </div>
  );
}

export function PortfolioExperience() {
  const [graphIndex, setGraphIndex] = useState(2);
  const [algorithm, setAlgorithm] = useState<keyof typeof algorithms>("Louvain");
  const [validationIndex, setValidationIndex] = useState(1);
  const [sensitivityIndex, setSensitivityIndex] = useState(0);

  const graph = graphVariants[graphIndex];
  const algorithmData = algorithms[algorithm];
  const validation = validationSchemes[validationIndex];
  const sensitivity = sensitivitySettings[sensitivityIndex];

  return (
    <div className="site-shell">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Toronto Airbnb Network home">
          <span className="brand-mark">TN</span>
          <span>Toronto Airbnb Network</span>
        </a>
        <nav className="nav-links" aria-label="Case study sections">
          <a href="#network">Network</a>
          <a href="#communities">Communities</a>
          <a href="#robustness">Robustness</a>
          <a href="#method">Method</a>
        </nav>
        <a className="nav-cta" href="https://github.com/souravC01/toronto-airbnb-market-network" target="_blank" rel="noreferrer">
          Repository <Arrow />
        </a>
      </header>

      <main id="main-content">
        <section className="hero" id="top">
          <div className="hero-copy">
            <p className="eyebrow"><span /> Network science · Toronto · 2025 snapshot</p>
            <h1>Toronto&apos;s Airbnb market doesn&apos;t stop at neighbourhood lines.</h1>
            <p className="hero-deck">
              A network of 15,809 listings reveals a connected market shaped by proximity, ownership, and product similarity—not only the city&apos;s administrative map.
            </p>
            <div className="hero-actions">
              <a className="button primary" href="#network">Explore the network <span aria-hidden="true">↓</span></a>
              <a className="button secondary" href="#method">View methodology <Arrow /></a>
            </div>
            <dl className="hero-stats">
              <div><dt>Listings</dt><dd>15,809</dd></div>
              <div><dt>Official areas</dt><dd>140</dd></div>
              <div><dt>Full-graph communities</dt><dd>17</dd></div>
            </dl>
          </div>
          <div className="hero-visual">
            <div className="visual-header">
              <span>Graph C · Louvain communities</span>
              <span className="live-dot">Portfolio case study</span>
            </div>
            <Image
              src="/figures/graph_c_louvain_community_map_labelled.png"
              alt="Map of Toronto Airbnb market communities with a citywide background and highlighted local segments"
              width={2250}
              height={2100}
              sizes="(max-width: 860px) 100vw, 47vw"
              priority
            />
            <div className="visual-caption">
              <div><span className="metric-number">1</span><span>connected component</span></div>
              <p>Market segments cross the 140 official neighbourhood boundaries.</p>
            </div>
          </div>
        </section>

        <section className="thesis-strip" aria-label="Case study thesis">
          <p>Administrative neighbourhoods explain <em>where</em>.</p>
          <p>The network explains <em>how the market connects</em>.</p>
        </section>

        <section className="section" id="network">
          <div className="section-heading">
            <div>
              <p className="section-kicker">01 · Build the network</p>
              <h2>Three views of the same market</h2>
            </div>
            <p>Relationships are added one layer at a time to isolate how geography, host ownership, and listing similarity reshape the network.</p>
          </div>

          <div className="graph-explorer">
            <div className="graph-tabs" role="tablist" aria-label="Graph variants">
              {graphVariants.map((item, index) => (
                <button key={item.id} type="button" role="tab" aria-selected={graphIndex === index} onClick={() => setGraphIndex(index)}>
                  <span>Graph {item.id}</span>
                  <strong>{item.name}</strong>
                </button>
              ))}
            </div>
            <div className="graph-panel" role="tabpanel" aria-live="polite">
              <div className="relationship-stack">
                <p className="mini-label">Relationships included</p>
                {graph.relationships.map((relationship, index) => (
                  <div className="relationship-row" key={relationship}>
                    <span className={`relationship-icon icon-${index + 1}`} aria-hidden="true" />
                    <span>{relationship}</span>
                  </div>
                ))}
                <p className="panel-takeaway">{graph.takeaway}</p>
              </div>
              <div className="graph-metrics">
                <div><span>Edges</span><strong>{graph.edges}</strong></div>
                <div><span>Components</span><strong>{graph.components}</strong></div>
                <div><span>Louvain communities</span><strong>{graph.communities}</strong></div>
                <div><span>Neighbourhood NMI</span><strong>{graph.nmi.toFixed(4)}</strong></div>
              </div>
            </div>
          </div>

          <div className="finding-banner">
            <span className="finding-index">Key shift</span>
            <p><strong>100 components become one.</strong> A relatively small set of non-spatial links changes global connectivity and reveals broader market structure.</p>
          </div>
        </section>

        <section className="section dark-section" id="communities">
          <div className="section-heading light">
            <div>
              <p className="section-kicker">02 · Detect communities</p>
              <h2>One market, multiple defensible partitions</h2>
            </div>
            <p>Louvain and Leiden agree on strong modular structure, but community IDs and exact membership remain algorithm-dependent.</p>
          </div>

          <div className="algorithm-toggle" role="tablist" aria-label="Community detection algorithm">
            {(Object.keys(algorithms) as Array<keyof typeof algorithms>).map((name) => (
              <button key={name} type="button" role="tab" aria-selected={algorithm === name} onClick={() => setAlgorithm(name)}>{name}</button>
            ))}
          </div>
          <div className="algorithm-layout" role="tabpanel" aria-live="polite">
            <figure className="map-card">
              <Image
                src={algorithmData.image}
                alt={`${algorithm} communities mapped across Toronto Airbnb listings`}
                width={2100}
                height={1800}
                sizes="(max-width: 860px) 100vw, 60vw"
              />
              <figcaption>Community colours are categorical; IDs should not be matched directly across algorithms.</figcaption>
            </figure>
            <div className="algorithm-summary">
              <p className="mini-label">{algorithm} result</p>
              <div className="large-metric"><strong>{algorithmData.communities}</strong><span>communities</span></div>
              <div className="summary-grid">
                <div><span>Modularity</span><strong>{algorithmData.modularity.toFixed(4)}</strong></div>
                <div><span>Largest segment</span><strong>{algorithmData.largest}</strong></div>
                <div><span>Neighbourhood NMI</span><strong>{algorithmData.nmi.toFixed(4)}</strong></div>
              </div>
              <p className="algorithm-note">{algorithmData.note}</p>
            </div>
          </div>

          <div className="segment-heading">
            <p className="section-kicker">What the communities represent</p>
            <h3>Not abstract clusters—recognizable rental-market segments.</h3>
          </div>
          <div className="segment-grid">
            <article>
              <span className="segment-type">Citywide background</span>
              <strong>5,857 listings</strong>
              <p>Lower-priced, predominantly entire-home inventory spread across 114 official neighbourhoods.</p>
            </article>
            <article>
              <span className="segment-type">Premium downtown</span>
              <strong>Up to $201 median</strong>
              <p>Compact waterfront and Bay Street segments with high local concentration and tourism-market pricing.</p>
            </article>
            <article>
              <span className="segment-type">Budget private room</span>
              <strong>$46–$51 median</strong>
              <p>Small student- and suburb-oriented pockets that remain distinct from the dominant entire-home market.</p>
            </article>
          </div>
          <figure className="wide-figure">
            <Image
              src="/figures/community_characterisation.png"
              alt="Detected communities ranked by median price, dominant neighbourhood, room type, and size"
              width={3000}
              height={2100}
              sizes="(max-width: 860px) 100vw, 90vw"
            />
            <figcaption>Largest communities profiled by price, room type, dominant neighbourhood, and listing count.</figcaption>
          </figure>
        </section>

        <section className="section" id="robustness">
          <div className="section-heading">
            <div>
              <p className="section-kicker">03 · Test the claim</p>
              <h2>Does the network improve price prediction?</h2>
            </div>
            <p>Paired validation separates a small raw association from a genuinely useful predictive gain.</p>
          </div>

          <div className="validation-explorer">
            <div className="validation-tabs" role="tablist" aria-label="Validation scheme">
              {validationSchemes.map((item, index) => (
                <button key={item.id} type="button" role="tab" aria-selected={validationIndex === index} onClick={() => setValidationIndex(index)}>{item.label}</button>
              ))}
            </div>
            <div className="validation-panel" role="tabpanel" aria-live="polite">
              <div className="score-card">
                <p className="mini-label">Mean test R² · five folds</p>
                <ScoreBar label="Baseline model" value={validation.baseline} tone="base" />
                <ScoreBar label="With network community" value={validation.expanded} tone="expanded" />
              </div>
              <div className="delta-grid">
                <div><span>Raw R² change</span><strong className="positive">{validation.delta}</strong></div>
                <div><span>Adjusted R² change</span><strong className="caution">{validation.adjustedDelta}</strong></div>
                <div><span>Approx. dollar MAE</span><strong>{validation.maeDelta}</strong></div>
                <div><span>Raw R² wins</span><strong>{validation.wins}</strong></div>
              </div>
              <p className="validation-note">{validation.note}</p>
            </div>
          </div>

          <div className="verdict-card">
            <p className="section-kicker">Evidence-based verdict</p>
            <h3>Interesting market signal. Not a material pricing boost.</h3>
            <p>Raw R² rises by only 0.0016–0.0024, while mean adjusted R² declines in every validation scheme. The network&apos;s strongest value is explanatory.</p>
          </div>

          <div className="sensitivity-block">
            <div className="sensitivity-copy">
              <p className="section-kicker">Parameter sensitivity</p>
              <h3>The broad pattern persists; exact boundaries move.</h3>
              <p>Explore seven reasonable Graph C settings. Every one produces a small set of high-modularity market segments, but no single partition should be treated as an immutable ground truth.</p>
              <div className="sensitivity-pills" role="tablist" aria-label="Parameter configuration">
                {sensitivitySettings.map((item, index) => (
                  <button key={item.id} type="button" role="tab" aria-selected={sensitivityIndex === index} onClick={() => setSensitivityIndex(index)}>{item.label}</button>
                ))}
              </div>
            </div>
            <div className="sensitivity-result" role="tabpanel" aria-live="polite">
              <div className="result-title"><span>{sensitivity.label}</span><strong>{sensitivity.communities} communities</strong></div>
              <div className="result-metrics">
                <div><span>Modularity</span><strong>{sensitivity.modularity.toFixed(4)}</strong></div>
                <div><span>NMI vs baseline</span><strong>{sensitivity.stability.toFixed(4)}</strong></div>
                <div><span>Neighbourhood NMI</span><strong>{sensitivity.neighbourhood.toFixed(4)}</strong></div>
              </div>
              <p>{sensitivity.note}</p>
            </div>
          </div>
        </section>

        <section className="section method-section" id="method">
          <div className="section-heading">
            <div>
              <p className="section-kicker">04 · Research integrity</p>
              <h2>Reproducible by design</h2>
            </div>
            <p>The portfolio refresh turns a course submission into an auditable case study with locked dependencies, tests, generated artifacts, and careful limitations.</p>
          </div>

          <div className="method-grid">
            <article><span>01</span><h3>Public snapshot</h3><p>November 2025 Toronto data from Inside Airbnb; 15,809 cleaned listings across 140 neighbourhoods.</p></article>
            <article><span>02</span><h3>Price-free graph</h3><p>Edges use location, host identity, and listing attributes. Price never enters graph construction.</p></article>
            <article><span>03</span><h3>Paired evaluation</h3><p>Baseline and expanded models use identical random, host-grouped, and spatial folds.</p></article>
            <article><span>04</span><h3>Honest scope</h3><p>The graph evaluation is transductive and represents one temporal snapshot—not a causal or deployment claim.</p></article>
          </div>

          <details className="method-details">
            <summary>Open the technical methodology</summary>
            <div>
              <p><strong>Network:</strong> weighted undirected graph with BallTree spatial edges, sparse nearest same-host links, and cosine-similarity attribute neighbours.</p>
              <p><strong>Communities:</strong> seeded Louvain and Leiden modularity optimization, compared using modularity, NMI, and Variation of Information.</p>
              <p><strong>Price model:</strong> ridge regression on log-transformed winsorized price, with listing attributes and official neighbourhoods as the baseline.</p>
              <p><strong>Quality checks:</strong> locked dependency versions, automated tests, artifact validation, and a visually verified report.</p>
            </div>
          </details>

          <div className="team-card">
            <div>
              <p className="section-kicker">Portfolio attribution</p>
              <h3>Research, analysis, and portfolio presentation by Sourav Chandhok.</h3>
              <p>Developed from an EECS 4414 research project and extended with reproducible analysis, grouped and spatial validation, parameter-sensitivity testing, and this interactive presentation.</p>
            </div>
            <ul>
              <li>Sourav Chandhok</li>
            </ul>
          </div>
        </section>

        <section className="closing">
          <p className="section-kicker">The takeaway</p>
          <h2>A market can be geographically local and structurally citywide at the same time.</h2>
          <p>This case study shows where network analysis adds real explanatory value—and where disciplined validation says it does not.</p>
          <div className="hero-actions closing-actions">
            <a className="button primary" href="https://github.com/souravC01/toronto-airbnb-market-network" target="_blank" rel="noreferrer">View repository <Arrow /></a>
            <a className="button secondary" href="#robustness">Review robustness <Arrow /></a>
          </div>
        </section>
      </main>

      <footer>
        <span>Toronto Airbnb Market Network</span>
        <span>Network science · reproducibility · honest evaluation</span>
        <span>York University · EECS 4414</span>
      </footer>
    </div>
  );
}
