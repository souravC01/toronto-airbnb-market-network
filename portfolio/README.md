# Toronto Airbnb Market Network — interactive case study

This site turns the research repository into a public-facing portfolio story. It presents the network design, graph evolution, Louvain/Leiden results, interpretable market segments, grouped and spatial price validation, parameter sensitivity, methodology, limitations, and portfolio attribution.

[View the live Vercel deployment](https://toronto-airbnb-market-network.vercel.app/).

## Run locally

Node.js 22 and pnpm are required.

```bash
pnpm install
pnpm run dev
```

Create the production build and run the rendered-page checks:

```bash
pnpm run build
node --test tests/rendered-html.test.mjs
```

The production build is a fully static Vite export. It prerenders the complete case study into `dist/index.html` for fast first paint and search visibility, then hydrates the interactive controls in the browser. Vercel deploys this directory with the Vite preset and serves the output from its CDN without a server runtime. The repository-level `render.yaml` is retained temporarily as a legacy fallback during the hosting cutover.

## Data and assets

The metrics shown in the experience come from the canonical CSV artifacts in `../results/tables/`, and publication figures are copied from `../results/figures/`.

The current presentation reflects the verified portfolio-refresh results:

- 15,809 listings and 140 official neighbourhoods
- 17 Graph C Louvain communities and 16 Leiden communities
- Random, host-grouped, and spatial-block five-fold validation
- Seven Graph C sensitivity configurations

## Portfolio credit

Research, analysis, reproducibility work, robustness experiments, and the interactive presentation are presented as Sourav Chandhok's portfolio case study.
