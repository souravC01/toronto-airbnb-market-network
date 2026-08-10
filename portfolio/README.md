# Toronto Airbnb Market Network — interactive case study

This site turns the research repository into a public-facing portfolio story. It presents the network design, graph evolution, Louvain/Leiden results, interpretable market segments, grouped and spatial price validation, parameter sensitivity, methodology, limitations, and portfolio attribution.

[View the live Render deployment](https://toronto-airbnb-market-network.onrender.com).

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

The production build is a self-contained Node server. The repository-level `render.yaml` configures Render to build this directory, start that server, and check the home page for service health.

## Data and assets

The metrics shown in the experience come from the canonical CSV artifacts in `../results/tables/`, and publication figures are copied from `../results/figures/`.

The current presentation reflects the verified portfolio-refresh results:

- 15,809 listings and 140 official neighbourhoods
- 17 Graph C Louvain communities and 16 Leiden communities
- Random, host-grouped, and spatial-block five-fold validation
- Seven Graph C sensitivity configurations

## Portfolio credit

Research, analysis, reproducibility work, robustness experiments, and the interactive presentation are presented as Sourav Chandhok's portfolio case study.
