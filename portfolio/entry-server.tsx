import { StrictMode } from "react";
import { renderToString } from "react-dom/server";

import { PortfolioExperience } from "./app/portfolio-experience";

export function renderPortfolio() {
  return renderToString(
    <StrictMode>
      <PortfolioExperience />
    </StrictMode>,
  );
}
