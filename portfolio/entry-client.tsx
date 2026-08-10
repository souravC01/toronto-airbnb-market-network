import { StrictMode } from "react";
import { hydrateRoot } from "react-dom/client";

import { PortfolioExperience } from "./app/portfolio-experience";
import "./app/globals.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("The portfolio root element is missing.");
}

hydrateRoot(
  root,
  <StrictMode>
    <PortfolioExperience />
  </StrictMode>,
);
