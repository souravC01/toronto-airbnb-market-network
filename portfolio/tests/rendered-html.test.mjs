import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const builtHtmlUrl = new URL("../dist/index.html", import.meta.url);

test("static export contains the completed interactive case study", async () => {
  const html = await readFile(builtHtmlUrl, "utf8");
  assert.match(html, /<title>Toronto Airbnb Market Network/);
  assert.match(html, /15,809/);
  assert.match(html, /Three views of the same market/);
  assert.match(html, /Does the network improve price prediction\?/);
  assert.match(html, /Host-grouped/);
  assert.match(html, /Parameter sensitivity/);
  assert.match(html, /Portfolio attribution/);
  assert.match(html, /Research, analysis, and portfolio presentation by Sourav Chandhok/);
  assert.match(html, /View report/);
  assert.match(html, /EECS4414-Airbnb-Network-Analysis-Final-Report\.pdf/);
  assert.match(html, /Open the original.*report in a new tab/);
  assert.match(html, /github\.com\/souravC01\/toronto-airbnb-market-network/);
  assert.doesNotMatch(html, /Team attribution|equal contribution/i);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("publishes absolute social metadata and required artifacts", async () => {
  const html = await readFile(builtHtmlUrl, "utf8");

  assert.match(
    html,
    /https:\/\/toronto-airbnb-market-network\.vercel\.app\/og\.png/,
  );
  assert.match(html, /summary_large_image/);

  await Promise.all([
    access(new URL("../dist/og.png", import.meta.url)),
    access(
      new URL(
        "../dist/report/EECS4414-Airbnb-Network-Analysis-Final-Report.pdf",
        import.meta.url,
      ),
    ),
    access(
      new URL(
        "../dist/figures/graph_c_louvain_community_map_labelled.png",
        import.meta.url,
      ),
    ),
  ]);

  const [experience, indexHtml, packageJson] = await Promise.all([
    readFile(new URL("../app/portfolio-experience.tsx", import.meta.url), "utf8"),
    readFile(new URL("../index.html", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.doesNotMatch(`${experience}\n${indexHtml}`, /_sites-preview|codex-preview/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.doesNotMatch(packageJson, /vinext|next start|standalone/);
});
