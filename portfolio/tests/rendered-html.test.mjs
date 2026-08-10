import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { access, readFile } from "node:fs/promises";
import { createServer } from "node:net";
import { fileURLToPath } from "node:url";
import test, { after, before } from "node:test";

const projectDirectory = fileURLToPath(new URL("..", import.meta.url));
let serverProcess;
let origin;
let serverOutput = "";

async function availablePort() {
  const probe = createServer();
  probe.listen(0, "127.0.0.1");
  await once(probe, "listening");
  const address = probe.address();
  const port = typeof address === "object" && address ? address.port : 0;
  probe.close();
  await once(probe, "close");
  return port;
}

before(async () => {
  const port = await availablePort();
  origin = `http://127.0.0.1:${port}`;
  serverProcess = spawn(
    process.execPath,
    ["dist/standalone/server.js"],
    {
      cwd: projectDirectory,
      env: { ...process.env, HOST: "127.0.0.1", PORT: String(port) },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  serverProcess.stdout.on("data", (chunk) => {
    serverOutput += chunk.toString();
  });
  serverProcess.stderr.on("data", (chunk) => {
    serverOutput += chunk.toString();
  });

  for (let attempt = 0; attempt < 80; attempt += 1) {
    if (serverProcess.exitCode !== null) {
      throw new Error(`Portfolio server exited before startup.\n${serverOutput}`);
    }
    try {
      const response = await fetch(origin);
      if (response.ok) return;
    } catch {
      // The standalone server is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  throw new Error(`Portfolio server did not become ready.\n${serverOutput}`);
}, { timeout: 15_000 });

after(async () => {
  if (!serverProcess || serverProcess.exitCode !== null) return;
  serverProcess.kill();
  await Promise.race([
    once(serverProcess, "exit"),
    new Promise((resolve) => setTimeout(resolve, 2_000)),
  ]);
});

async function render() {
  return fetch(`${origin}/`, {
    headers: {
      accept: "text/html",
      "x-forwarded-host": "portfolio.example",
      "x-forwarded-proto": "https",
    },
  });
}

test("standalone server renders the completed interactive case study", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Toronto Airbnb Market Network/);
  assert.match(html, /15,809/);
  assert.match(html, /Three views of the same market/);
  assert.match(html, /Does the network improve price prediction\?/);
  assert.match(html, /Host-grouped/);
  assert.match(html, /Parameter sensitivity/);
  assert.match(html, /Portfolio attribution/);
  assert.match(html, /Research, analysis, and portfolio presentation by Sourav Chandhok/);
  assert.match(html, /github\.com\/souravC01\/toronto-airbnb-market-network/);
  assert.doesNotMatch(html, /Team attribution|equal contribution/i);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("publishes absolute social metadata and required artifacts", async () => {
  const response = await render();
  const html = await response.text();

  assert.match(html, /https:\/\/portfolio\.example\/og\.png/);
  assert.match(html, /summary_large_image/);

  await Promise.all([
    access(new URL("../public/og.png", import.meta.url)),
    access(
      new URL(
        "../public/figures/graph_c_louvain_community_map_labelled.png",
        import.meta.url,
      ),
    ),
  ]);

  const [page, experience, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/portfolio-experience.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.doesNotMatch(`${page}\n${experience}\n${layout}`, /_sites-preview|codex-preview/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
