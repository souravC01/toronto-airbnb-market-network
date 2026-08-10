import { readFile, rm, writeFile } from "node:fs/promises";

const templatePath = new URL("../dist/index.html", import.meta.url);
const serverEntryPath = new URL(
  "../dist-server/entry-server.js",
  import.meta.url,
);

const template = await readFile(templatePath, "utf8");
const { renderPortfolio } = await import(serverEntryPath.href);
const appHtml = renderPortfolio();

if (!template.includes("<!--app-html-->")) {
  throw new Error("The static HTML template is missing the app marker.");
}

await writeFile(templatePath, template.replace("<!--app-html-->", appHtml));
await rm(new URL("../dist-server", import.meta.url), {
  recursive: true,
  force: true,
});
