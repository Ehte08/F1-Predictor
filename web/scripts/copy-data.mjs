// Copies ../data/site -> web/public/data so the static site can fetch /data/*.
// Runs automatically via predev / prebuild. Idempotent.
import { cp, mkdir, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const src = resolve(__dirname, "..", "..", "data", "site");
const dest = resolve(__dirname, "..", "public", "data");

async function main() {
  if (!existsSync(src)) {
    console.error(`[copy-data] source not found: ${src}`);
    process.exit(1);
  }
  await mkdir(dest, { recursive: true });
  await cp(src, dest, { recursive: true });
  const idx = join(dest, "index.json");
  const info = await stat(idx);
  console.log(`[copy-data] ${src} -> ${dest} (index.json ${info.size} bytes)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
