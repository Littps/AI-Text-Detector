// Icon build pipeline for AI-Text-Detector (D38, supersedes D37).
//
// Why a script and not `tauri icon`: that command takes one 1024 source and
// resamples it to every size, so the 16/24/32 bitmaps are blurred copies of the
// 1024 render. Here each size is rasterised from the vectors AT that size, and
// small sizes fall back to coarser variants so the magnifier survives 16 px.
//
// Three tiers over one 512-unit geometry (the user's own drawing):
//
//   A  complete  >= 32 : 4 text lines, frame 20, lens r72/stroke 20
//   B  reduced  21-31 : 3 text lines on an even rhythm, frame 24, lens r74/24
//   C  symbolic  <= 20 : 2 text lines, frame 32, lens r74/30
//
// The frame's bottom-right corner is left open where the lens crosses it
// (cutX/cutY).  The opening stays minimal on purpose: a round line-cap extends
// stroke/2 past the endpoint, so the lens' plate-coloured fill is what trims
// the stub.  Widening the opening instead made the frame read as an "L" at
// 16 px.
//
//   node scripts/gen_icons.cjs            # write all icons
//   node scripts/gen_icons.cjs --preview  # also write VERIFICATION/round17-final.png
//
// Outputs (all paths relative to the version folder):
//   src-tauri/icons/{32x32,128x128,128x128@2x,icon}.png
//   src-tauri/icons/{icon.ico,icon.icns}
//   src-tauri/icons/{icon.svg,icon-mid.svg,icon-small.svg}   <- editable sources
//   src-tauri/icons/{StoreLogo,Square*Logo}.png             <- Windows Store manifest
//   ui/public/favicon.svg
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const ROOT = path.resolve(__dirname, "..");
const ICONS = path.join(ROOT, "src-tauri", "icons");
const PREVIEW = process.argv.includes("--preview");

// ---- geometry -----------------------------------------------------------
// Pure white plate on purpose (user decision, round-17): #F1F0EA would vanish
// against the app's own cream background and let the black linework float.
const INK = "#000000";
const PLATE = "#FFFFFF";
const EDGE = "#F2F2F2";

// Bottom-right corner of the frame is left open for the lens.  cutX/cutY must
// grow with the stroke width, because a round cap extends w/2 past the endpoint
// and the gap to the lens is eaten from both sides.
const frame = (w, cutX = 270, cutY = 275) =>
  `<path d="M413 ${cutY} V60 Q413 35 388 35 H110 Q85 35 85 60 V412 Q85 437 110 437 H${cutX}" fill="none" stroke="${INK}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"/>`;

const plateRect = `<rect x="2" y="2" width="508" height="508" rx="72" fill="${PLATE}" stroke="${EDGE}" stroke-width="2"/>`;

const lines = (items, w) =>
  items
    .map(
      ([y, x2]) =>
        `<line x1="147" y1="${y}" x2="${x2}" y2="${y}" stroke="${INK}" stroke-width="${w}" stroke-linecap="round"/>`,
    )
    .join("");

// A — the original structure, with only its geometry bugs fixed:
//   - top-left corner was 20x23 while every other corner is 25 -> unified
//   - the 3rd text line ended 4 units from the lens (0.125 px at 16 px) -> 245
const SVG_A = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  ${plateRect}
  <g fill="none">
    ${frame(20)}
    ${lines([[127, 348], [200, 348], [270, 245], [343, 233]], 20)}
    <circle cx="351" cy="362" r="72" stroke="${INK}" stroke-width="20" fill="${PLATE}"/>
    <line x1="403" y1="414" x2="450" y2="461" stroke="${INK}" stroke-width="20" stroke-linecap="round"/>
  </g>
</svg>`;

// B — one line fewer placed on an even 108 rhythm (deleting one line from the
//      73/70/73 set would leave an uneven cadence that reads as noise at 24 px)
const SVG_B = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  ${plateRect}
  <g fill="none">
    ${frame(24, 290, 280)}
    ${lines([[127, 348], [235, 348], [343, 230]], 26)}
    <circle cx="351" cy="362" r="74" stroke="${INK}" stroke-width="24" fill="${PLATE}"/>
    <line x1="404" y1="415" x2="452" y2="463" stroke="${INK}" stroke-width="25" stroke-linecap="round"/>
  </g>
</svg>`;

// C — two thick lines, frame 32, lens 100 across: the smallest structure that
//      still reads as "document + magnifier" at 16 px
const SVG_C = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  ${plateRect}
  <g fill="none">
    ${frame(32, 290, 285)}
    ${lines([[160, 228], [270, 228]], 38)}
    <circle cx="352" cy="362" r="74" stroke="${INK}" stroke-width="30" fill="${PLATE}"/>
    <line x1="410" y1="420" x2="458" y2="468" stroke="${INK}" stroke-width="32" stroke-linecap="round"/>
  </g>
</svg>`;

const SMALL_MAX = 20; // <= 20 uses C
const MID_MAX = 31; // 21..31 uses B (24 and 30 are the only real callers)
const SVG_FULL = SVG_A;
const SVG_SMALL = SVG_C;

function variantFor(size) {
  if (size <= SMALL_MAX) return SVG_C;
  if (size <= MID_MAX) return SVG_B;
  return SVG_A;
}

// ---- containers ---------------------------------------------------------
function buildIco(entries) {
  const n = entries.length;
  const head = Buffer.alloc(6);
  head.writeUInt16LE(0, 0);
  head.writeUInt16LE(1, 2); // 1 = icon
  head.writeUInt16LE(n, 4);

  const dir = Buffer.alloc(16 * n);
  let offset = 6 + 16 * n;
  entries.forEach((e, i) => {
    const o = i * 16;
    const dim = e.size >= 256 ? 0 : e.size; // 0 means 256 in the ICO spec
    dir.writeUInt8(dim, o);
    dir.writeUInt8(dim, o + 1);
    dir.writeUInt8(0, o + 2); // palette count
    dir.writeUInt8(0, o + 3); // reserved
    dir.writeUInt16LE(1, o + 4); // planes
    dir.writeUInt16LE(32, o + 6); // bits per pixel
    dir.writeUInt32LE(e.png.length, o + 8);
    dir.writeUInt32LE(offset, o + 12);
    offset += e.png.length;
  });

  return Buffer.concat([head, dir, ...entries.map((e) => e.png)]);
}

// ICNS element types -> pixel size. ic11..ic14 are the @2x slots and reuse the
// same bitmaps as their 1x counterparts.
const ICNS_SLOTS = [
  ["icp4", 16], ["icp5", 32], ["icp6", 64],
  ["ic07", 128], ["ic08", 256], ["ic09", 512], ["ic10", 1024],
  ["ic11", 32], ["ic12", 64], ["ic13", 256], ["ic14", 512],
];

function buildIcns(bySize) {
  const parts = [];
  for (const [type, size] of ICNS_SLOTS) {
    const png = bySize.get(size);
    if (!png) throw new Error(`icns: missing ${size}px bitmap for ${type}`);
    const h = Buffer.alloc(8);
    h.write(type, 0, 4, "ascii");
    h.writeUInt32BE(png.length + 8, 4); // big-endian, includes the 8-byte header
    parts.push(h, png);
  }
  const body = Buffer.concat(parts);
  const head = Buffer.alloc(8);
  head.write("icns", 0, 4, "ascii");
  head.writeUInt32BE(body.length + 8, 4);
  return Buffer.concat([head, body]);
}

// ---- rasterisation ------------------------------------------------------
const SIZES = [16, 20, 24, 30, 32, 40, 44, 48, 50, 64, 71, 89, 107, 128, 142, 150, 256, 284, 310, 512, 1024];

const PAGE = "<!doctype html><meta charset=utf-8><body></body>";

const oneLine = (svg) => svg.replace(/\n\s*/g, " ") + "\n";

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 400 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.setContent(PAGE, { waitUntil: "load" });

  const bySize = new Map();

  for (const size of SIZES) {
    const svg = variantFor(size);
    // Inject width/height so the browser rasterises the vectors at exactly this
    // pixel count instead of scaling a 512px bitmap down.
    const sized = svg.replace("<svg ", `<svg width="${size}" height="${size}" `);
    const dataUrl = await page.evaluate(async ({ src, s }) => {
      const img = new Image();
      await new Promise((res, rej) => {
        img.onload = res;
        img.onerror = () => rej(new Error("svg failed to decode"));
        img.src = src;
      });
      const c = document.createElement("canvas");
      c.width = s;
      c.height = s;
      const g = c.getContext("2d");
      g.drawImage(img, 0, 0, s, s);
      return c.toDataURL("image/png");
    }, { src: "data:image/svg+xml;charset=utf-8," + encodeURIComponent(sized), s: size });

    bySize.set(size, Buffer.from(dataUrl.split(",")[1], "base64"));
  }

  if (errors.length) throw new Error("console errors: " + errors.join(" | "));

  // ---- write files ------------------------------------------------------
  fs.mkdirSync(ICONS, { recursive: true });
  const wrote = [];
  const write = (abs, buf) => {
    fs.writeFileSync(abs, buf);
    wrote.push([path.relative(ROOT, abs), buf.length]);
  };

  // editable sources, one per tier
  write(path.join(ICONS, "icon.svg"), Buffer.from(oneLine(SVG_A), "utf8"));
  write(path.join(ICONS, "icon-mid.svg"), Buffer.from(oneLine(SVG_B), "utf8"));
  write(path.join(ICONS, "icon-small.svg"), Buffer.from(oneLine(SVG_C), "utf8"));

  // tauri.conf.json listed set
  write(path.join(ICONS, "32x32.png"), bySize.get(32));
  write(path.join(ICONS, "128x128.png"), bySize.get(128));
  write(path.join(ICONS, "128x128@2x.png"), bySize.get(256));
  write(path.join(ICONS, "icon.png"), bySize.get(1024));

  // Windows .ico: 16/20/24/32/40 are the sizes Explorer and the taskbar ask for
  // at 100-250% scaling; 48+ carry the complete variant.
  const ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256];
  write(path.join(ICONS, "icon.ico"), buildIco(ICO_SIZES.map((s) => ({ size: s, png: bySize.get(s) }))));

  // macOS .icns
  write(path.join(ICONS, "icon.icns"), buildIcns(bySize));

  // Windows Store manifest tiles (referenced by the bundler on MSIX)
  const STORE = [
    ["StoreLogo.png", 50],
    ["Square30x30Logo.png", 30],
    ["Square44x44Logo.png", 44],
    ["Square71x71Logo.png", 71],
    ["Square89x89Logo.png", 89],
    ["Square107x107Logo.png", 107],
    ["Square142x142Logo.png", 142],
    ["Square150x150Logo.png", 150],
    ["Square284x284Logo.png", 284],
    ["Square310x310Logo.png", 310],
  ];
  for (const [name, size] of STORE) write(path.join(ICONS, name), bySize.get(size));

  // web favicon: the symbolic variant, so a 16 px tab keeps the magnifier
  write(path.join(ROOT, "ui", "public", "favicon.svg"), Buffer.from(oneLine(SVG_C), "utf8"));

  // the sidebar mark is inlined in App.tsx; keep it in sync with the C tier
  // (21 px on a 1x display — the thin detail strokes would turn grey)

  for (const [rel, bytes] of wrote) {
    console.log(String(bytes).padStart(8), rel);
  }
  console.log("tiers:", SIZES.map((s) => `${s}:${s <= SMALL_MAX ? "C" : s <= MID_MAX ? "B" : "A"}`).join(" "));

  // ---- optional preview sheet -------------------------------------------
  if (PREVIEW) {
    const shown = [16, 20, 24, 32, 40, 48, 64, 128, 256];
    const cells = shown.map((s) => {
      const b64 = bySize.get(s).toString("base64");
      const tier = s <= SMALL_MAX ? "C" : s <= MID_MAX ? "B" : "A";
      return `<figure>
        <div class="row">
          <div class="pad light"><img src="data:image/png;base64,${b64}" width="${s}" height="${s}"></div>
          <div class="pad dark"><img src="data:image/png;base64,${b64}" width="${s}" height="${s}"></div>
          <div class="pad app"><img src="data:image/png;base64,${b64}" width="${s}" height="${s}"></div>
        </div>
        <div class="zoom"><img src="data:image/png;base64,${b64}" width="${Math.min(s * 6, 192)}" height="${Math.min(s * 6, 192)}"></div>
        <figcaption><b>${s}px</b> · ${tier}</figcaption>
      </figure>`;
    }).join("");

    const html = `<!doctype html><meta charset=utf-8><style>
      *{box-sizing:border-box}
      body{margin:0;padding:22px 24px;background:#fff;color:#171717;font:12px/1.5 "Segoe UI",system-ui,sans-serif}
      h2{font-size:13px;font-weight:600;margin:0 0 4px}
      p.sub{margin:0 0 20px;color:#888;font-size:11px}
      .grid{display:flex;flex-wrap:wrap;gap:22px 26px;align-items:flex-start}
      figure{margin:0;text-align:center}
      .row{display:flex;gap:6px;margin-bottom:8px}
      .pad{display:grid;place-items:center;border-radius:6px;padding:9px;min-width:52px;min-height:52px}
      .light{background:#f3f3f3}
      .dark{background:#1f1f1f}
      .app{background:#F1F0EA}
      .zoom img{image-rendering:pixelated;display:block}
      figcaption{color:#777;font-size:11px;margin-top:4px}
      figcaption b{color:#333}
    </style>
    <h2>最终交付位图 — 每个尺寸由矢量按自身像素数直接栅格化</h2>
    <p class="sub">上排：浅色 / 深色任务栏 / 应用底色 #F1F0EA 1:1 · 下排：最近邻放大（最大 6×）· ≤20px 用 C 符号版，21–31px 用 B 简化版，≥32px 用 A 完整版</p>
    <div class="grid">${cells}</div>`;

    await page.setContent(html, { waitUntil: "load" });
    await page.waitForTimeout(300);
    const out = path.join(ROOT, "VERIFICATION", "round17-final.png");
    await page.screenshot({ path: out, fullPage: true });
    console.log("preview ->", path.relative(ROOT, out));
  }

  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
