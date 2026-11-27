/**
 * Unicode code-point helpers for backend spans.
 *
 * The backend reports offsets in Python string indices, which are Unicode
 * code points. JavaScript string indices are UTF-16 code units, so spreading
 * a large document into an array is both expensive and unnecessary. This view
 * keeps the normal BMP path O(1) per slice and builds a compact index only
 * when the document actually contains astral characters (emoji, rare scripts).
 */
export interface CodePointView {
  length: number;
  hasAstral: boolean;
  slice(start: number, end: number): string;
  toUtf16(offset: number): number;
}

function isPair(text: string, i: number): boolean {
  if (i + 1 >= text.length) return false;
  const hi = text.charCodeAt(i);
  const lo = text.charCodeAt(i + 1);
  return hi >= 0xd800 && hi <= 0xdbff && lo >= 0xdc00 && lo <= 0xdfff;
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

export function codePointLength(text: string): number {
  let count = 0;
  for (let i = 0; i < text.length; i += 1) {
    if (isPair(text, i)) i += 1;
    count += 1;
  }
  return count;
}

/**
 * Build a reusable view for one text value.
 *
 * For the common case (no surrogate pairs) this does not allocate a per
 * character array at all. For documents with astral characters, the Uint32
 * table stores UTF-16 starts and is much smaller than an array of JS strings.
 */
export function createCodePointView(text: string): CodePointView {
  let astral = false;
  let length = 0;
  for (let i = 0; i < text.length; i += 1) {
    if (isPair(text, i)) {
      astral = true;
      i += 1;
    }
    length += 1;
  }

  if (!astral) {
    return {
      length,
      hasAstral: false,
      slice: (start, end) =>
        text.slice(clamp(start, 0, length), clamp(end, 0, length)),
      toUtf16: (offset) => clamp(offset, 0, length),
    };
  }

  const offsets = new Uint32Array(length + 1);
  let cp = 0;
  for (let i = 0; i < text.length; i += 1) {
    offsets[cp] = i;
    cp += 1;
    if (isPair(text, i)) i += 1;
  }
  offsets[length] = text.length;

  return {
    length,
    hasAstral: true,
    slice: (start, end) =>
      text.slice(
        offsets[clamp(start, 0, length)],
        offsets[clamp(end, 0, length)],
      ),
    toUtf16: (offset) => offsets[clamp(offset, 0, length)],
  };
}
