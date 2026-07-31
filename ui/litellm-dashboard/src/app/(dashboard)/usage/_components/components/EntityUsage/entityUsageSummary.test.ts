import { describe, expect, it } from "vitest";
import { buildSummaryTiles } from "./entityUsageSummary";

const metadata = {
  total_spend: 100,
  total_flat_cost: 40,
  total_api_requests: 12,
  total_successful_requests: 10,
  total_failed_requests: 2,
  total_tokens: 3456,
};

describe("buildSummaryTiles", () => {
  it("omits flat-cost tiles when showFlatCost is false", () => {
    const titles = buildSummaryTiles(metadata, false).map((t) => t.title);
    expect(titles).not.toContain("Flat Cost");
    expect(titles).not.toContain("Total Cost");
    expect(titles).toHaveLength(5);
  });

  it("adds Flat Cost and Total Cost (spend + flat) when showFlatCost is true", () => {
    const tiles = buildSummaryTiles(metadata, true);
    const byTitle = Object.fromEntries(tiles.map((t) => [t.title, t.value]));
    expect(tiles).toHaveLength(7);
    expect(byTitle["Flat Cost"]).toBe("$40.00");
    expect(byTitle["Total Cost"]).toBe("$140.00");
  });

  it("treats a missing flat cost as zero", () => {
    const { total_flat_cost, ...noFlat } = metadata;
    const byTitle = Object.fromEntries(buildSummaryTiles(noFlat, true).map((t) => [t.title, t.value]));
    expect(byTitle["Flat Cost"]).toBe("$0.00");
    expect(byTitle["Total Cost"]).toBe("$100.00");
  });
});
