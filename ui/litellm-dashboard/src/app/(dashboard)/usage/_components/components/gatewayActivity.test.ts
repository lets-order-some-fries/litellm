import { describe, expect, it } from "vitest";
import {
  GATEWAY_TOP_ROUTES,
  gatewayRangeKey,
  selectGatewayActivity,
  topGatewayRoutes,
  type GatewayActivity,
} from "./gatewayActivity";

const activity = (total: number): GatewayActivity => ({
  total_successful_requests: total,
  total_failed_requests: 0,
  by_date: [{ date: "2025-01-01", successful_requests: total, failed_requests: 0 }],
  by_route: [{ category: "llm", route: "/chat/completions", successful_requests: total, failed_requests: 0 }],
});

const JANUARY = gatewayRangeKey(new Date("2025-01-01T00:00:00Z"), new Date("2025-01-31T00:00:00Z"));
const FEBRUARY = gatewayRangeKey(new Date("2025-02-01T00:00:00Z"), new Date("2025-02-28T00:00:00Z"));

describe("gatewayRangeKey", () => {
  it("distinguishes ranges that differ only in their end", () => {
    const start = new Date("2025-01-01T00:00:00Z");
    expect(gatewayRangeKey(start, new Date("2025-01-31T00:00:00Z"))).not.toEqual(
      gatewayRangeKey(start, new Date("2025-02-28T00:00:00Z")),
    );
  });

  it("is stable for equal instants held in different Date objects", () => {
    expect(gatewayRangeKey(new Date("2025-01-01T00:00:00Z"), new Date("2025-01-31T00:00:00Z"))).toEqual(JANUARY);
  });

  it("tolerates a range that has not been picked yet", () => {
    expect(gatewayRangeKey(null, null)).toEqual("|");
  });
});

describe("selectGatewayActivity", () => {
  it("returns the counts when an admin's result matches the selected range", () => {
    expect(selectGatewayActivity(true, { rangeKey: JANUARY, activity: activity(7) }, JANUARY)).toEqual(activity(7));
  });

  it("withholds the previous range's counts while a new range is in flight", () => {
    expect(selectGatewayActivity(true, { rangeKey: JANUARY, activity: activity(7) }, FEBRUARY)).toBeNull();
  });

  it("withholds deployment-wide counts from a non-admin", () => {
    expect(selectGatewayActivity(false, { rangeKey: JANUARY, activity: activity(7) }, JANUARY)).toBeNull();
  });

  it("returns null before anything has been fetched", () => {
    expect(selectGatewayActivity(true, null, JANUARY)).toBeNull();
  });
});

describe("topGatewayRoutes", () => {
  it("leaves an llm route unprefixed and prefixes the others so they stay distinguishable", () => {
    const bars = topGatewayRoutes({
      ...activity(0),
      by_route: [
        { category: "llm", route: "/chat/completions", successful_requests: 3, failed_requests: 1 },
        { category: "mcp", route: "/tools/call", successful_requests: 2, failed_requests: 0 },
        { category: "a2a", route: "/tools/call", successful_requests: 1, failed_requests: 0 },
      ],
    });
    expect(bars.map((bar) => bar.route)).toEqual(["/chat/completions", "mcp/tools/call", "a2a/tools/call"]);
    expect(bars[0]).toEqual({ route: "/chat/completions", successful_requests: 3, failed_requests: 1 });
  });

  it("caps the bars at the top N so a wide deployment stays readable", () => {
    const many = Array.from({ length: GATEWAY_TOP_ROUTES + 5 }, (_, i) => ({
      category: "llm",
      route: `/route-${i}`,
      successful_requests: 100 - i,
      failed_requests: 0,
    }));
    const bars = topGatewayRoutes({ ...activity(0), by_route: many });
    expect(bars).toHaveLength(GATEWAY_TOP_ROUTES);
    // The cap keeps the busiest endpoints, which is only true because it slices
    // the server's descending order rather than re-sorting.
    expect(bars[0].route).toEqual("/route-0");
    expect(bars[GATEWAY_TOP_ROUTES - 1].route).toEqual(`/route-${GATEWAY_TOP_ROUTES - 1}`);
  });

  it("renders no bars when there is nothing to show", () => {
    expect(topGatewayRoutes(null)).toEqual([]);
  });
});
