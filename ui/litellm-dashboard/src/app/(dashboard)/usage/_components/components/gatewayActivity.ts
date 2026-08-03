/**
 * Gateway request counts (SGR) from `/gateway/daily/activity`.
 *
 * Recorded by the proxy's request-metrics middleware rather than derived from
 * spend logs, so it counts what the gateway actually answered. Deployment-wide
 * with no per-key or per-user dimension, which is why it is admin-only and why
 * the per-key and per-model breakdowns on the usage page still come from the
 * spend tables.
 */

export const GATEWAY_TOP_ROUTES = 15;

export interface GatewayActivity {
  total_successful_requests: number;
  total_failed_requests: number;
  by_date: { date: string; successful_requests: number; failed_requests: number }[];
  by_route: { category: string; route: string; successful_requests: number; failed_requests: number }[];
}

export interface FetchedGatewayActivity {
  rangeKey: string;
  activity: GatewayActivity;
}

/** Extends Record so it satisfies the chart component's row constraint. */
export interface GatewayRouteBar extends Record<string, unknown> {
  route: string;
  successful_requests: number;
  failed_requests: number;
}

/** Identifies the date range a result was fetched for. */
export const gatewayRangeKey = (startTime: Date | null | undefined, endTime: Date | null | undefined): string =>
  `${startTime?.toISOString() ?? ""}|${endTime?.toISOString() ?? ""}`;

/**
 * The counts safe to render right now, or null to fall back.
 *
 * Two ways a result must not reach the screen: the viewer is not an admin, and
 * the result belongs to a range the viewer has already navigated away from. The
 * second is why the fetched value carries its own range key; without it the
 * previous range's totals stay on screen for the length of the new request.
 */
export const selectGatewayActivity = (
  isAdmin: boolean,
  fetched: FetchedGatewayActivity | null,
  currentRangeKey: string,
): GatewayActivity | null => (isAdmin && fetched?.rangeKey === currentRangeKey ? fetched.activity : null);

/**
 * Bars for the endpoint breakdown chart, capped so a deployment exercising many
 * endpoints does not render an unreadable axis. `by_route` arrives sorted by
 * successful_requests descending, so the cap keeps the busiest endpoints.
 */
export const topGatewayRoutes = (
  activity: GatewayActivity | null,
  limit: number = GATEWAY_TOP_ROUTES,
): GatewayRouteBar[] =>
  (activity?.by_route ?? []).slice(0, limit).map((entry) => ({
    // The llm routes are already fully qualified; mcp and a2a routes are not, so
    // their category prefix is what keeps "/mcp" apart from "/a2a".
    route: entry.category === "llm" ? entry.route : `${entry.category}${entry.route}`,
    successful_requests: entry.successful_requests,
    failed_requests: entry.failed_requests,
  }));
