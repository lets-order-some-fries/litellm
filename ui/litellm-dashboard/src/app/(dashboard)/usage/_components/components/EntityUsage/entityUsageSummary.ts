import { formatNumberWithCommas } from "@/utils/dataUtils";

export interface SummaryTile {
  title: string;
  value: string;
  className?: string;
}

interface SpendSummaryMetadata {
  total_spend: number;
  total_flat_cost?: number;
  total_api_requests: number;
  total_successful_requests: number;
  total_failed_requests: number;
  total_tokens: number;
}

export const buildSummaryTiles = (metadata: SpendSummaryMetadata, showFlatCost: boolean): SummaryTile[] => {
  const flatCost = metadata.total_flat_cost ?? 0;
  return [
    { title: "Total Spend", value: `$${formatNumberWithCommas(metadata.total_spend, 2)}` },
    ...(showFlatCost
      ? [
          { title: "Flat Cost", value: `$${formatNumberWithCommas(flatCost, 2)}` },
          { title: "Total Cost", value: `$${formatNumberWithCommas(metadata.total_spend + flatCost, 2)}` },
        ]
      : []),
    { title: "Total Requests", value: metadata.total_api_requests.toLocaleString() },
    {
      title: "Successful Requests",
      value: metadata.total_successful_requests.toLocaleString(),
      className: "text-green-600",
    },
    { title: "Failed Requests", value: metadata.total_failed_requests.toLocaleString(), className: "text-red-600" },
    { title: "Total Tokens", value: metadata.total_tokens.toLocaleString() },
  ];
};
