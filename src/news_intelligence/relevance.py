"""
Relevance assessment module for commodity news intelligence.

Evaluates whether an ingested news article is directly relevant to target
commodities, potentially relevant via macroeconomic/cross-market channels,
irrelevant (noise/false positive/homonym), or uncertain.

Transparently operates on available headline and metadata without fabricating
analysis of full article text.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RelevanceClass(str, Enum):
    """Relevance taxonomy for commodity news intelligence."""
    DIRECTLY_RELEVANT = "directly_relevant"
    POTENTIALLY_RELEVANT = "potentially_relevant"
    IRRELEVANT = "irrelevant"
    UNCERTAIN = "uncertain"


@dataclass
class RelevanceResult:
    """Structured assessment output for article relevance."""
    relevance_class: str
    relevance_score: float
    relevance_explanation: str
    matched_indicators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "relevance_class": self.relevance_class,
            "relevance_score": round(self.relevance_score, 2),
            "relevance_explanation": self.relevance_explanation,
            "matched_indicators": list(self.matched_indicators),
        }


# ---------------------------------------------------------------------------
# Indicator Regex Catalog
# ---------------------------------------------------------------------------

# Noise / False Positive Patterns
NOISE_PATTERNS: Dict[str, re.Pattern] = {
    "sports_competition": re.compile(
        r"\b(olympic[s]?|paralympic[s]?|gold\s+medal(?:ist)?|silver\s+medal(?:ist)?|"
        r"bronze\s+medal(?:ist)?|world\s+cup|championship[s]?|tournament|fifa|nba|nfl|"
        r"badminton|athletics|sprinter|podium|swimming|medalist[s]?)\b",
        re.IGNORECASE,
    ),
    "entertainment_culture": re.compile(
        r"\b(golden\s+globe[s]?|golden\s+raspberry|golden\s+boot|golden\s+boy|"
        r"box\s+office|grammy[s]?|oscar[s]?|hollywood|album|concert|cinema)\b",
        re.IGNORECASE,
    ),
    "idiomatic_figurative": re.compile(
        r"\b(crude\s+(?:joke|jokes|humor|humour|remark|remarks|gesture|gestures|language|comment|comments)|"
        r"golden\s+(?:age|rule|opportunity|jubilee|anniversary|handshake|ticket|years)|"
        r"heart\s+of\s+gold|golden\s+retriever)\b",
        re.IGNORECASE,
    ),
    "consumer_household_goods": re.compile(
        r"\b(copper\s+(?:pipe|pipes|pan|pans|pot|pots|cookware|mug|mugs|hair|wire\s+theft)|"
        r"gold\s+(?:earring|earrings|necklace|bracelet|pendant|jewel(?:e)?ry\s+collection))\b",
        re.IGNORECASE,
    ),
    "foreign_homonyms": re.compile(
        r"\b(đồng|tỷ\s+đồng|triệu\s+đồng|penny\s+stocks?|penny\s+pinching)\b",
        re.IGNORECASE,
    ),
}

# Commodity Identity Patterns
COMMODITY_PATTERNS: Dict[str, re.Pattern] = {
    "copper": re.compile(
        r"\b(copper|cathode[s]?|concentrate[s]?|cobre|red\s+metal|comex\s+copper|lme\s+copper|shfe\s+copper)\b",
        re.IGNORECASE,
    ),
    "gold": re.compile(
        r"\b(gold|bullion|yellow\s+metal|ingot[s]?|spot\s+gold|gold\s+futures|"
        r"gold\s+reserves|gold\s+mining|gold\s+etf[s]?|gold\s+bars?)\b",
        re.IGNORECASE,
    ),
    "crude oil": re.compile(
        r"\b(crude(?:\s+oil)?|brent(?:\s+crude)?|wti(?:\s+crude)?|petroleum|opec(?:\+)?|"
        r"barrels\s+(?:per\s+day|bpd|mbpd)|refinery|refining|drilling|oil\s+rig[s]?|"
        r"oil\s+tanker[s]?|supertanker[s]?|oil\s+pipeline[s]?|eia\s+crude|strategic\s+petroleum\s+reserve|spr)\b",
        re.IGNORECASE,
    ),
}

# Operational & Supply-Chain Direct Indicators
OPERATIONAL_PATTERNS: Dict[str, re.Pattern] = {
    "mining_production": re.compile(
        r"\b(mining|mine[s]?|miner[s]?|smelter[s]?|smelting|refinery|refining|"
        r"output|production|drilling|rig[s]?|extraction|capacity|producer[s]?|"
        r"bpd|mbpd|barrels|cathode[s]?|concentrate[s]?)\b",
        re.IGNORECASE,
    ),
    "supply_disruption": re.compile(
        r"\b(strike[s]?|walkout[s]?|outage[s]?|halt[s]?|halted|disrupt(?:ion|ions|ed)|"
        r"force\s+majeure|blockade[s]?|shutdown[s]?|embargo(?:es)?|sanction[s]?|"
        r"sabotage|pipeline\s+spill|explosion)\b",
        re.IGNORECASE,
    ),
    "inventory_storage": re.compile(
        r"\b(inventor(?:y|ies)|stockpile[s]?|storage|drawdown[s]?|build[s]?|"
        r"spr|strategic\s+petroleum\s+reserve|eia\s+report|stocks)\b",
        re.IGNORECASE,
    ),
    "market_price_movements": re.compile(
        r"\b(price[s]?|pricing|rally|rallies|slump[s]?|surge[s]?|plunge[s]?|"
        r"tumble[s]?|rebound[s]?|futures|spot|benchmark|trading|all-time\s+high|"
        r"lme|comex|cme|nymex|volatility|etf[s]?|inflow[s]?|outflow[s]?|holdings?|investor[s]?)\b",
        re.IGNORECASE,
    ),
    "trade_flows_demand": re.compile(
        r"\b(import[s]?|imported|export[s]?|exported|demand|consumption|"
        r"shipment[s]?|tariff[s]?|duties|trade\s+flow[s]?)\b",
        re.IGNORECASE,
    ),
}

# Macroeconomic & Cross-Market Patterns (Potentially Relevant)
MACRO_PATTERNS: Dict[str, re.Pattern] = {
    "monetary_policy": re.compile(
        r"\b(federal\s+reserve|the\s+fed|fed\s+chair|fed\s+rate|rate\s+cut[s]?|"
        r"rate\s+hike[s]?|interest\s+rate[s]?|fomc|jerome\s+powell|powell|"
        r"central\s+bank[s]?|monetary\s+policy|ecb|european\s+central\s+bank|"
        r"bank\s+of\s+england|bank\s+of\s+japan|boj|quantitative\s+easing|quantitative\s+tightening)\b",
        re.IGNORECASE,
    ),
    "inflation_data": re.compile(
        r"\b(inflation|cpi|ppi|pce|consumer\s+price\s+index|producer\s+price\s+index|"
        r"stagflation|deflation)\b",
        re.IGNORECASE,
    ),
    "currency_yields": re.compile(
        r"\b(us\s+dollar|dollar\s+index|dxy|greenback|treasury\s+yield[s]?|"
        r"bond\s+yield[s]?|10-year\s+yield|yield\s+curve)\b",
        re.IGNORECASE,
    ),
    "macro_growth_pmi": re.compile(
        r"\b(manufacturing\s+pmi|factory\s+activity|industrial\s+output|"
        r"industrial\s+production|gdp|gdp\s+growth|china\s+stimulus|"
        r"global\s+recession|economic\s+growth|nonfarm\s+payrolls)\b",
        re.IGNORECASE,
    ),
    "trade_geopolitics": re.compile(
        r"\b(trade\s+war|tariffs|tariff|import\s+tariffs|strait\s+of\s+hormuz|"
        r"red\s+sea|chokepoint|shipping\s+route[s]?)\b",
        re.IGNORECASE,
    ),
    "broad_commodities": re.compile(
        r"\b(commodity\s+market[s]?|commodities|raw\s+materials|broad\s+commodities|"
        r"energy\s+sector|industrial\s+metals|precious\s+metals)\b",
        re.IGNORECASE,
    ),
}


class RelevanceClassifier:
    """
    Transparent relevance classifier assessing headline text and metadata.
    """

    def __init__(self) -> None:
        pass

    def assess(
        self,
        headline: Optional[str],
        commodity: Optional[str] = None,
        language: Optional[str] = None,
        url: Optional[str] = None,
    ) -> RelevanceResult:
        """
        Assess relevance of a news record based on headline text and context.

        Args:
            headline: Headline or title string.
            commodity: Target commodity label (e.g. 'copper', 'gold', 'crude oil').
            language: Language label from ingestion (e.g. 'English').
            url: Article URL.

        Returns:
            RelevanceResult with classification, score, explanation, and matched indicators.
        """
        # Step 1: Check for missing, empty, or uninformative headline
        if not headline or not isinstance(headline, str) or not headline.strip():
            return RelevanceResult(
                relevance_class=RelevanceClass.UNCERTAIN.value,
                relevance_score=0.40,
                relevance_explanation=(
                    "Headline text is missing or empty. Cannot assess relevance from headline alone; "
                    "full article text was not analyzed."
                ),
                matched_indicators=["missing_headline"],
            )

        text = headline.strip()
        tokens = text.split()

        # Step 2: Check for uninformative boilerplates
        boilerplate_match = re.match(r"^(press\s+release|untitled|update|news|home|alert)$", text, re.IGNORECASE)
        if boilerplate_match:
            return RelevanceResult(
                relevance_class=RelevanceClass.UNCERTAIN.value,
                relevance_score=0.40,
                relevance_explanation=(
                    f"Headline '{text}' is generic boilerplate without informational context. "
                    "Assessment based on headline analysis; full article text was not analyzed."
                ),
                matched_indicators=["boilerplate_headline"],
            )

        # Step 3: Check for explicit noise / false-positive indicators
        matched_noise: List[str] = []
        for noise_name, pattern in NOISE_PATTERNS.items():
            if pattern.search(text):
                matched_noise.append(noise_name)

        # Step 4: Check commodity match
        norm_comm = commodity.strip().lower() if commodity else ""
        commodity_matched = False
        comm_indicators: List[str] = []

        if norm_comm in COMMODITY_PATTERNS:
            if COMMODITY_PATTERNS[norm_comm].search(text):
                commodity_matched = True
                comm_indicators.append(f"commodity_{norm_comm}")
        else:
            # Check across all known commodity patterns if commodity not explicitly provided
            for c_name, pat in COMMODITY_PATTERNS.items():
                if pat.search(text):
                    commodity_matched = True
                    comm_indicators.append(f"commodity_{c_name}")

        # Step 5: Check operational / supply-chain markers
        matched_ops: List[str] = []
        for op_name, pattern in OPERATIONAL_PATTERNS.items():
            if pattern.search(text):
                matched_ops.append(op_name)

        # Step 6: Check macroeconomic markers
        matched_macro: List[str] = []
        for macro_name, pattern in MACRO_PATTERNS.items():
            if pattern.search(text):
                matched_macro.append(macro_name)

        # Step 7: Classification Logic
        # Case A: Noise detected with no strong operational context
        if matched_noise:
            # If sports or crude humor is detected, it is irrelevant regardless of word presence
            # (e.g. 'Olympic gold medal', 'crude joke', 'golden retriever')
            if not (commodity_matched and matched_ops and "sports_competition" not in matched_noise and "idiomatic_figurative" not in matched_noise):
                return RelevanceResult(
                    relevance_class=RelevanceClass.IRRELEVANT.value,
                    relevance_score=0.10,
                    relevance_explanation=(
                        f"Headline matched noise/false-positive patterns ({', '.join(matched_noise)}) "
                        "unrelated to commodity supply chains. "
                        "Assessment based on headline analysis; full article text was not analyzed."
                    ),
                    matched_indicators=matched_noise,
                )

        # Case B: Direct commodity relevance (Commodity + Operational/Market context)
        if commodity_matched and (matched_ops or matched_macro):
            combined_indicators = comm_indicators + matched_ops + matched_macro
            # Score scaling: base 0.85, +0.03 per additional indicator up to 1.0
            score = min(1.0, 0.85 + (len(combined_indicators) - 2) * 0.03)
            return RelevanceResult(
                relevance_class=RelevanceClass.DIRECTLY_RELEVANT.value,
                relevance_score=score,
                relevance_explanation=(
                    f"Headline explicitly references target commodity and operational/market signals "
                    f"({', '.join(matched_ops + matched_macro)}). "
                    "Assessment based on headline analysis; full article text was not analyzed."
                ),
                matched_indicators=combined_indicators,
            )

        # Case C: Commodity mentioned alone without operational/market signals
        if commodity_matched and not matched_ops and not matched_macro:
            # Standalone commodity mention (e.g., "Exploring copper", "Gold trends")
            return RelevanceResult(
                relevance_class=RelevanceClass.DIRECTLY_RELEVANT.value,
                relevance_score=0.75,
                relevance_explanation=(
                    f"Headline references target commodity ({', '.join(comm_indicators)}) "
                    "without explicit supply-chain or market indicators. "
                    "Assessment based on headline analysis; full article text was not analyzed."
                ),
                matched_indicators=comm_indicators,
            )

        # Case D: Macroeconomic relevance without specific commodity mention
        # (e.g. "Fed cuts interest rates by 50 bps", "Global manufacturing PMI contracts", "US Dollar rallies")
        if matched_macro:
            score = min(0.80, 0.65 + (len(matched_macro) - 1) * 0.05)
            return RelevanceResult(
                relevance_class=RelevanceClass.POTENTIALLY_RELEVANT.value,
                relevance_score=score,
                relevance_explanation=(
                    f"Headline reports macroeconomic/cross-market drivers ({', '.join(matched_macro)}) "
                    "that materially influence commodity prices, though no specific commodity is named. "
                    "Assessment based on headline analysis; full article text was not analyzed."
                ),
                matched_indicators=matched_macro,
            )

        # Case E: Operational keywords matched, but target commodity is absent
        # (e.g., "Miners agree on new wage contract", "Refinery outage halts fuel exports")
        if matched_ops:
            return RelevanceResult(
                relevance_class=RelevanceClass.POTENTIALLY_RELEVANT.value,
                relevance_score=0.60,
                relevance_explanation=(
                    f"Headline mentions commodity-related operational activities ({', '.join(matched_ops)}) "
                    "without specifying target commodity in the headline. "
                    "Assessment based on headline analysis; full article text was not analyzed."
                ),
                matched_indicators=matched_ops,
            )

        # Case F: Short text or foreign language text without recognized indicators
        if language and language.lower() not in ("english", "en", "unknown"):
            return RelevanceResult(
                relevance_class=RelevanceClass.UNCERTAIN.value,
                relevance_score=0.45,
                relevance_explanation=(
                    f"Headline is in non-English language ('{language}') and contains no recognizable "
                    "English domain terms. Translation or full body extraction required. "
                    "Assessment based on headline analysis; full article text was not analyzed."
                ),
                matched_indicators=[f"non_english_{language.lower()}"],
            )

        # Case G: Unrelated headline with no commodity, macro, or operational terms
        # (e.g., "New Venues in Boston for Fall 2026 Meetings and Events")
        return RelevanceResult(
            relevance_class=RelevanceClass.IRRELEVANT.value,
            relevance_score=0.15,
            relevance_explanation=(
                "Headline contains no target commodity, operational supply-chain, or macroeconomic indicators. "
                "Assessment based on headline analysis; full article text was not analyzed."
            ),
            matched_indicators=["no_domain_signals"],
        )


# Global default classifier instance
_default_classifier = RelevanceClassifier()


def assess_relevance(
    headline: Optional[str],
    commodity: Optional[str] = None,
    language: Optional[str] = None,
    url: Optional[str] = None,
) -> RelevanceResult:
    """
    Assess relevance of an article headline against commodity intelligence criteria.

    Args:
        headline: Headline or title string.
        commodity: Target commodity label.
        language: Language identifier.
        url: Article URL.

    Returns:
        RelevanceResult dataclass instance.
    """
    return _default_classifier.assess(
        headline=headline,
        commodity=commodity,
        language=language,
        url=url,
    )


def enrich_record_with_relevance(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure non-destructive helper to enrich a news record with relevance metadata.

    Preserves all existing fields and appends:
      - relevance_class: str
      - relevance_score: float
      - relevance_explanation: str
      - matched_indicators: list of str
    """
    if not isinstance(record, dict):
        return {}

    title = record.get("title", "")
    commodity = record.get("commodity")
    language = record.get("language")
    url = record.get("url")

    result = assess_relevance(
        headline=title,
        commodity=commodity,
        language=language,
        url=url,
    )

    enriched = dict(record)
    enriched.update(result.to_dict())
    return enriched
