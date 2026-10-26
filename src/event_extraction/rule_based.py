"""
Rule-based supply-chain event extractor with explainable pattern matching,
negation detection, and commodity-specific economic rationale.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseEventExtractor, EventRecord
from .patterns import EVENT_PATTERNS, IRRELEVANT_PATTERNS, NEGATION_PATTERNS

logger = logging.getLogger(__name__)


class RuleBasedEventExtractor(BaseEventExtractor):
    """
    Explainable rule-based extractor for commodity supply-chain events.
    
    Identifies concrete operational, logistical, and geopolitical events,
    handles negations and false positives, and applies commodity-specific
    hypotheses for potential market impact.
    """

    def __init__(
        self,
        patterns: Optional[List[Dict[str, Any]]] = None,
        negation_patterns: Optional[List[Any]] = None,
        irrelevant_patterns: Optional[List[Any]] = None,
    ):
        self.patterns = patterns or EVENT_PATTERNS
        self.negation_patterns = negation_patterns or NEGATION_PATTERNS
        self.irrelevant_patterns = irrelevant_patterns or IRRELEVANT_PATTERNS

    def _resolve_commodity_specific_impact(
        self,
        event_type: str,
        initial_direction: str,
        commodity: str,
        full_title: str,
        matched_text: str,
        initial_rationale: str,
    ) -> Tuple[str, str]:
        """
        Adjust directional hypothesis based on commodity-specific market dynamics.
        
        Avoids treating every rate cut, geopolitical event, policy action,
        or production change as uniformly bullish or bearish across all commodities.
        """
        norm_comm = commodity.strip().lower()
        lower_title = full_title.lower()

        # 1. Geopolitical Risk Nuance
        if event_type == "geopolitical_risk":
            if norm_comm == "copper":
                # Copper is an industrial metal; broad geopolitical conflict, trade wars, or embargoes
                # hurt global manufacturing activity and cross-border trade.
                return (
                    "mixed",
                    "Geopolitical instability or trade friction may disrupt physical trade lanes but simultaneously threatens global industrial demand for copper.",
                )
            elif norm_comm in ("gold", "crude oil"):
                return (
                    "potentially_bullish",
                    f"Geopolitical conflict or sanctions expand risk premiums and safe-haven demand for {norm_comm}.",
                )

        # 2. Monetary Policy (e.g. Central Bank Rate Cuts)
        if event_type == "policy_action" and ("rate cut" in lower_title or "cuts rate" in lower_title):
            if norm_comm == "gold":
                return (
                    "potentially_bullish",
                    "Central bank interest rate cuts reduce the opportunity cost of holding non-yielding gold and tend to weaken the US dollar.",
                )
            elif norm_comm in ("copper", "crude oil"):
                # For industrial commodities, rate cuts stimulate activity but often occur during economic deceleration.
                return (
                    "mixed",
                    f"Monetary easing may stimulate longer-term demand, but rate cuts often coincide with macroeconomic growth deceleration for {norm_comm}.",
                )

        # 3. Production Changes Cross-Commodity Scope
        if event_type == "production_change":
            if "opec" in lower_title:
                if norm_comm != "crude oil":
                    return (
                        "unknown",
                        f"OPEC production quota decisions directly govern petroleum supply, not {norm_comm}.",
                    )
            elif "smelter" in lower_title or "mine" in lower_title:
                if norm_comm == "crude oil":
                    return (
                        "unknown",
                        "Metal mining or smelting adjustments do not directly constrain crude oil supply.",
                    )

        return initial_direction, initial_rationale

    def extract_event(self, record: Dict[str, Any]) -> EventRecord:
        """
        Extract a supply-chain event from an article record.

        Args:
            record: Article dictionary with 'title', 'url', 'commodity', and 'publication_timestamp'.

        Returns:
            EventRecord with structured fields.
        """
        if not isinstance(record, dict):
            return EventRecord(
                commodity="unknown",
                event_type="other",
                event_summary="Invalid record payload",
                impact_direction="unknown",
                confidence=0.0,
                evidence_url="",
                article_publication_timestamp=None,
                extraction_status="error",
                evidence_fact=None,
                implication_rationale="Input record was not a valid dictionary.",
            )

        # Preserve exact original identifiers without fabricating
        raw_url = record.get("url") or record.get("evidence_url")
        url = str(raw_url).strip() if raw_url is not None else ""

        commodity = str(record.get("commodity") or "unknown").strip().lower()

        raw_pub_ts = record.get("publication_timestamp") or record.get("article_publication_timestamp")
        pub_timestamp: Optional[str] = str(raw_pub_ts).strip() if raw_pub_ts is not None else None

        raw_title = record.get("title") or record.get("text")
        title = str(raw_title).strip() if raw_title is not None else ""

        # Safe fallback for missing or empty text
        if not title:
            return EventRecord(
                commodity=commodity,
                event_type="other",
                event_summary="No headline or text provided",
                impact_direction="unknown",
                confidence=0.0,
                evidence_url=url,
                article_publication_timestamp=pub_timestamp,
                extraction_status="insufficient_data",
                evidence_fact=None,
                implication_rationale="Article lacks textual headline or body content.",
            )

        # 1. Filter out obvious misleading non-commodity contexts (sports, lifestyle, slang)
        for irr_pat in self.irrelevant_patterns:
            if irr_pat.search(title):
                return EventRecord(
                    commodity=commodity,
                    event_type="other",
                    event_summary=title,
                    impact_direction="unknown",
                    confidence=0.0,
                    evidence_url=url,
                    article_publication_timestamp=pub_timestamp,
                    extraction_status="unclassified",
                    evidence_fact=None,
                    implication_rationale="Headline matches non-commodity or recreational context (e.g., sports, medals, slang).",
                )

        # 2. Check for explicit negations and averted disruptions
        for neg_pat in self.negation_patterns:
            match = neg_pat.search(title)
            if match:
                negated_fact = match.group(0)
                return EventRecord(
                    commodity=commodity,
                    event_type="other",
                    event_summary=title,
                    impact_direction="unknown",
                    confidence=0.85,
                    evidence_url=url,
                    article_publication_timestamp=pub_timestamp,
                    extraction_status="success",
                    evidence_fact=f"negation detected: {negated_fact}",
                    implication_rationale="Reported operational disruption or production cut was averted, resolved, or ruled out; no active physical constraint.",
                )

        # 3. Evaluate operational and economic pattern matches
        matched_groups: List[Tuple[Dict[str, Any], str]] = []
        for group in self.patterns:
            for pat in group["patterns"]:
                match = pat.search(title)
                if match:
                    matched_groups.append((group, match.group(0)))
                    break  # Avoid double-counting same group

        # If no operational patterns matched
        if not matched_groups:
            return EventRecord(
                commodity=commodity,
                event_type="other",
                event_summary=title,
                impact_direction="unknown",
                confidence=0.0,
                evidence_url=url,
                article_publication_timestamp=pub_timestamp,
                extraction_status="unclassified",
                evidence_fact=None,
                implication_rationale="No known supply-chain event keywords or operational markers detected.",
            )

        # Primary matched group is the most specific match
        primary_group, primary_fact = matched_groups[0]
        event_type = primary_group["event_type"]
        initial_direction = primary_group["direction"]
        initial_rationale = primary_group["rationale"]

        all_directions = {grp["direction"] for grp, _ in matched_groups}
        all_facts = [fact for _, fact in matched_groups]
        combined_facts = "; ".join(all_facts)

        # Check for multiple conflicting directions across matches
        if "potentially_bullish" in all_directions and "potentially_bearish" in all_directions:
            direction = "mixed"
            rationale = "Contradictory supply/demand or policy markers detected in reported headline."
            confidence = 0.60
        else:
            # Apply commodity-specific economic nuance
            direction, rationale = self._resolve_commodity_specific_impact(
                event_type=event_type,
                initial_direction=initial_direction,
                commodity=commodity,
                full_title=title,
                matched_text=combined_facts,
                initial_rationale=initial_rationale,
            )
            confidence = 0.90 if len(matched_groups) > 1 else 0.80

        # Construct concise factual event summary from reported headline
        event_summary = f"{title}"

        return EventRecord(
            commodity=commodity,
            event_type=event_type,
            event_summary=event_summary,
            impact_direction=direction,
            confidence=confidence,
            evidence_url=url,
            article_publication_timestamp=pub_timestamp,
            extraction_status="success",
            evidence_fact=combined_facts,
            implication_rationale=rationale,
        )
