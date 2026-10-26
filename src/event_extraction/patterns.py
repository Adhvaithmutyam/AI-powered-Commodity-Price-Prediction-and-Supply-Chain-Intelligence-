"""
Pattern definitions, negation detection, and domain logic for supply-chain event extraction.
"""

import re
from typing import Any, Dict, List, Pattern

# ----------------------------------------------------
# Negation and cancellation patterns
# Identifies headlines where a disruption or action was averted, ended, or ruled out
# ----------------------------------------------------
NEGATION_PATTERNS: List[Pattern] = [
    re.compile(r"\b(strike|closure|shutdown|disruption|halt|outage)\s+(avoided|averted|called\s+off|ended|resolved|ruled\s+out|unlikely|denied)\b", re.I),
    re.compile(r"\b(avoided|averted|called\s+off|ruled\s+out|denied|lifted|ended)\s+(strike|shutdown|disruption|halt|cuts?|sanctions?)\b", re.I),
    re.compile(r"\b(production\s+cuts?|output\s+cuts?|supply\s+cuts?)\s+(ruled\s+out|dismissed|unlikely|shelved|rejected)\b", re.I),
    re.compile(r"\bno\s+(supply\s+disruptions?|shortage|impact\s+expected|halt|shutdown|cuts?)\b", re.I),
    re.compile(r"\b(lift(s|ed|ing)?|ease(s|d|ing)?)\s+(sanctions?|embargo|tariffs?|export\s+ban)\b", re.I),
]

# ----------------------------------------------------
# Misleading non-commodity contexts (sports, lifestyle, slang)
# ----------------------------------------------------
IRRELEVANT_PATTERNS: List[Pattern] = [
    re.compile(r"\b(gold\s+medal|olympics?|world\s+cup|championship|tournament|sports?|athletics?|boxing)\b", re.I),
    re.compile(r"\b(baseball|bowling\s+strike|lightning\s+strike|hunger\s+strike)\b", re.I),
    re.compile(r"\b(crude\s+humou?r|crude\s+joke|crude\s+cartoon)\b", re.I),
]

# ----------------------------------------------------
# Event classification rules
# ----------------------------------------------------
EVENT_PATTERNS: List[Dict[str, Any]] = [
    # 1. Supply Disruption
    {
        "event_type": "supply_disruption",
        "direction": "potentially_bullish",
        "rationale": "Reported infrastructure stoppage, strike, or export halt may constrain physical market availability.",
        "patterns": [
            re.compile(r"\b(mine|smelter|refinery|pipeline|facility|terminal)\s+(closure|shutdown|halt|blast|fire|explosion|disruption|strike)\b", re.I),
            re.compile(r"\b(force\s+majeure|strikers?|workers?\s+strike|labou?r\s+dispute|supply\s+disruptions?|shipping\s+blockade|tanker\s+attacked|chokepoint\s+closure)\b", re.I),
            re.compile(r"\b(halted|suspended|blocked|shut\s+down)\s+(supply|shipment|exports?|transit|operations?)\b", re.I),
            re.compile(r"\b(drought|flood|weather|earthquake|storm)\s+(disrupts|halts|shuts|delays)\s+(mining|production|shipping|exports?)\b", re.I),
            re.compile(r"\b(export\s+ban|export\s+curtailment|curbs?\s+exports?)\b", re.I),
        ],
    },
    # 2. Demand Change
    {
        "event_type": "demand_change",
        "direction": "potentially_bullish",
        "rationale": "Accelerating industrial consumption or green-transition deployment may tighten commodity balances.",
        "patterns": [
            re.compile(r"\b(demand\s+(surge|growth|rebound|rises?|jumps?|climbs?|booms?|soars?))\b", re.I),
            re.compile(r"\b(consumption\s+(jumps?|rises?|surges?|expands?|accelerates?))\b", re.I),
            re.compile(r"\b(ev\s+boom|renewable\s+expansion|grid\s+expansion|infrastructure\s+spending)\s+(boosts?|drives?|spurs?)\s+demand\b", re.I),
            re.compile(r"\b(strong|robust|booming)\s+(demand|appetite|consumption|orders?)\b", re.I),
        ],
    },
    {
        "event_type": "demand_change",
        "direction": "potentially_bearish",
        "rationale": "Macroeconomic headwinds or manufacturing contractions may reduce end-user material demand.",
        "patterns": [
            re.compile(r"\b(demand\s+(slump|slowdown|drops?|falls?|weakens?|destruction|wanes?|cools?))\b", re.I),
            re.compile(r"\b(recession|inflation|slowdown)\s+(weighs?|dents?|drags?|hits?)\s+on\s+demand\b", re.I),
            re.compile(r"\b(factory|manufacturing|industrial)\s+(slump|contraction|slowdown|stall)\b", re.I),
            re.compile(r"\b(weak|tepid|soft|cooling)\s+(demand|consumption|appetite|purchases?)\b", re.I),
        ],
    },
    # 3. Geopolitical Risk
    {
        "event_type": "geopolitical_risk",
        "direction": "potentially_bullish",
        "rationale": "Geopolitical conflict, embargoes, or trade sanctions typically expand risk premiums and safe-haven bids.",
        "patterns": [
            re.compile(r"\b(sanctions?|embargo|war\s+threat|military\s+(strike|escalation|action)|missiles?\s+attack)\b", re.I),
            re.compile(r"\b(middle\s+east\s+tensions?|strait\s+of\s+hormuz|red\s+sea\s+(crisis|attack|drone)|gulf\s+conflict)\b", re.I),
            re.compile(r"\b(geopolitical\s+(risk|tension|premium|instability|conflict))\b", re.I),
        ],
    },
    {
        "event_type": "geopolitical_risk",
        "direction": "mixed",
        "rationale": "Broad trade conflict or tariff disputes may simultaneously disrupt supply lines and impair global economic growth.",
        "patterns": [
            re.compile(r"\b(trade\s+war|tariffs?\s+(hike|dispute|imposed|threat)|retaliatory\s+tariffs?)\b", re.I),
            re.compile(r"\b(diplomatic\s+standoff|border\s+friction)\b", re.I),
        ],
    },
    # 4. Inventory Change
    {
        "event_type": "inventory_change",
        "direction": "potentially_bullish",
        "rationale": "Commercial stockpile depletion signals prompt supply deficits.",
        "patterns": [
            re.compile(r"\b(inventor(y|ies)|stockpiles?|stocks?|reserves?)\s+(draws?|decline|drop|plunge|shrink|fall|depleted)\b", re.I),
            re.compile(r"\b(draw\s+in\s+(crude|copper|oil|stockpiles?|inventories))\b", re.I),
            re.compile(r"\b(lme|eia)\s+(copper|crude|oil|stocks?|inventories?)\s+(drop|fall|drain|shrink)\b", re.I),
        ],
    },
    {
        "event_type": "inventory_change",
        "direction": "potentially_bearish",
        "rationale": "Rising stockpiles or storage capacity saturation indicate slack or oversupplied fundamentals.",
        "patterns": [
            re.compile(r"\b(inventor(y|ies)|stockpiles?|stocks?|reserves?)\s+(builds?|rise|surge|climb|swell|accumulate)\b", re.I),
            re.compile(r"\b(build\s+in\s+(crude|copper|oil|stockpiles?|inventories))\b", re.I),
            re.compile(r"\bstorage\s+(glut|overflow|capacity\s+strained)\b", re.I),
            re.compile(r"\b(lme|eia)\s+(copper|crude|oil|stocks?|inventories?)\s+(rise|climb|swell|accumulate)\b", re.I),
        ],
    },
    # 5. Policy Action
    {
        "event_type": "policy_action",
        "direction": "potentially_bullish",
        "rationale": "Central bank easing or mining regulatory constraints support liquidity and elevate extraction costs.",
        "patterns": [
            re.compile(r"\b(fed|central\s+bank|ecb)\s+(rate\s+cut|cuts\s+rates?|eases?|easing)\b", re.I),
            re.compile(r"\b(mining\s+royalties?|mining\s+tax|environmental\s+curbs?|permit\s+revocation)\b", re.I),
            re.compile(r"\b(resource\s+nationalism|export\s+tax)\b", re.I),
        ],
    },
    {
        "event_type": "policy_action",
        "direction": "potentially_bearish",
        "rationale": "Monetary tightening or government strategic stockpile releases exert downward pressure on prices.",
        "patterns": [
            re.compile(r"\b(fed|central\s+bank|ecb)\s+(rate\s+hike|hikes?\s+rates?|tightening)\b", re.I),
            re.compile(r"\b(spr\s+release|strategic\s+petroleum\s+reserve\s+release|emergency\s+stockpile\s+sale)\b", re.I),
        ],
    },
    {
        "event_type": "policy_action",
        "direction": "mixed",
        "rationale": "Policy summits or institutional negotiations have uncertain or conditional market implications.",
        "patterns": [
            re.compile(r"\b(opec\s+meeting|cartel\s+gathering|energy\s+minister\s+talks?)\b", re.I),
            re.compile(r"\b(regulatory\s+review|legislation\s+debated|carbon\s+policy)\b", re.I),
        ],
    },
    # 6. Production Change
    {
        "event_type": "production_change",
        "direction": "potentially_bullish",
        "rationale": "Targeted production cuts or smelter run reductions curtail newly refined supply.",
        "patterns": [
            re.compile(r"\b(opec(\+)?\s+(agrees?\s+to\s+)?(cuts?|lowers?|slashes?)\s+(output|quotas?|production))\b", re.I),
            re.compile(r"\b(production|output|smelting)\s+(cuts?|reduction|curtailment|slashed)\b", re.I),
            re.compile(r"\b(miners?|producers?)\s+(cuts?|lowers?|restricts?)\s+output\s+guidance\b", re.I),
        ],
    },
    {
        "event_type": "production_change",
        "direction": "potentially_bearish",
        "rationale": "Output ramp-ups or new extraction capacity expand ongoing commodity flows.",
        "patterns": [
            re.compile(r"\b(opec(\+)?\s+(boosts?|hikes?|raises?)\s+(output|production|quotas?))\b", re.I),
            re.compile(r"\b(production|output)\s+(surges?|expands?|ramps?\s+up|record\s+high|jumps?)\b", re.I),
            re.compile(r"\b(new\s+(mine|well|refinery)\s+(commissioned|opened|starts\s+production))\b", re.I),
        ],
    },
]
