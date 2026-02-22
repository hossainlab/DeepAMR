"""Bangladesh-specific clinical guidelines for AMR management.

Based on:
- DGHS (Directorate General of Health Services) antibiotic policy
- IEDCR (Institute of Epidemiology, Disease Control and Research) surveillance data
- WHO Bangladesh AMR reports
"""

from typing import Dict, List

# Bangladesh local resistance prevalence estimates (IEDCR surveillance)
BANGLADESH_RESISTANCE_PREVALENCE: Dict[str, Dict] = {
    "aminoglycoside": {
        "prevalence_pct": 35,
        "trend": "stable",
        "common_organisms": ["E. coli", "Klebsiella pneumoniae"],
        "note": "Gentamicin resistance common in hospital-acquired UTIs",
    },
    "beta-lactam": {
        "prevalence_pct": 65,
        "trend": "increasing",
        "common_organisms": ["E. coli", "K. pneumoniae", "Acinetobacter"],
        "note": "ESBL-producing Enterobacteriaceae highly prevalent in Bangladesh",
    },
    "fosfomycin": {
        "prevalence_pct": 8,
        "trend": "stable",
        "common_organisms": ["E. coli"],
        "note": "Low resistance; good option for uncomplicated UTI",
    },
    "glycopeptide": {
        "prevalence_pct": 5,
        "trend": "stable",
        "common_organisms": ["Enterococcus"],
        "note": "VRE rare but emerging in ICU settings",
    },
    "macrolide": {
        "prevalence_pct": 50,
        "trend": "increasing",
        "common_organisms": ["Staphylococcus aureus", "Streptococcus pneumoniae"],
        "note": "High resistance due to over-the-counter availability",
    },
    "phenicol": {
        "prevalence_pct": 30,
        "trend": "stable",
        "common_organisms": ["Salmonella typhi", "Shigella"],
        "note": "Chloramphenicol resistance common in enteric pathogens",
    },
    "quinolone": {
        "prevalence_pct": 70,
        "trend": "increasing",
        "common_organisms": ["E. coli", "Salmonella", "Shigella"],
        "note": "Very high resistance; ciprofloxacin no longer reliable for empirical use in Bangladesh",
    },
    "rifampicin": {
        "prevalence_pct": 12,
        "trend": "stable",
        "common_organisms": ["M. tuberculosis"],
        "note": "MDR-TB prevalence ~1.6% in new cases, ~29% in retreatment (NTP data)",
    },
    "sulfonamide": {
        "prevalence_pct": 60,
        "trend": "stable",
        "common_organisms": ["E. coli", "Shigella"],
        "note": "Co-trimoxazole resistance widespread",
    },
    "tetracycline": {
        "prevalence_pct": 55,
        "trend": "stable",
        "common_organisms": ["E. coli", "Vibrio cholerae"],
        "note": "Commonly used in aquaculture contributing to environmental resistance",
    },
    "trimethoprim": {
        "prevalence_pct": 58,
        "trend": "stable",
        "common_organisms": ["E. coli", "Klebsiella"],
        "note": "High resistance often linked to sulfonamide co-resistance",
    },
}

# DGHS recommended treatment alternatives per drug class
DGHS_ALTERNATIVES: Dict[str, List[str]] = {
    "beta-lactam": [
        "Consider meropenem/imipenem for ESBL-producing organisms",
        "Piperacillin-tazobactam for moderate infections",
        "Nitrofurantoin or fosfomycin for uncomplicated UTI",
    ],
    "quinolone": [
        "Use nitrofurantoin for uncomplicated UTI instead of ciprofloxacin",
        "Consider ceftriaxone for enteric fever (replacing ciprofloxacin)",
        "Azithromycin for Shigella dysentery if quinolone-resistant",
    ],
    "aminoglycoside": [
        "Amikacin may retain activity when gentamicin-resistant",
        "Consider colistin as last resort for MDR Gram-negatives",
    ],
    "macrolide": [
        "Doxycycline as alternative for respiratory infections",
        "Amoxicillin-clavulanate for community-acquired pneumonia",
    ],
    "sulfonamide": [
        "Nitrofurantoin for UTI instead of co-trimoxazole",
        "Azithromycin for Shigella if co-trimoxazole resistant",
    ],
    "trimethoprim": [
        "Fosfomycin for uncomplicated UTI",
        "Nitrofurantoin as alternative",
    ],
    "rifampicin": [
        "Refer to National TB Programme (NTP) for MDR-TB regimen",
        "Bedaquiline-based regimen per WHO 2022 guidelines",
    ],
}

# Key referral centers in Bangladesh
REFERRAL_CENTERS = [
    {
        "name": "Institute of Epidemiology, Disease Control and Research (IEDCR)",
        "location": "Mohakhali, Dhaka",
        "services": "AMR surveillance, reference laboratory",
    },
    {
        "name": "International Centre for Diarrhoeal Disease Research, Bangladesh (icddr,b)",
        "location": "Mohakhali, Dhaka",
        "services": "AMR research, clinical microbiology reference lab",
    },
    {
        "name": "National Institute of Diseases of the Chest and Hospital (NIDCH)",
        "location": "Mohakhali, Dhaka",
        "services": "MDR-TB diagnosis and treatment",
    },
]


def get_bangladesh_context(drug_class: str) -> Dict:
    """Get Bangladesh-specific context for a drug class."""
    prevalence = BANGLADESH_RESISTANCE_PREVALENCE.get(drug_class, {})
    alternatives = DGHS_ALTERNATIVES.get(drug_class, [])
    return {
        "local_prevalence": prevalence,
        "dghs_alternatives": alternatives,
    }


def get_bangladesh_recommendations(resistant_classes: set) -> List[str]:
    """Generate Bangladesh-specific recommendations based on resistance profile."""
    recs: List[str] = []

    if "quinolone" in resistant_classes:
        recs.append(
            "BANGLADESH CONTEXT: Quinolone resistance is ~70% locally. "
            "DGHS recommends avoiding ciprofloxacin for empirical therapy. "
            "Use ceftriaxone for enteric fever."
        )

    if "beta-lactam" in resistant_classes:
        recs.append(
            "BANGLADESH CONTEXT: ESBL prevalence is very high (>60%). "
            "Consider carbapenems for serious infections. "
            "For UTI, fosfomycin or nitrofurantoin remain effective."
        )

    if "macrolide" in resistant_classes and "quinolone" in resistant_classes:
        recs.append(
            "BANGLADESH CONTEXT: Combined macrolide + quinolone resistance limits oral options. "
            "Consider IV therapy and infectious disease consultation."
        )

    if "rifampicin" in resistant_classes:
        recs.append(
            "BANGLADESH CONTEXT: Possible MDR-TB. Refer to National TB Programme (NTP). "
            "Contact NIDCH, Mohakhali, Dhaka for MDR-TB management."
        )

    if len(resistant_classes) >= 5:
        recs.append(
            "BANGLADESH CONTEXT: Extensive drug resistance detected. "
            "Recommend referral to icddr,b or IEDCR for susceptibility testing and guidance."
        )

    if not recs:
        recs.append(
            "BANGLADESH CONTEXT: Predicted susceptibility aligns with local empirical options. "
            "Follow DGHS standard treatment guidelines."
        )

    return recs
