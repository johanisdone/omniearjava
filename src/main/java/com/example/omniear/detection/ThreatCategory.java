package com.example.omniear.detection;

/**
 * Application-level threat categories. Mapping model labels onto these does not
 * add capability the underlying model does not have.
 */
public enum ThreatCategory {
    NORMAL,
    SCREAM_DISTRESS,
    GUNSHOT,
    EXPLOSION,
    IMPACT_CRASH,
    SIREN,
    GLASS_BREAK
}
