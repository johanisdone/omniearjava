package com.example.omniear.detection;

/**
 * Application-level threat categories. Mapping model labels onto these does not
 * add capability the underlying model does not have.
 */
public enum ThreatCategory {
    NORMAL,
    SCREAM_DISTRESS,
    /**
     * Gunshots and explosions share the same acoustic signature family (a loud,
     * broadband transient) and the model frequently confuses the two, so they are
     * treated as one P0 category: the score is the max of the two model labels.
     */
    GUNSHOT_EXPLOSION,
    IMPACT_CRASH,
    SIREN,
    GLASS_BREAK
}
