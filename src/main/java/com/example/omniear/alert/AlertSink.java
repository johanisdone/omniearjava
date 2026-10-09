package com.example.omniear.alert;

import com.example.omniear.detection.ThreatEvent;

/** Destination for threat events (console now; logs/integrations later). */
public interface AlertSink {
    void emit(ThreatEvent event);
}
