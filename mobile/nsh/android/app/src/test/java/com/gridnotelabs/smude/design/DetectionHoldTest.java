package com.gridnotelabs.smude.design;
import org.junit.Test;
import static org.junit.Assert.*;

public final class DetectionHoldTest {
    @Test public void briefUncertaintyKeepsEstablishedPageButDoesNotRefreshDeadline() {
        assertTrue(DetectionHold.retain(true,1200,1000,.02,false));
        assertTrue(DetectionHold.retain(true,1600,1000,.02,false));
        assertFalse(DetectionHold.retain(true,1601,1000,.02,false));
        assertFalse(DetectionHold.retain(false,1200,1000,.02,false));
    }
    @Test public void motionAndHardFailureImmediatelyInvalidate() {
        assertFalse(DetectionHold.retain(true,1001,1000,.2,false));
        assertFalse(DetectionHold.retain(true,1001,1000,.01,true));
        assertFalse(DetectionHold.retain(true,1001,1000,Double.NaN,false));
    }
    @Test public void exposureShiftIsNotSpatialMotion() {
        double[] a=new double[64],b=new double[64];
        for(int i=0;i<64;i++) { a[i]=(i%8)/10.; b[i]=a[i]+.1; }
        assertEquals(0,DetectionHold.sceneDifference(a,b),1e-10);
        for(int i=0;i<64;i++) b[i]=(i%2==0) ? 0 : 1;
        assertTrue(DetectionHold.sceneDifference(a,b)>.08);
        assertTrue(Double.isInfinite(DetectionHold.sceneDifference(a,new double[1])));
    }
}
