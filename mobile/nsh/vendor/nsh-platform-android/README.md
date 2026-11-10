# NSH Android scroll patch

`ScrollBox.java` comes from `nsh-platform-android` 0.4.0, with its original license alongside it. The only behavior change is the `enabled` check in `setOnScrollChangeListener`: programmatic scrolling while native touch scrolling is disabled must not cancel the raw touches used for page reordering. Normal native scrolling retains its existing touch cancellation.

Gradle removes the original ScrollBox and inner classes from its generated copy of the NSH runtime jar, then compiles this source in the same package. The installed NSH SDK and Cargo registry remain unchanged.
