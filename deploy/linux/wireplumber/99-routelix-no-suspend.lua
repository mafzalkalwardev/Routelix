-- WirePlumber 0.4 (Ubuntu 24.04). Keep Routelix null sinks from suspending.
-- A suspended PipeWire node makes paplay and pacat block until Python times out.
-- Install to /etc/wireplumber/main.lua.d/99-routelix-no-suspend.lua

alsa_monitor = alsa_monitor or {}
alsa_monitor.rules = alsa_monitor.rules or {}

table.insert(alsa_monitor.rules, {
  matches = {
    {
      { "node.name", "matches", "routelix_.*" },
    },
  },
  apply_properties = {
    ["session.suspend-timeout-seconds"] = 0,
  },
})
