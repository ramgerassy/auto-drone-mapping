"""The operator console's package: a top layer beside `cli`.

The console runs scenarios, installs uploaded rooms, and browses run history
(`docs/sprint-3-plan.md`, D2). This package sits *above* `cli` and may import
it; nothing in the core imports this package, so the simulator and CI never
depend on it. The UI itself (Feature 13) lives here too and is an optional
extra — nothing imported by this `__init__` needs it.
"""
