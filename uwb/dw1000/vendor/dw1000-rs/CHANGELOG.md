# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/nicolasfara/dw1000-rs/compare/v0.1.0...v0.2.0) - 2026-07-10

### Fixed

- add regression tests
- fix multi anchor setup

### Other

- improve readme

## [0.1.0](https://github.com/nicolasfara/dw1000-rs/releases/tag/v0.1.0) - 2026-06-02

### Added

- implement TX,RX led

### Fixed

- improve error and avoid paniking
- fix problem in receiving rx payload
- fix problem with led blinking on tx/rx DW1000 module
- avoid panic at runtime, use debug_assert instead
- fix bug leading to access out of bounds in array
- improve stability
- minor fixing improving stability
- trying to solve a problem causing the antenna to stuck
- trying to solve a problem causing the antenna to stuck

### Other

- format files
- update readme
- improve gitignore
- remove codex file
- setup workflow
- implement clippy with basic config
- refactor common logic
- avoid magic numbers as bitmasks
- remove old arduino references
- reorganized the example logic into different modules
- switch to embassy for examples
- general improvements
- semi-working porting
- some changes
- some migrations
- some progress
- some progress
- more on SPI interface
- some updates
- repo setup
