# CRT-Royale MSL Port -- Projektstatus

## Projektbeschreibung

**Ziel:** Port des CRT-Royale-Shader von Slang/GLSL -> Metal Shading Language (MSL) + Integration in RetroVisor.

**Betreuer:** Prof. Dr. Dirk W. Hoffmann

---

## Projektphasen

### Phase 1: Portierung
Umsetzung in MSL.

| Pass | Beschreibung | Status |
|------|-------------|--------|
| Pass 1 | Linearize CRT Gamma + Bob Interlaced Fields | Portiert (Tests ausstehend)|
| Pass 2 | Vertical Scanlines (Beam-Distribution) | Ausstehend |
| Pass 3 | Mask Resize Horizontal (Lanczos) | Ausstehend |
| Pass 4 | Mask Resize Vertical | Ausstehend |
| Pass 5 | Horizontal Scanlines + Mask Apply | Ausstehend |
| Pass 6 | Brightpass (Bloom-Extraktion) | Ausstehend |
| Pass 7 | Bloom Vertical (Gaussian Blur) | Ausstehend |
| Pass 8 | Bloom Horizontal + Reconstitute | Ausstehend |
| Pass 9 | Bloom Approximation | Ausstehend |
| Pass 12 | Geometry + Anti-Aliasing + Final Output | Ausstehend |

### Phase 2: Integration
RetroVisor Integration + Anpassung der Rendering-Pipeline.

| Aufgabe | Status |
|---------|--------|
| Swift-Klasse CrtRoyale erstellt (Shader-Subklasse) | Erledigt |
| Metal-Datei CrtRoyale.metal erstellt | Erledigt |
| In ShaderLibrary registriert | Erledigt |
| Xcode-Projekt konfiguriert (pbxproj) | Erledigt |
| Build erfolgreich | Erledigt |
| UI-Parameter (Gamma, Interlacing) angebunden | Erledigt |
| Multi-Pass-Pipeline in apply() orchestriert | Begonnen (1 von ~10 Passes) |

### Phase 3: Validierung
Vergleich mit Original als Referenz.

| Aufgabe | Status |
|---------|--------|
| Testumgebung eingerichtet (OpenEmu) | Erledigt |
| 240p Test Suite ROM vorhanden | Erledigt |
| RetroArch als Referenz installieren | Ausstehend |
| Screenshot-Vergleichspipeline aufsetzen | Ausstehend |
| Pixel-Differenz-Analyse | Ausstehend |

### Phase 4: Evaluierung
Analyse der Qualität GPU-Performance.

| Aufgabe | Status |
|---------|--------|
| Bildqualität: CRT-Royale vs. Sankara/CRT-Easy | Ausstehend |
| GPU-Auslastung messen (Metal System Trace) | Ausstehend |
| Frame-Time-Analyse | Ausstehend |
| Ergebnisse dokumentieren | Ausstehend |

---

## Was bereits erledigt ist

### Infrastruktur
- [x] Projektstruktur angelegt (vendor/, crt-royale-msl/)
- [x] RetroVisor Repository geklont und erfolgreich gebaut
- [x] libretro/slang-shaders geklont (CRT-Royale Originalquellen)
- [x] Eigenes Git-Repository erstellt und auf GitHub gepusht
- [x] Metal Toolchain installiert
- [x] OpenEmu als Testquelle installiert + 240p Test Suite ROM

### Analyse & Verständnis
- [x] RetroVisor Shader-Architektur analysiert (Swift + Metal Compute Pattern)
- [x] Shader-Integrationsmuster verstanden (Shader, Kernel, ShaderLibrary, ShaderSetting)
- [x] CRT-Royale 13-Pass-Pipeline dokumentiert
- [x] Alle 28 Quelldateien (17 .slang + 11 .h) identifiziert
- [x] Slang/GLSL zu MSL Mapping erarbeitet
- [x] Festgestellt: Keine existierende MSL-Portierung vorhanden

### Code
- [x] Pass 1 (Linearize + Bob Fields) nach MSL portiert
- [x] Temporären Final-Encode-Pass erstellt (für sichtbare Ausgabe)
- [x] CrtRoyale.swift -- Swift-Integration mit Uniforms, Kernels, Settings
- [x] CrtRoyale.metal -- Metal Compute Shader
- [x] In RetroVisor ShaderLibrary registriert
- [x] Xcode-Projekt aktualisiert
- [x] Build erfolgreich

---

## Nächste Schritte

1. **Pass 2 portieren (Vertical Scanlines)** -- erster sichtbarer CRT-Effekt
2. **Pass 5 portieren (Horizontal Scanlines + Mask Apply)** -- Phosphor-Maske
3. **Pass 6-9 portieren (Bloom-Pipeline)** -- Halation/Bloom-Effekte
4. **Pass 12 portieren (Geometry + AA)** -- Bildschirmkrümmung
5. **Validierung** gegen RetroArch-Referenz
6. **Evaluierung** der Performance

---

## Technische Referenzen

| Ressource | Pfad / URL |
|-----------|-----------|
| RetroVisor Repo | vendor/RetroVisor/ |
| CRT-Royale Quellen | vendor/slang-shaders/crt/shaders/crt-royale/src/ |
| Unsere MSL-Portierung | crt-royale-msl/src/ |
| Unser GitHub Repo | https://github.com/snowIsNotAvaiable/crt-royale-msl |
| CRT-Royale Dokumentation | https://docs.libretro.com/shader/crt_royale/ |
| RetroVisor Website | https://dirkwhoffmann.github.io/RetroVisor/ |
| 240p Test Suite | test-roms/240p-test-suite.nes |
