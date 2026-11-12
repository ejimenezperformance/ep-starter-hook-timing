# ep-starter-hook-timing

**Emerson Performance · Starter fatigue and hook timing · Statcast 2025–2026**

[English](#english) · [Español](#español)

---

## English

### Headline

**A starter's in-game velocity fade is a stable personal trait. His third-time-through split is not.**
Across 52 MLB starters, the late-start fastball velo drop in 2025 predicted the 2026 drop (r = 0.74). Each starter's own 2025 profile cut the 2026 prediction error by 31% compared with the league-average rule (better in all 8 sensitivity settings, but confirmed in only 3). The third-time-through-the-order (TTO3) xwOBA penalty, the split many hook decisions lean on, did not repeat (r = 0.05). Using a starter's own TTO3 split made predictions 35% *worse* than the league average.

**Decision rule:** pull starters by the league times-through-the-order baseline plus **live fastball velo against that pitcher's own expected curve**. Do not use pitcher-specific TTO3 splits.

| Test | Result | Label |
|---|---|---|
| H1 · Late velo drop repeats 2025→2026 | r = 0.74 (95% CI 0.56–0.85), n = 52 | `CONFIRMED` |
| H2 · TTO3 xwOBA penalty repeats 2025→2026 | r = 0.05 (95% CI −0.15–0.24), n = 101 | Not confirmed |
| H3a · Own 2025 velo profile predicts 2026 velo drop better than league rule | MAE 0.31 vs 0.44 mph (−31%, CI −7% to −49%) | `CONFIRMED` in base spec (moderate robustness, see sensitivity) |
| H3b · Own 2025 TTO3 split predicts 2026 TTO3 split better than league rule | MAE 0.050 vs 0.037 (35% worse) | Not confirmed |
| H3c · 2025 velo profile predicts 2026 TTO3 split | 2% worse than league rule | Not confirmed |
| H4 · Live velo drop comes with worse contact, same pitcher | +0.007 xwOBA per mph lost (CI 0.002–0.012), 42,033 PA | `CONFIRMED` (small; fragile, see sensitivity) |

All p-values are Benjamini-Hochberg adjusted across the 8 tests.

![Stability](figures/fig1_stability.png)

![Profile curves](figures/fig2_profile_curves.png)

The groups were set **only with 2025 data** (bottom and top 20% of velo fade). In 2026 they still separated by **1.1 mph** late in the start: early faders −0.92 mph, league-typical −0.34 mph, velo holders +0.13 mph.

### Decision question

Should a club use each starter's own 2025 fatigue profile to decide when to pull him in 2026, or the league-average rule?

### What this means for a front office

1. **Stop using pitcher-specific TTO3 splits.** Split-half reliability inside one season is 0.04, which means it is almost pure noise. The league TTO pattern is real (2026 xwOBA: .302 → .319 → .333 for the 1st, 2nd and 3rd time through), but it is a league effect, not a trait of each pitcher. This lines up with Brill, Deshpande & Wyner (2023), who found that pitcher performance declines continuously across the game rather than jumping at the third time through.
2. **Use each starter's own velo curve as his "normal."** A −0.9 mph late drop is normal for an early fader and a red flag for a velo holder. The profile is stable (split-half reliability 0.84) and reaches about 0.89 reliability after roughly 23 starts that reach 76+ pitches.
3. **Treat live velo as a signal, not a trigger.** Every mph lost (measured before the PA starts) came with +0.007 xwOBA in the same pitcher. That is real but modest: a 2 mph drop is worth about half of the league TTO3 penalty (~.030).
4. **What this does NOT show:** that early faders get hit harder the third time through. A 2025 velo fade did not predict the 2026 TTO3 split (H3c). That outcome is too noisy at the pitcher level to predict with anything.

### Comparison line (league rule)

Without individual information, the best guess for any starter is the league average: a −0.37 to −0.39 mph fade from pitches 1–25 to pitches 76–100, plus the league TTO pattern. Every model is judged against that rule using leave-one-pitcher-out cross-validation.

### Application: offseason plan for early faders

`INTERPRETATION`, not tested in this repo.

Lab studies link late-game velo loss to changes in the delivery: less shoulder external rotation and less knee flexion at release (Murray et al., 2001), a more upright trunk as fatigue approaches (Escamilla et al., 2007), and selective strength loss in shoulder internal rotation, flexion and adduction after ~99 pitches (Mullaney et al., 2005). For the 29 starters classified as early faders (`outputs/starter_fatigue_profiles_2025_2026.csv`):

| Layer | Content |
|---|---|
| Data | Late velo drop beyond −0.80 mph (bottom 20%), shrunk toward the league mean by reliability |
| Finding | The fade repeats year to year (`CONFIRMED`) |
| Biomechanics | Loss of power transfer and shoulder rotator fatigue late in outings (`INTERPRETATION`) |
| Training | Repeat-effort capacity: high-intent throwing blocks with short rest building toward 90–100 pitches, shoulder rotator strength-endurance, lower-body power maintained in-season (`INTERPRETATION`) |
| Monitoring | Simulated-game velo decay test (first 15 vs last 15 fastballs) and pre/post force-plate CMJ; in season, live velo against the pitcher's own curve |

Before acting on any name, check medical history. Several extreme profiles had injuries (see `data/injury_exclusions.csv`), and a fade can be an early injury signal rather than a conditioning problem.

### Data and filters

- Statcast pitch-level data, **regular season only** (`game_type == "R"`): 2025 (712,528 pitches) and 2026 (716,792 pitches). `python src/pull_data.py`
- Starter = first pitcher of each team in each game. Pitch count = his running count in that game.
- Primary fastball = FF or SI, whichever he throws more (FC if neither reaches 15%).
- **Velo drop per start** = mean fastball velo on pitches 76–100 minus pitches 1–25 *of the same start* (at least 5 fastballs in each window). Pitcher-season qualifies with 8+ such starts.
- **TTO3 penalty** = xwOBA 3rd time through minus 1st time through (at least 40 and 80 PA). xwOBA = Statcast expected wOBA on contact, actual wOBA value on walks, HBP and strikeouts.
- **Injury and absence filter, applied before any analysis:** 1,277 of 9,718 starts excluded. Reasons: start followed by a 14+ day gap (318), return start after a 14+ day gap (316), season ended 14+ days early (227), opener or bulk pitcher with a median below 60 pitches (416). Ten pitcher-seasons with public injury reports are annotated in `data/injury_exclusions.csv` with sources.
- **H4 live velo** = mean of the last 5 fastballs thrown *before* the PA started, minus the start's first-25 baseline. The model is OLS of PA xwOBA on live drop + pitch count + times through + platoon + pitcher fixed effects, with SEs clustered by game, on PAs after pitch 50.

### Pre-registered thresholds

`CONFIRMED` requires BH-adjusted p < 0.05 **and** a practical threshold. For stability tests: r ≥ 0.30 with the bootstrap CI above 0. For decision tests: a 5% or larger MAE reduction with the paired-bootstrap CI above 0. For H4: CI above 0.

### Sensitivity

Reproduce with `python src/run_sensitivity.py` (writes `outputs/sensitivity.csv`). ✔ = `CONFIRMED`, ✘ = not confirmed.

| Setting | H1 n | H1 r | H2 r | H3a gain | H4 xwOBA/mph |
|---|---|---|---|---|---|
| Base | 52 | 0.74 ✔ | 0.05 ✘ | 31% ✔ | 0.0070 ✔ |
| Min 5 late starts | 82 | 0.68 ✔ | 0.05 ✘ | 23% ✔ | 0.0070 ✔ |
| Min 12 late starts | 34 | 0.72 ✔ | 0.05 ✘ | 30% ✘ | 0.0070 ✔ |
| Baseline pitches 10–30 | 47 | 0.71 ✔ | 0.05 ✘ | 24% ✘ | 0.0056 ✘ |
| Late window 81–105 | 31 | 0.73 ✔ | 0.05 ✘ | 29% ✘ | 0.0070 ✔ |
| Min 60 TTO3 PA | 52 | 0.74 ✔ | 0.08 ✘ | 28% ✔ | 0.0070 ✔ |
| Also drop starts within 30 days before a gap | 43 | 0.61 ✔ | 0.02 ✘ | 18% ✘ | 0.0064 ✔ |
| Drop all annotated injury seasons | 45 | 0.56 ✔ | 0.03 ✘ | 13% ✘ | 0.0072 ✔ |

- **H1 (the velo fade repeats) and H2 (the TTO3 split does not) hold in all 8 settings.** These are the robust findings.
- **H3a:** the starter's own profile beats the league rule in all 8 settings (13–31% lower error), but the gain is confirmed in only 3 of 8. With smaller samples the bootstrap interval touches zero. Read it as "consistently better, size uncertain."
- The stability shrinks (0.74 → 0.56) when injury seasons are removed, so part of the year-to-year signal comes from pitchers with injury histories.
- **H4 is the weakest finding:** with a baseline of pitches 10–30 it is no longer significant after correction.

### Limitations

- Survivorship: managers pull tired-looking starters, so the late window only sees starts where the pitcher was allowed to continue. Velo fades and their cost are probably understated.
- H4 is associational. Pitcher fixed effects remove stable differences, but not same-day factors such as illness or weather.
- Two seasons. n = 52 for the strictest stability test.
- The training plan is not tested here.

### Development notes (transparency)

- An internal first version of H4 measured live velo *including* the pitch that ended the PA. That inflated the effect to +0.016 xwOBA per mph. The published version uses only pitches thrown before the PA started (+0.007).
- An internal first version inverted the Spearman-Brown formula incorrectly for per-start reliability. Fixed before publication.

### Reproduce

```bash
pip install -r requirements.txt
python src/pull_data.py          # ~15 min, writes data/statcast_2025/2026.csv.gz
python tests/test_synthetic.py   # known-answer tests in both directions (8 PASS)
python src/run_analysis.py       # outputs/results.json and tables
python src/run_sensitivity.py    # outputs/sensitivity.csv (~2 min)
python src/make_figures.py       # figures/
```

The synthetic tests build fake seasons where the fatigue profile **does** repeat and velo **does** cost xwOBA (H1, H3a and H4 must be confirmed), and fake seasons where it does not (they must not be). All 8 checks pass.

### References

- Brill, R. S., Deshpande, S. K., & Wyner, A. J. (2023). A Bayesian analysis of the time through the order penalty in baseball. *Journal of Quantitative Analysis in Sports, 19*(4), 245–262. https://doi.org/10.1515/jqas-2022-0116
- Escamilla, R. F., Barrentine, S. W., Fleisig, G. S., Zheng, N., Takada, Y., Kingsley, D., & Andrews, J. R. (2007). Pitching biomechanics as a pitcher approaches muscular fatigue during a simulated baseball game. *American Journal of Sports Medicine, 35*(1), 23–33. https://doi.org/10.1177/0363546506293025
- Mullaney, M. J., McHugh, M. P., Donofrio, T. M., & Nicholas, S. J. (2005). Upper and lower extremity muscle fatigue after a baseball pitching performance. *American Journal of Sports Medicine, 33*(1), 108–113. https://doi.org/10.1177/0363546504266071
- Murray, T. A., Cook, T. D., Werner, S. L., Schlegel, T. F., & Hawkins, R. J. (2001). The effects of extended play on professional baseball pitchers. *American Journal of Sports Medicine, 29*(2), 137–142.

---

## Español

### Titular

**La caída de velocidad de un abridor dentro del juego es un rasgo personal estable. Su split de la tercera vuelta al lineup no lo es.**
En 52 abridores de MLB, la caída de velocidad de la recta al final de la salida en 2025 predijo la de 2026 (r = 0.74). El perfil propio de 2025 de cada abridor redujo el error de predicción de 2026 en un 31% frente a la regla promedio de la liga (mejor en las 8 configuraciones de sensibilidad, pero confirmado solo en 3). La penalización de xwOBA en la tercera vuelta al lineup (TTO3), el split en el que se apoyan muchas decisiones de sacar al abridor, no se repitió (r = 0.05). Usar el split TTO3 propio de cada pitcher empeoró las predicciones en un 35% frente al promedio de la liga.

**Regla de decisión:** sacar al abridor según la referencia de la liga por vuelta al lineup más la **velocidad en vivo comparada con la curva esperada de ese mismo pitcher**. No usar splits TTO3 individuales.

| Prueba | Resultado | Etiqueta |
|---|---|---|
| H1 · La caída tardía de velocidad se repite 2025→2026 | r = 0.74 (IC 95% 0.56–0.85), n = 52 | `CONFIRMED` |
| H2 · La penalización TTO3 se repite 2025→2026 | r = 0.05 (IC 95% −0.15–0.24), n = 101 | No confirmado |
| H3a · El perfil propio de 2025 predice la caída de 2026 mejor que la regla de la liga | MAE 0.31 vs 0.44 mph (−31%, IC −7% a −49%) | `CONFIRMED` en la especificación base (robustez moderada) |
| H3b · El split TTO3 propio de 2025 predice el de 2026 mejor que la regla de la liga | MAE 0.050 vs 0.037 (35% peor) | No confirmado |
| H3c · El perfil de velocidad de 2025 predice el split TTO3 de 2026 | 2% peor que la regla de la liga | No confirmado |
| H4 · La caída de velocidad en vivo vino con peor contacto, en el mismo pitcher | +0.007 xwOBA por mph perdida (IC 0.002–0.012), 42,033 PA | `CONFIRMED` (pequeño y frágil; ver sensibilidad) |

Todos los p-valores están ajustados por Benjamini-Hochberg en las 8 pruebas.

Los grupos del gráfico 2 se definieron **solo con datos de 2025** (20% inferior y superior de caída). En 2026 siguieron separados por **1.1 mph** al final de la salida: los que se apagan −0.92 mph, los típicos −0.34 mph y los que aguantan +0.13 mph.

### Pregunta de decisión

¿Debe un club usar el perfil de fatiga de 2025 de cada abridor para decidir cuándo sacarlo en 2026, o la regla promedio de la liga?

### Qué significa para un front office

1. **Dejar de usar splits TTO3 por pitcher.** Su confiabilidad dentro de una misma temporada (mitad contra mitad) es 0.04, es decir, casi puro ruido. El patrón de la liga sí es real (xwOBA 2026: .302 → .319 → .333 en la 1ª, 2ª y 3ª vuelta), pero es un efecto de liga, no un rasgo de cada pitcher. Esto coincide con Brill, Deshpande y Wyner (2023), que encontraron que el rendimiento del pitcher baja de forma continua durante el juego en lugar de dar un salto en la tercera vuelta.
2. **Usar la curva de velocidad propia de cada abridor como su "normal".** Una caída de −0.9 mph es normal para alguien que se apaga temprano y es una alerta para alguien que aguanta. El perfil es estable (confiabilidad 0.84) y alcanza ~0.89 tras unas 23 salidas que lleguen a 76+ lanzamientos.
3. **Tratar la velocidad en vivo como una señal, no como un gatillo automático.** Cada mph perdida (medida antes de empezar el turno) vino con +0.007 de xwOBA en el mismo pitcher. Es un efecto real pero modesto: perder 2 mph equivale a cerca de la mitad de la penalización TTO3 de la liga (~.030).
4. **Lo que esto NO muestra:** que los que se apagan temprano reciban más castigo en la tercera vuelta. La caída de velocidad de 2025 no predijo el split TTO3 de 2026 (H3c). Ese resultado es demasiado ruidoso a nivel de pitcher para predecirlo con cualquier variable.

### Línea de comparación (regla de la liga)

Sin información individual, la mejor estimación para cualquier abridor es el promedio de la liga: una caída de −0.37 a −0.39 mph entre los lanzamientos 1–25 y 76–100, más el patrón de la liga por vuelta al lineup. Cada modelo se compara contra esa regla con validación cruzada dejando fuera un pitcher a la vez.

### Aplicación: plan de offseason para los que se apagan temprano

`INTERPRETATION`, no probado en este repo.

Los estudios de laboratorio asocian la pérdida de velocidad al final del juego con cambios en la mecánica: menos rotación externa del hombro y menos flexión de rodilla al soltar (Murray et al., 2001), un tronco más vertical al acercarse la fatiga (Escamilla et al., 2007) y una pérdida selectiva de fuerza en rotación interna, flexión y aducción del hombro tras ~99 lanzamientos (Mullaney et al., 2005). Para los 29 abridores clasificados como "early fade" (`outputs/starter_fatigue_profiles_2025_2026.csv`):

| Capa | Contenido |
|---|---|
| Datos | Caída tardía mayor de −0.80 mph (20% inferior), ajustada hacia el promedio de la liga según la confiabilidad |
| Hallazgo | La caída se repite de un año a otro (`CONFIRMED`) |
| Biomecánica | Pérdida de transferencia de potencia y fatiga de los rotadores del hombro al final de la salida (`INTERPRETATION`) |
| Entrenamiento | Capacidad de esfuerzo repetido: bloques de lanzamiento de alta intención con descanso corto progresando hacia 90–100 lanzamientos, fuerza-resistencia de los rotadores del hombro y mantenimiento de la potencia del tren inferior durante la temporada (`INTERPRETATION`) |
| Monitoreo | Test de caída de velocidad en juego simulado (primeras 15 vs últimas 15 rectas) y CMJ en force plates antes y después; en temporada, velocidad en vivo comparada con la curva propia |

Antes de actuar sobre cualquier nombre, hay que revisar el historial médico. Varios perfiles extremos tuvieron lesiones (ver `data/injury_exclusions.csv`), y una caída puede ser una señal temprana de lesión y no un problema de acondicionamiento.

### Datos y filtros

- Datos de Statcast por lanzamiento, **solo temporada regular**: 2025 (712,528 lanzamientos) y 2026 (716,792).
- Abridor = primer pitcher de cada equipo en cada juego.
- **Caída por salida** = velocidad media de la recta en los lanzamientos 76–100 menos la de los lanzamientos 1–25 de la misma salida.
- **Filtro de lesiones y ausencias, aplicado antes del análisis:** se excluyeron 1,277 de 9,718 salidas (huecos de 14+ días, regresos, temporadas terminadas antes de tiempo, openers). Diez temporadas con lesiones públicas quedaron anotadas con su fuente.

### Sensibilidad y limitaciones

H1 (la caída se repite) y H2 (el split TTO3 no se repite) se mantienen en las 8 configuraciones (ver la tabla en inglés); son los hallazgos sólidos. H3a es mejor que la regla de la liga en las 8 (13–31% menos error), pero solo queda confirmado en 3: con muestras más pequeñas el intervalo toca el cero. La estabilidad baja de 0.74 a 0.56 al quitar las temporadas con lesión, así que parte de la señal viene de pitchers con historial de lesión. **H4 es el hallazgo más débil:** con una ventana base de lanzamientos 10–30 deja de ser significativo tras la corrección.

Limitaciones: survivorship (los managers sacan a los abridores que se ven cansados, así que la caída real probablemente se subestima), H4 es una asociación y no un efecto causal, solo hay dos temporadas y el plan de entrenamiento no se prueba aquí.

### Notas de desarrollo (transparencia)

Una primera versión interna de H4 incluía el lanzamiento que terminaba el turno dentro de la velocidad "en vivo", y eso inflaba el efecto a +0.016. La versión publicada usa solo los lanzamientos anteriores al turno (+0.007). También se corrigió antes de publicar la inversión de la fórmula de Spearman-Brown.

---

*Emerson Jimenez · Emerson Performance · [ejimenezperformance.github.io](https://ejimenezperformance.github.io)*
