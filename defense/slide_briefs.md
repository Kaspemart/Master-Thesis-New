# Defense Presentation — Slide Briefs

Thesis: "Estimating Parameters of Stochastic Volatility Models using Neural Networks"
Language: **Czech** (thesis is English). Talk length: ~15 min.
Fixed 10-slide university structure. Content authority = this file. Design done in separate window.

---

## Slide 3 — Úvod

**Hlavní sdělení:** Volatilita je klíčová, ale její odhad ve SV modelech je výpočetně náročný — a neuronové sítě slibují radikální zrychlení, jehož cena zatím není jasná.

**Odrážky na slajd (visible, keep to 4):**
- Volatilita je ústřední pro řízení rizika, oceňování derivátů a předpovědi na finančních trzích
- Modely stochastické volatility (SV) popisují volatilitu jako nepozorovaný (latentní) proces v čase
- Věrohodnostní funkce SV modelů nemá uzavřený tvar → odhad vyžaduje simulační metody
- Klasické metody jsou přesné, ale pomalé; neuronové sítě slibují odhad v milisekundách

**Vizuál:** `figures/fig_intro_volatility_clustering.png` — S&P 500 denní log-výnosy 2010–2025, shlukování volatility (COVID 2020 spike atd.).

**Poznámky k přednesu (~1.5–2 min):**
Volatilita, tedy míra kolísání výnosů, patří k nejdůležitějším veličinám ve financích — vstupuje do řízení rizika, oceňování derivátů i do předpovědí. Potíž je, že volatilita není přímo pozorovatelná a v čase se mění; typicky se shlukuje, kdy období klidu střídají období turbulence. Modely stochastické volatility tuto dynamiku zachycují tím, že volatilitu modelují jako samostatný, latentní náhodný proces. Právě to je ale i jádro obtíže: protože volatilitu nepozorujeme, věrohodnostní funkce těchto modelů nemá uzavřený tvar a nelze ji přímo maximalizovat. Odhad parametrů proto vyžaduje simulační metody. Klasické přístupy jako MCMC nebo částicové filtry jsou statisticky velmi přesné, ale výpočetně náročné — a hlavně, pro každý nový dataset je nutné celý odhad spustit znovu, což může trvat minuty až hodiny. Neuronové sítě nabízejí zásadně jiný přístup: síť se jednou natrénuje na simulovaných datech se známými parametry a poté odhaduje parametry pro nová data prakticky okamžitě. Tím vzniká otázka, kterou tato práce zkoumá: je ta obrovská úspora času vykoupena přijatelnou, nebo naopak nepřijatelnou ztrátou přesnosti a odolnosti vůči nesprávné specifikaci modelu?

**Status:** locked. Figure: `figures/fig_intro_volatility_clustering.png` (S&P 500, returns + 30-day rolling vol overlay).

---

## Slide 4 — Struktura práce

**Hlavní sdělení:** Práce jde od teorie (kap. 1–3) k empirii (kap. 4–7); kapitoly 5–7 nesou vlastní přínos.

**Na slajd (Czech):**
Nadpis: Struktura práce — Dvě části, sedm kapitol.
Teoretická část: 1. Stochastická volatilita · 2. Klasické metody odhadu · 3. Neuronové sítě pro odhad SV modelů
Empirická část: 4. Metodologie · 5. Simulační studie · 6. Analýza chybné specifikace · 7. Aplikace na reálná data

**Vizuál:** dvě sekce/sloupce (teoretická / empirická), lehký orientační slajd; empirická část jemně zvýrazněná.

**Poznámky (Czech):** viz paste block — orientace ~30–45 s; zmínit, že kap. 6 (chybná specifikace) je hlavní přínos, ale cíl neuvádět (patří na slide 5).

**Status:** locked.

---

## Slide 5 — Cíl práce

**Hlavní cíl:** Posoudit, za jakých podmínek NN odhad SV parametrů konkuruje MCMC a jak to ovlivňuje chybná specifikace.
**Dílčí:** architektury → vliv délky T → robustnost vůči chybné specifikaci (hlavní přínos) → reálná data.
**Status:** locked.

---

## Slide 6 — Metody práce

**Hlavní sdělení:** Kontrolované porovnání dvou odhadů téhož SV modelu — NN (amortizovaná inference) vs MCMC — hodnocené na přesnosti parametrů i prediktivní věrohodnosti mimo vzorek.
**Skupiny:** Model (SV + leverage) · NN (trénink na simulacích, 5 architektur) · Benchmark (stochvol: směs + ASIS) · Data (100k řad/T, oddělené množiny) · Vyhodnocení (RMSE/bias + prediktivní věrohodnost přes částicový filtr).
**Vizuál:** horizontální pipeline diagram (Simulace → NN / MCMC → Vyhodnocení).

**REVISION (verified against code):**
- **Model bullet (broadened):** „Diskrétní SV model a jeho rozšíření: pákový efekt (leverage), těžké chvosty (Student-t) a obojí (ASV-t)". Předchozí verze (jen base SV + leverage) podceňovala scope — SV-t a ASV-t se v misspec 2×2 „správný model" buňkách odhadují a ASV-t i v kap. 7. Čtyři modely celkem: base SV, ASV, SV-t, ASV-t.
- **Input bullet (fixed):** vstup je `log(r²)` (logaritmus druhé mocniny výnosu), NE „logaritmus výnosu" — ověřeno v `src/models/tcn.py:114`. U leverage modelu druhý kanál `sign(r)` (tcn.py:118). Parametry v transformovaném (neomezeném) tvaru.
- **Benchmark (clarified):** stochvol = primární; PyMC NUTS s uniform priory jen pro referenci (zkrácený bullet).
- **Caveat:** LSTM implementována, ale výpočetně neúnosná → do finálního srovnání nevstoupila (řeší slide 7).
- **Do NOT put on slide (→ notes/backup):** konkrétní transformace (logit φ, log σ_η, arctanh ρ, log(ν−2)); rozsahy μ∈(−10,0), φ∈(0.5,0.999), σ_η∈(0.05,1.0); vnořený test set z T=2000.

**Status:** locked (revised content sent to designer 2026-09-04).

---

## VÝSLEDKY — section plan (multiple slides)

Proposed 4 slides (numbering of later slides shifts accordingly):
- Výsledky 1 — Simulační studie: architektury + přesnost vs MCMC (fig1_architecture_comparison.png)
- Výsledky 2 — Vliv délky řady T (fig2_sample_size_analysis.png)
- Výsledky 3 — Chybná specifikace (hlavní přínos) (fig_ch6_rho_signfix.png / rho_identification)
- Výsledky 4 — Aplikace na reálná data (fig_ch7_oos_gap.png / asvt_gain)

**Status:** planning. Decisions: keep English figures as-is; phrase findings from repo results files.

---

## Slide 7 — Výsledky (1/4): Simulační studie

**Hlavní sdělení:** TCN je nejlepší architektura a při T=1000 dosahuje přesnosti srovnatelné s MCMC benchmarkem.
**Na slajd (Czech):** 5 architektur (LSTM neúnosná) · TCN nejlepší (~88k params, nejrychlejší) · při T=1000 srovnatelná s MCMC (stochvol) · TCN lepší v φ, MCMC nepatrně lepší v σ_η.
**Čísla (T=1000):** TCN μ0.279/φ0.081/σ0.082; stochvol μ0.281/φ0.107/σ0.080.
**Poctivá nuance (v poznámkách):** stochvol φ nafouknuté Beta(7,1) priorem; NUTS s flat priorem φ lepší než síť → síť je v φ konkurenceschopná, ne jednoznačně lepší.
**Figure:** `figures/fig1_architecture_comparison.png` (English labels, kept).
**Status:** locked.

---

## Slide 8 — Výsledky (2/4): Vliv délky řady

**Hlavní sdělení:** Přesnost roste s T u všech metod; vítěz závisí na parametru i délce. TCN nejlepší v φ; MCMC se prosazuje v σ_η při T=2000.
**Figure:** `figures/fig2_sample_size_analysis.png` (3-panel, English labels).
**Status:** locked.

---

## Slide 9 — Výsledky (3/4): Chybná specifikace (HLAVNÍ PŘÍNOS)

**Hlavní sdělení:** Zdánlivá slabina sítě u ρ = artefakt vstupu (log(r²) zahazuje znaménko); sign(r) kanál obnovil ρ na úroveň MCMC (corr 0,07→0,92; MCMC 0,93). Síť dorovnává MCMC u dobře identifikovaných modelů, ztrácí u slabě identifikovaných.
**Design:** 2×2 {síť, MCMC} × {base-SV, správný model}, 3 DGP (leverage, t, obojí); hodnoceno RMSE+bias a OOS prediktivní věrohodností.
**Figure:** `figures/fig_ch6_rho_signfix.png` (3-panel, English, kept as-is per user).
**Backup figures (Q&A):** `fig_ch6_rho_identification.png`, `fig_ch6_nu_identification.png`.
**Status:** locked.

---

## Slide 10 — Výsledky (4/4): Aplikace na reálná data

**Hlavní sdělení:** MCMC celkově > TCN na reálných datech (významně), ale TCN se vyrovná u akcií (dobře identifikováno) a ztrácí u měn (páka ~0). ASV-t zlepšuje fit 13/15. Potvrzení misspec zjištění.
**Figure:** `figures/fig_ch7_oos_gap.png` (2-panel, English). Backup: `fig_ch7_asvt_gain.png`.
**Status:** locked.

---

## Slide 11 — Diskuse

**Hlavní sdělení:** Síť konkuruje MCMC u dobře identifikovaných modelů, hlavní přednost rychlost; omezení: podceňuje φ blízko 1, akciový leverage bias, omezena tréninkovým rozsahem, MCMC lepší v σ_η a celkové OOS LL. Doporučení: síť pro rychlý screening, MCMC pro definitivní inferenci.
**Figure:** none (text synthesis).
**Status:** locked.

---

## Slide 12 — Vybraná literatura

**9 refs, two groups (from user's bibliography):**
SV & klasický odhad: Taylor (1994); Jacquier, Polson & Rossi (1994); Kim, Shephard & Chib (1998); Omori et al. (2007); Kastner & Frühwirth-Schnatter (2014); Kastner (2016).
Neuronové sítě: Bai, Kolter & Koltun (2018) [TCN]; Fičura & Witzany (2023); Witzany & Fičura (2023).
**Note:** user's pasted bib had OCR garbling — verify entries against clean thesis .bib. Taylor is 1994 (not 1982/86).
**Status:** locked.

---

## Slide 13 — Otázky z posudků (vedoucí)

Q1 training-distribution = implicit prior; weakly-identified params shrink to training mean (F4 ρ, F5 φ). Q2 misspec → no true param; predictive LL is right criterion (core thesis argument). Q3 uncertainty via amortized Bayesian inference (normalizing flow / BayesFlow), quantile reg, ensembles, conformal.
Full Czech answers in paste blocks (delivered in chat).
**Status:** locked.

---

## Slide 14 — Otázky z posudků (oponent)

Q1 TCN = 1D dilated causal conv; filters learn multi-scale vol features + sign-asymmetry (ties to sign-channel result). Q2 other misspec: jumps, long memory, regime switching, breaks, skew, multi-factor (jumps most relevant per F7). Q3 NN for fast/high-volume/well-identified, MCMC for definitive/weakly-identified (= slide 11).
Full Czech answers in paste blocks (delivered in chat).
**Status:** locked.

---

## FINAL ORDER (14 slides)
1 Title · 2 Struktura prezentace · 3 Úvod · 4 Struktura práce · 5 Cíl práce ·
6 Metody práce · 7 Výsledky 1/4 (simulační studie) · 8 Výsledky 2/4 (délka řady) ·
9 Výsledky 3/4 (chybná specifikace — přínos) · 10 Výsledky 4/4 (reálná data) ·
11 Diskuse · 12 Vybraná literatura · 13 Otázky (vedoucí) · 14 Otázky (oponent)
