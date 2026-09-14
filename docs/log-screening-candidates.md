# Candidate public event logs, and why each was set aside

**Screened:** 2026-07-27 · **Machine-readable copy:** [`data/log-screening-candidates.json`](data/log-screening-candidates.json)
· **The screen itself:** `python -m procposets.logscreen LOG.xes.gz --report ties|intervals`
(`procposets.logscreen`)

**What this is.** `procposets` reads a case as a *partial order* of activity occurrences, so it
needs a log that carries **within-case concurrency**: two activities of one case genuinely under way
at once. `procposets.logscreen` decides that question for a single log. This table is the result of
running that question across the public corpus — one row per source, recording which requirements it
satisfies, the single decisive reason it was set aside, and — the column that matters most for
judging whether the corpus can be repaired — **the smallest change to that source that would make it
usable**, with an honest rating of who could make that change.

It is published for two reasons: so that "we looked and found none" can be checked rather than
taken on trust, and so that nobody repeats the search from scratch.

**"The incumbent"**, referred to throughout, is the simulated fleet a candidate would have had to
displace: the search was run for a study whose exhibit was already synthetic, which is why being
simulated (R0) disqualifies a source here. A reader with a different purpose should read R0 as
optional and the rest as the substantive bar.

## 0. Read this first: 120 registry rows are not 120 candidate logs

| | count |
|---|---|
| distinct candidate **logs** screened | **59** |
| duplicate registry entries (same dataset, second DOI/mirror/name) | 21 |
| **generators** (produce logs; are not logs) | 29 |
| **indexes** (catalogues/link-lists of other logs) | 9 |
| **libraries** (software, no events) | 2 |
| **total registry rows** | **120** |

Quoting "120 sources screened" would overstate the work. The honest number is **59 distinct
candidate logs**, of which **20 were measured on disk** and the rest rejected on documented schema
facts or on access. Both the duplicates (§3) and the non-datasets (§4) are listed in full so that
nothing appears to have been quietly dropped.

## 1. How to read the table

**Requirements.** A usable source must satisfy all of:

| | requirement | why it is required |
|---|---|---|
| **R0** | **real**, not simulated | the incumbent exhibit is *already* a simulated fleet, so a simulated source cannot displace it |
| **R1** | a declared, **site-like** fleet key, G ≥ 3 groups × n_g ≥ 3 cases | the comparison is a statement about a *fleet* of models, not a single one |
| **R2** | a **case notion** | one poset is built per case |
| **R3a** | timestamps with a **sub-second** component | simultaneity is read as timestamp *equality*. At a 1-second ceiling a tie may be simultaneity or batching and **nothing in the log can distinguish them** |
| **R3b** | tie blocks **≥80% distinct-activity and ≥50% cross-resource** | separates genuine multi-resource simultaneity from *one actor emitting several fields at one instant* |
| **R4** | order structure varying **independently of activity frequency** | otherwise the fleet separates on *what* is done, not *in what order* |
| **R5** | usable **variant structure** | ≥90% singleton traces saturates the order distance; **†** marks the opposite failure, a degenerate 1–5 variants carrying no signal |

**The R1 axis column** is the requirement most open to challenge, so the standard is stated
explicitly and applied mechanically. A grouping attribute is one of:

- **site** — independent organisations, plants, hospitals, authorities, countries running *the same
  process*. **Only this yields R1 = Y.**
- **vocab** — groups defined by a *different activity alphabet* (different processes, products, game
  civilisations, declaration types). These separate on activity identity, failing R4 by construction.
- **resource** — individual actors, handlers, mailboxes, devices *within one organisation*. A
  resource cohort is not a fleet of organisations, and is not among the three permitted label
  surrogates. **This is why BPI 2012's `org:resource` and Enron's mailbox axis both score N** — an
  earlier draft of this table scored them differently, which was the single largest inconsistency an
  adversarial review found.

**Symbols.** `Y` satisfied · **`N`** fails · `·` **not measured** — an earlier requirement already
decided the row, so measuring this one would have bought nothing. A `·` is an honest statement of
what we did not do, not a hidden failure.

**Evidence column (`ev`).** `●` measured on disk by us · `○` judged from published schema/metadata
· `✕` cannot be measured at all (no obtainable data).

**Feasibility.** `impossible` no change to this source could work · `upstream` only the data owner
or depositor could act, by re-instrumenting or re-exporting at source · `plausible` we or a
collaborator could obtain or derive it · `trivial` a re-export we could simply request.
**Governance-blocked is not schema-death:** DREAM (53), PurpleLab (63), EHDEN (73) and HCUP (100)
are technically plausible fleets withheld by access control, and they are rated `upstream`, not
`impossible`. That distinction is the honest one and referees care about it.

---
## 2. The table — 59 distinct candidate logs

| # | source | R0 real | R1 fleet | axis | R2 case | R3a gran | R3b comp | R4 ord⊥frq | R5 var | ev | rejected because | minimum change that would make it usable | feasible? |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|---|---|
| 0 | OCEL 2.0 order-management (simulated) | **N** | **N** | resource | Y | · | · | · | **N**† | ● | Generated in CPN-Tools, not observed -- a simulated log cannot displace the simulated incumbent fleet; also <=0.9% tie cases. | None -- being simulated is not a fixable defect; it would have to be replaced by an observed order-to-cash log from the real ERP the CPN model imitates. | impossible |
| 1 | OCEL 2.0 procure-to-pay (P2P) | **N** | **N** | — | Y | **N** | **N** | · | **N**† | ● | Simulated SAP replica; its 25.9% tie cases are 100% same-activity at minute granularity, i.e. a rounded batch-emission artifact. | None -- replace with a genuine multi-plant SAP change-document extract; a CPN generator cannot be re-run into realness, and vendor is a supplier segment not a site. | impossible |
| 2 | OCEL 2.0 container-logistics | **N** | **N** | resource | Y | · | · | · | **N**† | ● | Simulated showcase log of a single operation -- the same category of evidence as the Synthea fleet it would replace; 1-5 trace variants. | None -- needs an observed multi-terminal TOS handling log; even if real, the container/vehicle key is a within-operation asset axis, not a site fleet. | impossible |
| 3 | Angular GitHub-commits OCEL 2.0 | Y | **N** | resource | Y | · | **N** | · | · | ● | One repository, no site axis: only directory-prefix (707) or release-era slices; and 88.9% of its tie blocks are same-activity. | A multi-repo fleet we could build ourselves, but git stamps at 1 s and the ties are rebase/squash re-stamping -- no source change yields simultaneity. | impossible |
| 4 | Enron email OCEL | Y | **N** | resource | Y | **N** | **N** | · | · | ● | Single organisation grouped by individual mailbox -- no site axis (registry cross_site 'no'); 69.5% of stamps are minute-precision. | None: RFC-2822 Date headers stop at seconds, so no re-extraction from the PST corpus can yield sub-second ties, whatever grouping is bolted on. | impossible |
| 5 | BPI Challenge 2012 (Dutch bank) | Y | **N** | resource | Y | Y | **N** | · | Y | ● | Only 2 of 11,361 sub-second tie blocks are cross-resource (0.0% vs a declared >=50%) -- 99.6% are one handler's multi-field emission. | None -- the back office is a case-handling queue; the ties are one handler writing several fields at one ms, which no re-export can turn into concurrency. | impossible |
| 6 | BPI Challenge 2017 (Dutch bank) | Y | **N** | — | Y | Y | **N** | · | · | ● | Exactly zero ties: 1,202,209 distinct timestamps for 1,202,267 events, so the R3b population is empty by measurement. | None -- the engine hands out one work item per case at a time; serialisation is in the process, not the export, so no finer clock creates equal stamps. | impossible |
| 7 | BPI Challenge 2019 (P2P, ~60 subs) | Y | Y | site | Y | **N** | · | · | · | ● | 95.6% of stamps are minute-rounded and 17.8% of events sit on four nightly batch minutes (21:59 x144,454) -- ties are rounding buckets. | Re-export from SAP keeping each change document's UTIME instead of the rounded posting minute -- but SAP's change clock stops at 1 s, so batches stay tied. | upstream |
| 8 | BPI Challenge 2011 (Dutch hospital) | Y | **N** | — | Y | **N** | **N** | · | Y | ● | Zero sub-hour timestamps anywhere in the log (57.9% hour + 42.1% date-only), so its 97.0% tie events are a billing-batch artifact. | Re-export from the hospital HIS with the original chart-entry clock rather than hour/date rounding -- whether that 2005-08 system held finer stamps is unknown. | upstream |
| 9 | BPI Challenge 2015 (5 municipalities) | Y | Y | site | Y | · | · | · | **N** | ● | 94.6-99.5% singleton traces -- the order distance saturates regardless of tie rate; the genuine 5-municipality fleet cannot rescue it. | The five authorities re-export from Squit at the system write clock (42.6-49.0% is date-only now) and at phase-level codes to cut variant count. | upstream |
| 10 | BPI Challenge 2018 (EU CAP subsidies) | Y | Y | site | Y | Y | **N** | · | · | ● | Composition fails: 33.4% of its millisecond tie blocks are a literal begin-X/finish-X pair of one action, not two resources acting at once. | Re-export from the paying agency with the DB commit timestamp at ms for all 2.5M events, not just the 9.5% machine-written stratum. | upstream |
| 11 | BPI Challenge 2020 (TU/e travel claims) | Y | **N** | site | Y | · | **N** | · | Y | ● | One organisation (TU Eindhoven); the 5 sub-logs are declaration TYPES, already refuted as a fleet surrogate at ARI 0-3, so no site axis exists. | Attach the claimant's faculty/department code to all ~40k declarations; only 2017's two-department slice is exposed, and two is below G>=3. | upstream |
| 12 | Sepsis Cases (one Dutch hospital) | Y | **N** | — | Y | · | **N** | · | Y | ● | Single Dutch hospital with no grouping attribute at all (fleet key = none, measured), so G>=3 site-like cohorts cannot be formed. | Impossible: this is one Dutch hospital's ED extract with no site field to expose; nothing the depositor can re-export turns a single ER into >=3 comparable sites. | impossible |
| 13 | Road Traffic Fine Management (Italy) | Y | **N** | — | Y | **N** | **N** | · | Y | ● | 100.00% date-only stamps: 4,905 distinct timestamps for 561,470 events, so timestamp equality degenerates to 'same day' and R3 is undefined. | Re-export the fine DB's row-insert transaction time at second/ms precision; the legal process clock is day-counted, so that field may not be retained. | upstream |
| 14 | CoSeLoG WABO (5 municipalities) | Y | Y | site | Y | · | **N** | · | **N** | ● | 93.2-99.7% singleton traces saturate the order distance, far above the 90% bar, despite a genuine 5-municipality fleet key. | Re-export at DB ms precision and pool cases per variant -- but the same authority's Receipt log at 100% sub-second has ZERO ties, so a finer export deletes them. | impossible |
| 19 | sOCEL 2.0 hinge production line | **N** | **N** | resource | Y | · | · | · | · | ○ | 'Built in CPN Tools with Climatiq API': a generated 3-workstation line in one facility, so it cannot beat the simulated incumbent. | None -- simulation is not fixable; it would need MES/PLC telemetry from the real hinge line, ideally the same line run at >=3 plants. | impossible |
| 20 | (Un)Fair fairness OCEL logs (4 processes) | **N** | **N** | vocab | Y | · | · | **N** | · | ○ | Synthetic by construction (generated fairness benchmark); the only grouping axis is a protected-attribute cohort built into the generator. | None -- being simulated is not fixable; it would need observed hiring/lending/admission logs of the same decisions, which the benchmark by design lacks. | impossible |
| 21 | Simulated inventory management OCEL | **N** | Y | site | Y | · | · | **N** | · | ○ | Simulated ERP log, and its plant axis is plant='Plant'+str(random.randint(1,5)) -- an i.i.d. draw, so R4 fails by construction too. | None -- simulated; needs an observed multi-plant SAP MM extract, and its plant key is an i.i.d. random draw, so the site-shaped axis groups nothing. | impossible |
| 22 | Bundestag OCEL (legislative procedures) | Y | **N** | vocab | Y | **N** | · | · | · | ● | Measured: 974 dates but 64 times-of-day, all in 00:00:00-00:01:03 and monotone in date (r=0.9993) -- the seconds field is a rank, not a clock. | The upstream DIP API records date-only; the Bundestag would have to stamp each procedure step to the second at source -- no re-export can add it. | upstream |
| 23 | Ethereum DApp execution OCEL (9.6 GB) | Y | **N** | vocab | Y | · | · | · | · | ○ | Block timestamps are integer seconds, identical for every event in a ~12 s block: sub-second is unrepresentable and every co-stamp is batching. | Per-call wall-clock stamps at node execution time instead of the block clock -- but the EVM runs calls strictly serially, so genuine ties still cannot occur. | impossible |
| 24 | Age of Empires 2 gameplay OCEL | Y | **N** | vocab | Y | · | · | · | · | ○ | Its only grouping axis (civilisation) is a vocabulary axis -- civs have different units/buildings -- so groups separate on frequency and R4 dies. | Re-group the 1,000 matches by server region with civilisation held fixed, giving one shared vocabulary -- and confirm tick stamps survive export sub-second. | upstream |
| 27 | BPI 2013 Volvo IT VINST (3 logs) | Y | Y | site | Y | **N** | **N** | · | Y | ● | Measured: zero sub-second stamps in 74,544 events and zero within-case ties at all -- R3 has nothing to read, despite the best fleet key. | Re-export from VINST with the DB transaction time at ms precision -- but VINST stamps one status change at a time, so even then no two events would coincide. | impossible |
| 28 | BPI 2016 UWV werk.nl clickstream | Y | **N** | vocab | Y | · | · | · | · | ○ | A web clickstream: one user, one browser, one click at a time, so within-case simultaneity is impossible and any co-stamp is a page-load pair. | Join the clicks to concurrent back-office handling of the same claim and expose the UWV office as a site key -- the click log alone can never carry ties. | upstream |
| 29 | BPI 2014 Rabobank ICT (HPSM) | Y | **N** | — | Y | · | · | · | · | ○ | An HPSM ticket queue hands out one work item at a time (as BPI 2013/2017), so R3b is empty; and it declares no site key at all. | Impossible: the HPSM extract is one Rabobank ICT organisation with no site column, and its incident/interaction tables log one assignment at a time, so no ties arise. | impossible |
| 30 | Hospital Billing (one regional hospital) | Y | **N** | — | Y | · | · | · | · | ○ | Case = one billing package in an ERP state machine: one work item at a time, so co-timestamps can only be batch or duplicate records (R3b). | Impossible: one regional hospital's billing ERP, no site column; its ties are a single clerk's batched package posting, which a finer commit clock would only split. | impossible |
| 32 | eICU Collaborative Research DB v2.0 | Y | Y | site | Y | **N** | **N** | · | · | ● | Every event time is an integer minute offset from unit admit, so timestamp equality ties are a rounding artifact; no sub-second component exists at all. | PhysioNet would have to re-derive eICU from Philips eCareManager retaining absolute sub-second device stamps -- precision stripped for de-identification. | upstream |
| 33 | MIMICEL (MIMIC-IV-ED, single hospital) | Y | **N** | — | Y | · | · | · | · | ○ | One tertiary ED (Beth Israel Deaconess); the schema declares only demographics and outcome, so no site-like axis with G>=3 cohorts exists. | Impossible: MIMICEL is derived from MIMIC-IV-ED, which is BIDMC alone, so no re-derivation or finer stamp can give it a second site, let alone a fleet of three. | impossible |
| 36 | LLM-generated OCEL 2.0 four-log set | **N** | **N** | vocab | · | · | · | **N** | · | ○ | LLM-generated (o1-preview via llm-ocel-simulator): fully synthetic, so no gain over the synthetic incumbent it would have to replace. | None -- synthetic is not fixable; and its only group axis is four different processes, so even a real version would be a vocabulary axis, not a site. | impossible |
| 38 | KYPO Cyber Range training logs | Y | **N** | resource | Y | · | · | · | · | ○ | Free-text Bash/Metasploit command strings (8,160 command events over 52 trainees) make traces effectively all-singleton, saturating the order distance. | Impossible: KYPO logs one trainee per isolated sandbox; its only keys are participant and time-limit condition, so no re-export of it yields a fleet of organisations. | impossible |
| 39 | Fraunhofer FIT multimedia ITAM OCEL | Y | **N** | — | Y | · | · | · | · | ○ | One lab room on one day: the only candidate grouping is 6 different processes, and 121 instances split 6 ways leaves no fleet at n_g>=3. | Re-run the identical ITAM protocol at >=3 sites or >=3 staff teams, >=3 instances each, with the site id in the OCEL; the sensor rig already gives +. | upstream |
| 40 | Claims Management + Digital Documents | Y | **N** | — | Y | · | · | · | · | ○ | Single-adjudicator sequential claim handling: no within-case co-timestamps can exist, so R3b is empty by construction. | Only the insurer can act: the published CSVs carry no office or handler column, so the claim-system audit trail would have to be re-exported with branch and actor ids. | upstream |
| 41 | IoT-Enriched Smart Factory XES (Zenodo) | Y | **N** | — | Y | · | · | · | · | ○ | Not rejected -- the one open candidate; blocked on R1: a single testbed with no declared site axis beyond product/program. | Publish a declared station/line/plant key (or sibling testbed installs) so R1 has a site axis -- its multi-station sub-second stamps are already the rare part. | upstream |
| 46 | OMOP-CDM to OCEL (MIMIC-IV) | Y | **N** | — | Y | · | · | · | · | ○ | MIMIC-IV charttime is minute-rounded -- eICU's permanent kill under another column name -- so every tie is a charting-batch artifact. | Impossible: the ETL's input is MIMIC-IV, one hospital (BIDMC), so no OMOP mapping choice creates a site fleet; running it on other OHDSI databases is a different source. | impossible |
| 47 | Scientific Publications OCEL | Y | Y | site | Y | · | · | · | · | ○ | Publication events carry only year/date stamps: no sub-day clock exists in scientometric metadata, so every tie is a date-only artifact. | Rebuild from a publisher's editorial-system audit trail (submission/review/decision stamped to the second) instead of date-level bibliographic metadata. | upstream |
| 49 | Public Jira Dataset (MSR 2022) | Y | Y | site | Y | · | · | · | · | ○ | Issue tracking serialises work (one assignee, one status); the only co-stamps are one transition writing several fields -- R3b artifact. | None -- Jira records one status transition per issue at a time, so cross-resource simultaneity cannot occur however finely the changelog is stamped. | impossible |
| 53 | DREAM multi-site EHR (Australia) | Y | Y | site | Y | · | · | · | · | ✕ | Not publicly released -- empty download URL, Australian health-system governance; no measurement can be run and no exhibit can be shipped. | Only the Australian health service holding DREAM can act: it would have to publish the site-labelled encounter events with their EHR write times under a releasable DUA. | upstream |
| 63 | PurpleLab HealthNexus / CLEAR Claims | Y | Y | site | Y | · | · | · | · | ○ | Claims carry a service/adjudication DATE, not an event clock, so timestamp equality equality is undefined exactly as on eICU -- despite a real payer/provider fleet. | Impossible: HealthNexus serves X12-derived claim lines dated to the service day; no export option adds adjudication clock stamps, and the licence bars raw release. | impossible |
| 68 | PMLRM-Bench LRM reasoning steps OCEL | **N** | **N** | vocab | Y | · | · | · | · | ○ | A case (MODQUE) is one autoregressive decode emitting steps strictly one at a time, so within-case co-timestamps are structurally impossible: R3b is empty. | None -- a serial decode has no concurrent resources to co-stamp, and model/question is a vocabulary axis, not a site axis. | impossible |
| 70 | Production Analysis (mfg, 225 cases) | Y | **N** | — | Y | · | · | · | · | ○ | 225 spaghetti cases with near-unique traces, and it is a start/complete INTERVAL log, which timestamp equality (timestamp equality) does not read. | Re-export the MES clock as per-machine instants instead of start/complete intervals, and add a plant key -- one 225-order factory offers neither. | upstream |
| 71 | 4TU teaching doc/e-invoicing logs | · | **N** | — | Y | · | · | · | · | ○ | Cross_site='no' with no organisation attribute, at teaching scale -- one clerk holds one document at a time, so there is no fleet axis. | Impossible: these are illustrative teaching extracts of one office with no organisation field; nothing in the deposit can be re-cut into >=3 comparable authorities. | impossible |
| 72 | opentender.eu / DIGIWHIST procurement | Y | Y | site | Y | · | · | · | · | ○ | The only clocks are publication and award DATES; a 3-stage date-level case has no within-case order structure to read. | TED/DIGIWHIST would have to publish the notice-submission transaction timestamps behind the award dates, plus more than three lifecycle events per tender. | upstream |
| 73 | EHDEN / OHDSI federated OMOP network | Y | Y | site | Y | · | · | · | · | ✕ | No obtainable log -- federated analytics only, data never leaves each partner's firewall, so nothing can ever be measured. | A partner study agreement releasing OMOP *_datetime at sub-second from >=3 EHDEN sites; governance is the block, but date-level visit stamps likely follow. | upstream |
| 74 | AGESIC Uruguay immigration (ICPM'25) | Y | **N** | site | Y | · | · | · | · | ○ | An online-procedures traceability engine issues one work item per application at a time, so cross-resource co-timestamps cannot arise. | AGESIC releasing the log unrestricted with the office key AND back-office system events, not just the one-work-item-at-a-time front-end trace. | upstream |
| 75 | KNS Automotive Work Center (ERP) | Y | **N** | resource | Y | · | · | · | · | ○ | Its only categorical besides case and timestamps is Work Center -- the activity label itself; no plant/site key (and 2.04 events/case). | A HarmonyERP re-export carrying the plant field plus per-operation instant stamps rather than Start/Finish intervals -- KNS is one manufacturer, so no plant key may exist. | upstream |
| 76 | Noise-robustness benchmark logs | **N** | **N** | — | Y | · | · | **N** | · | ○ | The group axis is INJECTED noise (0.5-2.0%, 10 iterations x 5 types incl. 'Ordering'), so any order-vs-frequency signal is planted. | None -- injected-noise cohorts cannot become observed ones; the clean base logs are the standard real XES set already screened and rejected. | impossible |
| 77 | Synthetic concept-drift logs | **N** | **N** | — | Y | · | · | · | · | ○ | Self-declared 'synthetically generated ... with injected concept drifts'; simulated cannot beat the already-simulated Synthea incumbent. | None -- being simulated is not a fixable defect; it would have to be replaced by an observed log of the same process across real sites. | impossible |
| 79 | processminingdata.com library | **N** | **N** | vocab | Y | · | · | · | · | ○ | The vendor's own line reads 'realistic simulated', and each log is one synthetic organisation with no site key. | None -- simulated by the vendor's own description; the commercial licence would also block redistribution even if it were observed. | impossible |
| 80 | Kaggle car-insurance claims log | **N** | **N** | — | Y | · | · | · | · | ○ | Registry records the log as artificially generated/curated -- a synthetic log cannot beat the already-simulated incumbent fleet. | None -- being artificially generated is not fixable; it would take an observed multi-branch insurer claims log, and claims handling serialises one work item at a time anyway. | impossible |
| 81 | Kaggle ITSM incident-management log | · | **N** | resource | Y | · | · | · | · | ○ | ITSM ticketing serialises: one assignment group holds a ticket at a time and every stamp is a human-entered state change, so R3b is empty by construction. | Obtain the underlying ITSM audit table with the DB commit clock and per-actor rows, so parallel group/CI/SLA updates on one ticket stay distinct actors at one instant. | upstream |
| 87 | coPM German state parliaments | Y | Y | site | Y | **N** | · | · | Y | ● | 100.0% date-only timestamps in all 520,679 events -- timestamp equality makes every same-day event in a case concurrent; the fleet axis is fine, the clock is not. | Landtag document registries would need to expose the entry time-of-day, not just the sitting date; the 2006-2020 records were never stamped finer, so only future sittings qualify. | upstream |
| 90 | PDC 2025 discovery-contest logs | **N** | **N** | — | Y | · | · | · | · | ○ | Artificial by construction: 288+96+96+96 XES logs generated from PNML workflow nets, so it cannot beat an already-simulated Synthea fleet. | None -- logs generated from PNML workflow nets cannot be de-simulated; only real logs of the processes those nets were fitted to would help, and each log also has no site attribute. | impossible |
| 93 | CMS DE-SynPUF synthetic Medicare claims | **N** | Y | site | Y | · | · | · | · | ○ | Fully synthetic per its own DUA -- every state is drawn from one CMS generator, so the 54-state fleet variation is injected, not observed. | None as published; and the real CMS claims it imitates carry only CLM_FROM_DT/CLM_THRU_DT claim DATES, so even de-synthesising it leaves timestamp equality undefined. | impossible |
| 94 | OHDSI Eunomia OMOP sample datasets | **N** | Y | site | Y | · | · | · | · | ○ | The shipped sets are Synthea-derived plus CMS SynPUF -- i.e. the incumbent generator itself; the registry itself says multi-site variation must be generated. | None -- swapping Eunomia's bundles for a real OMOP instance is a different source, and OMOP's day-granular visit/condition dates would still fail timestamp equality there. | impossible |
| 95 | JIRA Social Repository (4 OSS orgs) | Y | Y | site | Y | · | · | · | · | ○ | Not killed by a measurement: R3 is undecidable from metadata, and the audit files it -- Jira co-stamps are one transaction writing several fields, not simultaneity. | Re-fetch ISSUE_CHANGELOG from the live Apache/Spring Jira REST API, which returns ms-precise 'created' -- the social-repo dump keeps only the coarser export dates. | plausible |
| 97 | GH Archive GitHub event timeline | Y | Y | site | Y | **N** | **N** | · | Y | ● | 1 s is the schema ceiling: zero sub-second stamps in 272,954 events, so its 18.3% tie cases can never be told from batching -- R3b has no population to score. | GitHub would have to emit created_at at sub-second precision in the public events timeline; the 1 s truncation is applied at the API source, so no re-download recovers it. | upstream |
| 98 | TravisTorrent CI builds | Y | Y | site | **N** | · | · | · | · | ○ | Schema-fatal: one row and one timestamp per build (gh_build_started_at), so no within-case event sequence exists to order across its 1,359 projects. | Re-mine per-job start/finish stamps from the Travis job API -- but the canonical host is defunct and a build's jobs form one antichain, so order variation stays zero. | upstream |
| 99 | DataCo Smart Supply Chain (Kaggle) | Y | **N** | — | **N** | · | · | · | · | ○ | No event rows exist: only order date and shipping date, with transit as integer 'days for shipping', so a 2-3 point trace must be invented at day granularity. | The source ERP would have to export order-line status transitions (picked/packed/shipped/delivered) with clock timestamps; this extract has two date columns and one company. | upstream |
| 100 | HCUP SID/SEDD/SASD + NIS/NEDS (AHRQ) | Y | Y | site | **N** | · | · | · | · | ○ | Cross-sectional encounter abstracts stamped by date or QUARTER with no within-encounter event sequence -- no per-case poset exists at any resolution. | Impossible: HCUP repackages billing discharge abstracts -- one row per encounter, no intra-stay event clock -- so no AHRQ release adds times the hospitals never submit. | impossible |
| 101 | FOPPA French procurement awards | Y | Y | site | Y | · | · | · | · | ○ | Award notices carry lot DATES only -- the same date-level granularity that convicted Road Traffic Fine (100% date-only) and coPM (520,679 events). | Release the underlying procurement-platform workflow log (notice drafted/validated/published events) with clock timestamps, not just the legal award date. | upstream |

---

## 3. Duplicate registry entries — 21 rows, no independent evidence

Same dataset re-entered under a second name, DOI or mirror. Each inherits its canonical row's verdict; listed so a referee checking the registry can see nothing was quietly dropped.

| # | source | same dataset as | verdict inherited |
|---|---|---|---|
| 15 | Container Logistics OCEL 2.0 | row 2 | Simulated showcase log of a single operation -- the same category of evidence as the Synthea fleet it would re |
| 16 | OCEL 2.0 Order Management (v2) | row see group | Generated in CPN-Tools, not observed -- a simulated log cannot displace the simulated incumbent fleet; a |
| 17 | OCEL 2.0 Order Management (v3, 2026) | row see group | Generated in CPN-Tools, not observed -- a simulated log cannot displace the simulated incumbent fleet; a |
| 18 | Angular GitHub commits OCEL 2.0 | row 3 | One repository, no site axis: only directory-prefix (707) or release-era slices; and 88.9% of its tie blocks a |
| 26 | CoSeLoG WABO (5 municipalities) | row 14 | 93.2-99.7% singleton traces saturate the order distance, far above the 90% bar, despite a genuine 5-municipali |
| 31 | BPI 2013 Volvo IT VINST (3 logs) | row 27 | Measured: zero sub-second stamps in 74,544 events and zero within-case ties at all -- R3 has nothing to rea |
| 34 | Simulated Inventory Mgmt OCEL (Berti) | row 21 | Simulated ERP log, and its plant axis is plant='Plant'+str(random.randint(1,5)) -- an i.i.d. draw, so R4 fa |
| 37 | Fairness benchmark logs (CPN Tools) | row 20 | Synthetic by construction (generated fairness benchmark); the only grouping axis is a protected-attribute coho |
| 45 | Enron Email OCEL 2.0 | row 4 | Single organisation grouped by individual mailbox -- no site axis (registry cross_site 'no'); 69.5% of stamps  |
| 48 | OCEL 2.0 new simulated logs (4) | row 19 | 'Built in CPN Tools with Climatiq API': a generated 3-workstation line in one facility, so it cannot beat the  |
| 50 | Tawosi Agile OSS projects (Jira, MSR'22) | row 49 | Issue tracking serialises work (one assignee, one status); the only co-stamps are one transition writing sever |
| 51 | Container Logistics OCEL 2.0 (simulated) | row 2 | Simulated showcase log of a single operation -- the same category of evidence as the Synthea fleet it would re |
| 52 | Enron Email OCEL 2.0 | row 4 | Single organisation grouped by individual mailbox -- no site axis (registry cross_site 'no'); 69.5% of stamps  |
| 60 | llm-ocel-simulator (LLM-emitted OCEL 2.0) | row 36 | LLM-generated (o1-preview via llm-ocel-simulator): fully synthetic, so no gain over the synthetic incumbent it |
| 65 | Synthea / SyntheticMass (the incumbent) | row 54 | This is the incumbent: simulated, with concurrency planted as timestamp equality (737/737 co-stamped), so it c |
| 66 | PLG2 (Processes Logs Generator) | row 55 | A generator, not a log: random stochastic-grammar models played out to XES, and its sampled durations produce  |
| 67 | OCEL 2.0 Container Logistics v3 | row 2 | Simulated showcase log of a single operation -- the same category of evidence as the Synthea fleet it would re |
| 69 | pm4py bundled OCEL 2.0 example_log | row 25 | Hand-written OCEL format-conformance test fixture (a handful of events/objects): synthetic, and too small to c |
| 88 | PMLRM-Bench LRM reasoning OCEL | row 68 | A case (MODQUE) is one autoregressive decode emitting steps strictly one at a time, so within-case co-timestam |
| 96 | OpenTender / EU TED procurement (OCDS) | row 72 | The only clocks are publication and award DATES; a 3-stage date-level case has no within-case order structure  |
| 112 | Berti inventory-management sim (OCEL 2.0) | row 21 | Simulated ERP log, and its plant axis is plant='Plant'+str(random.randint(1,5)) -- an i.i.d. draw, so R4 fa |

---

## 4. Not a dataset — 40 rows

Log *generators*, catalogue *indexes* and software *libraries*. None can be an exhibit: a generator emits simulated logs, an index hosts none of its own, a library contains no events. Kept in the registry because they were named as candidates and a referee is entitled to see why each was set aside.

| # | source | what it is | why it cannot be an exhibit |
|---|---|---|---|
| 25 | pm4py ocel20_example fixture | library | Not a dataset -- it is a library test fixture; no change to it yields a real multi-site log. |
| 35 | Collection of 98 Event Logs (index) | index | Not a dataset -- an index. Useful only as a filter via its '#resources' and real/synthetic metadata columns; no change makes the bundle an exhibit. |
| 42 | Neo4j re-encoding of 5 BPI logs | index | None at this layer -- a graph re-encoding adds no stamps, and BPIC-2015 permit cases are near-bespoke, so no municipal re-export can create repeated variants. |
| 43 | ERamaM/ProcessMiningDatasets (index) | index | Not a dataset -- a conversion/index repo over logs already inside our 28-file screen; no change to it could add a log we have not seen. |
| 44 | TheWoops/awesome-processmining (list) | index | Not a dataset -- a link list whose targets are already registry rows. |
| 54 | Synthea generator (the incumbent) | generator | None -- simulation is not a fixable defect, and kind=generator means it emits logs rather than being one; only an observed multi-hospital log could replace it. |
| 55 | PLG2 log generator | generator | Not a dataset -- a generator; it stamps by sampled durations so it has no ties, and site identity is user-assigned rather than read from an event field. |
| 56 | Scylla BPMN simulator | generator | Not a dataset -- a BPMN discrete-event simulator; any concurrency it emitted would be planted by the modeller, the tautology that already sank the incumbent. |
| 57 | Simod (discover + Prosimos replay) | generator | Not a dataset -- a discover-and-replay pipeline whose output is synthetic however real the input; the input logs are the thing to audit, not Simod. |
| 58 | Prosimos BPS simulator | generator | Not a dataset -- a BPMN+JSON simulator; the 'site' is the config you chose to run, and sampled durations leave essentially no co-timestamps. |
| 59 | AgentSimulator (agent-based BPS) | generator | Not a dataset -- a generator learned from a single log, with no site axis at all; only the real input log it was trained on could ever be an exhibit. |
| 61 | CPN Tools (CPN simulator -> OCEL 2.0) | generator | Not a dataset -- and a site colour set only designs the fleet in, reproducing exactly the incumbent's weakness rather than replacing it. |
| 62 | SimPN (Python CPN simulation library) | generator | Not a dataset -- a play-out library can only emit simulated logs; no configuration of it produces an observed multi-site process. |
| 64 | ocpa (object-centric analysis library) | library | Not a dataset -- ocpa consumes OCEL, it does not contain any; no re-export of a library can produce events. |
| 78 | GEDI log generator | generator | Not a dataset -- a generator cannot supply observed data; keep only as a synthetic positive control, never as an exhibit. |
| 82 | RWTH PADS event-log index | index | Not a dataset -- an index cannot be fixed; it would take a new sub-second multi-site log being deposited in the 4TU collections it redirects to. |
| 83 | IEEE DataPort process-mining index | index | Not a dataset -- IEEE DataPort would have to host a new instrumented multi-site log; its present process-mining inventory (rows 75-77) is exhausted. |
| 84 | GitHub computertechworld PM list | index | Not a dataset -- a tools bibliography links no logs at all, so no change to the list itself could ever yield one. |
| 85 | GitHub lisenkovkv PM knowledge base | index | Not a dataset -- a resource list hosts nothing; only if it began hosting an original multi-site log would it enter the screen at all. |
| 86 | Loghub system-log collection | index | Re-collect HDFS/OpenStack with microsecond syslog stamps across datanodes, and from many clusters of ONE system -- cross-system groups are disjoint vocabularies. |
| 89 | FrOG OCEL generator | generator | None -- being a generator is not fixable; it would take an observed multi-warehouse fulfilment log, and FrOG has no built-in site parameter to encode one. |
| 91 | AVOCADO streaming challenge (generator) | generator | Not a dataset -- a framework that synthesises streams on demand; no re-export of it is a log, so it would have to be replaced by the observed stream it stands in for. |
| 92 | IOTEL IoT-OCEL generator | generator | None -- the concurrency it makes attractive is manufactured; it would have to be the real MES/OPC-UA export of a multi-plant factory. Keep only as a positive-control rig. |
| 102 | PySynthea (Synthea reimplementation) | generator | None -- it generates the incumbent's own logs; only an observed multi-provider EHR log of the same care pathway could substitute for it. |
| 103 | AMLGentex AML generator (AI Sweden) | generator | None -- simulated output is not fixable; a real multi-bank transaction log would be needed. Keep it only as a cross-institution positive control. |
| 104 | AMLSim (IBM AML simulator) | generator | None -- a simulator; its bank split is a config parameter, so even adding banks yields a configured, not an observed, site axis. |
| 105 | PaySim mobile-money simulator | generator | None -- being simulated is not fixable; only release of the operator's raw timestamped transaction log, which PaySim saw only in aggregate, would serve. |
| 106 | BankSim card-payment simulator | generator | None -- simulated; only the Spanish bank's raw card-authorisation log, the aggregate BankSim was fitted to, would serve, and it is unreleased. |
| 107 | RetSim retail-store simulator | generator | None -- simulated and effectively unobtainable; the retailer's real POS log with per-till sub-second stamps would be a different source, not a fix. |
| 108 | CARD-AI CDR-Generator | generator | None -- it emits simulated records; even publishing the real CDR set behind it would need per-call leg/switch events before any within-case order exists. |
| 109 | cdr-data-generator (deshpandetanmay) | generator | None -- a random-record utility, not a dataset; nothing was ever observed, so there is no export that could be re-cut at finer precision. |
| 110 | anyLogistix supply-chain digital twin | generator | None -- a digital twin's output is simulated by definition; it would have to be replaced by the modelled client's real WMS/ERP shipment log with facility IDs and ms stamps. |
| 111 | ISOMORPH logistics digital twin | generator | None -- synthetic by charter, and it emits per-node demand/inventory time series rather than cases; only a real multi-echelon carrier's shipment log would substitute. |
| 113 | OCEL 2.0 CPN generator models | generator | None -- these ARE the generator; only an observed order-to-cash or container log from the modelled firm helps. Adding a site colour set would just manufacture a fleet. |
| 114 | PySPN SPN log generator | generator | Not a dataset -- a simulation library; usable only as a positive-control rig, and the repo already has one (BPIC 2015), so building another adds nothing. |
| 115 | Smart-factory synthetic sensor-log gen | generator | Not a dataset, but it names the target: a real multi-plant MES/sensor log with per-device ms stamps on one work order would clear the bar this tool only mimics. |
| 116 | OpenSynth synthetic smart-meter data | generator | None -- synthetic by charter; and even the real feeds behind it (RTE7000 at 5-min) carry no activity vocabulary or case notion to build a poset over. |
| 117 | GridDS (LLNL) synthetic grid toolkit | generator | Not a dataset -- a toolkit whose relevant feature is synthetic generation; meter/feeder identity is a sensor, not a process site, so no fleet axis exists either. |
| 118 | GENA Petri-net/BPMN log generator | generator | Not a dataset -- a log generator (development on hold); same class as the already-rejected ed_multiorg generator, and no configuration makes generated events observed. |
| 119 | BIMP/QBP BPMN simulator | generator | Not a dataset -- a BPMN simulator; its resource pools also hand out one task per case at a time, reproducing the serialisation that killed BPI 2017 (zero ties). |
---

## 5. What the table shows

**Where the 59 distinct logs die — earliest failing requirement:**

| first requirement to fail | logs | what this says |
|---|---|---|
| **R1** no site-like fleet | 26 | the commonest defect: a single organisation, or a grouping key that is a vocabulary or a resource cohort |
| **R0** simulated | 15 | cannot beat an already-simulated incumbent |
| **R3a** granularity ceiling | 5 | date-only or minute-rounded; permanent, since precision is a schema property |
| **R3b** tie composition | 2 | **BPI 2012 and BPIC 2018** — ties exist and are an emission artifact |
| **R2** no case notion | 2 | |
| **R5** variant structure | 1 | BPIC 2015, 94.6–99.5% singleton |
| rejected on documented schema fact or access, no flag asserted | 8 | date-level procurement/claims registries (47, 63, 72, 101), Jira serialisation (49, 95), and the two unmeasurable ones (53, 73) |

**Nothing reaches R4.** Requirement 4 — order structure varying independently of activity frequency
— is the claim this screen exists to test, and **no real log in the registry survives long enough to test it.**
It remains validated on zero real datasets. That is not an oversight in the screen; it is what the
screen found.

**Feasibility of repair, across the 59 logs:** 32 `impossible`, 26 `upstream-only`, **1
`plausible`**. The single plausible repair is row 95 (JIRA Social — re-fetch the changelog from the
live REST API, which returns millisecond `created`), and it is expected to fail R3b anyway on the
same serialisation argument that killed rows 49/50.

**The two most instructive rows are the two that nearly worked:**

- **BPI 2012 (row 5)** — passes R0, R2, R3a, R5. Its ties are real, abundant (11,361 sub-second
  blocks) and **100% distinct-activity**. It fails only R3b, at **0.0% cross-resource**: 99.6% of
  those blocks are a *single handler* writing several activities at one millisecond. Precision is
  necessary and nowhere near sufficient.
- **BPI 2013 (row 27)** — passes R1 with the best multi-country fleet key in the registry
  (`organization country` G=23, 20 at n_g≥3) and R5 with the best singleton rates in the corpus
  (8.1–16.1%). It has **zero within-case ties at any granularity** in 74,544 events. The fleet axis
  was never the scarce resource; observed simultaneity is.

## 6. The structural finding this table is evidence for

Public event data is instrumented in one of two ways, and both destroy the relation the method
reads:

1. **Workflow engines hand out one work item at a time per case.** They record *serialisation*, not
   concurrency. BPI 2017 is the extremal case: 1,202,209 distinct timestamps for 1,202,267 events —
   ties are not rare, they are *absent*.
2. **Exports are coarser than the relation.** Road Traffic Fine is 100.00% date-only; coPM's 520,679
   events are 100.0% date-only; eICU is integer minutes.

The decisive control sits inside a single authority: **CoSeLoG's Receipt-phase log, exported at
100.00% sub-second, has ZERO ties, while its WABO siblings at ~29% date-only carry 46–63% tie
events.** Same organisation, same process, opposite tie rates — so **ties in public logs are
overwhelmingly an export property, not a process property.**

**What class of source would clear the bar.** Three properties must co-occur; every near-miss above
fails exactly one:

1. machine-generated stamps at sub-second resolution (human workflow stamps are either rounded or so
   fine that ties never occur);
2. physically parallel resources on one case — several machines, sensors or clinicians acting at
   once, *not* a pipelined line, a case-handling queue or an autoregressive chain;
3. a fleet axis that is a **site**, not a vocabulary.

The realistic sources are multi-plant MES/shop-floor logs, multi-hospital **device telemetry** (ICU
monitors, infusion pumps — not charting systems, which are minute-rounded), and multi-terminal port
handling. **Every one is behind an industrial NDA or health-system governance** — which is precisely
why the four best fleet keys in this registry (DREAM 53, PurpleLab 63, EHDEN 73, HCUP 100) are rated
`upstream`, not `impossible`. They are not technically deficient. They are withheld.

## 7. How this table was built, and its known limits

Twelve independent agents scored ten sources each against the requirement set, followed by an
adversarial audit pass whose brief was to find what a hostile referee would attack. That audit
returned **"not yet fit to put in front of a referee"** and named six defect classes; all six were
corrected before this table was written:

1. **one invented positive measurement** (row 49 asserted sub-second stamps in a dataset nobody
   opened) — demoted to `·`;
2. **every unmeasured hard flag** on R3a/R3b/R4/R5 demoted to `·`, reserving Y/N for the 20 logs
   actually measured — this is why the table has so many `·` and why that is a feature;
3. **six sources carrying mutually contradictory rows** reconciled to one row each, with the
   duplicates listed in §3;
4. **R1 applied inconsistently across resource-like axes** — fixed by the site/vocab/resource
   standard in §1 and re-checked across all rows;
5. **row 10's invented "≥5% bar"** removed (no such threshold exists in the requirement set; the
   kill now rests on the measured 33.4% begin/finish figure);
6. **feasibility relabelled** where a stated fix conceded it would not clear the bar (`upstream` →
   `impossible`) and where a fix was not in our hands (`plausible` → `upstream`).

**Known limits, stated so they are not discovered.** 39 of the 59 logs were rejected without being
downloaded — on published schema documentation, not measurement. For a granularity or fleet claim
that is usually decisive (precision and declared attributes are documented properties), but it is
weaker evidence than the 20 measured rows and is marked `○` throughout. Requirement 4 is untested
everywhere. And the `minimum_change` column is a judgement about upstream systems we do not
administer; it is our best reading of what would have to change, not a commitment that it would
then pass.
