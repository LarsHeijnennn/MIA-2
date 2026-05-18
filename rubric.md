# Assessment rubric

The following rubric will be used to grade the project work and report on point-based and intensity-based medical image registration using the provided (MRBrainS) dataset.

## Grading scale

Note on grading thresholds: A correctly completed minimal solution (the guided project work, the programming tasks, and the theory questions) accompanied by a suitable report is graded at most as **sufficient** (5.5 – 7.5). Higher grades (**8+**, excellent) require going beyond the minimal solution by formulating and investigating a well-motivated research question using the implemented methods and available data. Work that does not meet the minimal solution falls in the **insufficient** range (< 5.5).

## Project work
Component  | Insufficient | Satisfactory | Excellent
--- | --- | --- | ---
**Guided project work (minimal solution)** | Programming tasks and theory questions incomplete, incorrect, or missing; notebook does not run end-to-end | All programming tasks and theory questions completed correctly; notebook runs end-to-end and reproduces the minimal results | Minimal solution completed with additional implementation choices justified and explored (e.g., extra study based on the similarity metrics, optimizers, or transformation models)
**Design of research question and hypothesis** | No research question beyond the minimal working example, or research question is poorly motivated or methodologically unsound | Clear research question that goes beyond the minimal example, with an justified hypothesis and well-motivated, correct methodological choices | Novel and insightful research question with a testable hypothesis that probes the method or the problem
**Study design** | Components of the study design are missing or arbitrary; data usage, method requirements, or design choices are unjustified | Study design components (data, method requirements, choices) are present and motivated; appropriate use of the MRBrainS data (3 patients × 3 slices × T1/T2-FLAIR) | Complete, rigorous study design with justification of choices and evaluation strategy
**Experiments** | Do not go much further than the minimal working example; experiments lack structure | Goes beyond the minimal working example; experiments are somewhat structured and systematic, with appropriate evaluation metrics | A set of experiments that demonstrate a systematic approach to the research question, including ablations or sensitivity analyses where appropriate
**Code (notebook + code)** | Missing or incomplete; runs with errors; lacks documentation; is not self-contained or cannot reproduce reported results | Self-contained zip archive, notebook included, runs without errors, contains some documentation, and reproduces the results in the report | User-friendly, well-structured (clear separation of general functionality from experiments), detailed documentation, optimized for speed; easy for a reader to extend

## Report

The report must be a single PDF. The main body (parts 1–4 - Introduction, Study design, Method, experiments and results, and Discussion) has a maximum length of five pages combined; additional information may go in an appendix. The lecturers already know the background, the data, and the general methodology, so the report should not introduce the clinical problem or explain the background of image registration.

Component  | Insufficient | Satisfactory | Excellent
--- | --- | --- | ---
**General** | Poor structure, unclear line of thought, inconsistent referencing, severe grammar mistakes, exceeds the five-page limit for items 1–4 | Legible structure following the suggested section order, mostly clear line of thought, consistent referencing, within the page limit, spell-checkers do not detect errors | Excellent structure, easy to follow line of thought, concise and within the page limit while still complete
**Research question and motivation (≈0.5 page)** | Research question is absent, vague, or unmotivated; no hypothesis stated | Research question and hypothesis are clearly stated and motivated; scope is appropriate for the project | Well-scoped research question with a motivation and a clearly testable hypothesis that demonstrates insight into the methods or problem
**Study design (≈0.5 page)** | Missing or incomplete description of data used, method requirements, and design choices | Describes data used, method requirements, and the motivation behind key design choices; components of a study design are largely covered | Complete and well-justified study design; explicitly addresses evaluation, controls, and limitations of the chosen design
**Methods, experiments and results (≈2 pages)** | Methods not described in own words or convey little understanding; poor choice of experiments; results poorly presented | Methods described in own words and convey good understanding; appropriate experiments tied to the research question; clear and concise presentation of results | Methods convey excellent understanding and make connections between methods; extensive, well-motivated set of experiments; excellent presentation of results
**Discussion (≈1 page)** | Repetitive, analysis is missing, no discussion of strengths and weaknesses, no link back to the research question | Concise analysis tied to the research question, discussion of strengths and weaknesses, sensible conclusions | In-depth and critical analysis, clear summary of insights gained, discussion of limitations, and suggestions for further analysis
**Tables and figures** | Inconsistent formatting, not referenced in text, present irrelevant or incorrect information, poor captioning | Consistent formatting, correctly referenced in text, present all relevant information, captions contain all necessary information | Excellent formatting, self-contained (can be easily understood on their own), each figure/table is nec\cessary
**Contributions** | Missing, or each group member's role is unclear | Brief description by each group member of their activities, sufficient to support individual grade adjustment | Clear and specific contribution statements that allow fair individual assessment
**LLM usage declaration** | Missing, despite LLM use being evident, or no reflection on the use | If LLMs (e.g., ChatGPT) were used, their use is declared with what they were used for and a brief reflection | Reflection on how LLM use affected the work, what was verified, and the limitations encountered

## Deliverables checklist
- Single PDF report (main body items 1–4 within five pages; appendix optional).
- Single archive file (zip or 7z) containing the completed notebook with the guided project work and theory answers, and the documented Python code, self-contained and able to reproduce the reported results.
- Contributions section and (if applicable) LLM usage declaration included in the report.


