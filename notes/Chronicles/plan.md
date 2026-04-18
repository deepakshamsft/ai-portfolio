# The Pitch Chronicles — Storyboard Plan

> A three-book manga arc that wraps the entire learning path: **Pre-Requisites → ML → AI**. The running thread is one character's journey from apprentice striker to master scout to club manager. Each technical chapter gets **one manga chapter** — a cover/splash plus a short storyboard that dramatises the concept. The body of every technical chapter under `notes/PreRequisites/`, `notes/ML/`, and `notes/AI/` stays exactly as it is; the manga is a parallel layer that sits alongside, generated via Perchance (free AI-image tool) and linked from each chapter README.

---

## 1 · The Protagonist and the Arc

| Book | Title | Role | What it dramatises |
|---|---|---|---|
| Book I | **The Apprentice** | a teenage striker | Physics + math behind the perfect knuckleball free kick (Pre-Requisites Ch.1–7) |
| Book II | **The Scout** | a retired player turned talent scout | Reading players through FIFA-style attribute ratings; every ML algorithm as a scouting tool (ML Ch.1–19) |
| Book III | **The Manager** | a head coach of a rebuilding club | Building an AI war-room of agent-scouts, analysts, and LLM pundits (AI track) |

One protagonist, three life stages. Physics → statistics → intelligence. The same man who once read the flight of a ball now reads a dressing room.

---

## 2 · Recurring Cast

| Character | Role | Visual anchor |
|---|---|---|
| **The Protagonist** | One man, three life stages | Dark hair, left-footed, always in a fitted technical jacket (colour shifts with book: white → navy → charcoal) |
| **The Gaffer** | The protagonist's mentor — old coach, later director of football | Silver hair, tracksuit, always holding a notepad |
| **The Analyst** | Data/AI specialist introduced in Book II | Glasses, hoodie, laptop, permanent headphones |
| **The Oracle** | A veteran scout who speaks only in probabilities | Long grey coat, weathered face, appears from Book II Ch.1 onwards |
| **The Rival Manager** | Antagonist through Book III | Sharp suit, sunglasses indoors, runs a rival club with deeper pockets |

---

## 3 · Visual Style (applies to every prompt)

Young-adult shōnen × seinen, cinematic and grounded. Think **Blue Lock** (energy, tactics), **Slam Dunk / Real** by Inoue (grounded athlete portraits), **Haikyuu!!** final-arc colour palettes, and **Captain Tsubasa** for iconic freeze-frames. Full colour, clean confident linework, warm cinematic lighting, light cel shading, no chibi, no heavy horror blacks. Aspect ratio **2:3 portrait** for covers and single pages.

---

## 4 · Book I — The Apprentice (Pre-Requisites)

**Setting.** A sun-baked training ground at a small football academy. Our protagonist is 16, obsessed with mastering the direct knuckleball free kick — a no-spin strike that moves on pure gravity. The Gaffer is teaching him the physics and math beneath the shot.

| # | Folder | Manga chapter title | Concept dramatised | Key visual |
|---|---|---|---|---|
| 1 | [ch01-linear-algebra](../PreRequisites/ch01-linear-algebra/README.md) | **The First 0.1 Seconds** | $y = wx + b$, weights & biases | Freeze-frame of the ball leaving the boot, straight glowing line trailing it, equation floating above |
| 2 | [ch02-nonlinear-algebra](../PreRequisites/ch02-nonlinear-algebra/README.md) | **Gravity's Arc** | Parabolas + feature expansion | Full trajectory over the wall, curve bending under gravity, parabola overlaid |
| 3 | [ch03-calculus-intro](../PreRequisites/ch03-calculus-intro/README.md) | **Slicing Time** | Derivatives (tangent) and integrals (area) | Secant-to-tangent animation on the ball's flight; Riemann rectangles under the velocity curve |
| 4 | [ch04-small-steps](../PreRequisites/ch04-small-steps/README.md) | **The Angle Game** | Gradient descent / ascent | The striker trying angle after angle on the training ground; a ghostly arrow on the turf points "downhill" |
| 5 | [ch05-matrices](../PreRequisites/ch05-matrices/README.md) | **The Notebook** | Matrices, design matrix, normal equations | The Gaffer opens a leather notebook of 500 recorded kicks; rows fly off the page into a matrix |
| 6 | [ch06-gradient-chain-rule](../PreRequisites/ch06-gradient-chain-rule/README.md) | **Eight Knobs** | Gradient, Jacobian, chain rule | The striker at a mixing console with 8 dials (speed, angle, wind, wall height…); the console glows with a gradient field |
| 7 | [ch07-probability-statistics](../PreRequisites/ch07-probability-statistics/README.md) | **The Missed Kick** | Distributions, MLE, noise models | He misses one; a Gaussian bell-curve ghosts around where the ball *could* have landed |

**Book I cover concept.** Low-angle hero shot of the striker over the ball at dusk, a defensive wall silhouetted in the distance, the crossbar framing his head like a halo. Title: *The Apprentice*. Subtitle: *The Knuckleball Chronicles*.

---

## 5 · Book II — The Scout (ML)

**Setting.** Ten years later. His knee gave out; he now scouts for a mid-table club with FFP problems. He works out of a cramped office plastered with printed **attribute cards** — FIFA-style player ratings: PAC / SHO / PAS / DRI / DEF / PHY plus 30 sub-attributes. Every scouting assignment is a chapter. The Analyst (young, hoodie, laptop) joins him around Ch.4.

**Running metaphor.** Every ML algorithm is a *type of scout*. Some are cheap and fast (KNN — "just find the nearest comparable"). Some are deep and slow (deep nets — "a committee of seers"). Some cheat (overfit scouts who promise the world on one season's data).

| # | Folder | Manga chapter title | Concept dramatised | Key visual |
|---|---|---|---|---|
| 1 | [ch01-linear-regression](../ML/ch01-linear-regression/README.md) | **Market Value** | Linear regression, OLS | The scout fits a line through 50 dots: age-vs-market-value scatter; a glowing trend line cuts through |
| 2 | [ch02-logistic-regression](../ML/ch02-logistic-regression/README.md) | **The Transfer Deal** | Logistic regression, sigmoid | A sigmoid S-curve overlays a decision: *sign* vs *pass* on a rising-talent graph |
| 3 | [ch03-xor-problem](../ML/ch03-xor-problem/README.md) | **The Unscoutable Talent** | XOR — why a line isn't enough | Two scouts (Linear & Logistic) fail to classify a player who only shines in *both* pace **and** vision |
| 4 | [ch04-neural-networks](../ML/ch04-neural-networks/README.md) | **The Scout Committee** | MLP — layered non-linear judgement | A round table of 8 scouts, each whispering into the next layer, final verdict emerges |
| 5 | [ch05-backprop-optimisers](../ML/ch05-backprop-optimisers/README.md) | **The Feedback Loop** | Backprop, Adam, SGD | A signing fails; The scout traces the mistake back through the committee, each scout "learning" |
| 6 | [ch06-regularisation](../ML/ch06-regularisation/README.md) | **The Discipline** | L1/L2, dropout | The Analyst deletes 30% of the scout's notes before every meeting to force humility |
| 7 | [ch07-cnns](../ML/ch07-cnns/README.md) | **Heatmap Analysis** | CNNs, convolution, pooling | A pitch heatmap divided into tiles; a magnifying glass slides across, each tile lighting up local patterns |
| 8 | [ch08-rnns-lstms](../ML/ch08-rnns-lstms/README.md) | **The Season's Memory** | RNNs, LSTMs, gates | A season ledger unrolls; a hidden lantern carries the player's form from match 1 to match 38 |
| 9 | [ch09-metrics](../ML/ch09-metrics/README.md) | **The Report Card** | Accuracy, precision, recall, F1, ROC | A confusion matrix rendered as a 2×2 report card of signings vs rejections |
| 10 | [ch10-classical-classifiers](../ML/ch10-classical-classifiers/README.md) | **The Old Guard** | KNN, trees, Naive Bayes | Three grey-haired scouts, each with a different style (nearest neighbour, decision tree, probabilistic gut) |
| 11 | [ch11-svm-ensembles](../ML/ch11-svm-ensembles/README.md) | **The Council of Scouts** | SVM margin; bagging & boosting | A wide margin is drawn on a pitch map between two archetypes; then a round of 100 voices votes |
| 12 | [ch12-clustering](../ML/ch12-clustering/README.md) | **Archetypes** | K-means, hierarchical, DBSCAN | Hundreds of player cards cluster themselves into recognisable roles: poacher, regista, libero |
| 13 | [ch13-dimensionality-reduction](../ML/ch13-dimensionality-reduction/README.md) | **The Radar Chart** | PCA, t-SNE, UMAP | 30 attributes collapse into a six-pointed radar; the same player seen in 30-D and in 2-D |
| 14 | [ch14-unsupervised-metrics](../ML/ch14-unsupervised-metrics/README.md) | **Trust Without Labels** | Silhouette, Davies-Bouldin | The Oracle grades a clustering without ever seeing the answer key |
| 15 | [ch15-mle-loss-functions](../ML/ch15-mle-loss-functions/README.md) | **The Art of Being Wrong** | MLE, loss-to-noise mapping | The scout weighs three different noise models over the same residuals; the loss curve shifts |
| 16 | [ch16-tensorboard](../ML/ch16-tensorboard/README.md) | **The Tactics Board** | TensorBoard, experiment tracking | A giant glass tactics board behind the scout's desk, live-updating with loss curves per signing |
| 17 | [ch17-sequences-to-attention](../ML/ch17-sequences-to-attention/README.md) | **Reading the Play** | Attention mechanism | In the stands, The scout's eyes glow — coloured arrows radiate toward the 3 players that matter in this moment |
| 18 | [ch18-transformers](../ML/ch18-transformers/README.md) | **Full-Pitch View** | Transformer = all-to-all attention | Overhead stadium shot; every player's gaze draws a line to every other, the match seen as a Q·Kᵀ matrix |
| 19 | [ch19-hyperparameter-tuning](../ML/ch19-hyperparameter-tuning/README.md) | **The Training Regimen** | Grid, random, Bayesian search | The scout and the Analyst try 100 pre-season regimens on a practice XI; a bar chart declares the winner |

**Book II cover concept.** The scout, stubble, in a navy technical jacket, an attribute-card of a star player floating open in front of him like a hologram, a wall of printed player cards behind him.

---

## 6 · Book III — The Manager (AI)

**Setting.** The protagonist has been promoted. He walks into a glass-walled "war room" at the training ground: wall-screens showing match data, live injury feeds, transfer rumours, expected-goals maps. The Analyst has built a crew of **AI agents** — each one a specialist. The Oracle is now on the board. The Rival Manager runs a super-club with an even bigger AI setup.

**Running metaphor.** Every AI concept is a specialist in the war room. LLMs are the **pundits**. RAG is the **archive vault**. ReAct is the **scout-who-acts**. Agents are the **federation**.

| # | Folder | Manga chapter title | Concept dramatised | Key visual |
|---|---|---|---|---|
| 1 | [AIPrimer.md](../AI/AIPrimer.md) | **The War Room** | Overview of the AI stack | Wide establishing shot of the war room — screens, whiteboards, five silhouettes of specialist agents |
| 2 | [LLMFundamentals](../AI/LLMFundamentals/) | **The Pundit** | LLM basics, tokens, context | A broadcaster-figure made of text tokens streams commentary from a screen, reading word-by-word |
| 3 | [PromptEngineering](../AI/PromptEngineering/) | **Briefing the Pundit** | Prompt design, few-shot | The coach hands the Pundit a clipboard; the clearer the brief, the sharper the commentary returned |
| 4 | [RAGAndEmbeddings](../AI/RAGAndEmbeddings/) | **The Vault** | RAG pipeline, embeddings | A vault of filing cabinets; each drawer glows with coloured vectors; the Pundit retrieves only three folders before answering |
| 5 | [VectorDBs](../AI/VectorDBs/) | **The Archive** | Vector databases, ANN search | Infinite starfield of points, one query-star pulls in its six nearest neighbours at the speed of light |
| 6 | [FineTuning](../AI/FineTuning/) | **The Club's DNA** | Fine-tuning, LoRA, adapters | The generic Pundit steps into a tailor's shop; the Analyst stitches club colours and playing style into her jacket |
| 7 | [CoTReasoning](../AI/CoTReasoning/) | **The Step-by-Step Scout** | Chain-of-thought reasoning | The Scout-agent draws her reasoning on a glass board — five numbered steps lit one at a time |
| 8 | [ReActAndSemanticKernel](../AI/ReActAndSemanticKernel/) | **The Acting Scout** | ReAct loop, tool use | The Scout-agent reasons, *then reaches through the screen* to pull a live stat from an external feed |
| 9 | [EvaluatingAISystems](../AI/EvaluatingAISystems/) | **Post-Match Review** | Evaluation, benchmarks, LLM-as-judge | The coach sits with the Oracle grading each agent's call from last week against outcomes |
| 10 | [SafetyAndHallucination](../AI/SafetyAndHallucination/) | **The Overconfident Pundit** | Hallucination, guardrails | The Pundit confidently invents a transfer rumour; an alarm flashes red; The coach installs a guard-rail |
| 11 | [CostAndLatency](../AI/CostAndLatency/) | **The Budget** | Token cost, latency, caching | A split screen — left: the Rival's super-club GPU farm; right: the coach's lean, cached, quantised setup |
| 12 | [AgenticAI_ReadingMap](../AI/AgenticAI_ReadingMap.md) | **The Federation** | Agent teams, A2A, MCP, orchestration | Full-page spread of the war room at full strength — seven agents, each with a specialty, the coach at the centre tactics-board |

**Book III cover concept.** The coach in a charcoal technical coat in the glass war-room at night, seven agent-silhouettes behind him in a line like a starting XI, stadium lights bleeding through the glass.

---

## 7 · Perchance Prompt Catalogue

One composite multi-panel prompt per chapter — each produces a single manga page that captures the whole chapter at once. Three book covers are single-page hero shots.

### 7.0 · Shared anchors (paste once per generation)

**Style prefix** (prepend to every prompt below):
```
Young-adult shōnen seinen manga, full colour, clean confident linework,
cinematic lighting, light cel shading. Style references: Blue Lock for
tactics energy, Slam Dunk / Real by Takehiko Inoue for grounded athlete
portraits, Haikyuu!! final-arc colour palette. Grounded, mature, not
childish, not grim. 2:3 portrait aspect ratio. Empty speech bubbles —
text will be added later, do not render any text or equations.
```

**Negative prompt** (append to every generation):
```
chibi, mascot, cute, childish, horror, gritty, sparkles, heavy shadows,
monochrome, black and white, low detail, disfigured hands, extra fingers,
text, gibberish letters, watermark, logo, signature
```

**Character anchors** (reuse verbatim wherever a character appears):

- **The Striker (Book I, age 16):** lean teenage footballer, messy dark hair, intense dark brown eyes, white technical training jacket with thin navy piping, black shorts, athletic tape on left ankle, left-footed.
- **The Scout (Book II, age 27):** same man aged up, short dark hair, light stubble, composed expression, fitted navy technical jacket, scout lanyard around neck, leather notebook in hand.
- **The Coach (Book III, age 35):** same man again, short dark hair, clean-shaven, quietly authoritative, tailored charcoal technical coat, silver watch, club crest pin on lapel.
- **The Gaffer:** silver-haired veteran coach in his sixties, navy tracksuit, whistle, always holding a clipboard, kind-but-stern eyes.
- **The Analyst:** woman late twenties, oval glasses, oversized grey hoodie, laptop tucked under arm, black over-ear headphones around neck.
- **The Oracle:** weathered grey-coated scout in his sixties, flat cap, small leather notebook, eyes that look past people.
- **The Rival Manager:** sharp forties, pinstripe charcoal suit, sunglasses worn indoors, smug half-smile.

---

### 7.1 · Book I — The Apprentice

#### Cover
```
[STYLE PREFIX]

Single-page manga cover illustration, full-bleed.

Low-angle hero shot at golden hour on a sun-drenched training ground.
the Striker stands over a ball positioned on the turf, boot drawn back
mid-strike, eyes locked on a distant goal. A defensive wall of four
blurred silhouettes stands in the background at 9 metres; further back,
the 2.44 m crossbar frames his head like a halo. Warm amber rim-light
from the setting sun, long soft shadows across the wood-chip markings.

Floating quietly in the air above him, drawn like elegant chalk-line
strokes: a single glowing trajectory arc (no numbers, no text) curving
from his boot over the wall and dipping under the crossbar. Fine dust
particles lit by the sunset drift around his standing foot.

Mood: quiet confidence, the breath before the strike.

Title area reserved at top (leave a clean rectangular space for the
title to be typeset later). Subtitle area reserved bottom-right.

[NEGATIVE PROMPT]
```

#### Ch.1 — The First 0.1 Seconds *(linear algebra)*
```
[STYLE PREFIX]

Manga page, 4-panel vertical layout, clean white gutters between panels.

PANEL 1 (top wide, establishing): Training ground at late afternoon.
the Striker crouches beside a ball on the turf, running his palm across
the laces, intensely focused. Empty thought bubble above him.

PANEL 2 (middle-left, tight close-up): the striker's planted foot and striking
boot, boot a blur of motion just after contact with the ball. A thin
straight glowing line — ruler-straight, no curve yet — extends from the
boot into the air in the direction of motion. The ball is centimetres
off the turf.

PANEL 3 (middle-right, medium shot): The Gaffer (silver-haired coach in
navy tracksuit) stands beside the striker, holding a clipboard, tapping the
page with a pencil. Empty speech bubble directed at the striker.

PANEL 4 (bottom wide, cinematic): The ball freezes one-tenth of a
second after the strike, still travelling along a perfectly straight
line. Two thin elegant labels float in the margin like annotation arrows
— one pointing at the slope of the line, one at the turf-level origin.
No text, just the annotation shapes.

Consistent character across all panels: The Striker (Book I — lean teenage
footballer, messy dark hair, white technical jacket, black shorts,
athletic tape on left ankle).

[NEGATIVE PROMPT]
```

#### Ch.2 — Gravity's Arc *(non-linear algebra / parabolas)*
```
[STYLE PREFIX]

Manga page, 3-panel layout: one top wide, one tall middle, one bottom
wide. Clean white gutters.

PANEL 1 (top wide): the Striker at the moment of strike, body torqued
through the ball, turf churning beneath his planted foot.

PANEL 2 (middle, tall dramatic): Full trajectory shown across a night-
lit training ground — the ball arcs in a perfect parabola from the striker's
boot, over a four-man defensive wall at 9 metres, and dips under the
crossbar at 20 metres. The arc itself is rendered as a glowing translucent
curve, softly illuminated. Stars above, floodlights along the edges.

PANEL 3 (bottom wide): Close-up of the Gaffer unfurling a chalkboard
sketch. On the chalkboard: three clean curves — a flat horizontal line,
a straight diagonal line, and a bold parabola — drawn in contrasting
chalk colours. His finger taps the parabola. the striker leans in, studying.

Consistent character across all panels: the Striker. Also: the Gaffer
(silver-haired coach, navy tracksuit, clipboard).

[NEGATIVE PROMPT]
```

#### Ch.3 — Slicing Time *(derivatives and integrals)*
```
[STYLE PREFIX]

Manga page, 4-panel vertical layout.

PANEL 1 (top wide): Slow-motion freeze of a knuckleball mid-flight
against a dusk sky. Around the ball, three semi-transparent chord-lines
of different slopes collapse onto one sharp tangent line — drawn in
descending opacity to suggest the limit being taken.

PANEL 2 (middle-left): The Gaffer holds a stopwatch to the air, his
expression serious, arm outstretched. The stopwatch face is blank —
no numbers.

PANEL 3 (middle-right): Below the flight path, warm orange translucent
rectangles fill the area under the curve like Riemann slabs, each
rectangle slimmer than the last from left to right, illustrating an
integral accumulating.

PANEL 4 (bottom wide, symbolic): Split-image: left half shows the
tangent line kissing the trajectory at its apex; right half shows the
stacked orange rectangles under the same curve. A thin double-headed
arrow connects the two halves, signalling they are two faces of the
same coin.

Consistent characters: the Striker, the Gaffer.

[NEGATIVE PROMPT]
```

#### Ch.4 — The Angle Game *(gradient descent / ascent)*
```
[STYLE PREFIX]

Manga page, 4-panel layout (2×2 grid).

PANEL 1 (top-left): the Striker strikes a free kick at a steep upward
angle; the ball sails pathetically short. He frowns.

PANEL 2 (top-right): Same the striker strikes a second attempt almost flat
along the turf; ball skims under the wall and into their shins. He
winces.

PANEL 3 (bottom-left): Third attempt at an angle in between; ball arcs
cleanly over the wall. His eyes widen with realisation.

PANEL 4 (bottom-right, overhead diagram view): A stylised top-down
curve on a chalkboard showing launch angle on the horizontal axis and
range on the vertical; three bright dots plot the striker's attempts, each a
step closer to the peak. A dashed arrow traces the path of improvement.

Consistent character: the Striker. Mood: determined, learning-by-doing.

[NEGATIVE PROMPT]
```

#### Ch.5 — The Notebook *(matrices, design matrix)*
```
[STYLE PREFIX]

Manga page, 3-panel layout: one top wide, one large centre, one bottom
wide.

PANEL 1 (top wide): The Gaffer sets a thick leather-bound notebook onto
a wooden bench. The cover is worn, embossed with a club crest. the striker
(Book I) stands beside him, curious.

PANEL 2 (centre, dramatic): The notebook opens and its pages explode
upward in a tornado of individual sheets — each sheet a neat record of
one free kick with columns of numbers. The sheets reorganise mid-air
into a clean glowing rectangular grid suspended between the striker and the
Gaffer — a design matrix visualisation, rows stacking cleanly.

PANEL 3 (bottom wide): The glowing grid folds down into a single
elegant page in the Gaffer's hand. He offers it to the striker. On the page:
one clean parabola drawn in thick ink, perfectly fitting a scatter of
500 small dots.

Consistent characters: the Striker, the Gaffer.

[NEGATIVE PROMPT]
```

#### Ch.6 — Eight Knobs *(gradient, Jacobian, chain rule)*
```
[STYLE PREFIX]

Manga page, 2-panel layout: one large top panel, one wide bottom panel.

PANEL 1 (large top): the Striker stands before a studio-style mixing
console mounted on the sideline of the pitch. The console has eight
glowing dials arranged in a row, each labelled with a different icon
(a boot, a wind-arrow, a wall, a raindrop, an angle, a stopwatch, a
flame for fatigue, a heart). His hands hover over two of the dials.
Small coloured light-trails radiate off each dial suggesting subtle
adjustments.

PANEL 2 (wide bottom, cinematic): Above the pitch, a vector field of
glowing arrows blankets the air — every arrow points toward a single
peak in the distance where the ball is sailing cleanly through the
goal. the striker stands small in the foreground, silhouetted against this
field, understanding it for the first time.

Consistent character: the Striker. Mood: awe, comprehension.

[NEGATIVE PROMPT]
```

#### Ch.7 — The Missed Kick *(probability, MLE, noise)*
```
[STYLE PREFIX]

Manga page, 4-panel vertical layout.

PANEL 1 (top wide): the Striker strikes a free kick confidently;
technique looks identical to his best attempts.

PANEL 2 (middle-left): The ball sails wide of the goal and thuds into
the side netting. His shoulders drop.

PANEL 3 (middle-right, diagrammatic): Overhead view of the goal. The
intended target point is marked with an X at the top-right corner.
Around it, a soft translucent bell-shaped cloud fades outward in warm
orange, showing the ghost-distribution of where the ball could have
landed across many attempts. The actual impact point sits at the edge
of the cloud.

PANEL 4 (bottom wide): The Gaffer crouches beside the striker on the turf,
pointing at the orange distribution cloud floating in the air above
them. His expression is gentle but honest. the striker listens.

Consistent characters: the Striker, the Gaffer. Mood: reflective,
teaching moment.

[NEGATIVE PROMPT]
```

---

### 7.2 · Book II — The Scout

#### Cover
```
[STYLE PREFIX]

Single-page manga cover illustration, full-bleed.

The Scout (Book II — 27, stubble, navy technical jacket, scout lanyard) sits
in a dim scouting office at night. A desk lamp casts a warm pool of
light across his face and hands. In front of him, suspended in the air
like a hologram, floats a glowing six-sided radar chart — a FIFA-style
attribute card for a player (six axes: pace, shooting, passing,
dribbling, defending, physical). The chart is translucent, edges
catching the lamp glow.

The wall behind the scout is covered in dozens of printed player cards pinned
with coloured thumb-tacks, arranged in tactical formations on a giant
cork board. A half-empty coffee mug steams on the desk. A laptop
screen glows faintly at his elbow.

Mood: analytical, patient, post-career.

Title area reserved at top. Subtitle area reserved bottom-right.

[NEGATIVE PROMPT]
```

#### Ch.1 — Market Value *(linear regression)*
```
[STYLE PREFIX]

Manga page, 3-panel layout: top wide, middle wide, bottom wide.

PANEL 1 (top wide): the Scout sits at his desk, pen poised over a
scatter plot on paper. The plot shows fifty dots with player ages on
the horizontal axis and market values on the vertical axis.

PANEL 2 (middle wide): Close-up of the scout's hand drawing a clean diagonal
line through the cloud of dots. The line glows faintly blue as it
appears, cleaving the scatter into "above" and "below".

PANEL 3 (bottom wide): the scout holds the completed chart up in front of
him. Behind the chart, softly ghosted, a young prospect player jogs on
a training pitch — the dot representing him sits well above the scout's
trend line, circled in red ink.

Consistent character: the Scout.

[NEGATIVE PROMPT]
```

#### Ch.2 — The Transfer Deal *(logistic regression)*
```
[STYLE PREFIX]

Manga page, 4-panel layout.

PANEL 1 (top wide): the Scout faces a club director across a
polished boardroom table. Between them, a manila folder sits closed.

PANEL 2 (middle-left, diagrammatic): A glowing sigmoid S-curve floats
above the folder, with the horizontal axis labelled only by small
player-silhouette icons increasing in rating. A clean threshold line
cuts the curve at its midpoint, separating "pass" (left, dim) from
"sign" (right, warm glow).

PANEL 3 (middle-right): the scout's finger taps a point on the curve far to
the right, in the warm zone. His expression: decided.

PANEL 4 (bottom wide): A handshake between the scout and the director, close-
up on the clasped hands with the folder now open beside them.

Consistent character: the Scout.

[NEGATIVE PROMPT]
```

#### Ch.3 — The Unscoutable Talent *(XOR)*
```
[STYLE PREFIX]

Manga page, 4-panel layout (2×2 grid).

PANEL 1 (top-left): Two older scouts in grey coats frown at a player
profile pinned to a board. The profile shows a 2×2 matrix of four
player dots in the classic XOR pattern — diagonals are signings,
off-diagonals are passes. The scouts try to draw a single straight
line through it and fail.

PANEL 2 (top-right): The scouts throw up their hands and walk away,
muttering.

PANEL 3 (bottom-left): the Scout steps up to the same board,
tilting his head at the pattern. The Analyst (late twenties, oval
glasses, grey hoodie, headphones around neck) stands beside him with
her laptop open, pointing at something on the screen.

PANEL 4 (bottom-right, dramatic reveal): The 2×2 pattern resolves — two
curved decision boundaries (not straight lines) loop around the diagonal
dots in glowing colour. the scout and the Analyst exchange a knowing look.

Consistent characters: the Scout, the Analyst.

[NEGATIVE PROMPT]
```

#### Ch.4 — The Scout Committee *(MLPs)*
```
[STYLE PREFIX]

Manga page, 2-panel layout: one large top panel, one wide bottom.

PANEL 1 (large top): A round wooden table in a warmly lit room. Eight
diverse scout figures sit around it in layered arcs — three in the
front row, three behind, two at the back — each whispering into the
ear of the scout in front of them. Coloured thread-lines of light
connect every scout to every other scout in the adjacent layer,
forming a glowing spider-web of connections.

PANEL 2 (wide bottom): At the front of the table, the Scout and
the Analyst read the final verdict on a single card that has emerged
from the committee. The card glows with a composite colour blended
from all eight scouts' threads.

Consistent characters: the Scout, the Analyst.

[NEGATIVE PROMPT]
```

#### Ch.5 — The Feedback Loop *(backprop, optimisers)*
```
[STYLE PREFIX]

Manga page, 4-panel layout.

PANEL 1 (top wide): Newspaper headline on a desk — a signed player is
shown failing in a big match. the Scout stares at it, jaw tight.

PANEL 2 (middle-left): the scout sits at the committee table from Ch.4, but
now the glowing thread-lines reverse direction — signals flow from the
front of the table backward through each layer of scouts, each scout
recoiling slightly as the "error" reaches them.

PANEL 3 (middle-right, close-up): A single scout's notes visibly
rewrite themselves in real time, pencil marks moving on the page. He
looks startled, then thoughtful.

PANEL 4 (bottom wide): The Analyst at her laptop, soft blue screen
light on her face. Behind her on the wall, a loss curve drops from
high to low across a chart.

Consistent characters: the Scout, the Analyst.

[NEGATIVE PROMPT]
```

#### Ch.6 — The Discipline *(regularisation, dropout)*
```
[STYLE PREFIX]

Manga page, 3-panel layout.

PANEL 1 (top wide): The Analyst stands in front of the cork-board wall
of printed player cards. She holds a black marker. Her posture is
deliberate, almost ceremonial.

PANEL 2 (middle wide, dramatic): She draws dark Xs over roughly thirty
percent of the pinned cards in a random pattern, crossing them out.
The remaining cards glow a little more brightly in response, as if
compensating.

PANEL 3 (bottom wide): the Scout watches from the doorway, arms
folded, the faintest smile on his face. Empty speech bubble between
them.

Consistent characters: the Scout, the Analyst.

[NEGATIVE PROMPT]
```

#### Ch.7 — Heatmap Analysis *(CNNs)*
```
[STYLE PREFIX]

Manga page, 4-panel layout.

PANEL 1 (top wide): Overhead view of a football pitch divided into a
grid of small rectangular tiles. Each tile is shaded a different
intensity of warm colour showing where a specific player spent time
during a match — a heatmap.

PANEL 2 (middle-left): A glowing translucent square — a convolutional
filter — hovers over a small cluster of tiles, reading them as a
single local pattern. It slides to the next cluster, then the next,
leaving faint trails.

PANEL 3 (middle-right): The filter's outputs aggregate into smaller
summary tiles in a second layer above the pitch, fewer and brighter.

PANEL 4 (bottom wide): the Scout and the Analyst study the
summarised heatmap on a tablet. The final image shows a clear player-
role signature — a false-nine pattern — highlighted by a glowing outline.

Consistent characters: the Scout, the Analyst.

[NEGATIVE PROMPT]
```

#### Ch.8 — The Season's Memory *(RNNs, LSTMs)*
```
[STYLE PREFIX]

Manga page, 3-panel layout.

PANEL 1 (top wide): A long horizontal ledger unrolls across a wooden
table like an ancient scroll. Each segment of the ledger represents
one match of a season, 38 segments in total.

PANEL 2 (middle wide, cinematic): Along the ledger, a small glowing
lantern-figure walks from segment to segment, carrying a warm flame
that represents the player's form. At some segments the flame brightens,
at others it dims, but it is always the same flame — continuous memory.

PANEL 3 (bottom wide): the Scout and the Analyst stand at the end
of the ledger, watching the lantern-figure arrive. The final segment
glows the warmest, and the figure hands the flame to them.

Consistent characters: the Scout, the Analyst. Mood: reverent,
almost fable-like.

[NEGATIVE PROMPT]
```

#### Ch.9 — The Report Card *(metrics)*
```
[STYLE PREFIX]

Manga page, 4-panel layout (2×2 grid).

PANEL 1 (top-left): the Scout at his desk holding a 2×2 grid drawn
on a white card — a confusion matrix. The four cells are colour-coded:
true positive (green), true negative (green), false positive (red),
false negative (red).

PANEL 2 (top-right): Close-up of his pen circling the false-negative
cell — a player he passed on who later succeeded. His expression is
pained but honest.

PANEL 3 (bottom-left): Beside the matrix, a clean ROC curve rises from
bottom-left to top-right on graph paper, the area under it shaded
softly.

PANEL 4 (bottom-right): the scout pins the report card to the cork board
next to a printed player's photo. The card has no text, just the grid
and the curve.

Consistent character: the Scout.

[NEGATIVE PROMPT]
```

#### Ch.10 — The Old Guard *(KNN, trees, Naive Bayes)*
```
[STYLE PREFIX]

Manga page, 3-panel layout.

PANEL 1 (top wide): Three senior scouts stand side by side in a
stadium corridor, each distinct: one looks through a monocle at a
cluster of player photos pinned close together (nearest-neighbour);
one holds a flowchart branching into decisions on a clipboard (decision
tree); one rubs his chin with probabilistic gut instinct, eyes half-
closed (Naive Bayes). Each is lit by a subtly different coloured
light.

PANEL 2 (middle wide): The three turn to face the Scout who stands
across from them, leather notebook in hand. An empty speech bubble from
each scout, directed at the scout.

PANEL 3 (bottom wide): the scout writes a single line in his notebook while
the three scouts watch approvingly. The old guard nods — tradition
honoured.

Consistent character: the Scout.

[NEGATIVE PROMPT]
```

#### Ch.11 — The Council of Scouts *(SVM, ensembles)*
```
[STYLE PREFIX]

Manga page, 3-panel layout.

PANEL 1 (top wide, overhead): Top-down view of a training pitch. Two
archetypal player-shapes are scattered across it in two colours — warm
orange (signings) and cool blue (passes). Between them, a wide bright
corridor runs edge to edge — the maximum-margin lane — flanked by two
thin support vectors touching the nearest orange and blue dots.

PANEL 2 (middle wide): Around a massive circular war-room table, a
hundred miniature scout-silhouettes raise small voting paddles in
unison. A vote tally counter above the table clicks upward.

PANEL 3 (bottom wide): the Scout at the head of the table, reading
the aggregated verdict on a single card. Empty speech bubble above him.

Consistent character: the Scout.

[NEGATIVE PROMPT]
```

#### Ch.12 — Archetypes *(clustering)*
```
[STYLE PREFIX]

Manga page, 2-panel layout: one large top, one wide bottom.

PANEL 1 (large top, overhead): Hundreds of small player-card
silhouettes float in the air like a swarm of fireflies. Slowly, they
drift and coalesce into distinct glowing clusters — each cluster a
different warm colour — each shaped vaguely like the role it
represents (a compact forward-shape, a long midfielder-shape, a dense
defender-shape).

PANEL 2 (wide bottom): the Scout and the Analyst walk through the
cluster-garden, pointing out each grouping. Labels float faintly above
each cluster — the Poacher, the Regista, the Libero, the Box-to-Box —
rendered only as empty placeholder banners (no readable text).

Consistent characters: the Scout, the Analyst.

[NEGATIVE PROMPT]
```

#### Ch.13 — The Radar Chart *(dimensionality reduction)*
```
[STYLE PREFIX]

Manga page, 3-panel layout.

PANEL 1 (top wide): On the left, a dense unreadable table of numbers —
thirty columns of attribute data for a single player. the Scout
looks at it overwhelmed.

PANEL 2 (middle wide, dramatic): The thirty columns collapse inward
through a glowing funnel, reforming on the other side as a clean six-
axis radar chart — the familiar FIFA hexagon — glowing softly with the
player's shape.

PANEL 3 (bottom wide): the scout holds the radar chart in his hands,
recognising the player at a glance. Behind him on the wall, a 2-D
scatter plot shows hundreds of players dotted, the same player
highlighted among them — a t-SNE-style constellation.

Consistent character: the Scout.

[NEGATIVE PROMPT]
```

#### Ch.14 — Trust Without Labels *(unsupervised metrics)*
```
[STYLE PREFIX]

Manga page, 3-panel layout.

PANEL 1 (top wide): The Oracle (weathered grey-coated scout, flat cap,
notebook) stands in front of a cluster-garden — the same glowing
groupings from Ch.12 — but without any labels attached to them.

PANEL 2 (middle wide): He walks slowly between two clusters, measuring
the gap between them with an outstretched hand like a sculptor judging
proportion. Faint glowing distance-lines appear in the air where his
hand moves — inter-cluster separation visualised.

PANEL 3 (bottom wide): The Oracle writes a single number in his
notebook, nods once, and hands the book to the Scout. No labels
have been revealed.

Consistent characters: the Scout, the Oracle.

[NEGATIVE PROMPT]
```

#### Ch.15 — The Art of Being Wrong *(MLE, loss functions)*
```
[STYLE PREFIX]

Manga page, 4-panel layout.

PANEL 1 (top wide): the Scout stands before a wall of residual-
dots pinned to a board — dozens of small "error" markers representing
how wrong past predictions were.

PANEL 2 (middle-left): Above the dots, three different translucent
bell-curves float side by side — a tall narrow Gaussian, a flatter
Laplace, a skewed asymmetric one — each in a different warm colour.

PANEL 3 (middle-right): the scout selects one curve with his hand; the
residuals on the wall reshade themselves subtly to match its profile.

PANEL 4 (bottom wide): The corresponding loss-shape appears on a
chalkboard behind him — a clean parabolic bowl when the Gaussian is
picked — signalling the choice of noise model has chosen the loss.

Consistent character: the Scout.

[NEGATIVE PROMPT]
```

#### Ch.16 — The Tactics Board *(experiment tracking / TensorBoard)*
```
[STYLE PREFIX]

Manga page, 2-panel layout: one large top, one wide bottom.

PANEL 1 (large top, wide room-shot): A floor-to-ceiling glass tactics
board covers the back wall of a dim scouting office. Across its
surface, dozens of live loss curves, accuracy charts, and scatter plots
glow in different colours, each labelled with a small placeholder
banner. The whole wall pulses with gentle animation-lines suggesting
live data.

PANEL 2 (wide bottom): the Scout and the Analyst stand facing the
glass wall, the cold blue light washing over their faces. She points
to one curve that has begun to rise above the others. He nods.

Consistent characters: the Scout, the Analyst.

[NEGATIVE PROMPT]
```

#### Ch.17 — Reading the Play *(attention)*
```
[STYLE PREFIX]

Manga page, 3-panel layout.

PANEL 1 (top wide): the Scout sits high in the stands of a
stadium, watching a live match unfold below. He is still, eyes wide
open.

PANEL 2 (middle wide, cinematic): The scout's eyes glow faintly. From his
eye-line, a fan of coloured attention-arrows radiates across the pitch
below — most arrows fade, but three arrows burn bright, each pointing
at a different player in the current build-up. The rest of the twenty-
two players are dimmed.

PANEL 3 (bottom wide): The Analyst beside him glances at her tablet —
the same three arrows have been logged as a weighted distribution in a
bar chart on her screen, the three bright players dominating.

Consistent characters: the Scout, the Analyst.

[NEGATIVE PROMPT]
```

#### Ch.18 — Full-Pitch View *(transformers)*
```
[STYLE PREFIX]

Manga page, 2-panel layout: one very large top, one wide bottom.

PANEL 1 (very large top, god's-eye): Overhead stadium view at night
under floodlights. On the pitch, every one of the twenty-two players
has faint coloured gaze-lines connecting them to every other player —
a dense glowing web of attention across the entire field. The
geometry forms an elegant all-to-all lattice.

PANEL 2 (wide bottom): the Scout stands on the touchline at pitch
level, head slightly raised, taking the whole pattern in at once.
Empty thought bubble above him.

Consistent character: the Scout. Mood: revelation.

[NEGATIVE PROMPT]
```

#### Ch.19 — The Training Regimen *(hyperparameter tuning)*
```
[STYLE PREFIX]

Manga page, 4-panel layout.

PANEL 1 (top wide): The training ground. the Scout and the Analyst
set up a small grid of cones on the turf — a literal grid search laid
out across the pitch.

PANEL 2 (middle-left): A practice XI of young players run drills; each
cone corresponds to a different training regimen being tested.

PANEL 3 (middle-right): The Analyst scatters a few cones at random
elsewhere on the pitch — a handful of random-search trials breaking
the grid pattern.

PANEL 4 (bottom wide): A bar chart appears in the air above the pitch.
One bar stands taller than the rest, glowing gold — the winning
regimen.

Consistent characters: the Scout, the Analyst.

[NEGATIVE PROMPT]
```

---

### 7.3 · Book III — The Manager

#### Cover
```
[STYLE PREFIX]

Single-page manga cover illustration, full-bleed.

The Coach (Book III — 35, charcoal technical coat, clean-shaven, club crest
pin) stands at the centre of a glass-walled war-room at night. Stadium
floodlights bleed through the glass behind him, casting long cool
shadows across the floor. Behind him, seven humanoid agent-silhouettes
stand in a line like a starting XI, each rendered in a different warm
accent colour, each with a distinct visual signature: one is a
broadcaster made of streaming text, one is a vault with glowing
drawers, one is a star-chart navigator, one is a tailor with thread,
one is a step-by-step chalkboard, one is a reaching arm of light, one
is a shielded guardian.

At the coach's centre: a transparent tactics-desk glows with live match
data, heatmaps, and xG charts. His hand rests lightly on the desk.

Mood: command, composure, a quiet orchestra.

Title area reserved at top. Subtitle area reserved bottom-right.

[NEGATIVE PROMPT]
```

#### Ch.1 — The War Room *(AI overview)*
```
[STYLE PREFIX]

Manga page, 2-panel layout: one very wide establishing, one wide
bottom.

PANEL 1 (very wide top, establishing): Wide shot of a glass-walled
war-room at the training ground. Wall-screens display live match
feeds, injury lists, transfer tickers, heatmaps. Five humanoid agent-
silhouettes stand at separate workstations, each doing their own task
in quiet focus — one reading text, one pulling vectors from a vault-
drawer, one drawing reasoning on glass, one reaching through a screen,
one watching for errors.

PANEL 2 (wide bottom): the Coach walks into frame from the
doorway, charcoal coat flaring slightly. The agents do not look up;
they continue their work. He pauses, taking it all in.

Consistent character: the Coach.

[NEGATIVE PROMPT]
```

#### Ch.2 — The Pundit *(LLM fundamentals)*
```
[STYLE PREFIX]

Manga page, 3-panel layout.

PANEL 1 (top wide): A broadcast-booth screen on the war-room wall. On
the screen, a humanoid pundit figure is visible, but her body is
composed entirely of drifting text-tokens — small rectangular tiles of
language, each a different warm pastel colour, flowing like a river
through her outline.

PANEL 2 (middle wide): the Coach stands in front of the screen,
arms crossed, listening. A trail of coloured token-tiles streams from
the pundit's mouth and floats into his empty thought bubble.

PANEL 3 (bottom wide, close-up): A single token-tile falls into the coach's
open palm and dissolves into light. His expression: considering.

Consistent character: the Coach. The Pundit is a humanoid figure
made of streaming text-tokens in warm pastels — reuse this description
in every Book III chapter where she appears.

[NEGATIVE PROMPT]
```

#### Ch.3 — Briefing the Pundit *(prompt engineering)*
```
[STYLE PREFIX]

Manga page, 4-panel layout.

PANEL 1 (top wide): the Coach hands a clipboard across a desk to
the Pundit (humanoid figure made of streaming text-tokens). The
clipboard is blank — no text on it.

PANEL 2 (middle-left): The Pundit returns commentary in a thin,
scattered, pale stream of tokens that drift apart and dissipate.

PANEL 3 (middle-right): the coach hands her a second clipboard, more densely
detailed in visual weight (still blank of readable text). This time
the returning stream of tokens is thick, bright, tightly ordered.

PANEL 4 (bottom wide): The Analyst stands nearby, holding three
example clipboards arranged like trading cards — few-shot examples —
watching the coach refine his briefing technique.

Consistent characters: the Coach, the Analyst, the Pundit.

[NEGATIVE PROMPT]
```

#### Ch.4 — The Vault *(RAG, embeddings)*
```
[STYLE PREFIX]

Manga page, 2-panel layout: one large top, one wide bottom.

PANEL 1 (large top, cavernous): A vast underground vault stretching
into darkness, lined with thousands of shallow metal filing drawers.
Each drawer glows softly with a different coloured vector-bar along
its front edge — like tiny embedded spectra. The air itself shimmers
with indexed light.

PANEL 2 (wide bottom): The Pundit (streaming-tokens figure) walks
between the rows. A beam of pale gold light extends from her head out
to three specific drawers, which obediently slide open. She retrieves
only those three folders, leaves the rest untouched, and walks back.
the Coach watches from the mouth of the vault.

Consistent characters: the Coach, the Pundit.

[NEGATIVE PROMPT]
```

#### Ch.5 — The Archive *(vector databases, ANN)*
```
[STYLE PREFIX]

Manga page, 3-panel layout.

PANEL 1 (top wide): Infinite starfield — a cosmic 3-D space dotted
with millions of tiny glowing points, each point a different colour.
The Analyst stands on a small platform in the centre, laptop open.

PANEL 2 (middle wide): A bright query-star forms in the Analyst's
hand. She releases it; the star shoots outward and instantly pulls six
of the nearest stars toward her in a soft tractor-beam of light —
nearest neighbours arriving as a tight cluster.

PANEL 3 (bottom wide): The six retrieved stars resolve into six small
floating cards of player information in her palm. the Coach stands
beside her, nodding once.

Consistent characters: the Coach, the Analyst.

[NEGATIVE PROMPT]
```

#### Ch.6 — The Club's DNA *(fine-tuning, LoRA)*
```
[STYLE PREFIX]

Manga page, 4-panel layout.

PANEL 1 (top wide): The Pundit (streaming-tokens figure) stands in a
plain grey robe in a tailor's workshop. Her figure is generic, uncoloured.

PANEL 2 (middle-left): The Analyst approaches with a spool of glowing
thread in the club's warm accent colour.

PANEL 3 (middle-right): The Analyst stitches the coloured thread
across the Pundit's robe in a careful embroidered pattern — no large
redesign, just thin tracks of club colour along the seams (LoRA
adapters as surgical additions).

PANEL 4 (bottom wide): The Pundit emerges transformed — her streaming
tokens now carry the club's colour woven through them. the Coach
nods approvingly in the doorway.

Consistent characters: the Coach, the Analyst, the Pundit.

[NEGATIVE PROMPT]
```

#### Ch.7 — The Step-by-Step Scout *(chain-of-thought)*
```
[STYLE PREFIX]

Manga page, 2-panel layout: one large top, one wide bottom.

PANEL 1 (large top): A floor-to-ceiling transparent glass board in the
war-room. A humanoid Scout-agent (sleek, featureless silver-outlined
figure with a single glowing orange eye-slit) draws her reasoning across
the glass in warm marker. Five numbered step-blocks are visible, each
step illuminated a fraction later than the one before it, a chain of
glowing causal arrows connecting them.

PANEL 2 (wide bottom): the Coach stands on the opposite side of
the glass, reading the reasoning from left to right, eye following
each step in turn.

Consistent character: the Coach. Reuse the Scout-agent design
(silver-outlined humanoid, single orange eye-slit) in Ch.8.

[NEGATIVE PROMPT]
```

#### Ch.8 — The Acting Scout *(ReAct loop, tool use)*
```
[STYLE PREFIX]

Manga page, 4-panel layout.

PANEL 1 (top wide): The Scout-agent (silver-outlined humanoid, single
orange eye-slit) stands in front of a live data screen showing a player
in action.

PANEL 2 (middle-left): She draws a reasoning step on a glass board
beside her — one glowing numbered block.

PANEL 3 (middle-right, dramatic): She reaches her arm directly through
the data screen, her hand emerging on the other side to pluck a glowing
statistic-token from a live match feed. The screen glass ripples around
her wrist like liquid light.

PANEL 4 (bottom wide): She withdraws the stat-token and adds it as a
new reasoning block on the glass board. the Coach watches,
satisfied.

Consistent characters: the Coach, the Scout-agent.

[NEGATIVE PROMPT]
```

#### Ch.9 — Post-Match Review *(evaluation)*
```
[STYLE PREFIX]

Manga page, 3-panel layout.

PANEL 1 (top wide): A dim briefing room. the Coach and the Oracle
(weathered scout, grey coat, flat cap) sit across from each other at a
long table. Between them, a stack of small cards — each card a
decision one of the AI agents made during last week's match.

PANEL 2 (middle wide): The Oracle picks up each card in turn and
places it into one of two piles — a green "right call" pile and a red
"wrong call" pile. The scale of his judgement is clearly ruthless but
fair.

PANEL 3 (bottom wide): the coach holds the final tally — a small scoreboard
card summarising the grading. He and the Oracle exchange a measured
look.

Consistent characters: the Coach, the Oracle.

[NEGATIVE PROMPT]
```

#### Ch.10 — The Overconfident Pundit *(hallucination, guardrails)*
```
[STYLE PREFIX]

Manga page, 4-panel layout.

PANEL 1 (top wide): The Pundit (streaming-tokens figure) stands on her
broadcast screen, gesturing confidently. Her tokens stream brightly.

PANEL 2 (middle-left): A fabricated transfer-rumour card appears in
her hand — glowing suspiciously magenta, the colour of invention.

PANEL 3 (middle-right, alarm): A red alarm-light strobes on the war-
room wall. the Coach snaps his head up from his desk, expression
sharpening.

PANEL 4 (bottom wide): the coach installs a translucent guard-rail — a pane
of thin glowing shield — directly in front of the Pundit's broadcast
screen. The magenta fabrication card hits the shield and dissolves
harmlessly.

Consistent characters: the Coach, the Pundit.

[NEGATIVE PROMPT]
```

#### Ch.11 — The Budget *(cost and latency)*
```
[STYLE PREFIX]

Manga page, 2-panel split-screen layout (left half vs right half).

LEFT HALF: A vast industrial GPU-farm hall at the Rival Manager's
super-club — rows upon rows of humming server racks, cold blue light,
excessive, wasteful. The Rival Manager (sharp forties, pinstripe suit,
sunglasses indoors) stands at the entrance with a smug half-smile,
arms folded.

RIGHT HALF: The coach's (Book III) lean setup — a single elegant workstation
in a warm-lit room. A small local server hums quietly. A cache-shelf
behind him glows with pre-computed results in neat jars. The Analyst
sits beside him, content. The whole scene is composed, efficient, warm.

A thin vertical divider line separates the two halves, with a faint
balance-scale icon at the top centre tipping slightly toward the coach's
side.

Consistent characters: the Coach, the Analyst, the Rival Manager.

[NEGATIVE PROMPT]
```

#### Ch.12 — The Federation *(agents, A2A, MCP, orchestration)*
```
[STYLE PREFIX]

Single full-page manga spread, no panel division — one grand tableau.

The war-room at full operational strength. the Coach stands at
the centre of a transparent tactics-desk, hands resting on its glowing
surface. Arranged in an arc around him, seven specialist agents work
in concert: the Pundit (streaming-tokens figure) on her broadcast
screen; the Scout-agent (silver-outlined, orange eye-slit) at her
glass reasoning board; a Vault-keeper agent pulling glowing drawers
from the wall behind her; an Archive-navigator standing in a pool of
starfield light; a Tailor-agent with threads of club colour between
her fingers; a Guardian-agent holding a translucent shield panel; a
Budget-agent balancing small weighted tokens in both hands.

Thin coloured lines of communication run between every pair of agents
and into the centre of the coach's tactics-desk — an agent-to-agent mesh.
The whole scene is lit warmly from within the desk and coolly from
the glass walls behind; floodlights from a distant stadium seep
through.

Mood: orchestra at full tempo, quietly triumphant.

[NEGATIVE PROMPT]
```

